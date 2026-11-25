"""Trainer hooks for turn-level credit and for removing truncated responses.

Two functions of Agent Lightning's verl integration are wrapped; the installed package is not modified.

``RolloutAdapter.get_train_data_batch`` (builds the training rows of an update):
  * rows whose response filled the whole generation budget are truncated completions; they are dropped
    (the overlong filter of DAPO, Yu et al. 2025);
  * every row is tagged with its turn (episode, agent, type) read from its own prompt, and with the turn's
    reward read from the rollout's credit file (``prove.credit``); rows of a type that is not trained, and
    rows whose answer does not have the form its type requires, are dropped.

``compute_rollout_level_advantage`` / ``compute_advantage`` (trainer): the advantage of every tagged row is
replaced by the turn-level advantage, standardized across the rollouts of the row's group.

Active when PROVE_CREDIT_DIR is set; without it training uses the rollout-level advantage for every row.
"""
from __future__ import annotations

import os

from prove.hooks._post_import import patch_after_import

_ADAPTER = "agentlightning.verl.rollout_adapter"
_TRAINER = "agentlightning.verl.trainer"


def mark_overlong(batch, max_response_length: int) -> int:  # noqa: ANN001
    """Drop rows whose response has max_response_length tokens; returns how many rows were newly dropped."""
    attn = batch["attention_mask"]
    resp_len = attn[:, -max_response_length:].sum(-1)
    overlong = resp_len >= max_response_length
    drop = batch["is_drop_mask"].bool()
    newly = int((overlong & ~drop).sum().item())
    batch["is_drop_mask"] = drop | overlong
    return newly


def _decode_tail_for_header(tok, ids: list[int]) -> str:  # noqa: ANN001
    """Decode the shortest prompt suffix that contains an episode header (prompts can be very long)."""
    from prove.credit import HEADER

    for n in (4096, 16384, 65536):
        text = tok.decode(ids[-n:], skip_special_tokens=False)
        if HEADER.search(text) or n >= len(ids):
            return text
    return tok.decode(ids, skip_special_tokens=False)


def tag_rows(adapter, data_proto, credit_dir: str) -> dict:  # noqa: ANN001
    """Attach prove_turn_key / prove_turn_reward to every row and drop the rows that must not be trained."""
    import numpy as np
    import torch

    from prove.credit import DROP, load_credit, response_shape_ok, tag_prompt, trained_types, turn_credit

    tok, b, nt = adapter.tokenizer, data_proto.batch, data_proto.non_tensor_batch
    R = int(adapter.max_response_length)
    prompt_attn, resp_attn = b["attention_mask"][:, :-R], b["attention_mask"][:, -R:]
    eos_ids = {i for i in (getattr(tok, "eos_token_id", None), tok.convert_tokens_to_ids("<|im_end|>")) if i is not None}
    rids = nt["rollout_id_list"]
    credit_of = {r: load_credit(credit_dir, str(r)) for r in set(rids.tolist())}
    types = trained_types()

    keys, rewards, drop = [], [], []
    n_untagged = n_reflection = n_shape = 0
    type_counts: dict[str, int] = {}
    for i in range(len(b)):
        p_ids = b["prompts"][i][prompt_attn[i].bool()].tolist()
        head = tok.decode(p_ids[:1024], skip_special_tokens=False)
        ep, _, kind = tag_prompt(_decode_tail_for_header(tok, p_ids))
        _, agent, _ = tag_prompt(head)
        type_counts[str(kind)] = type_counts.get(str(kind), 0) + 1
        verdict_rewards, task_ok = credit_of[rids[i]]
        k, v = turn_credit(verdict_rewards, ep, agent, kind, task_ok if "task" in types else None)
        d = k == DROP or kind not in types
        if not d:
            r_ids = b["responses"][i][resp_attn[i].bool()].tolist()
            ok = response_shape_ok(kind, tok.decode(r_ids, skip_special_tokens=False),
                                   bool(r_ids) and r_ids[-1] in eos_ids)
            d = not ok
            n_shape += not ok
        if d:
            k, v = None, None
        n_untagged += k is None and not d
        n_reflection += bool(k) and k.endswith("|reflection")
        keys.append(k)
        rewards.append(float("nan") if v is None else v)
        drop.append(d)
    nt["prove_turn_key"] = np.array(keys, dtype=object)
    nt["prove_turn_reward"] = np.array(rewards, dtype=np.float64)
    dm = torch.tensor(drop, dtype=torch.bool, device=b["is_drop_mask"].device)
    newly = int((dm & ~b["is_drop_mask"].bool()).sum())
    b["is_drop_mask"] = b["is_drop_mask"].bool() | dm
    return {"prove/rollouts_with_credit": sum(v[0] is not None for v in credit_of.values()),
            "prove/rollouts": len(credit_of), "prove/rows_untagged": n_untagged,
            "prove/rows_reflection": n_reflection, "prove/rows_type_dropped": newly,
            "prove/rows_shape_dropped": n_shape,
            **{f"prove/rows_by_type/{k}": v for k, v in type_counts.items()}}


def apply_turn_advantages(batch, norm_by_std: bool) -> dict:  # noqa: ANN001
    """Overwrite advantages (and returns) of tagged rows with turn-level advantages."""
    import torch

    from prove.credit import turn_advantages

    b, nt = batch.batch, batch.non_tensor_batch
    keys = list(nt["prove_turn_key"])
    rewards = [None if v != v else float(v) for v in nt["prove_turn_reward"]]
    adv, m = turn_advantages(list(nt["uid"]), list(nt["rollout_id_list"]), keys, rewards, norm_by_std)
    mask = b["response_mask"].to(b["advantages"].dtype)
    old_row = [float(b["advantages"][i][mask[i].bool()][0]) if mask[i].any() else 0.0 for i in range(len(b))]
    for i, a in enumerate(adv):
        if a is not None:
            b["advantages"][i] = a * mask[i]
            if "returns" in b.keys():
                b["returns"][i] = a * mask[i]
    pairs = [(o, a) for o, a in zip(old_row, adv) if a is not None]
    if len(pairs) > 2:
        xo = torch.tensor([p[0] for p in pairs])
        xa = torch.tensor([p[1] for p in pairs])
        m["prove/corr_with_rollout_advantage"] = float(torch.corrcoef(torch.stack([xo, xa]))[0, 1])
        m["prove/advantage_abs_mean"] = float(xa.abs().mean())
    return m


def _patch_adapter(module) -> None:  # noqa: ANN001
    cls = getattr(module, "RolloutAdapter", None)
    if cls is None or getattr(cls.get_train_data_batch, "_prove", False):
        return
    orig = cls.get_train_data_batch

    def get_train_data_batch(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
        data_proto, metrics = orig(self, *a, **k)
        n = mark_overlong(data_proto.batch, int(self.max_response_length))
        metrics["training/prove_overlong_dropped"] = n
        print(f"[prove] overlong filter: dropped {n}/{len(data_proto.batch)} rows "
              f"(response >= {self.max_response_length} tokens)", flush=True)
        credit_dir = os.environ.get("PROVE_CREDIT_DIR")
        if credit_dir:  # turn-level credit is part of the objective: fail loudly, never skip silently
            m = tag_rows(self, data_proto, credit_dir)
            metrics.update(m)
            print(f"[prove] turn credit: {m}", flush=True)
        return data_proto, metrics

    get_train_data_batch._prove = True
    cls.get_train_data_batch = get_train_data_batch
    print(f"[prove] {_ADAPTER}.RolloutAdapter: overlong filter on, turn credit "
          f"{'on' if os.environ.get('PROVE_CREDIT_DIR') else 'off (PROVE_CREDIT_DIR not set)'}", flush=True)


def _patch_trainer(module) -> None:  # noqa: ANN001
    for name in ("compute_rollout_level_advantage", "compute_advantage"):
        fn = getattr(module, name, None)
        if fn is None or getattr(fn, "_prove", False):
            continue

        def wrapped(*a, _fn=fn, **k):  # noqa: ANN002, ANN003
            out = _fn(*a, **k)
            batch = out[0] if isinstance(out, tuple) else out
            if "prove_turn_key" in batch.non_tensor_batch:
                m = apply_turn_advantages(batch, bool(k.get("norm_adv_by_std_in_grpo", True)))
                if isinstance(out, tuple) and isinstance(out[1], dict):
                    out[1].update(m)
                print(f"[prove] turn advantages: {m}", flush=True)
            return out

        wrapped._prove = True
        setattr(module, name, wrapped)


def apply() -> None:
    patch_after_import("turn_credit", {_ADAPTER: _patch_adapter, _TRAINER: _patch_trainer})
