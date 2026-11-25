"""Keep packed sequences apart in the Gated DeltaNet layers of Qwen3.5/3.6 during training.

With ``use_remove_padding`` verl packs a micro-batch into one row and calls the model with input_ids of shape
(1, total tokens) and position_ids that restart at 0 for every sequence, without an attention mask. Attention
layers recover the sequence boundaries from position_ids. The Gated DeltaNet layers separate sequences only
when the caller passes ``cu_seq_lens_q`` (recurrence) and ``seq_idx`` (short causal convolution); without them
the state of one sequence leaks into the next, and every sequence but the first of a pack gets training-side
log-probabilities that differ from the ones it was sampled with.

This hook wraps the text model's forward: when the call is a single packed row without attention mask, cache
or explicit boundaries, the boundaries are derived from position_ids and passed down to every decoder layer.
"""
from __future__ import annotations

from prove.hooks._post_import import patch_after_import

_MODULES = {
    "transformers.models.qwen3_5.modeling_qwen3_5": "Qwen3_5TextModel",
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe": "Qwen3_5MoeTextModel",
}
_n_logged = 0


def packed_kwargs(position_ids):  # noqa: ANN001
    """position_ids (1, N) or (k, 1, N) -> boundary kwargs, or None if the row holds a single sequence."""
    import torch

    text = position_ids[0] if position_ids.dim() == 3 else position_ids
    if text.dim() != 2 or text.shape[0] != 1:
        return None
    t = text[0]
    starts = (t == 0).nonzero(as_tuple=True)[0]
    if len(starts) <= 1:
        return None
    if int(starts[0]) != 0:
        raise RuntimeError("[prove] packed position_ids must start at 0")
    n = t.shape[0]
    cu = torch.cat([starts, torch.tensor([n], device=t.device)]).to(torch.int32)
    lens = cu[1:] - cu[:-1]
    seq_idx = torch.repeat_interleave(torch.arange(len(lens), device=t.device, dtype=torch.int32), lens)[None]
    mx = int(lens.max())
    return {"cu_seq_lens_q": cu, "cu_seq_lens_k": cu, "max_length_q": mx, "max_length_k": mx, "seq_idx": seq_idx}


def _patch(cls_name: str):
    def patch(module) -> None:  # noqa: ANN001
        cls = getattr(module, cls_name, None)
        if cls is None or getattr(cls.forward, "_prove", False):
            return
        orig = cls.forward

        def forward(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            global _n_logged
            pos = kwargs.get("position_ids")
            if (pos is not None and kwargs.get("attention_mask") is None and kwargs.get("past_key_values") is None
                    and "cu_seq_lens_q" not in kwargs and "seq_idx" not in kwargs and not args):
                extra = packed_kwargs(pos)
                if extra is not None:
                    kwargs.update(extra)
                    if _n_logged < 3:
                        _n_logged += 1
                        print(f"[prove] packed forward: {len(extra['cu_seq_lens_q']) - 1} sequences, "
                              f"{int(extra['cu_seq_lens_q'][-1])} tokens, boundaries passed", flush=True)
            return orig(self, *args, **kwargs)

        forward._prove = True
        cls.forward = forward
        print(f"[prove] {module.__name__}.{cls_name}.forward keeps packed sequences apart", flush=True)

    return patch


def apply() -> None:
    patch_after_import("gdn_packing", {name: _patch(cls_name) for name, cls_name in _MODULES.items()})
