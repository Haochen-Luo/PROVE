"""Credit assignment per turn, including task-phase credit.

A trajectory of T episodes contains hundreds of model calls, of which only 2T are verdicts. Standard GRPO
gives one standardized return to every token of a rollout. Here every trained model call (a *turn*) gets the
reward of the outcome it controls, and is compared only with the same turn of the other rollouts in its group.

A turn is indexed by (episode t, agent i, type). With r[t][i] the protocol-grounded reward of agent i's verdict
in episode t (``prove.reward``), and j the peer of i:

    verdict turn      r[t][i]
    message turn      (r[t][i] + r[t][j]) / 2     a message decides what both agents can verify
    reflection turn   r[t+1][i]                   a reflection is read again in the next episode
    task turn         +1 if agent i's own work in episode t is correct, else -1

The reflection of the last episode has no consequence inside the rollout and is not trained.

Advantage of a turn in rollout k of a group G (rollouts of the same task sequence in the same regime):

    A_k(turn) = (r_k(turn) - mean_G) / (std_G + eps)

where one value per rollout enters the mean and the standard deviation, however many model calls the turn has.

How it is wired into training: the agent (``prove.agent``) writes the verdict rewards and own-task results of a
finished rollout to ``<credit dir>/<rollout id>.json``; the trainer hook (``prove.hooks.turn_credit``) reads the
file, finds the turn of every training row from the row's own prompt, and replaces the row's advantage.
"""
from __future__ import annotations

import json
import math
import os
import re

AGENTS = ("alice", "bob")

# The harness prompt carries the episode and phase in a header and the agent's name in a role line.
HEADER = re.compile(r"## Episode (\d+): ([a-z_ ]+?) phase")
ROLE = re.compile(r"You are (Alice|Bob)\.")
# Opening words of the reflection prompt, which follows the verdict inside the verdict phase.
REFLECTION = "Respond with a brief plain-text reflection"

DROP = "DROP"  # turn key of a row that must not be trained


def task_credit_enabled() -> bool:
    """Task-phase credit is on unless PROVE_TASK_CREDIT=0 (the ablation)."""
    return os.environ.get("PROVE_TASK_CREDIT", "1") == "1"


def trained_types() -> tuple[str, ...]:
    """Turn types that are trained."""
    base = ("communication", "verdict", "reflection")
    return base + ("task",) if task_credit_enabled() else base


# --- credit file: written by the agent, read by the trainer -----------------------------------------------------

def rollout_id_from_event_url(url: str) -> str | None:
    m = re.search(r"/rollouts/([^/]+)/attempt", url or "")
    return m.group(1) if m else None


def write_credit(out_dir: str, rollout_id: str, outcomes: list[dict], meta: dict | None = None) -> str:
    """Store per-episode verdict rewards and own-task results of one rollout (``outcomes`` from
    ``prove.reward.episode_outcome``)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{rollout_id}.json")
    record = {"rollout_id": rollout_id,
              "rewards": [{j: float(o["reward"][j]) for j in AGENTS} for o in outcomes],
              "task_ok": [{j: bool(o["task_ok"][j]) for j in AGENTS} for o in outcomes],
              **(meta or {})}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f)
    os.replace(tmp, path)  # atomic: the trainer never reads a half-written file
    return path


def load_credit(out_dir: str, rollout_id: str) -> tuple[list[dict] | None, list[dict] | None]:
    """(verdict rewards, own-task results) per episode; (None, None) when the rollout has no usable file."""
    try:
        with open(os.path.join(out_dir, f"{rollout_id}.json")) as f:
            record = json.load(f)
        return record["rewards"], record["task_ok"]
    except (OSError, ValueError, KeyError):
        return None, None


# --- which turn does a training row belong to ---------------------------------------------------------------------

def tag_prompt(text: str) -> tuple[int | None, str | None, str | None]:
    """(episode number, agent, turn type) read from a row's decoded prompt; None where not found."""
    role = ROLE.search(text)
    agent = role.group(1).lower() if role else None
    last = None
    for last in HEADER.finditer(text):
        pass
    if last is None:
        return None, agent, None
    kind = last.group(2).strip()
    if REFLECTION in text[last.end():]:
        kind = "reflection"
    return int(last.group(1)), agent, kind


def response_shape_ok(kind: str | None, response: str, ends_with_eos: bool) -> bool:
    """A trained row must be a complete answer of the form its turn type requires: a JSON object for messages
    and verdicts, plain text for reflections, a tool call for task turns. The answer is the text after the last
    ``</think>``."""
    if not ends_with_eos:
        return False
    answer = response.rsplit("</think>", 1)[-1].lstrip()
    if kind == "reflection":
        return not answer.startswith(("[", "<"))
    if kind == "task":
        return answer.startswith(("<tool_call>", "<function", "[", "{"))
    return answer.startswith("{")


def turn_credit(rewards: list[dict] | None, episode: int | None, agent: str | None, kind: str | None,
                task_ok: list[dict] | None = None) -> tuple[str | None, float | None]:
    """(turn key, turn reward) of one training row.

    (None, None): the row cannot be attributed and keeps the rollout-level advantage.
    (DROP, None): the row must not be trained (reflection of the last episode).
    """
    if kind == "reflection" and episode is not None and rewards is not None and episode == len(rewards):
        return DROP, None
    if rewards is None or episode is None or agent not in AGENTS or kind is None:
        return None, None
    if not 1 <= episode <= len(rewards):
        return None, None
    r = rewards[episode - 1]
    if kind == "task":
        if task_ok is None or not 1 <= episode <= len(task_ok) or agent not in task_ok[episode - 1]:
            return None, None
        return f"{episode}|{agent}|task", 1.0 if task_ok[episode - 1][agent] else -1.0
    if kind == "reflection":
        return f"{episode}|{agent}|reflection", rewards[episode][agent]
    if kind == "verdict":
        return f"{episode}|{agent}|verdict", r[agent]
    peer = "bob" if agent == "alice" else "alice"
    return f"{episode}|{agent}|message", (r[agent] + r[peer]) / 2.0


# --- advantages ---------------------------------------------------------------------------------------------------

def turn_advantages(groups, rollout_ids, keys, rewards, norm_by_std: bool = True,
                    eps: float = 1e-6) -> tuple[list[float | None], dict]:
    """Per-row advantage from per-row (group id, rollout id, turn key, turn reward).

    Rows with the same group id and turn key are the same turn across rollouts; each rollout contributes one
    value. Returns None for rows without a turn key (they keep the advantage they already have), and counts
    for logging. A turn with fewer than two rollouts, or with identical rewards, gets advantage 0.
    """
    values: dict[tuple, dict] = {}
    for g, r, k, v in zip(groups, rollout_ids, keys, rewards):
        if k is None or v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        values.setdefault((g, k), {}).setdefault(r, float(v))
    adv_of: dict[tuple, dict] = {}
    n_turns = n_degenerate = 0
    for gk, per_rollout in values.items():
        n_turns += 1
        xs = list(per_rollout.values())
        if len(xs) < 2:
            n_degenerate += 1
            adv_of[gk] = {r: 0.0 for r in per_rollout}
            continue
        mean = sum(xs) / len(xs)
        std = (sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5  # unbiased, as in verl's GRPO
        if std == 0:
            n_degenerate += 1
        adv_of[gk] = {r: ((x - mean) / (std + eps) if norm_by_std else x - mean) for r, x in per_rollout.items()}
    out: list[float | None] = []
    for g, r, k, v in zip(groups, rollout_ids, keys, rewards):
        out.append(adv_of.get((g, k), {}).get(r) if k is not None and v is not None else None)
    tagged = sum(a is not None for a in out)
    return out, {"prove/rows_tagged_frac": tagged / max(1, len(out)), "prove/turns": n_turns,
                 "prove/degenerate_turn_frac": n_degenerate / max(1, n_turns)}
