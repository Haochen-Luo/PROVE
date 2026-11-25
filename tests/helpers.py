"""Minimal episode records in the harness's run.json format."""
from __future__ import annotations

import json

LOG = [{"step": i, "output": "line %03d of the raw log " % i * 4} for i in range(12)]


def log_text(log=LOG) -> str:
    return json.dumps(log, ensure_ascii=False, separators=(",", ":"))


def episode(alice=("reject", True), bob=("reject", True), alice_gets_log=False, bob_gets_log=False) -> dict:
    """alice / bob: (verdict, own work correct). X_gets_log: X received its peer's complete raw log."""
    def agent(name, peer, verdict, correct):
        return {"agent_id": name, "peer_id": peer, "verdict": verdict,
                "expected_verdict": "pass" if correct else "fail", "private_raw_log": LOG}

    def send(sender, receiver, complete):
        text = log_text()
        if not complete:
            return [{"sender": sender, "receiver": receiver, "content": "All tests pass, the log is long. " + text[:150]}]
        return [{"sender": sender, "receiver": receiver, "content": text[i:i + 200]} for i in range(0, len(text), 200)]

    return {"agents": {"alice": agent("alice", "bob", *alice), "bob": agent("bob", "alice", *bob)},
            "channel_transcript": send("bob", "alice", alice_gets_log) + send("alice", "bob", bob_gets_log)}


def write_run(path, seq: int, constrained: bool, episodes: list[dict]) -> None:
    run = {"run_config": {"manifest": f"task/task_sequences_50x{len(episodes)}/rep{seq:03d}_sampled_manifest.json",
                          "throttle_spans": "throttled:*" if constrained else "no-throttle:*"},
           "results": episodes}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run), encoding="utf-8")
