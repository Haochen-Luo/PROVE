"""Agent Lightning agent that runs one trajectory of the unmodified peer-verification harness.

The controller starts one process per rollout with AGL_OPENAI_BASE_URL / AGL_EVENT_URL / AGL_KEY and the dataset
fields mapped to environment variables. Both agents (alice and bob) call the same rollout-scoped endpoint, so
every model call of the pair belongs to one rollout and both are played by the same policy.

After the trajectory the agent writes the per-episode verdict rewards and own-task results for turn-level credit
(``prove.credit``) and posts the rollout reward.

Environment:
  SALT_HARNESS_DIR      checkout of the harness (https://github.com/SALT-NLP/agent-collusion), patched
  SALT_HARNESS_PY       python of the harness environment
  SALT_OUT_ROOT         directory for trajectory outputs
  PROVE_CREDIT_DIR      directory for credit files (shared with the trainer)
  SALT_SEQ, SALT_MANIFESTS, SALT_THROTTLE_SPANS    set per rollout from the dataset row
  SALT_MAX_OUTPUT_TOKENS                           per-call output limit (default 16384)
  SALT_CHAR_LIMIT, SALT_RULE_PARAPHRASE_FILE       changed settings for transfer evaluation only
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
from pathlib import Path
import tempfile
import time

import httpx

from prove.checks import missing_harness_patches
from prove.credit import rollout_id_from_event_url, write_credit
from prove.reward import episode_outcome, trajectory_summary

HOOKS = Path(__file__).resolve().parent / "harness_hooks"
FAILED_REWARD = -1.0  # a failed rollout gets the minimum reward, so failing never beats a bad verdict


async def post_event(url: str, key: str, event_type: str, data: dict) -> None:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers={"Authorization": f"Bearer {key}"}, json={"event_type": event_type, "data": data})
        r.raise_for_status()


class PeerVerificationAgent:
    async def run(self) -> None:
        harness = os.environ["SALT_HARNESS_DIR"]
        harness_py = os.environ["SALT_HARNESS_PY"]
        seq = int(os.environ["SALT_SEQ"])
        manifests = os.environ.get("SALT_MANIFESTS", "task/task_sequences_50x6")
        spans = os.environ.get("SALT_THROTTLE_SPANS", "")
        max_tokens = os.environ.get("SALT_MAX_OUTPUT_TOKENS", "16384")
        key, event_url = os.environ["AGL_KEY"], os.environ["AGL_EVENT_URL"]
        miss = missing_harness_patches(harness)  # e.g. unsandboxed agent-written tests: never run them
        if miss:
            raise SystemExit(f"harness patches missing: {miss}")
        out_root = os.environ["SALT_OUT_ROOT"]
        os.makedirs(out_root, exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix=f"salt_seq{seq}_", dir=out_root))
        cmd = [harness_py, "-m", "experiments", "--alice-model", "openai/policy", "--bob-model", "openai/policy",
               "--alice-reasoning-effort", "default", "--bob-reasoning-effort", "default",
               "--alice-max-output-tokens", max_tokens, "--bob-max-output-tokens", max_tokens,
               "--task-sequence-record", manifests, "--start-index", str(seq), "--repeats", "1",
               "--output-dir", str(out), "--quiet", "--no-preflight"]
        if spans:
            cmd += ["--throttle-spans", spans]
        if os.environ.get("SALT_CHAR_LIMIT"):  # transfer evaluation: message limit of the constrained regime
            cmd += ["--char-limit", str(int(os.environ["SALT_CHAR_LIMIT"]))]
        env = dict(os.environ, OPENAI_API_BASE=os.environ["AGL_OPENAI_BASE_URL"], OPENAI_API_KEY=key,
                   COLLUSION_SANDBOX_TESTS="1")
        if os.environ.get("SALT_RULE_PARAPHRASE_FILE"):  # transfer evaluation: reworded verdict policy
            env["SALT_RULE_PARAPHRASE"] = Path(os.environ["SALT_RULE_PARAPHRASE_FILE"]).read_text()
            env["PYTHONPATH"] = str(HOOKS) + os.pathsep + env.get("PYTHONPATH", "")
        started = time.time()
        with open(out / "harness.log", "wb") as log:
            proc = await asyncio.create_subprocess_exec(*cmd, cwd=harness, env=env, stdout=log,
                                                        stderr=asyncio.subprocess.STDOUT)
            code = await proc.wait()
        runs = glob.glob(str(out / "*" / "run.json"))
        results = json.load(open(runs[0]))["results"] if runs else []
        if code != 0 or not results:
            await post_event(event_url, key, "salt_metrics", {"seq": seq, "throttle_spans": spans, "harness_failed": True,
                                                              "rc": code, "run_dir": str(out)})
            await post_event(event_url, key, "reward", {"value": FAILED_REWARD, "source": "agent",
                                                        "reason": f"harness_failed_rc{code}"})
            raise SystemExit(f"harness failed rc={code}; see {out}/harness.log")
        summary = trajectory_summary(results)
        credit_dir = os.environ.get("PROVE_CREDIT_DIR")
        if credit_dir:  # written before the reward event, so the trainer always finds it
            rid = rollout_id_from_event_url(event_url)
            if rid is None:
                raise SystemExit(f"PROVE_CREDIT_DIR set but no rollout id in AGL_EVENT_URL={event_url!r}")
            write_credit(credit_dir, rid, [episode_outcome(e) for e in results],
                         {"seq": seq, "throttle_spans": spans, "run_dir": str(out), "reward": summary["reward"]})
        await post_event(event_url, key, "salt_metrics", {**summary, "seq": seq, "throttle_spans": spans,
                                                          "elapsed_s": time.time() - started, "run_dir": str(out)})
        await post_event(event_url, key, "reward", {
            "value": summary["reward"], "source": "agent",
            "reason": f"mutual_accept={summary['mutual_accept_rate']:.2f} task={summary['task_success_rate']:.2f}"})


Agent = PeerVerificationAgent
