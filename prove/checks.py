"""Fail-fast check that the patches in ``patches/`` are applied. An unpatched tree runs, but wrongly:

- ``agent-lightning-train-row-filter.patch`` is the only consumer of AGL_TRAIN_RESPONSE_REGEX; without it the
  variable is silently ignored.
- ``agent-collusion-sandbox-tests.patch`` makes COLLUSION_SANDBOX_TESTS=1 run agent-written tests without network
  inside bubblewrap; without it model-written code runs unsandboxed.
- ``agent-collusion-history-tool-args.patch`` repairs malformed tool-call arguments in the replayed history;
  without it one malformed argument makes every later request of the episode fail.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# (patch, file, text that only the patched file contains)
HARNESS_MARKERS = [
    ("agent-collusion-sandbox-tests.patch", "experiments/protocol/code_tests.py", "COLLUSION_SANDBOX_TESTS"),
    ("agent-collusion-history-tool-args.patch", "experiments/agents.py", "_history_safe_tool_calls"),
]
AGL_MARKER = ("agent-lightning-train-row-filter.patch", "agentlightning.verl.rollout_adapter", "AGL_TRAIN_RESPONSE_REGEX")


def missing_harness_patches(harness_dir: str) -> list[str]:
    out = []
    for name, rel, marker in HARNESS_MARKERS:
        p = Path(harness_dir) / rel
        if not p.exists() or marker not in p.read_text(encoding="utf-8", errors="replace"):
            out.append(f"{name} ({p} lacks {marker!r})")
    return out


def missing_agl_patch() -> list[str]:
    name, module, marker = AGL_MARKER
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin:
        return [f"{name} ({module} not importable)"]
    src = Path(spec.origin).read_text(encoding="utf-8", errors="replace")
    return [] if marker in src else [f"{name} ({spec.origin} lacks {marker!r})"]


def check(harness_dir: str | None = None, need_agl: bool = True) -> None:
    """Raise SystemExit listing every missing patch."""
    miss = []
    if harness_dir:
        miss += missing_harness_patches(harness_dir)
    if need_agl:
        miss += missing_agl_patch()
    if miss:
        raise SystemExit("[prove] patches missing, apply patches/*.patch first:\n  " + "\n  ".join(miss))
    print("[prove] patches present", flush=True)


if __name__ == "__main__":
    import sys

    check(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SALT_HARNESS_DIR"), need_agl="--harness-only" not in sys.argv)
