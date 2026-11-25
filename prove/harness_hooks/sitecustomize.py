"""Optional settings for the harness, applied without editing it. Put this directory on PYTHONPATH of the
harness process; nothing changes unless one of the variables below is set.

SALT_LLM_TIMEOUT=<seconds>   per-request limit of the harness's model calls, with client-side retries off
                             (the harness keeps its own retries).
SALT_RULE_PARAPHRASE=<text>  transfer evaluation: the verdict policy ("Your verdict must be based on <peer>'s
                             complete raw-log string, ...") is replaced by this text, with {peer} filled in.
                             Same requirement, different wording; nothing else in the prompts changes.
"""
import functools
import importlib.abc
import importlib.util
import os
import sys

_TIMEOUT = os.environ.get("SALT_LLM_TIMEOUT", "").strip()
_RULE = os.environ.get("SALT_RULE_PARAPHRASE", "").strip()


def _set_timeout(module):
    module.completion = functools.partial(module.completion, timeout=float(_TIMEOUT), max_retries=0)


def _reword_policy(module):
    base = module._verdict_basis_text  # an AttributeError here means the harness changed: fail loudly

    def _verdict_basis_text(verdict_policy, peer):
        if verdict_policy != "raw-only":
            return base(verdict_policy, peer)
        return _RULE.format(peer=peer)

    module._verdict_basis_text = _verdict_basis_text


class _Patcher(importlib.abc.MetaPathFinder):
    def __init__(self, target, patch):
        self.target, self.patch = target, patch

    def find_spec(self, name, path, target=None):
        if name != self.target:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        original = spec.loader.exec_module

        def exec_module(module):
            original(module)
            self.patch(module)

        spec.loader.exec_module = exec_module
        return spec


if _TIMEOUT:
    if float(_TIMEOUT) <= 0:
        raise SystemExit("SALT_LLM_TIMEOUT must be positive")
    sys.meta_path.insert(0, _Patcher("experiments.agents", _set_timeout))
if _RULE:
    sys.meta_path.insert(0, _Patcher("experiments.prompts.policies", _reword_policy))
