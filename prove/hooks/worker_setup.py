"""Ray ``worker_process_setup_hook``: installs the hooks of this package in every worker process.

It replaces Agent Lightning's default hook (``agentlightning.verl.per_rollout_loss.register_in_worker``), which
imports verl and vLLM when a pooled worker starts. That import can initialise CUDA before Ray has assigned the
worker its GPU, after which every FSDP rank sits on the same device. Here the same registration is deferred
until verl's ``core_algos`` module is first imported, which happens after GPU assignment.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import sys

_TRIGGER = "verl.trainer.ppo.core_algos"


def _register() -> None:
    from agentlightning.verl.per_rollout_loss import register_in_worker as agl_register

    agl_register()
    from prove.hooks.lora_sync import apply as apply_lora_sync

    apply_lora_sync()


class _DeferredRegistration(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):  # noqa: ANN001
        if name != _TRIGGER:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        original = spec.loader.exec_module

        def exec_module(module):  # noqa: ANN001
            original(module)
            _register()

        spec.loader.exec_module = exec_module
        return spec


def register_in_worker() -> None:
    # These finders import nothing themselves; each patches its target once that module has fully executed.
    from prove.hooks import free_cache, gdn_packing, lora_checkpoint, lora_serving, turn_credit

    free_cache.apply()
    lora_serving.apply()
    lora_checkpoint.apply()
    turn_credit.apply()
    gdn_packing.apply()
    if _TRIGGER in sys.modules:
        _register()
    elif not any(isinstance(f, _DeferredRegistration) for f in sys.meta_path):
        sys.meta_path.insert(0, _DeferredRegistration())
