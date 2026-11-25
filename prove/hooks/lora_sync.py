"""Collect the trained LoRA weights for the sync to vLLM (verl 0.7 / 0.8, FSDP1).

verl's LoRA wrap policy makes every trainable LoRA leaf (``...q_proj.lora_A.default``) its own FSDP unit. The
upstream ``layered_summon_lora_params`` summons decoder layers and filters their state dicts with PEFT's
adapter-name filter, which never matches those unit-relative names, so it returns an empty adapter: vLLM then
either fails or keeps serving the base model, and rollouts never see the trained weights.

This collector gathers each LoRA leaf by its full module name and refuses to return a partial adapter. Keys
follow PEFT's adapter state-dict format (``base_model.model.<path>.lora_A.weight``).
"""
from __future__ import annotations

import re
from collections import OrderedDict

from prove.hooks._post_import import patch_after_import

_LEAF = re.compile(r"\.(lora_A|lora_B)\.[^.]+$")
_MODULES = ("verl.utils.fsdp_utils", "verl.workers.fsdp_workers")
_logged = False


def _clean(name: str) -> str:
    return ".".join(p for p in name.split(".") if p != "_fsdp_wrapped_module")


def _peft_key(clean_name: str) -> str:
    return _LEAF.sub(lambda m: f".{m.group(1)}.weight", clean_name)


def layered_summon_lora_params(fsdp_module) -> OrderedDict:  # noqa: ANN001
    import torch.nn as nn
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from verl.utils.device import get_torch_device
    from verl.utils.fsdp_utils import fsdp_version

    expected = {_clean(n) for n, m in fsdp_module.named_modules()
                if isinstance(m, nn.Linear) and _LEAF.search(_clean(n))}
    lora_params: OrderedDict = OrderedDict()
    for name, unit in fsdp_module.named_modules():
        if unit is fsdp_module or fsdp_version(unit) == 0:
            continue
        clean = _clean(name)
        if not _LEAF.search(clean):
            continue
        with FSDP.summon_full_params(unit, writeback=False, recurse=False):
            w = unit.module.weight
            w = w.full_tensor() if hasattr(w, "full_tensor") else w
            lora_params[_peft_key(clean)] = w.detach().cpu().clone()
        # summon_full_params lazily initialises this unit as a root; reset it (as upstream does), or the next
        # root forward asserts that a non-root instance has _is_root set.
        unit._is_root = False
    got = set(lora_params)
    want = {_peft_key(c) for c in expected}
    if got != want:
        missing = sorted(want - got)[:5]
        raise RuntimeError(f"[prove] LoRA collector got {len(got)}/{len(want)} adapter tensors; "
                           f"missing e.g. {missing} (LoRA leaves not wrapped as their own FSDP units?)")
    get_torch_device().empty_cache()
    global _logged
    if not _logged:
        _logged = True
        print(f"[prove] LoRA collector: {len(lora_params)} adapter tensors", flush=True)
    return lora_params


def _patch(module) -> None:  # noqa: ANN001
    current = getattr(module, "layered_summon_lora_params", None)
    if current is not None and getattr(current, "__module__", "") != __name__:
        module.layered_summon_lora_params = layered_summon_lora_params
        print(f"[prove] {module.__name__}.layered_summon_lora_params replaced", flush=True)


def apply() -> None:
    patch_after_import("lora_sync", {name: _patch for name in _MODULES})
