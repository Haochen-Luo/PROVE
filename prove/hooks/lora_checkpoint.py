"""Adapter-only checkpoints for verl 0.8 with FSDP (on unless PROVE_FULL_CKPT=1).

verl's ``FSDPCheckpointManager`` saves ``model.state_dict()``, which for a LoRA run is the whole frozen base
model next to a small adapter and optimizer state. Newer verl has a native ``save_lora_only`` option; this is
the same idea for verl 0.8.0.

  save: the per-rank model shard keeps only adapter entries; optimizer and extra state are saved unchanged.
        A PEFT adapter (``adapter_model.safetensors`` + ``adapter_config.json``) is written to
        ``<checkpoint>/actor/lora_adapter/`` for evaluation; it holds the same tensors that are synced to vLLM.
  load: a shard whose keys are all adapter keys is merged into the current state dict (the base weights are the
        frozen pretrained ones) and fails if any adapter key of the model is missing. Full checkpoints load as
        upstream.
"""
from __future__ import annotations

import json
import os

from prove.hooks._post_import import patch_after_import

_MODULE = "verl.utils.checkpoint.fsdp_checkpoint_manager"
_MARKER = "lora_only.json"


def _is_adapter_key(key: str) -> bool:
    return "lora_" in key or ".adapter_" in key


def _enabled() -> bool:
    return os.environ.get("PROVE_FULL_CKPT", "0") != "1"


def _has_lora(model) -> bool:  # noqa: ANN001
    unwrap = getattr(model, "_fsdp_wrapped_module", model)
    return hasattr(unwrap, "peft_config") or any("lora_" in n for n, _ in model.named_parameters())


def _export_adapter(model, local_path: str, rank: int) -> None:  # noqa: ANN001
    from prove.hooks.lora_sync import layered_summon_lora_params

    params = layered_summon_lora_params(model)  # collective: every rank must call it
    if rank != 0:
        return
    try:
        _write_adapter(params, local_path)
    except Exception as e:  # noqa: BLE001  (an export for evaluation must never break checkpointing)
        print(f"[prove] WARNING LoRA adapter export failed: {e!r}", flush=True)


def _write_adapter(params, local_path: str) -> None:  # noqa: ANN001
    from safetensors.torch import save_file

    out = os.path.join(local_path, "lora_adapter")
    os.makedirs(out, exist_ok=True)
    save_file({k: v.contiguous() for k, v in params.items()}, os.path.join(out, "adapter_model.safetensors"))
    meta_path = os.path.join(local_path, "lora_train_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    targets = sorted({k.split(".lora_")[0].rsplit(".", 1)[-1] for k in params})
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": meta.get("r"), "lora_alpha": meta.get("lora_alpha"),
           "target_modules": targets, "lora_dropout": 0.0, "bias": "none",
           "base_model_name_or_path": os.environ.get("MODEL", "")}
    json.dump(cfg, open(os.path.join(out, "adapter_config.json"), "w"), indent=2)
    print(f"[prove] LoRA adapter exported: {len(params)} tensors -> {out}", flush=True)


def _patch(module) -> None:  # noqa: ANN001
    cls = getattr(module, "FSDPCheckpointManager", None)
    if cls is None or getattr(cls.save_checkpoint, "_prove", False):
        return
    orig_save, orig_load = cls.save_checkpoint, cls.load_checkpoint

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):  # noqa: ANN001
        if _enabled() and getattr(self, "should_save_hf_model", False):
            print("[prove] WARNING hf_model saving is on: adapter-only save disabled for this checkpoint", flush=True)
        if not (_enabled() and self.model is not None and _has_lora(self.model)
                and not getattr(self, "should_save_hf_model", False)):
            return orig_save(self, local_path, hdfs_path=hdfs_path, global_step=global_step,
                             max_ckpt_to_keep=max_ckpt_to_keep)
        model, real_sd = self.model, self.model.state_dict

        def lora_state_dict(*a, **k):  # noqa: ANN002, ANN003
            sd = real_sd(*a, **k)
            keep = type(sd)((key, v) for key, v in sd.items() if _is_adapter_key(key))
            if not keep:
                raise RuntimeError("[prove] adapter-only checkpoint found no lora_ keys")
            return keep

        model.state_dict = lora_state_dict  # instance attribute shadows nn.Module.state_dict
        try:
            out = orig_save(self, local_path, hdfs_path=hdfs_path, global_step=global_step,
                            max_ckpt_to_keep=max_ckpt_to_keep)
        finally:
            del model.state_dict
        _export_adapter(model, local_path, self.rank)
        if self.rank == 0:
            json.dump({"lora_only": True, "global_step": global_step}, open(os.path.join(local_path, _MARKER), "w"))
        return out

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):  # noqa: ANN001
        if local_path is None or self.model is None:
            return orig_load(self, local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)
        model, real_load, real_sd = self.model, self.model.load_state_dict, self.model.state_dict

        def merged_load(state_dict, *a, **k):  # noqa: ANN001, ANN002, ANN003
            if not (state_dict and all(_is_adapter_key(key) for key in state_dict)):
                return real_load(state_dict, *a, **k)  # a full checkpoint
            # Under FSDP1 the sharded pre-load hook pops every managed parameter key, so a missing base key
            # would raise even with strict=False. Merge into the current sharded state dict instead.
            full = real_sd()
            unknown = set(state_dict) - set(full)
            if unknown:
                raise RuntimeError(f"[prove] adapter-only checkpoint has keys the model lacks: {sorted(unknown)[:3]}")
            missing = {key for key in full if _is_adapter_key(key)} - set(state_dict)
            if missing:  # a partial restore would silently keep freshly initialised adapter weights
                raise RuntimeError(f"[prove] adapter-only checkpoint misses {len(missing)} adapter keys, "
                                   f"e.g. {sorted(missing)[:3]}")
            full.update(state_dict)
            print(f"[prove] adapter-only checkpoint: restored {len(state_dict)} entries (rank {self.rank})", flush=True)
            return real_load(full, *a, **k)

        model.load_state_dict = merged_load
        try:
            return orig_load(self, local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)
        finally:
            del model.load_state_dict

    save_checkpoint._prove = True
    cls.save_checkpoint, cls.load_checkpoint = save_checkpoint, load_checkpoint
    print(f"[prove] {_MODULE}.FSDPCheckpointManager: adapter-only checkpoints "
          f"{'on' if _enabled() else 'off (PROVE_FULL_CKPT=1)'}", flush=True)


def apply() -> None:
    patch_after_import("lora_checkpoint", {_MODULE: _patch})
