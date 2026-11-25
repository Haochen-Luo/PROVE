"""Release the trainer's cached CUDA memory right after the policy update (verl 0.8 engine workers).

verl resumes the colocated vLLM weights before it empties the PyTorch cache. After an update on long sequences
the caching allocator still reserves most of the GPU, and vLLM can fail to wake up with an out-of-memory error
although the update itself fitted. Emptying the cache when ``TrainingWorker.train_mini_batch`` returns frees that
memory before the weight sync. It does not change any numerics.
"""
from __future__ import annotations

import functools

from prove.hooks._post_import import patch_after_import

_MODULE = "verl.workers.engine_workers"


def _patch(module) -> None:  # noqa: ANN001
    cls = getattr(module, "TrainingWorker", None)
    if cls is None or getattr(cls.train_mini_batch, "_prove", False):
        return
    original = cls.train_mini_batch

    @functools.wraps(original)  # keeps verl's register()/dispatch attributes
    def train_mini_batch(self, *args, **kwargs):  # noqa: ANN001
        out = original(self, *args, **kwargs)
        from verl.utils.memory_utils import aggressive_empty_cache

        aggressive_empty_cache(force_sync=True)
        return out

    train_mini_batch._prove = True
    cls.train_mini_batch = train_mini_batch
    print(f"[prove] {_MODULE}.TrainingWorker.train_mini_batch frees the CUDA cache after an update", flush=True)


def apply() -> None:
    patch_after_import("free_cache", {_MODULE: _patch})
