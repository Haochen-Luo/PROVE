import importlib
import sys

import pytest

from prove.hooks._post_import import patch_after_import


def test_patch_after_import(tmp_path, monkeypatch):
    (tmp_path / "prove_hook_target.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    calls = []

    def patch(module):
        calls.append(module.VALUE)
        module.VALUE = 2

    patch_after_import("test", {"prove_hook_target": patch})
    assert "prove_hook_target" not in sys.modules and calls == []   # nothing is imported by installing the hook
    module = importlib.import_module("prove_hook_target")
    assert calls == [1] and module.VALUE == 2                        # patched right after the module executed
    patch_after_import("test", {"prove_hook_target": patch})         # loaded modules are patched immediately
    assert calls == [1, 2]
    sys.modules.pop("prove_hook_target")


def test_packed_sequence_boundaries():
    torch = pytest.importorskip("torch")
    from prove.hooks.gdn_packing import packed_kwargs

    # three sequences of 3, 2 and 4 tokens packed into one row
    pos = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]])
    out = packed_kwargs(pos)
    assert out["cu_seq_lens_q"].tolist() == [0, 3, 5, 9]
    assert out["seq_idx"].tolist() == [[0, 0, 0, 1, 1, 2, 2, 2, 2]]
    assert out["max_length_q"] == 4
    assert packed_kwargs(torch.tensor([[0, 1, 2, 3]])) is None       # a single sequence needs nothing
    assert packed_kwargs(torch.tensor([[0, 1], [0, 1]])) is None     # a padded batch is not a packed row


def test_overlong_rows_are_dropped():
    torch = pytest.importorskip("torch")
    from prove.hooks.turn_credit import mark_overlong

    # prompt length 2, response budget 3; row 0 fills the budget, row 1 does not
    batch = {"attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
             "is_drop_mask": torch.tensor([False, False])}
    assert mark_overlong(batch, 3) == 1
    assert batch["is_drop_mask"].tolist() == [True, False]
