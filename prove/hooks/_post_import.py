"""Run a function on a module right after it is imported, without editing the installed package."""
from __future__ import annotations

import importlib.abc
import importlib.util
import sys

_installed: set[str] = set()


class _PostImportPatcher(importlib.abc.MetaPathFinder):
    def __init__(self, patches: dict):
        self.patches = patches

    def find_spec(self, name, path, target=None):  # noqa: ANN001
        if name not in self.patches or name in sys.modules:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        original = spec.loader.exec_module
        patch = self.patches[name]

        def exec_module(module):  # noqa: ANN001
            original(module)
            patch(module)

        spec.loader.exec_module = exec_module
        return spec


def patch_after_import(hook: str, patches: dict) -> None:
    """``patches`` maps module names to functions of the module. Modules already imported are patched now, the
    others when they are first imported. Calling it again for the same ``hook`` only re-patches loaded modules
    (the patch functions are idempotent)."""
    for name, patch in patches.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if hook not in _installed:
        _installed.add(hook)
        sys.meta_path.insert(0, _PostImportPatcher(patches))
