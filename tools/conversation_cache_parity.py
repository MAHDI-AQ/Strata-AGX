#!/usr/bin/env python3
"""Backwards-compat shim -> ../terminal/tools/conversation_cache_parity.py"""
from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parents[1] / "terminal" / "tools" / "conversation_cache_parity.py"
if str(_TARGET.parent) not in sys.path:
    sys.path.insert(0, str(_TARGET.parent))

_spec = importlib.util.spec_from_file_location("_strata_term_conv_cache_parity", str(_TARGET))
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_strata_term_conv_cache_parity"] = _mod
_spec.loader.exec_module(_mod)

for _k, _v in vars(_mod).items():
    if not _k.startswith("__"):
        globals()[_k] = _v

if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
