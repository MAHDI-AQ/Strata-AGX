#!/usr/bin/env python3
"""Backwards-compat shim -> ../terminal/tools/needle_bench.py"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parents[1] / "terminal" / "tools" / "needle_bench.py"
if str(_TARGET.parent) not in sys.path:
    sys.path.insert(0, str(_TARGET.parent))

if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
