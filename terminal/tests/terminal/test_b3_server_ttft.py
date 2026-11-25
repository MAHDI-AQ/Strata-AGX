"""Tests for Wave B3 server.py TTFT/GIL optimizations."""
from __future__ import annotations

import array
import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import server as srv  # noqa: E402


def _full_crc32(tokens: list[int]) -> int:
    return zlib.crc32(array.array("i", tokens).tobytes()) & 0xFFFFFFFF


def test_rolling_slot_held_crc_bit_exact() -> None:
    eng = srv.StrataEngine.__new__(srv.StrataEngine)
    eng.batch = 2
    eng.slot_held = [[] for _ in range(eng.batch)]
    eng.slot_held_crc = [None] * eng.batch

    # Turn 0 (cold): 1000 tokens
    t0 = list(range(100, 1100))
    eng._set_slot_held(0, t0)
    assert eng.slot_held_crc[0] == (len(t0), _full_crc32(t0))
    assert eng._get_slot_held_crc(0, t0) == _full_crc32(t0)

    # Turn 1 (warm delta): append 257 tokens incrementally
    prev_len = len(t0)
    prev_crc = eng._get_slot_held_crc(0, t0)
    t1 = t0 + [9999 + i for i in range(257)]
    eng._set_slot_held(0, t1, prev_len=prev_len, prev_crc=prev_crc)
    assert eng.slot_held_crc[0] == (len(t1), _full_crc32(t1))
    assert eng._get_slot_held_crc(0, t1) == _full_crc32(t1)

    # Turn 2 (warm delta): append 0 tokens (exact prefix)
    prev_len2 = len(t1)
    prev_crc2 = eng._get_slot_held_crc(0, t1)
    t2 = list(t1)
    eng._set_slot_held(0, t2, prev_len=prev_len2, prev_crc=prev_crc2)
    assert eng.slot_held_crc[0] == (len(t2), _full_crc32(t2))

    # Turn 3 (warm delta): append 5000 tokens
    t3 = t2 + [i * 3 + 7 for i in range(5000)]
    eng._set_slot_held(0, t3, prev_len=len(t2), prev_crc=prev_crc2)
    assert eng._get_slot_held_crc(0, t3) == _full_crc32(t3)

    # Shrink / non-append fallback must recompute cleanly
    t_shrink = t0[:500]
    eng._set_slot_held(0, t_shrink, prev_len=len(t3), prev_crc=eng.slot_held_crc[0][1])
    assert eng._get_slot_held_crc(0, t_shrink) == _full_crc32(t_shrink)

    # Clear invalidates cache
    eng._set_slot_held(0, [])
    assert eng.slot_held[0] == []
    assert eng.slot_held_crc[0] is None

    # Stale cache length mismatch recomputes cleanly
    eng.slot_held[1] = [1, 2, 3, 4]
    eng.slot_held_crc[1] = (2, 0xDEADBEEF)
    assert eng._get_slot_held_crc(1, eng.slot_held[1]) == _full_crc32([1, 2, 3, 4])


def test_tok_worker_array_i_roundtrip() -> None:
    class _FakeTok:
        def encode(self, text, *args, **kwargs):
            return [10, 20, 300, 151645]

        def encode_marked(self, text, *args, **kwargs):
            return [1, 2, 3, 4, 151643], [(len(text), 5)]

    old_inst = srv._TOK_WORKER_INST
    try:
        srv._TOK_WORKER_INST = _FakeTok()
        arr = srv._tok_worker_encode("hello", (), {})
        assert isinstance(arr, array.array) and arr.typecode == "I"
        assert arr.tolist() == [10, 20, 300, 151645]

        arr_m, marks = srv._tok_worker_encode_marked("hello", (), {})
        assert isinstance(arr_m, array.array) and arr_m.typecode == "I"
        assert arr_m.tolist() == [1, 2, 3, 4, 151643]
        assert marks == [(5, 5)]

        class _FakeFuture:
            def __init__(self, val):
                self._val = val

            def result(self):
                return self._val

        class _FakePool:
            def submit(self, fn, *args):
                return _FakeFuture(fn(*args))

        proxy = srv._OffloadTokenizerProxy(_FakeTok(), _FakePool())
        proxy._min_chars = 1
        out_enc = proxy.encode("hello")
        assert isinstance(out_enc, list) and out_enc == [10, 20, 300, 151645]
        out_ids, out_marks = proxy.encode_marked("hello")
        assert isinstance(out_ids, list) and out_ids == [1, 2, 3, 4, 151643]
        assert out_marks == [(5, 5)]
    finally:
        srv._TOK_WORKER_INST = old_inst


def test_json_dumps_bytes_sse_equivalence() -> None:
    part = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1234567,
        "model": "qwen",
        "choices": [],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    }
    raw = srv._json_dumps_bytes(part)
    assert isinstance(raw, bytes)
    assert json.loads(raw.decode("utf-8")) == part
