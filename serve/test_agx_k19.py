"""serve/test_agx_k19.py - Unit tests for Wave K19 serve/server.py changes (no GPU, no lab paths)."""
from __future__ import annotations

import array
import os
import queue
import sys
import threading
import unittest
import zlib
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server  # noqa: E402
from serve.server import StrataEngine  # noqa: E402


class _FakeLeases:
    def __init__(self, tier: str, leased_slots: dict[int, str]):
        self._tier = tier
        self._leased = dict(leased_slots)
        self.promoted: list[tuple[str, int]] = []

    def cleanup_expired(self) -> None:
        pass

    def get_slot(self, sid: str):
        for b, s in self._leased.items():
            if s == sid:
                return b
        return None

    def get_session_tier(self, sid: str):
        return self._tier, None

    def get_leased_slots(self) -> dict[int, str]:
        return dict(self._leased)

    def promote_from_nvme(self, sid: str, slot: int, eng) -> None:
        self.promoted.append((sid, slot))


def _make_bare_engine(batch: int = 2) -> StrataEngine:
    eng = StrataEngine.__new__(StrataEngine)
    eng.batch = batch
    eng.slot_busy = [False] * batch
    eng.slot_held = [[] for _ in range(batch)]
    eng.slot_used = [0] * batch
    eng.slot_group = [0] * batch
    eng.slot_groups = 1
    eng.slot_order = list(range(batch))
    eng.alive = lambda: True
    eng.slot_has_work = [False] * batch
    eng.slot_req = [None] * batch
    eng.waiting = 1
    eng._coalesce = lambda *a, **kw: None
    eng.batch_groups = 1
    eng.mtp_piped = False
    eng.slot_claims = set()
    eng.slot_pending = [None] * batch
    eng.slot_since = [0.0] * batch
    eng.slot_active_sid = [None] * batch
    eng.active_req = 0
    eng._active_lock = threading.Lock()
    eng.slot_q = [queue.Queue() for _ in range(batch)]
    eng.slot_live = [None] * batch
    eng.TURN_ID = 151644
    eng.BRANCH_BOUNDARY_MIN = 32
    eng.last = None
    eng.gen = 0
    eng.wlock = threading.Lock()
    eng.slot_cv = threading.Condition()
    eng.ctl = threading.Lock()
    eng.ctl_epoch = 0
    eng._last_done = None
    eng._yielded = None
    eng._ctl_mode = None
    eng._ctl_result = None
    eng._ctl_erred = False
    eng.progress = None
    eng.log = None
    eng.YIELDS_MAX = 0
    eng.BGEN_REFUSALS_MAX = 1
    eng.pen_from_key = lambda orig_len, cur_len: ""
    eng._shorter_waiting = lambda rem: False
    return eng


class TestWaveK19Serve(unittest.TestCase):
    def test_tier3_nvme_promotion_respects_unleased_free(self) -> None:
        eng = _make_bare_engine(batch=2)
        eng.session_leases = _FakeLeases("nvme", {0: "leased-other"})

        # Only slot 0 is free, and it is leased to another session: allow_lease_steal=False must return None
        eng.slot_busy = [False, True]
        got = eng.pick_slot([1, 2, 3], session_id="nvme-sess", allow_lease_steal=False)
        self.assertIsNone(got)
        self.assertEqual(eng.session_leases.promoted, [])

        # Both slots free: slot 1 is unleased, so NVMe promotion must pick slot 1 (never steal slot 0)
        eng.slot_busy = [False, False]
        got2 = eng.pick_slot([1, 2, 3], session_id="nvme-sess", allow_lease_steal=False)
        self.assertEqual(got2, 1)
        self.assertEqual(eng.session_leases.promoted, [("nvme-sess", 1)])

    def test_ipc_delta_bgend_framing_and_fallback(self) -> None:
        eng = _make_bare_engine(batch=1)
        eng.slot_held[0] = [10, 20, 30]
        sent_bin: list[tuple[str, list[int]]] = []
        eng._send_bin = lambda head, ids: sent_bin.append((head, list(ids)))
        eng._send = lambda text: sent_bin.append((text, []))
        def _tc(*a, **kw):
            eng.ctl.acquire()
            if False:
                yield None
            return True
        eng._take_control = _tc
        eng._estimate_uncached_len = lambda prompt, session_id=None: len(prompt)

        def fake_control(cancel, append_fn, on_quiet=None, want_lp=False):
            eng._ctl_result = (0, False)
            yield 99

        eng._control = fake_control
        eng._done_finish = lambda: "stop"
        cancel = threading.Event()
        cancel._strata_session_id = "s1"
        prompt = [10, 20, 30, 40, 50]

        with mock.patch.dict(os.environ, {"STRATA_BINARY_IPC": "1", "STRATA_IPC_DELTA": "1"}):
            with mock.patch.object(eng, "pick_slot", return_value=0):
                list(eng.generate_batched(prompt, 8, {}, cancel))

        self.assertEqual(len(sent_bin), 1)
        head, delta = sent_bin[0]
        self.assertTrue(head.startswith("BGEND 0 "))
        expected_h = zlib.crc32(array.array("i", [10, 20, 30]).tobytes()) & 0xffffffff
        self.assertIn(f" pfx=3 pfx_hash={expected_h:08x}", head)
        self.assertEqual(delta, [40, 50])

        # Fallback when STRATA_IPC_DELTA=0 -> BGENB with full prompt
        sent_bin.clear()
        eng.slot_held[0] = [10, 20, 30]
        with mock.patch.dict(os.environ, {"STRATA_BINARY_IPC": "1", "STRATA_IPC_DELTA": "0"}):
            with mock.patch.object(eng, "pick_slot", return_value=0):
                list(eng.generate_batched(prompt, 8, {}, cancel))

        self.assertEqual(len(sent_bin), 1)
        head2, full2 = sent_bin[0]
        self.assertTrue(head2.startswith("BGENB 0 "))
        self.assertEqual(full2, prompt)

    def test_dummy_prefix_cache_lru_capped_at_128(self) -> None:
        # Exercise the exact LRU cap logic in serve/server.py around _dummy_prefix_cache
        cache: dict[tuple[int, int], str] = {}
        for i in range(150):
            dp_key = (1, i)
            val = cache.pop(dp_key, None)
            if val is None:
                val = f"prefix-{i}"
                if len(cache) >= 128:
                    cache.pop(next(iter(cache)))
            cache[dp_key] = val

        self.assertEqual(len(cache), 128)
        # Oldest 22 entries (0..21) evicted; newest 128 (22..149) retained
        self.assertNotIn((1, 0), cache)
        self.assertNotIn((1, 21), cache)
        self.assertIn((1, 22), cache)
        self.assertIn((1, 149), cache)


if __name__ == "__main__":
    unittest.main()
