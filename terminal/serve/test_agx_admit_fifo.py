"""Tests for Wave K20c _refresh_env_settings() and opt-in STRATA_ADMIT_FIFO=1 admission."""
from __future__ import annotations

import collections
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

SERVE_DIR = Path(__file__).resolve().parent
TERMINAL_DIR = SERVE_DIR.parent
if str(TERMINAL_DIR) not in sys.path:
    sys.path.insert(0, str(TERMINAL_DIR))

from serve import server as srv  # noqa: E402


def _make_fake_engine() -> srv.StrataEngine:
    eng = srv.StrataEngine.__new__(srv.StrataEngine)
    eng.slot_cv = threading.Condition()
    eng.waiting = 0
    eng.wait_lens = []
    eng.arrivals = collections.deque(maxlen=64)
    eng.ctl_epoch = 0
    eng.ctl = threading.Lock()
    eng._ctl_owner = None
    eng._ctl_fifo_q = collections.deque()
    eng._ctl_held = False
    eng.starting = False
    eng.ended = False
    eng.closing = False
    eng.alive = lambda: True
    return eng


class TestRefreshEnvSettings(unittest.TestCase):
    def test_refresh_updates_admit_and_coalesce_globals(self):
        env_patch = {
            "STRATA_ADMIT_SHORT_FIRST": "2048",
            "STRATA_ADMIT_MAX_WAIT_S": "12.5",
            "STRATA_ADMIT_FIFO": "1",
            "STRATA_COALESCE_PREFILL": "4",
        }
        with mock.patch.dict(os.environ, env_patch, clear=False):
            try:
                srv._refresh_env_settings()
                self.assertEqual(srv.ADMIT_SHORT_FIRST, 2048)
                self.assertAlmostEqual(srv.ADMIT_MAX_WAIT_S, 12.5)
                self.assertTrue(srv.ADMIT_FIFO)
                self.assertEqual(srv.COALESCE_PREFILL, 4)
            finally:
                pass
        srv._refresh_env_settings()
        self.assertFalse(srv.ADMIT_FIFO)


class TestAdmitFifo(unittest.TestCase):
    def tearDown(self):
        srv._refresh_env_settings()

    def test_8_thread_strict_fifo_zero_starvation(self):
        eng = _make_fake_engine()
        with mock.patch.dict(os.environ, {"STRATA_ADMIT_FIFO": "1", "STRATA_ADMIT_SHORT_FIRST": "0"}):
            srv._refresh_env_settings()
            self.assertTrue(srv.ADMIT_FIFO)

            # Hold ctl initially so all 8 threads queue up in deterministic order
            eng.ctl.acquire()
            order = []
            ready_events = [threading.Event() for _ in range(8)]
            errors = []

            def worker(idx: int):
                try:
                    cancel = threading.Event()
                    gen = eng._take_control(cancel, 1000 + idx, new=(idx % 2 == 0))
                    # Step once to register in _ctl_fifo_q? Actually _take_control enters wait_lens
                    # before the loop; let's run the generator to completion.
                    ready_events[idx].set()
                    for _ in gen:
                        pass
                    order.append(idx)
                    eng._release_control()
                except BaseException as exc:  # noqa: BLE001
                    errors.append((idx, exc))

            threads = []
            for i in range(8):
                t = threading.Thread(target=worker, args=(i,))
                t.start()
                # Wait until thread i is registered in _ctl_fifo_q
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    with eng.slot_cv:
                        if len(eng._ctl_fifo_q) == i + 1:
                            break
                    time.sleep(0.005)
                threads.append(t)

            with eng.slot_cv:
                self.assertEqual(len(eng._ctl_fifo_q), 8)
            eng.ctl.release()
            with eng.slot_cv:
                eng.slot_cv.notify_all()

            for t in threads:
                t.join(timeout=5.0)
                self.assertFalse(t.is_alive())

            self.assertEqual(errors, [])
            self.assertEqual(order, list(range(8)))
            self.assertIsNone(eng._ctl_owner)

    def test_short_first_preempts_and_max_wait_bounds_starvation(self):
        eng = _make_fake_engine()
        with mock.patch.dict(
            os.environ,
            {"STRATA_ADMIT_FIFO": "1", "STRATA_ADMIT_SHORT_FIRST": "512", "STRATA_ADMIT_MAX_WAIT_S": "0.15"},
        ):
            srv._refresh_env_settings()
            eng.ctl.acquire()
            order = []

            def worker(tag: str, plen: int, new: bool):
                cancel = threading.Event()
                for _ in eng._take_control(cancel, plen, new=new):
                    pass
                order.append(tag)
                time.sleep(0.02)
                eng._release_control()

            # 1. Long waiter arrives first
            t_long = threading.Thread(target=worker, args=("long", 4096, True))
            t_long.start()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                with eng.slot_cv:
                    if len(eng._ctl_fifo_q) == 1:
                        break
                time.sleep(0.005)

            # 2. Short waiter (new=False retry too!) arrives second -> should preempt long waiter
            t_short1 = threading.Thread(target=worker, args=("short1", 256, False))
            t_short1.start()
            while time.monotonic() < deadline:
                with eng.slot_cv:
                    if len(eng._ctl_fifo_q) == 2:
                        break
                time.sleep(0.005)

            # Release ctl: short1 should go before long
            eng.ctl.release()
            with eng.slot_cv:
                eng.slot_cv.notify_all()

            # Wait 0.16s (> ADMIT_MAX_WAIT_S=0.15) then enqueue short2 -> long has waited >= max_wait, so long goes before short2
            time.sleep(0.16)
            t_short2 = threading.Thread(target=worker, args=("short2", 256, True))
            t_short2.start()

            for t in (t_long, t_short1, t_short2):
                t.join(timeout=5.0)
                self.assertFalse(t.is_alive())

            self.assertEqual(order[0], "short1")
            self.assertEqual(order[1], "long")
            self.assertEqual(order[2], "short2")

    def test_default_admit_fifo_zero_preserves_upstream(self):
        eng = _make_fake_engine()
        with mock.patch.dict(os.environ, {"STRATA_ADMIT_FIFO": "0"}):
            srv._refresh_env_settings()
            self.assertFalse(srv.ADMIT_FIFO)
            cancel = threading.Event()
            for _ in eng._take_control(cancel, 128, new=True):
                pass
            self.assertIsNone(eng._ctl_owner)
            eng._release_control()


if __name__ == "__main__":
    unittest.main()
