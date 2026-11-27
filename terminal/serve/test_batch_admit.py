"""serve/test_batch_admit.py - CPU-only unit tests for STRATA_BATCH_ADMIT (no GPU, no lab paths).

    python -m unittest serve.test_batch_admit -v
"""
from __future__ import annotations

import array
import collections
import os
import queue
import sys
import threading
import time
import unittest
import zlib
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server  # noqa: E402
from serve.server import StrataEngine  # noqa: E402


class _Stdin:
    def __init__(self):
        self.chunks: list[bytes] = []
        self.flushes = 0
        self.buffer = self

    def write(self, data) -> None:
        if isinstance(data, str):
            data = data.encode("utf-8")
        self.chunks.append(bytes(data))

    def flush(self) -> None:
        self.flushes += 1

    @property
    def raw(self) -> bytes:
        return b"".join(self.chunks)


class _Proc:
    def __init__(self):
        self.stdin = _Stdin()


class _TimedLines(queue.Queue):
    """lines queue with a hook after each get, to cancel a member mid-flight."""

    def __init__(self, items, on_get=None):
        super().__init__()
        self.count = 0
        self.on_get = on_get
        for item in items:
            self.put(item)

    def get(self, timeout=None):
        item = super().get(timeout=timeout)
        self.count += 1
        if self.on_get is not None:
            self.on_get(self.count, item)
        return item


def _make_engine(batch: int = 4) -> StrataEngine:
    """A StrataEngine with the admission state of a started one and a fake stdin."""
    eng = StrataEngine.__new__(StrataEngine)
    eng.batch = batch
    eng.slot_busy = [False] * batch
    eng.slot_held = [[] for _ in range(batch)]
    eng.slot_held_crc = [None] * batch
    eng.slot_used = [0.0] * batch
    eng.slot_group = [0] * batch
    eng.slot_groups = 1
    eng.slot_order = list(range(batch))
    eng.slot_gs = 0
    eng.mtp_piped = False
    eng.slot_cv = threading.Condition()
    eng.waiting = 0
    eng.wait_lens = []
    eng.arrivals = collections.deque(maxlen=64)
    eng.ctl_epoch = 0
    eng.ctl = threading.Lock()
    eng._ctl_fifo_q = collections.deque()
    eng._ctl_owner = None
    eng._admit_tickets = []
    eng.slot_claims: set[int] = set()
    eng.slot_releasing: set[int] = set()
    eng.slot_q = [queue.Queue() for _ in range(batch)]
    eng.slot_live = [None] * batch
    eng.slot_since = [0.0] * batch
    eng.slot_active_sid = [None] * batch
    eng.active_req = 0
    eng._active_lock = threading.Lock()
    eng.slot_has_work = [False] * batch
    eng.slot_req = [None] * batch
    eng.lines = queue.Queue()
    eng.proc = _Proc()
    eng.wlock = threading.Lock()
    eng.gen = 1
    eng.up = True
    eng.alive = lambda: eng.up
    eng.ended = False
    eng.closing = False
    eng.starting = False
    eng.log = None
    eng.progress = None
    eng.info = {}   # slot_read is derived from info
    eng.TURN_ID = 151644
    eng.BRANCH_BOUNDARY_MIN = 32
    eng.YIELDS_MAX = 0
    eng.BGEN_REFUSALS_MAX = 1
    eng._ctl_held = False
    eng._ctl_erred = False
    eng._ctl_result = None
    eng._ctl_mode = None
    eng._yielded = None
    eng._last_done = None
    eng.last = None
    eng.pen_from_key = lambda orig_len, cur_len: ""
    eng._shorter_waiting = lambda rem: False
    eng._estimate_uncached_len = lambda prompt, session_id=None: len(prompt)
    eng._set_slot_held = lambda slot, tokens, prev_len=0, prev_crc=None: None
    eng._raise_if_closed_under = lambda finish, cancel: None
    eng._apply_bdone = lambda f: None
    eng._may_go_solo = lambda *a, **kw: False
    eng._done_finish = lambda: "stop"
    eng._park_slot = lambda slot: None
    eng._release_slot_when_done = lambda slot, stream=None, born=None: None
    eng._send_for = lambda born, text: None
    return eng


def _ticket(eng, prompt, cancel=None, max_new=8):
    cancel = cancel or threading.Event()
    entry = [len(prompt), time.monotonic(), True, list(prompt), cancel, None]
    ticket = server.AdmitTicket(entry, cancel, list(prompt), len(prompt), "", None, max_new, eng.gen)
    eng._admit_tickets.append(ticket)
    return ticket


def _run(gen):
    """Drive a generator to its StopIteration value."""
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value


class TestBatchAdmitEnv(unittest.TestCase):
    def tearDown(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            server._refresh_env_settings()

    def test_defaults_off_and_clamped_refresh(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            server._refresh_env_settings()
        self.assertFalse(server.BATCH_ADMIT)
        self.assertEqual(server.BATCH_ADMIT_WINDOW_MS, 12)
        self.assertEqual(server.BATCH_ADMIT_MAX, 8)
        self.assertEqual(server.BATCH_ADMIT_MAX_DELTA, 2048)

        with mock.patch.dict(os.environ, {
            "STRATA_BATCH_ADMIT": "1",
            "STRATA_BATCH_ADMIT_WINDOW_MS": "0",
            "STRATA_BATCH_ADMIT_MAX": "200",
            "STRATA_BATCH_ADMIT_MAX_DELTA": "5",
        }, clear=True):
            server._refresh_env_settings()
        self.assertTrue(server.BATCH_ADMIT)
        self.assertEqual(server.BATCH_ADMIT_WINDOW_MS, 0)
        self.assertEqual(server.BATCH_ADMIT_MAX, 64)
        self.assertEqual(server.BATCH_ADMIT_MAX_DELTA, 256)

    def test_bad_values_fall_back_without_enabling(self):
        with mock.patch.dict(os.environ, {"STRATA_BATCH_ADMIT": "1", "STRATA_BATCH_ADMIT_MAX": "nope"}, clear=True):
            server._refresh_env_settings()
        self.assertFalse(server.BATCH_ADMIT)
        self.assertEqual(server.BATCH_ADMIT_MAX, 8)


class TestBatchAdmitFastPath(unittest.TestCase):
    def test_c1_fast_path_has_no_badmit_and_off_is_byte_identical(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                eng = _make_engine(batch=2)
                eng.arrivals.clear()
                cancel = threading.Event()
                with mock.patch.object(server, "BATCH_ADMIT", enabled), \
                     mock.patch.object(server, "ADMIT_FIFO", False), \
                     mock.patch.object(server, "ADMIT_SHORT_FIRST", 0):
                    start = time.perf_counter_ns()
                    ok = _run(eng._take_control(cancel, 64, new=True, prompt=[1, 2, 3]))
                    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                self.assertTrue(ok)
                self.assertLessEqual(elapsed_ms, 5.0)
                self.assertNotIn(b"BADMIT", eng.proc.stdin.raw)
                self.assertEqual(eng.proc.stdin.raw, b"")
                self.assertEqual(eng._admit_tickets, [])
                self.assertTrue(eng.ctl.locked())
                eng.ctl.release()


class TestBatchAdmitShortFirstGate(unittest.TestCase):
    def test_short_first_2048_does_not_disable_admit_ticket_registration(self):
        eng = _make_engine(batch=4)
        eng.waiting = 1                     # another request is waiting: no c=1 fast path
        now = time.monotonic()
        eng.arrivals.extend([now, now])
        eng.ctl.acquire()                   # somebody else holds the lines: this one must be a follower
        cancel = threading.Event()
        seen = []

        def publish_outcome():
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                with eng.slot_cv:
                    if eng._admit_tickets:
                        tk = eng._admit_tickets[0]
                        seen.append((tk.plen, list(tk.prompt)))
                        tk.outcome, tk.slot = "cohort", 0
                        tk.ready_ev.set()
                        eng.slot_cv.notify_all()
                        return
                time.sleep(0.005)

        threading.Thread(target=publish_outcome, daemon=True).start()
        try:
            with mock.patch.object(server, "BATCH_ADMIT", True), \
                 mock.patch.object(server, "ADMIT_SHORT_FIRST", 2048), \
                 mock.patch.object(server, "ADMIT_FIFO", False):
                gen = eng._take_control(cancel, 64, new=True, prompt=[1, 2, 3])
                ok = _run(gen)              # the published cohort outcome ends the wait immediately
                self.assertTrue(ok)
        finally:
            eng.ctl.release()
        self.assertEqual(seen, [(64, [1, 2, 3])])
        self.assertEqual(eng._admit_tickets, [])


class TestBatchAdmitCohort(unittest.TestCase):
    def setUp(self):
        self.eng = _make_engine(batch=4)
        self.slots = iter(range(4))
        self.eng.pick_slot = lambda prompt, session_id=None, allow_lease_steal=True, allow_nvme_promote=True: next(
            self.slots, None)
        self.released = []
        original = self.eng._release_control

        def release_once():
            self.released.append(1)
            original()

        self.eng._release_control = release_once
        self.tickets = [_ticket(self.eng, [10 + i, 20 + i, 30 + i, 40 + i]) for i in range(4)]

    def _lead(self, lines, short_first=0):
        self.eng.lines = lines
        with mock.patch.object(server, "BATCH_ADMIT", True), \
             mock.patch.object(server, "ADMIT_SHORT_FIRST", short_first), \
             mock.patch.object(server, "BATCH_ADMIT_WINDOW_MS", 1), \
             mock.patch.dict(os.environ, {"STRATA_BATCH_ADMIT_WINDOW_MS": "1"}):
            self.eng.ctl.acquire()
            try:
                result = _run(self.eng._admit_cohort(self.tickets[0], self.eng.gen, self.eng.lines))
            finally:
                self.eng._release_control()
        return result

    def test_burst_four_atomic_badmit_distinct_slots_single_release(self):
        self.eng.slot_q[0].put("BT 0 999\n")   # a stale frame from a previous request
        lead = self._lead(_TimedLines([line for i in range(4) for line in (
            f"DONE 0 {i} 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0 chunk=0", f"BADM {i} 1")]))
        self.assertTrue(lead)
        raw = self.eng.proc.stdin.raw
        self.assertTrue(raw.startswith(b"BADMIT 4 0 1 2 3\n"), raw[:80])
        self.assertEqual(raw.count(b"BGBD "), 4)
        self.assertEqual(self.eng.proc.stdin.flushes, 1)
        self.assertEqual(self.eng.slot_busy, [True] * 4)
        self.assertEqual(self.eng.slot_claims, {0, 1, 2, 3})
        self.assertEqual(len(self.released), 1)
        self.assertFalse(self.eng.ctl.locked())
        for t in self.tickets:
            self.assertEqual(t.outcome, "cohort")
            self.assertTrue(t.ready_ev.is_set())
        self.assertEqual(self.eng._admit_tickets, [])
        self.assertTrue(self.eng.slot_q[0].empty())   # stale frame drained before the cohort write

    def test_short_first_2048_still_forms_a_cohort(self):
        lead = self._lead(_TimedLines([line for i in range(4) for line in (
            f"DONE 0 {i} 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0 chunk=0", f"BADM {i} 1")]),
            short_first=2048)
        self.assertTrue(lead)
        raw = self.eng.proc.stdin.raw
        self.assertTrue(raw.startswith(b"BADMIT 4 0 1 2 3\n"), raw[:80])
        for t in self.tickets:
            self.assertEqual(t.outcome, "cohort")

    def test_partial_bgend_mismatch_retries_as_badmit_one_cold(self):
        lines = _TimedLines(["DONE 0 1 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0", "BADM 0 1",
                 "ERR BGEND_MISMATCH 1 0",
                 "DONE 0 2 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0", "BADM 2 1",
                 "DONE 0 3 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0", "BADM 3 1",
                 "DONE 0 1 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0", "BADM 1 0"])
        original_send = self.eng._send_batch_admit
        sends = []

        def traced_send(records):
            sends.append((len(records), lines.qsize()))
            original_send(records)
            if len(records) == 1:                       # the cold retry's answer arrives afterwards
                lines.put("DONE 0 1 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0")
                lines.put("BADM 1 1")

        self.eng._send_batch_admit = traced_send
        self._lead(lines)
        self.assertEqual(sends[0][0], 4)
        self.assertEqual(sends[1][0], 1)
        self.assertEqual(sends[1][1], 0)                # BADM 1 0 consumed before the retry
        self.assertTrue(lines.empty())                  # no BADM left for the next request
        raw = self.eng.proc.stdin.raw
        self.assertEqual(raw.count(b"BADMIT 4 "), 1)
        self.assertIn(b"BADMIT 1 1\n", raw)
        self.assertEqual(raw.count(b"BGBD "), 5)   # 4 original + 1 cold retry
        self.assertIn(b"BGBD 1 8 pfx=0 pfx_hash=00000000 bin=4\n", raw)
        self.assertNotIn("BGENB ", raw.decode("utf-8", "replace"))
        for t in self.tickets:
            self.assertEqual(t.outcome, "cohort")
        self.assertEqual(len(self.released), 1)

    def test_cancel_before_write_and_midflight_bstop_no_double_release(self):
        # A member cancelled before the leader writes is not collected at all.
        self.tickets[3].cancel.set()
        self.tickets[3].state = "waiting"

        def on_get(count, _line):
            if count == 1:
                self.tickets[2].cancel.set()

        stopped = []
        self.eng._release_slot_when_done = lambda slot, stream=None, born=None: stopped.append(slot)
        lines = _TimedLines(["DONE 0 1 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0", "BADM 0 1",
                      "DONE 0 1 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0", "BADM 1 1",
                      "DONE 0 1 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0", "BADM 2 1"], on_get=on_get)
        self._lead(lines)
        self.assertTrue(self.eng.proc.stdin.raw.startswith(b"BADMIT 3 "))
        self.assertEqual(self.tickets[2].outcome, "cancel")
        self.assertEqual(stopped, [self.tickets[2].slot])   # one BSTOP, registered once
        self.assertTrue(lines.empty())                     # its BADM was drained before ctl was freed
        # Slot 2 stays claimed until BDONE (the release is registered, not executed here).
        self.assertTrue(self.eng.slot_busy[self.tickets[2].slot])
        self.assertIn(self.tickets[2].slot, self.eng.slot_claims)
        self.assertEqual(len(self.released), 1)
        self.assertFalse(self.eng.ctl.locked())


    def test_multi_leg_stale_ticket_does_not_leak_control(self):
        # Leg 1 is admitted by BADMIT on this cancel Event and leaves its ticket attached, as the
        # real generate_batched did before the fix.  Leg 2 must take the c=1 fast path and BGEN.
        cancel = self.tickets[0].cancel
        self._lead(_TimedLines([line for i in range(4) for line in (
            f"DONE 0 {i} 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0 chunk=0", f"BADM {i} 1")]))
        setattr(cancel, "_strata_admit_ticket", self.tickets[0])
        self.eng.proc.stdin.chunks.clear()                 # only look at Leg 2's wire traffic
        self.eng.arrivals.clear()                          # no recent arrival: c=1 fast path applies
        self.eng.slot_busy = [False] + [True] * (self.eng.batch - 1)   # not alone: take the batch path
        self.eng.slot_claims = set(range(1, self.eng.batch))
        self.eng.slot_live = [None] * self.eng.batch
        self.eng.pick_slot = lambda prompt, session_id=None, allow_lease_steal=True, allow_nvme_promote=True: 0
        sent = []

        def record_bin(head, ids):
            sent.append((head, list(ids)))
            self.eng.slot_q[0].put("BT 0 555\n")     # the engine's answer arrives after the BGENB
            self.eng.slot_q[0].put("BDONE 0 0 0 ok\n")

        self.eng._send_bin = record_bin

        def fake_control(cancel, append_fn, **kw):
            self.eng._ctl_result = (0, True)
            yield 555

        self.eng._control = fake_control
        with mock.patch.dict(os.environ, {"STRATA_BINARY_IPC": "1"}), \
             mock.patch.object(server, "BATCH_ADMIT", True):
            tokens = list(self.eng.generate_batched([10, 11], 8, {}, cancel))
        self.assertEqual(tokens, [555],
                         (sent, self.eng.proc.stdin.raw[:200], list(self.eng._admit_tickets),
                          self.eng.ctl.locked(), list(self.eng.slot_busy),
                          [q.qsize() for q in self.eng.slot_q]))
        self.assertTrue(sent and sent[0][0].startswith("BGENB 0 "), sent)
        self.assertNotIn(b"BADMIT", self.eng.proc.stdin.raw)
        self.assertFalse(self.eng.ctl.locked())            # no leaked control lines
        self.assertIsNone(self.eng.slot_live[0])
        self.assertIsNone(getattr(cancel, "_strata_admit_ticket", None))

    def test_generator_exit_mid_drain_quarantines_ctl_and_badm(self):
        for i, tk in enumerate(self.tickets):
            tk.slot = i
            setattr(tk.cancel, "_strata_slot", i)
        class ClosingLines(_TimedLines):
            closed = False

            def get(self, timeout=None):
                if not self.closed:
                    self.closed = True
                    raise GeneratorExit()                  # the client closed us mid-drain
                return super().get(timeout=timeout)

        lines = ClosingLines([line for i in range(4) for line in (
            f"DONE 0 {i} 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0", f"BADM {i} 1")])
        ownership = {"handed": False}
        self.eng.ctl.acquire()
        with self.assertRaises(GeneratorExit):
            next(self.eng._drain_admit_outcomes(self.tickets, self.eng.gen, lines, ownership))
        self.assertTrue(ownership["handed"])
        deadline = time.monotonic() + 2.0
        while self.eng.ctl.locked() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.eng.ctl.locked())            # quarantine released it exactly once
        self.assertTrue(lines.empty())                     # no BADM left for the next request
        for tk in self.tickets:
            self.assertIsNotNone(tk.outcome)


    def test_badmit_slot_error_waits_for_its_badm_zero(self):
        lines = _TimedLines(["DONE 0 1 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0", "BADM 0 1",
                 "ERR BADMIT_SLOT 1 the K/V cannot grow",
                 "DONE 0 1 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0", "BADM 1 0",
                 "DONE 0 2 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0", "BADM 2 1",
                 "DONE 0 3 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0", "BADM 3 1"])
        slot1 = 1
        self._lead(lines)
        self.assertTrue(lines.empty())                  # BADM 1 0 was consumed here, not later
        self.assertEqual(self.tickets[1].outcome, "serial")
        self.assertNotIn(slot1, self.eng.slot_claims)
        self.assertFalse(self.eng.slot_busy[slot1])
        self.assertEqual(len(self.released), 1)
        self.assertFalse(self.eng.ctl.locked())

    def test_quarantine_waits_for_slot_error_badm_and_only_stops_accepted_slots(self):
        stopped = []
        self.eng._release_slot_when_done = lambda slot, stream=None, born=None: stopped.append(slot)
        lines = _TimedLines(["ERR BADMIT_SLOT 1 bad cold prompt size", "BADM 1 0",
                            "DONE 0 1 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0", "BADM 0 1"])
        for i, tk in enumerate(self.tickets[:2]):
            tk.slot = i
        self.eng.ctl.acquire()
        self.eng._quarantine_admit_drain(self.tickets[:2], self.eng.gen, lines, set())
        deadline = time.monotonic() + 2.0
        while self.eng.ctl.locked() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(lines.empty())
        self.assertEqual(stopped, [0])                  # never BSTOP a slot that never started
        self.assertEqual(len(self.released), 1)
        self.assertFalse(self.eng.ctl.locked())
        for tk in self.tickets[:2]:
            self.assertIsNotNone(tk.outcome)

    def test_post_pick_delta_cap_leaves_long_prompt_unclaimed(self):
        self.eng._admit_tickets.clear()
        self.slots = iter(range(4))
        long_one = _ticket(self.eng, list(range(3000)))
        short = _ticket(self.eng, [1, 2, 3, 4])
        # The pre-pick estimate looks warm, but the slot it picks holds no prefix: the real delta
        # is 3000 > BATCH_ADMIT_MAX_DELTA and must not be sent in a BADMIT.
        self.eng._estimate_uncached_len = lambda prompt, session_id=None: 1 if len(prompt) > 100 else len(prompt)
        with mock.patch.object(server, "BATCH_ADMIT", True), \
             mock.patch.object(server, "BATCH_ADMIT_MAX_DELTA", 64), \
             mock.patch.object(server, "BATCH_ADMIT_WINDOW_MS", 1):
            self.eng.ctl.acquire()
            try:
                _run(self.eng._admit_cohort(long_one, self.eng.gen, _TimedLines([])))
            finally:
                self.eng._release_control()
        self.assertNotIn(b"BADMIT", self.eng.proc.stdin.raw)
        self.assertEqual(self.eng.slot_claims, set())
        self.assertEqual(long_one.outcome, "serial")
        self.assertEqual(short.outcome, "serial")

    def test_cold_warm_mixed_cohort_and_max_delta_exclusion(self):
        self.eng._admit_tickets.clear()
        self.slots = iter(range(4))
        warm = _ticket(self.eng, [10, 20, 30, 40])
        cold = _ticket(self.eng, [7, 8, 9])
        long_one = _ticket(self.eng, list(range(3000)))
        self.eng.slot_held[0] = [10, 20, 30]
        with mock.patch.object(server, "BATCH_ADMIT", True), \
             mock.patch.object(server, "BATCH_ADMIT_MAX_DELTA", 64), \
             mock.patch.object(server, "BATCH_ADMIT_WINDOW_MS", 1):
            self.eng.ctl.acquire()
            try:
                _run(self.eng._admit_cohort(warm, self.eng.gen,
                                            _TimedLines(["DONE 0 1 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0", "BADM 0 1",
                                                 "DONE 0 1 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0", "BADM 1 1"])))
            finally:
                self.eng._release_control()
        raw = self.eng.proc.stdin.raw
        self.assertTrue(raw.startswith(b"BADMIT 2 0 1\n"), raw[:80])
        self.assertIn(b"pfx=3 ", raw)
        self.assertIn(b"pfx=0 ", raw)
        self.assertEqual(warm.outcome, "cohort")
        self.assertEqual(cold.outcome, "cohort")
        self.assertEqual(long_one.outcome, "serial")
        self.assertEqual(len(self.released), 1)

    def test_follower_enters_slot_phase_and_reads_slot_queue(self):
        ticket = self.tickets[1]
        ticket.outcome = "cohort"
        ticket.slot = 1
        ticket.delta_prev_len = 0
        ticket.delta_prev_crc = None
        setattr(ticket.cancel, "_strata_admit_ticket", ticket)
        self.eng.slot_busy[1] = True
        self.eng.slot_claims.add(1)
        self.eng.slot_q[1].put("BT 1 777\n")
        self.eng.slot_q[1].put("BDONE 1 0 0 ok\n")

        calls = []

        def forbidden(*a, **kw):
            raise AssertionError("a cohort follower must not take control lines again")

        def take_once(*a, **kw):
            calls.append(1)
            if len(calls) > 1:
                raise AssertionError("a cohort follower must not take control lines again")
            if False:
                yield None                # keep this a generator
            return True

        with mock.patch.object(self.eng, "_take_control", take_once), \
             mock.patch.object(self.eng, "_release_control", forbidden), \
             mock.patch.object(server, "BATCH_ADMIT", True):
            tokens = list(self.eng.generate_batched([11, 22], 8, {}, ticket.cancel))
        self.assertEqual(tokens, [777])
        self.assertEqual(len(calls), 1)


class TestBatchAdmitStress(unittest.TestCase):
    def test_eight_threads_mixed_cohorts_no_protocol_desync(self):
        eng = _make_engine(batch=8)
        eng.lines = queue.Queue()
        active: list[threading.Event] = []
        overlaps: list[str] = []
        doubles: list[int] = []
        errors: list[tuple[int, str]] = []
        sent: list[str] = []
        guard = threading.Lock()
        inflight = [0]
        flips = [0]

        def enter():
            with guard:
                inflight[0] += 1
                if inflight[0] > 1:
                    overlaps.append("stdin commands overlapped")

        def leave():
            with guard:
                inflight[0] -= 1

        original_send = eng._send_batch_admit

        def send_batch(records):
            enter()
            try:
                original_send(records)
                with guard:
                    flips[0] += 1
                    n = flips[0]
                    if active:
                        target = active[0]
                slots = [int(r[0]) for r in records]
                if len(records) == 1:                   # a cold retry's BADM arrives afterwards
                    eng.lines.put("DONE 0 1 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0")
                    eng.lines.put(f"BADM {slots[0]} 1")
                    eng.slot_q[slots[0]].put(f"BT {slots[0]} 700\n")
                    eng.slot_q[slots[0]].put(f"BDONE {slots[0]} 0 0 ok\n")
                    return
                if n % 4 == 3 and 'target' in locals():  # a client leaves mid-cohort
                    target.set()
                for s in slots:
                    if n % 4 == 1 and s == slots[0]:
                        eng.lines.put(f"ERR BADMIT_SLOT {s} bad checkpoint")
                        eng.lines.put(f"BADM {s} 0")
                    elif n % 4 == 2 and s == slots[0]:
                        eng.lines.put(f"ERR BGEND_MISMATCH {s}")
                        eng.lines.put(f"BADM {s} 0")
                    else:
                        eng.lines.put("DONE 0 1 0.0 0.0 length 0 0 0 0 0 0 0 0 0.0 0 0")
                        eng.lines.put(f"BADM {s} 1")
                        eng.slot_q[s].put(f"BT {s} 700\n")
                        eng.slot_q[s].put(f"BDONE {s} 0 0 ok\n")
            finally:
                leave()

        eng._send_batch_admit = send_batch

        def send_bin(head, ids):
            enter()
            try:
                sent.append(head)
                parts = head.split()
                if len(parts) > 1 and parts[0] in ("BGENB", "BGBD") and parts[1].isdigit():
                    slot = int(parts[1])
                    if slot < len(eng.slot_q):
                        eng.slot_q[slot].put(f"BT {slot} 700\n")
                        eng.slot_q[slot].put(f"BDONE {slot} 0 0 ok\n")
            finally:
                leave()

        eng._send_bin = send_bin

        def send_text(text):
            enter()
            try:
                sent.append(text)
            finally:
                leave()

        eng._send = send_text

        def fake_control(cancel, append_fn, **kw):
            eng._ctl_result = (0, True)
            if False:
                yield None
            return

        eng._control = fake_control

        def release_at_bdone(slot, stream=None, born=None):
            with eng.slot_cv:
                eng.slot_busy[slot] = False
                eng.slot_claims.discard(slot)
                eng.slot_cv.notify_all()

        eng._release_slot_when_done = release_at_bdone
        original_pick = eng.pick_slot

        def checked_pick(prompt, session_id=None, **kw):
            slot = original_pick(prompt, session_id=session_id, **kw)
            if slot is not None and eng.slot_busy[slot]:
                doubles.append(slot)
            return slot

        eng.pick_slot = checked_pick

        def worker(idx):
            try:
                for rnd in range(3):
                    cancel = threading.Event()
                    cancel._strata_request_id = f"w{idx}-{rnd}"
                    with guard:
                        active.append(cancel)
                    try:
                        list(eng.generate_batched([idx, rnd, 7], 8, {}, cancel))
                    finally:
                        with guard:
                            if cancel in active:
                                active.remove(cancel)
            except BaseException as e:                  # noqa: BLE001
                errors.append((idx, __import__("traceback").format_exc()))

        with mock.patch.object(server, "BATCH_ADMIT", True), \
             mock.patch.object(server, "ADMIT_SHORT_FIRST", 0), \
             mock.patch.object(server, "ADMIT_FIFO", False), \
             mock.patch.object(server, "BATCH_ADMIT_WINDOW_MS", 5), \
             mock.patch.dict(os.environ, {"STRATA_BINARY_IPC": "1", "STRATA_BATCH_ADMIT_WINDOW_MS": "5"}):
            threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(8)]
            for th in threads:
                th.start()
            for th in threads:
                th.join(timeout=15)
                self.assertFalse(th.is_alive(), "a request did not finish")

        self.assertEqual(errors, [], errors)
        self.assertEqual(overlaps, [], overlaps)
        self.assertEqual(doubles, [], doubles)
        self.assertTrue(eng.lines.empty(), "a BADM was left for the next request")
        self.assertFalse(eng.ctl.locked(), "ctl was leaked")
        self.assertEqual(eng.slot_claims, set())
        self.assertFalse(any(eng.slot_busy))
        self.assertTrue(sent, "no request reached the engine")


class TestBatchAdmitPickSlot(unittest.TestCase):
    def test_cohort_slot_pick_skips_nvme_promotion(self):
        class Leases:
            promoted = []

            def cleanup_expired(self):
                pass

            def get_slot(self, sid):
                return None

            def get_session_tier(self, sid):
                return "nvme", None

            def get_leased_slots(self):
                return {}

            def promote_from_nvme(self, sid, slot, eng):
                self.promoted.append((sid, slot))

        eng = _make_engine(batch=2)
        eng.session_leases = Leases()
        with eng.slot_cv:
            got = eng.pick_slot([1, 2, 3], session_id="nvme-sess", allow_nvme_promote=False)
            self.assertEqual(got, 0)
        self.assertEqual(eng.session_leases.promoted, [])


if __name__ == "__main__":
    unittest.main()
