"""Unit tests for server slot-group spread and prompt-cache safety."""
from __future__ import annotations

import os
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import ConvCacheLog, Service, StrataEngine  # noqa: E402


def _make_8slot_4group_engine(*, busy=None) -> StrataEngine:
    eng = StrataEngine.__new__(StrataEngine)
    eng.batch = 8
    gs = 2
    groups = 4
    eng.slot_order = [g * gs + t for t in range(gs) for g in range(groups)]
    eng.slot_group = [b // gs for b in range(eng.batch)]
    eng.slot_groups = groups
    eng.slot_gs = gs
    eng.mtp_piped = False
    eng.slot_busy = list(busy) if busy is not None else [False] * 8
    eng.slot_held = [[] for _ in range(8)]
    eng.slot_used = [0] * 8
    eng.slot_claims = set()
    return eng


class TestPickSlotGroupSpread(unittest.TestCase):
    def test_default_fills_hole_clusters_on_3_groups(self):
        # Slots 1, 3, 5 busy -> groups 0, 1, 2 busy; group 3 completely idle.
        busy = [False, True, False, True, False, True, False, False]
        eng = _make_8slot_4group_engine(busy=busy)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_PICK_SLOT_GROUP_SPREAD", None)
            slot = eng.pick_slot([100, 101, 102])
        # Default fills_hole(0)=True -> picks slot 0 in group 0 (clusters onto 3 groups)
        self.assertEqual(slot, 0)
        self.assertEqual(eng.slot_group[slot], 0)

    def test_zero_behaves_like_default(self):
        busy = [False, True, False, True, False, True, False, False]
        eng = _make_8slot_4group_engine(busy=busy)
        with mock.patch.dict(os.environ, {"STRATA_PICK_SLOT_GROUP_SPREAD": "0"}):
            slot = eng.pick_slot([100, 101, 102])
        self.assertEqual(slot, 0)

    def test_spread_enabled_picks_idle_group_3(self):
        busy = [False, True, False, True, False, True, False, False]
        eng = _make_8slot_4group_engine(busy=busy)
        with mock.patch.dict(os.environ, {"STRATA_PICK_SLOT_GROUP_SPREAD": "1"}):
            slot = eng.pick_slot([100, 101, 102])
        # Spread picks group 3 (slot 6, busy_in=0) -> all 4 groups active
        self.assertEqual(slot, 6)
        self.assertEqual(eng.slot_group[slot], 3)

    def test_four_concurrent_idle_picks_spread_across_all_4_groups(self):
        eng = _make_8slot_4group_engine()
        picked_groups = []
        with mock.patch.dict(os.environ, {"STRATA_PICK_SLOT_GROUP_SPREAD": "1"}):
            for _ in range(4):
                s = eng.pick_slot([10, 20, 30])
                self.assertIsNotNone(s)
                eng.slot_busy[s] = True
                picked_groups.append(eng.slot_group[s])
        self.assertEqual(sorted(picked_groups), [0, 1, 2, 3])

    def test_sequential_warmup_spreads_across_4_groups_even_with_mtp_piped(self):
        # 8 held slots, mtp_piped=True, busy=[False]*8, clustered LRU timestamps
        held = [[1] for _ in range(8)]
        used = [1.0, 2.0, 5.0, 6.0, 4.0, 7.0, 3.0, 8.0]

        eng_off = _make_8slot_4group_engine()
        eng_off.mtp_piped = True
        eng_off.slot_held = [list(h) for h in held]
        eng_off.slot_used = list(used)
        picks_off = []
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_PICK_SLOT_GROUP_SPREAD", None)
            for i in range(4):
                s = eng_off.pick_slot([100 + i])
                picks_off.append(s)
                eng_off.slot_used[s] = 100.0 + i
        self.assertEqual(picks_off, [0, 1, 6, 4])
        self.assertEqual(len({eng_off.slot_group[s] for s in picks_off}), 3)

        eng_on = _make_8slot_4group_engine()
        eng_on.mtp_piped = True
        eng_on.slot_held = [list(h) for h in held]
        eng_on.slot_used = list(used)
        picks_on = []
        with mock.patch.dict(os.environ, {"STRATA_PICK_SLOT_GROUP_SPREAD": "1"}):
            for i in range(4):
                s = eng_on.pick_slot([100 + i])
                picks_on.append(s)
                eng_on.slot_used[s] = 100.0 + i
        self.assertEqual(picks_on, [0, 2, 4, 6])
        self.assertEqual(sorted(eng_on.slot_group[s] for s in picks_on), [0, 1, 2, 3])


class TestMergeSafetyRegressions(unittest.TestCase):
    def test_conv_cache_log_poll_has_lock(self):
        ccl = ConvCacheLog()
        st = ccl.poll(None, None)
        self.assertIsInstance(st, dict)
        self.assertIn("refused", st)

    def test_encode_prompt_msg_cache_filters_empty_assistant_without_nameerror(self):
        svc = Service.__new__(Service)
        svc._msg_prefix_cache = {}
        svc._msg_prefix_lock = threading.Lock()
        svc.effort_end = False
        svc.template = mock.MagicMock()
        svc._prompt_msg_cache_lock = mock.MagicMock()
        svc._prompt_msg_cache_lock.__enter__ = mock.MagicMock(return_value=None)
        svc._prompt_msg_cache_lock.__exit__ = mock.MagicMock(return_value=False)
        svc.chat = mock.MagicMock()
        svc.chat.encode.return_value = [1, 2, 3]
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "next"},
        ]
        out = svc._encode_prompt_msg_cache(msgs, None, {}, False)
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
