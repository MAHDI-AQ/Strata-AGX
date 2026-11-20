"""serve/test_agx_vision_lazy.py - K21 Lever 4A: the opt-in lazy image reader and the tool-image idle fix.

No GPU, no encoder process and no pack.  The pure env switch (apply_vision_lazy_env) is exercised directly; the
elastic idle behaviour (_vision_idle_down "busy"/"down"/"nothing") against a mock reader on a real Service (the
same MockEngine construction serve/test_server.py uses); and the structural guarantees -- the tool-image
try/finally that refreshes last_image_at, the elastic no-spawn guard, /health reporting images while unloaded,
the config wiring and shutdown cleanup -- as source guards, the AGX convention (terminal/tools/test_uma_drop_guard.py).

    python -m pytest terminal/serve/test_agx_vision_lazy.py
"""
from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (  # noqa: E402
    ByteTokenizer, MockEngine, Service, apply_vision_lazy_env,
)

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "serve" / "server.py"


def _service() -> Service:
    tok = ByteTokenizer()
    eng = MockEngine(tok, "ok", max_context=4096)
    return Service(eng, tok, ChatTemplate(ROOT / "serve" / "chat_template.jinja"))


def _reader(elastic=True, idle=300.0, alive=True):
    return SimpleNamespace(elastic=elastic, elastic_idle_s=idle,
                           alive=lambda: alive, unload=mock.Mock())


class VisionLazyEnv(unittest.TestCase):
    """STRATA_VISION_LAZY default OFF: unset / 0 is the identity; 1 makes the reader elastic with a long idle."""

    def test_unset_is_identity(self):
        cfg = {"vision": {"elastic": False, "max_tokens": 1024}, "backend": "cuda"}
        with mock.patch.dict(os.environ, {}, clear=True):
            out = apply_vision_lazy_env(cfg)
        self.assertIs(out, cfg)                                   # the same object, untouched
        self.assertEqual(cfg["vision"], {"elastic": False, "max_tokens": 1024})

    def test_zero_is_identity(self):
        cfg = {"vision": {"elastic": False}}
        with mock.patch.dict(os.environ, {"STRATA_VISION_LAZY": "0"}, clear=True):
            out = apply_vision_lazy_env(cfg)
        self.assertIs(out, cfg)
        self.assertFalse(cfg["vision"]["elastic"])

    def test_one_elastic_default_idle(self):
        cfg = {"vision": {"max_tokens": 1024}}
        with mock.patch.dict(os.environ, {"STRATA_VISION_LAZY": "1"}, clear=True):
            out = apply_vision_lazy_env(cfg)
        self.assertIs(out, cfg)
        self.assertIs(cfg["vision"]["elastic"], True)
        self.assertEqual(cfg["vision"]["elastic_idle_s"], 300.0)
        self.assertEqual(cfg["vision"]["max_tokens"], 1024)       # the other vision keys survive

    def test_one_custom_idle(self):
        cfg = {"vision": {}}
        with mock.patch.dict(os.environ, {"STRATA_VISION_LAZY": "1", "STRATA_VISION_IDLE_S": "45"}, clear=True):
            apply_vision_lazy_env(cfg)
        self.assertIs(cfg["vision"]["elastic"], True)
        self.assertEqual(cfg["vision"]["elastic_idle_s"], 45.0)

    def test_one_invalid_idle_falls_back_to_default(self):
        cfg = {"vision": {}}
        with mock.patch.dict(os.environ, {"STRATA_VISION_LAZY": "1", "STRATA_VISION_IDLE_S": "abc"}, clear=True):
            apply_vision_lazy_env(cfg)
        self.assertEqual(cfg["vision"]["elastic_idle_s"], 300.0)

    def test_one_creates_vision_section_when_absent(self):
        cfg = {}
        with mock.patch.dict(os.environ, {"STRATA_VISION_LAZY": "1"}, clear=True):
            apply_vision_lazy_env(cfg)
        self.assertIs(cfg["vision"]["elastic"], True)
        self.assertEqual(cfg["vision"]["elastic_idle_s"], 300.0)


class VisionIdleDown(unittest.TestCase):
    """_vision_idle_down: a recent picture (user or tool) keeps the elastic reader up; a long idle stops it."""

    def test_recent_image_is_busy(self):
        svc = _service()
        svc.vision = _reader()
        svc.vision_held = True
        svc.last_image_at = time.time()                          # a picture just ended
        self.assertEqual(svc._vision_idle_down(), "busy")
        svc.vision.unload.assert_not_called()

    def test_long_idle_is_down(self):
        svc = _service()
        svc.vision = _reader(idle=300.0)
        svc.vision_held = True
        svc.last_image_at = time.time() - 1000.0                 # well past elastic_idle_s
        self.assertEqual(svc._vision_idle_down(), "down")
        svc.vision.unload.assert_called_once()
        self.assertFalse(svc.vision_held)

    def test_not_elastic_is_nothing(self):
        svc = _service()
        svc.vision = _reader(elastic=False)
        svc.vision_held = True
        svc.last_image_at = time.time() - 1000.0
        self.assertEqual(svc._vision_idle_down(), "nothing")
        svc.vision.unload.assert_not_called()

    def test_force_down_even_when_recent(self):
        svc = _service()
        svc.vision = _reader()
        svc.vision_held = True
        svc.last_image_at = time.time()
        self.assertEqual(svc._vision_idle_down(force=True), "down")
        svc.vision.unload.assert_called_once()


class VisionLazySourceGuards(unittest.TestCase):
    """The structural guarantees, read from the source (the AGX convention): no encoder process is needed."""

    @classmethod
    def setUpClass(cls):
        cls.lines = SERVER.read_text(encoding="utf-8").splitlines()

    def test_tool_image_refreshes_last_image_at(self):
        # the tool-image single encode is wrapped try/finally so a tool picture keeps the elastic reader warm,
        # exactly as the user-image path does (otherwise the next idle poll unloads it immediately).
        at = [i for i, l in enumerate(self.lines) if l.strip() == "self.vision.encode(src)"]
        self.assertTrue(at, "the tool-image encode call is present")
        ok = False
        for i in at:
            window = "\n".join(self.lines[max(0, i - 3):i + 4])
            if "try:" in window and "finally:" in window and "self.last_image_at = time.time()" in window:
                ok = True
        self.assertTrue(ok, "the tool-image encode refreshes last_image_at in a finally (K21 4A)")

    def test_elastic_init_spawns_no_subprocess(self):
        at = [i for i, l in enumerate(self.lines) if l.strip() == "if not lazy and not self.elastic:"]
        self.assertTrue(at, "the elastic no-spawn guard is present")
        self.assertTrue(any(self.lines[i + 1].strip() == "self._start()" for i in at),
                        "_start() runs only when not lazy and not elastic")

    def test_health_reports_images_while_unloaded(self):
        self.assertTrue(any('"images": self.vision is not None' in l for l in self.lines),
                        "/health reports images from the reader's presence, not whether it is loaded")

    def test_lazy_env_is_wired_into_the_config(self):
        self.assertTrue(any("apply_vision_lazy_env(apply_vision_defaults(with_component_engine(cfg)))" in l
                            for l in self.lines),
                        "apply_vision_lazy_env wraps apply_vision_defaults at the strata config site")

    def test_shutdown_removes_the_encoder_directory(self):
        at = [i for i, l in enumerate(self.lines) if l.strip() == "def shutdown(self):"]
        self.assertTrue(at)
        body = "\n".join(self.lines[at[0]:at[0] + 6])
        self.assertIn("self.close()", body)
        self.assertIn("shutil.rmtree(self.dir", body)


if __name__ == "__main__":
    unittest.main()