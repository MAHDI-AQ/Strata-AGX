#!/usr/bin/env python3
"""Stall watchdog v3 for the Strata serve (single engine; restarts via restart-campaign).

INCIDENT LESSON (2026-09-30, verified by lane needle-forensics):
v1/v2 judged liveness by /metrics live.generated alone, which does NOT move during PREFILL.
A 96K prompt prefills longer than the 90s window -> false "stall" -> the watchdog killed the
server mid-request (5 needle cases ERRORed: RemoteDisconnected + 4x ConnectionRefused) and its
restart raced the campaign's phase restart -> 4 engine boots in 1s -> 4x23.42 GiB arenas on a
91 GiB box -> user-space wedge, power cycle required.  v3 fixes the criterion and adds a
restart throttle:
  - liveness = generated delta OR prefill advance parsed from the server log
    ("reading the prompt: N tokens" / "thinking: X of") OR both idle;
  - stall window default 420s (must exceed a 96K prefill at the measured ~363 tok/s);
  - never restart while a sanctioned restart is younger than --min-restart-gap
    (restart-campaign.sh / restart-pair.sh touch the marker file);
  - never starts servers that are simply down (operator's call);
  - 2026-10-01: prefill liveness at concurrency > 1 uses /metrics live.prompt_read/prompt_total
    (the server log's "reading the prompt" line carries the CONSTANT prompt size there and its
    "s s so far" clock ticks through a stall - neither is a work signal); the restart config is the
    pointer the sanctioned recipe writes (scratch/.current-campaign-config, then the
    the canonical q3 recipe config), never a hard-coded one - and the restart runs that
    recipe's own start script (TERM-first, teardown-gated); with no real prefill
    channel visible while reading it stands down instead of restarting.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

WS = "/home/mhd67/strata-serving-lab"
MARKER = f"{WS}/.last-serving-restart"
CONFIG_PTR = f"{WS}/scratch/.current-campaign-config"           # written by tools/restart-campaign.sh
FALLBACK_CONFIG = "/srv/lab/recipes/q3-xxs-3x262k/config.json"      # the canonical q3 recipe config


def http_json(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)}


def sh(cmd, timeout=30):
    try:
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout).stdout
    except Exception as e:  # noqa: BLE001
        return f"(cmd failed: {e})"


def log(msg):
    with open(f"{WS}/watchdog.log", "a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}Z] {msg}\n")


def prefill_progress():
    """Advancing prompt-read counter from the server window, single-request mode only ("X of Y
    tokens": X moves).  At concurrency > 1 the same line carries the CONSTANT prompt size - the
    old constant-form match made every concurrent prefill look frozen (the 2026-10-01 false
    fire); that shape is covered by /metrics live.prompt_read instead."""
    cfg, _ = current_config()
    log_candidates = [
        f"{os.path.dirname(cfg)}/logs/server.log" if cfg else None,
        "/srv/lab/recipes/q3-xxs-3x262k/logs/server.log",
        f"{WS}/serve-8096.log"
    ]
    log_path = next((p for p in log_candidates if p and os.path.exists(p)), None)
    if not log_path:
        return None
    tail = sh(f"tail -n 80 {log_path} 2>/dev/null")
    vals = [int(m.group(1).replace(",", "")) for m in re.finditer(r"reading the prompt:\s*([\d,]+) of [\d,]+ tokens", tail)]
    return max(vals) if vals else None


def current_config():
    """The config the lab is on: the pointer the sanctioned restart recipe writes (a file whose
    one line is the config path), then the canonical q3 recipe config.  Never a hard-coded config: the
    old serve-coder.json toy came back both times the watchdog restarted the lab (2026-10-01)."""
    try:
        with open(CONFIG_PTR, encoding="utf-8") as fh:
            cfg = fh.read().strip()
    except OSError:
        cfg = ""
    if cfg and os.path.exists(cfg):
        return cfg, CONFIG_PTR
    if cfg:
        log(f"config pointer {CONFIG_PTR} names a missing config ({cfg}); falling back to {FALLBACK_CONFIG}")
    if os.path.exists(FALLBACK_CONFIG):
        return FALLBACK_CONFIG, FALLBACK_CONFIG
    return None, None


def snapshot(port):
    with open(f"{WS}/watchdog.log", "a", encoding="utf-8") as f:
        f.write(f"\n===== STALL SNAPSHOT {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} =====\n")
        f.write("-- metrics --\n" + json.dumps(http_json(f"http://127.0.0.1:{port}/metrics"), indent=1)[:1500] + "\n")
        f.write("-- ps --\n" + sh("ps -ef | grep 'build/strata' | grep -v grep") + "\n")
        f.write("-- nvidia-smi --\n" + sh("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader") + "\n")
        for pid in sh("pgrep -f 'build/strata --serve'").split():
            f.write(f"-- gdb bt (pid {pid}) --\n")
            f.write(sh(f"sudo -n timeout 25 gdb -p {pid} -batch -ex 'thread apply all bt' 2>&1 | head -120", timeout=40))
        f.write("-- server log tail --\n" + sh(f"tail -30 {WS}/serve-8096.log") + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--stall-seconds", type=int, default=420)
    ap.add_argument("--interval", type=int, default=15)
    ap.add_argument("--min-restart-gap", type=int, default=900)
    a = ap.parse_args()

    log(f"watchdog v3 start (gmtime); {a.interval}s polls, stall window {a.stall_seconds}s, restart gap {a.min_restart_gap}s")
    last_progress = time.time()
    last_gen = None
    last_pr = None
    last_pf = None
    last_slots = None
    down_logged = 0.0
    no_channel_logged = 0.0

    while True:
        m = http_json(f"http://127.0.0.1:{a.port}/metrics")
        if "_error" in m:
            if time.time() - down_logged > 300:
                log(f"server unreachable ({m['_error'][:80]}) - not restarting (operator's call)")
                down_logged = time.time()
            last_gen, last_pf, last_slots = None, None, None
            last_progress = time.time()
            time.sleep(a.interval)
            continue

        live = m.get("live", {}) or {}
        state = live.get("state")
        gen = live.get("generated")
        active = live.get("active_requests") or 0
        pr = live.get("prompt_read")        # real prefill progress (chunk reads), multiplex-aware
        pf = prefill_progress()
        # CRIT-SCHED-05: Monitor per-slot progress across all concurrent slots
        slots = m.get("slots") or live.get("slots") or []
        slots_progress = tuple(
            (s.get("id"), s.get("read") or s.get("prompt_read"), s.get("tokens") or s.get("generated"))
            for s in slots
        )
        if (gen != last_gen or (pr is not None and pr != last_pr) or (pf is not None and pf != last_pf) or
            (slots_progress and slots_progress != last_slots)):
            last_progress = time.time()
        if not active:
            last_progress = time.time()
        last_gen, last_pr, last_pf, last_slots = gen, pr, pf, slots_progress

        stuck = time.time() - last_progress
        if active and stuck > a.stall_seconds:
            if state == "reading" and pr is None and pf is None:
                # No channel that MOVES during a concurrent prefill is visible (the server log
                # line carries a constant token count there; live.prompt_read needs the
                # serve/server.py progress patch).  Restarting on this silence killed a legitimate
                # 432 s prefill on 2026-10-01 - stand down and say so.
                if time.time() - no_channel_logged > 300:
                    log(f"silence {int(stuck)}s while reading, but no real prefill progress channel "
                        f"is visible (live.prompt_read is null - server patch missing?); standing down, NOT restarting")
                    no_channel_logged = time.time()
                last_progress = time.time()
            else:
                try:
                    age = time.time() - os.path.getmtime(MARKER)
                except OSError:
                    age = 10**9
                if age < a.min_restart_gap:
                    log(f"stall-like silence ({int(stuck)}s) but a sanctioned restart is {int(age)}s old - standing down")
                    last_progress = time.time()
                else:
                    cfg, src = current_config()
                    if cfg is None:
                        log(f"STALL: active={active} generated={gen} prefill={pf} no progress {int(stuck)}s, but no "
                            f"usable campaign config pointer ({CONFIG_PTR} / {FALLBACK_CONFIG}) - NOT restarting (operator's call)")
                        last_progress = time.time()
                    else:
                        log(f"STALL: active={active} generated={gen} prefill={pf} prompt_read={pr} no progress "
                            f"{int(stuck)}s - restarting 8096 with {cfg} (pointer: {src})")
                        snapshot(a.port)
                        start_script = os.path.join(os.path.dirname(cfg), "start_server.sh")
                        if os.path.exists(start_script):
                            out = sh(f"bash {start_script}", timeout=500)
                        else:
                            out = f"no start script at {start_script} - NOT restarted"
                        log("restart result: " + out.strip().replace("\n", " | ")[:400])
                        last_progress = time.time()
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
