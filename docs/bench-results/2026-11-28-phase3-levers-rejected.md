---
title: Phase 3 L1/L2 online gate - clean rejection and revert
truth_class: evidence
project: local-agent-lab
updated: 2026-11-28
status: "REJECTED - both levers deleted from source; baseline binary/config retained"
---

# Phase 3 — `STRATA_PF_FUSED_SMEM_LB` / `STRATA_PIPED_SPEC_COST` online rejection

Both opt-in levers in commit `bae0172bf` failed the live serving gate. Per
`strata-lever-measurement`, a fast change that fails its gate is deleted, not tuned and not left in
the tree switched off. Both are reverted to `a315e6e0a` source; no flag or registration survives.

## Evidence

Hardware: `ai-server`, 4x RTX 4090, clean after power cycle, `nvtop` absent, 0 Xids throughout.
Every service transition used the zero-NVML PID-only teardown gate (PID exit, 7 s silent settle,
ONE `nvidia-smi`, 5 s settle); `AGX_GPU_POLL_S >= 20.0`, `AGX_ENABLE_DMON=0`.

1. Interleaved matrix `A1 / L1 / L12 / A2` (`/srv/lab/scratch/2026-11-27-phase3/online2/`):
   - `A1` and `A2` (bin `4573e5df`, cfg `d97f39c2`) passed `burst65`, `burst196` and Suite-D alpha:
     8/8 needles, 0 leaks, 0 Xids (e.g. `A1` burst65 361.17 turns/min, p50 609.28 ms;
     `A2` burst65 363.00 turns/min, p50 608.11 ms).
   - Both candidate arms booted with the expected `/proc/<pid>/exe` SHA and exact flag values in
     `/proc/<pid>/environ`, then **every** probe returned HTTP 503
     (`Pre-flight slot hygiene failed: HTTP 503 x8`); journal recorded
     `ERROR: strata-host exited under live server.py; terminating for restart`.
2. Binary provenance was ruled out: a clean `--clean-first` rebuild of the same commit produced a
   **different** binary (`a24e00904687f43c800bbb1ce87de556cfa266fb624fa67cfd9ba36de68fb7fc`) from
   the stale `ed547fba4f75d7e580190f88ce9890a906739150b0d1d2ea3ebeec00d591df16` used in step 1.
3. Focused control with the fresh binary (`/srv/lab/scratch/2026-11-27-phase3/smoke_fresh/`):
   - candidate + L1-only config (`STRATA_PF_FUSED_SMEM_LB=1`, verified in `/proc`): health loaded,
     first request returned 503 with `the engine stopped unexpectedly (exit code 1)`, and
     `pgrep -x strata-host` was empty immediately afterwards.
   - the **same binary** with the unflagged baseline config returned HTTP 200 and stayed alive;
     the baseline binary with the baseline config also returned HTTP 200.
   - Therefore the crash is caused by the L1 flag, not by a stale build, the gateway, or the host.

No speed number exists for either lever: an engine that dies on its first request cannot be A/B'd.
No quality battery was spent on a lever already rejected by its primary gate.

## Disposition

- `engine/src/prefill/moe_fused_iq.cu`, `engine/src/program/generate.cpp`,
  `terminal/data/settings-register.json`, `docs/SETTINGS.md` restored byte-for-byte to
  `a315e6e0a` (the manual `## Serve-side image reader switches (Terminal)` footer is intact).
- Rebuilt Release (CUDA 12.4.131, arch `86;89`); the rebuilt binary matches the served baseline
  `4573e5df68895b99d9d78fea37ff4bf30e3f276eb03f721b5317c09cb744ce16`. CUDA clean rebuilds
  are not byte-deterministic on this host (`b2d33e43...`), so identity is proven by the empty
  source diff against `a315e6e0a` and by zero Phase 3 flag strings in the rebuilt binary.
- Served binary/config remain `/usr/local/bin/strata` `4573e5df...` and
  `/srv/lab/recipes/stock-q3-s/config.json` `d97f39c2...`, service active, `NRestarts=0`, 0 Xids.
- See also [[2026-11-28-phase3-q4-imma-rejected]] for the third lever's clean rejection.

Raw artifacts: `/srv/lab/scratch/2026-11-27-phase3/{online2,online3,rebuild,revert_rebuild,diag,diag_l1,smoke_fresh}/`.
Board record: swarm 57e8a06a seq 37-40.
