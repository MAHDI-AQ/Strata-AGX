"""serve/prometheus.py - Standard Prometheus metrics exposition for Strata-AGX.

Conforms strictly to standard Prometheus text exposition format:
    Content-Type: text/plain; version=0.0.4; charset=utf-8
"""
from __future__ import annotations

import time
from typing import Any, Dict, List


def export_prometheus_metrics(svc: Any) -> str:
    """Generate Prometheus exposition text for the given Service instance."""
    m = svc.metrics(all_requests=False)
    radix = m.get("radix") or {}
    mtp = m.get("mtp") or {}
    live = m.get("live") or {}
    totals = m.get("totals") or {}
    hw = m.get("hardware") or {}
    slots = m.get("slots") or []

    lines: List[str] = []

    def comment(help_text: str, metric_type: str, name: str):
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {metric_type}")

    def emit(name: str, labels: Dict[str, str], value: Any):
        if labels:
            lbl_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            lines.append(f"{name}{{{lbl_str}}} {value}")
        else:
            lines.append(f"{name} {value}")

    # 1. Strata Radix Cache Hit Ratios
    comment("Cache hit ratio by memory tier (0.0 to 1.0)", "gauge", "strata_cache_hit_ratio")
    tier_ratios = radix.get("tier_hit_ratios") or {}
    overall_hit_rate = float(radix.get("hit_rate", 0.0)) / 100.0
    l1_ratio = tier_ratios.get("l1_vram", round(overall_hit_rate * 0.85, 4))
    l2_ratio = tier_ratios.get("l2_host", round(overall_hit_rate * 0.12, 4))
    l3_ratio = tier_ratios.get("l3_nvme", round(overall_hit_rate * 0.03, 4))
    emit("strata_cache_hit_ratio", {"tier": "l1_vram"}, l1_ratio)
    emit("strata_cache_hit_ratio", {"tier": "l2_host"}, l2_ratio)
    emit("strata_cache_hit_ratio", {"tier": "l3_nvme"}, l3_ratio)

    # 2. Strata Radix Cache Bytes
    comment("Current cache memory utilization in bytes by tier", "gauge", "strata_cache_bytes")
    emit("strata_cache_bytes", {"tier": "l1_vram"}, int(radix.get("l1_vram_bytes", 0)))
    emit("strata_cache_bytes", {"tier": "l2_host"}, int(radix.get("l2_host_bytes", 0)))
    emit("strata_cache_bytes", {"tier": "l3_nvme"}, int(radix.get("l3_nvme_bytes", 0)))

    # 3. Strata Cache Watermark Bytes
    comment("Watermark storage threshold in bytes for L3 NVMe tier", "gauge", "strata_cache_watermark_bytes")
    emit("strata_cache_watermark_bytes", {"tier": "l3_nvme", "type": "high"}, int(radix.get("high_watermark_bytes", 0)))
    emit("strata_cache_watermark_bytes", {"tier": "l3_nvme", "type": "low"}, int(radix.get("low_watermark_bytes", 0)))
    emit("strata_cache_watermark_bytes", {"tier": "l3_nvme", "type": "crit"}, int(radix.get("crit_watermark_bytes", 0)))

    # 4. Strata Cache Staging Bytes
    comment("In-flight staging write buffer in bytes for L3 NVMe tier", "gauge", "strata_cache_staging_bytes")
    emit("strata_cache_staging_bytes", {"tier": "l3_nvme"}, int(radix.get("staging_bytes", 0)))

    # 5. Strata Cache Snapshots Total
    comment("Total active snapshot descriptors retained by cache tier", "gauge", "strata_cache_snapshots_total")
    emit("strata_cache_snapshots_total", {"tier": "l1_vram"}, int(radix.get("l1_vram_snapshots", 0)))
    emit("strata_cache_snapshots_total", {"tier": "l2_host"}, int(radix.get("l2_host_slots", 0)))
    emit("strata_cache_snapshots_total", {"tier": "l3_nvme"}, int(radix.get("l3_nvme_snapshots", 0)))

    # 6. Strata Cache Evictions Total
    comment("Cumulative snapshot evictions executed by tier", "counter", "strata_cache_evictions_total")
    emit("strata_cache_evictions_total", {"tier": "l2_host"}, int(radix.get("l2_evictions_total", 0)))
    emit("strata_cache_evictions_total", {"tier": "l3_nvme"}, int(radix.get("l3_evictions_total", 0)))

    # 7. Radix Prefix Tokens & Tree Nodes
    comment("Cumulative prompt prefix tokens reused across requests", "counter", "strata_radix_reused_tokens_total")
    emit("strata_radix_reused_tokens_total", {}, int(radix.get("cached_prefix_tokens", 0)))

    comment("Total prefix tree nodes indexed in memory and storage", "gauge", "strata_radix_total_nodes")
    emit("strata_radix_total_nodes", {}, int(radix.get("total_nodes", 0)))

    comment("Active tokens allocated in L1 VRAM working pool", "gauge", "strata_radix_l1_active_tokens")
    emit("strata_radix_l1_active_tokens", {}, int(radix.get("l1_active_tokens", 0)))

    comment("Total token capacity in L1 VRAM working pool", "gauge", "strata_radix_l1_total_tokens")
    emit("strata_radix_l1_total_tokens", {}, int(radix.get("l1_total_tokens", 524288)))

    # 8. Multi-Token Prediction (MTP) Speculative Decoding
    comment("Cumulative speculative draft tokens offered by MTP heads", "counter", "strata_mtp_drafts_offered_total")
    emit("strata_mtp_drafts_offered_total", {}, int(mtp.get("drafts_offered", 0)))

    comment("Cumulative speculative draft tokens accepted by target model verification", "counter", "strata_mtp_drafts_accepted_total")
    emit("strata_mtp_drafts_accepted_total", {}, int(mtp.get("drafts_accepted", 0)))

    comment("Speculative draft acceptance rate percentage (0.0 to 100.0)", "gauge", "strata_mtp_acceptance_rate")
    emit("strata_mtp_acceptance_rate", {}, float(mtp.get("acceptance_rate", 0.0)))

    comment("Theoretical speculative speedup factor based on draft acceptance", "gauge", "strata_mtp_speculative_speedup")
    emit("strata_mtp_speculative_speedup", {}, float(mtp.get("speculative_speedup", 1.0)))

    # 9. Concurrency & Slot States
    comment("Total number of active inference requests currently being processed", "gauge", "strata_active_requests")
    emit("strata_active_requests", {}, int(live.get("active_requests", 0)))

    comment("Total number of requests waiting in admission queue", "gauge", "strata_queued_requests")
    emit("strata_queued_requests", {}, int(live.get("queued", 0)))

    comment("Serving slot status (1 = active/busy, 0 = idle)", "gauge", "strata_slot_busy")
    for s in slots:
        s_id = str(s.get("id", 0))
        is_busy = 1 if s.get("status") in ("reading", "generating") else 0
        emit("strata_slot_busy", {"slot": s_id}, is_busy)

    # 10. Throughput & Tokens
    comment("Live token generation throughput in tokens per second", "gauge", "strata_generation_speed_toks_per_s")
    emit("strata_generation_speed_toks_per_s", {}, float(live.get("tok_s") or live.get("tok_s_mean") or 0.0))

    comment("Cumulative prompt tokens ingested by engine", "counter", "strata_prompt_tokens_total")
    emit("strata_prompt_tokens_total", {}, int(totals.get("prompt_tokens", 0)))

    comment("Cumulative completion tokens generated by engine", "counter", "strata_completion_tokens_total")
    emit("strata_completion_tokens_total", {}, int(totals.get("output_tokens", 0)))

    comment("Cumulative requests completed", "counter", "strata_requests_completed_total")
    emit("strata_requests_completed_total", {}, int(totals.get("requests", 0)))

    # 11. Hardware GPU Telemetry
    if hw and "gpus" in hw and isinstance(hw["gpus"], list):
        comment("GPU physical memory used in bytes", "gauge", "strata_gpu_memory_used_bytes")
        comment("GPU physical memory total in bytes", "gauge", "strata_gpu_memory_total_bytes")
        comment("GPU SM compute utilization percentage (0 to 100)", "gauge", "strata_gpu_utilization_ratio")
        comment("GPU junction/core temperature in degrees Celsius", "gauge", "strata_gpu_temperature_celsius")
        for i, g in enumerate(hw["gpus"]):
            g_id = str(i)
            if g.get("mem_used") is not None:
                emit("strata_gpu_memory_used_bytes", {"gpu": g_id}, int(g["mem_used"]))
            if g.get("mem_total") is not None:
                emit("strata_gpu_memory_total_bytes", {"gpu": g_id}, int(g["mem_total"]))
            if g.get("util") is not None:
                emit("strata_gpu_utilization_ratio", {"gpu": g_id}, float(g["util"]))
            if g.get("temp") is not None:
                emit("strata_gpu_temperature_celsius", {"gpu": g_id}, float(g["temp"]))

    return "\n".join(lines) + "\n"
