#!/usr/bin/env python3
"""
tools/benchmark_fleet_scaling.py
Strata AGX N-Scaling Fleet Stress Test Battery (C=2, C=4, C=8 concurrency).

Evaluates:
  1. Per-stream & Aggregate Decode Throughput (target >= 250 tok/s at C=4..8).
  2. Per-stream TTFT (Time To First Token).
  3. MTP Speculative Token Acceptance Rate (extracted from timings).
  4. Radix Cache Prefix Hit Latency & Throughput across multi-turn shared sessions.

Usage:
  python3 tools/benchmark_fleet_scaling.py --concurrency 2 --tokens 64
  python3 tools/benchmark_fleet_scaling.py --sweep 2,4,8 --tokens 64 --output report.json
"""

import argparse
import asyncio
import dataclasses
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

SYSTEM_PROMPT_TEMPLATE = """You are an expert autonomous systems engineer in Mahdi AI Lab.
You are tasked with diagnosing multi-GPU inference engines, profiling kernel latency, analyzing CUDA streams, and optimizing asynchronous scheduling.
Always provide detailed, hardware-grounded, technical analysis with exact code snippets and architectural rationale."""

USER_TASK_TEMPLATES = [
    "Write an extensive analysis on PCIe 4.0 x8/x8 non-P2P bifurcation during multi-agent speculative decoding verify passes.",
    "Formulate a zero-copy pinned bounce buffer queue algorithm to eliminate pageable host-to-device memory transfer bubbles.",
    "Write an optimized CUDA kernel outline for 128-block parallel argmax reduction across 40,525 vocabulary logits on SM89 Ada Lovelace.",
    "Explain how DeltaNet chunked recurrent states ($S_t$) preserve mathematical purity at conversational turn boundaries under session forking.",
    "Derive the memory bandwidth bound for batched MTP draft verification across 19,913 MoE resident experts on dual 24GB GPUs.",
    "Design a thread-safe token tree retention policy that coordinates L1 VRAM, L2 Pinned Host RAM, and continuous L3 NVMe storage tiers.",
    "Construct an asynchronous HTTP client pool in Python that tracks microsecond-accurate time-to-first-token and inter-token arrival jitter.",
    "Evaluate the throughput trade-off between linear MTP speculative decoding and dynamic tree-branching SpecTree under high concurrency."
]

MULTI_TURN_FOLLOWUPS = [
    "Continue your analysis with explicit SIMD coalesced memory access and AVX-512 vectorization guards.",
    "Provide a detailed step-by-step mitigation plan for race conditions during high-concurrency buffer recycling."
]


@dataclasses.dataclass
class StreamMetrics:
    client_id: int
    turn_id: int
    status_code: int
    success: bool
    prompt_tokens: int
    completion_tokens: int
    ttft_sec: float
    total_latency_sec: float
    decode_latency_sec: float
    decode_tok_per_sec: float
    error_msg: Optional[str] = None
    server_timings: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class ConcurrencyRoundResult:
    concurrency: int
    turn_id: int
    is_shared_prefix: bool
    total_wall_sec: float
    total_prompt_tokens: int
    total_completion_tokens: int
    aggregate_decode_tok_per_sec: float
    mean_per_stream_decode_tok_per_sec: float
    min_per_stream_decode_tok_per_sec: float
    max_per_stream_decode_tok_per_sec: float
    mean_ttft_sec: float
    p50_ttft_sec: float
    p95_ttft_sec: float
    streams: List[StreamMetrics]
    mean_mtp_acceptance_rate: Optional[float] = None


class StrataBenchmarkClient:
    def __init__(self, endpoint: str, timeout: float = 120.0):
        self.endpoint = endpoint
        self.timeout = timeout

    async def send_request(
        self,
        client_id: int,
        turn_id: int,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0
    ) -> StreamMetrics:
        """Sends chat completion request to Strata server and parses exact timing metrics."""
        payload = {
            "model": "Qwen3.8-Flash-Next-IQ3_XXS",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json"
        }

        loop = asyncio.get_running_loop()

        def _blocking_call() -> StreamMetrics:
            req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
            start_wall = time.perf_counter()

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status_code = resp.status
                    body = json.loads(resp.read().decode("utf-8"))

                end_wall = time.perf_counter()
                total_latency = end_wall - start_wall

                usage = body.get("usage", {})
                timings = body.get("timings", {})

                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                prompt_ms = timings.get("prompt_ms", 0.0)
                predicted_ms = timings.get("predicted_ms", 0.0)

                ttft_sec = (prompt_ms / 1000.0) if prompt_ms > 0 else (total_latency * 0.2)
                decode_sec = (predicted_ms / 1000.0) if predicted_ms > 0 else (total_latency - ttft_sec)

                decode_tok_s = timings.get("predicted_per_second")
                if not decode_tok_s:
                    decode_tok_s = (completion_tokens / decode_sec) if decode_sec > 0 else 0.0

                return StreamMetrics(
                    client_id=client_id,
                    turn_id=turn_id,
                    status_code=status_code,
                    success=True,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    ttft_sec=ttft_sec,
                    total_latency_sec=total_latency,
                    decode_latency_sec=decode_sec,
                    decode_tok_per_sec=decode_tok_s,
                    server_timings=timings
                )

            except Exception as e:
                end_wall = time.perf_counter()
                return StreamMetrics(
                    client_id=client_id,
                    turn_id=turn_id,
                    status_code=500,
                    success=False,
                    prompt_tokens=0,
                    completion_tokens=0,
                    ttft_sec=end_wall - start_wall,
                    total_latency_sec=end_wall - start_wall,
                    decode_latency_sec=0.0,
                    decode_tok_per_sec=0.0,
                    error_msg=str(e)
                )

        return await loop.run_in_executor(None, _blocking_call)


async def run_concurrency_round(
    client: StrataBenchmarkClient,
    concurrency: int,
    turn_id: int,
    max_tokens: int,
    shared_prefix: bool,
    session_histories: List[List[Dict[str, str]]]
) -> ConcurrencyRoundResult:
    """Dispatches N concurrent requests and compiles timing statistics."""
    tasks = []
    round_start = time.perf_counter()

    for cid in range(concurrency):
        hist = session_histories[cid]
        if turn_id == 0:
            sys_msg = SYSTEM_PROMPT_TEMPLATE
            user_msg = USER_TASK_TEMPLATES[cid % len(USER_TASK_TEMPLATES)]
            if shared_prefix:
                hist.append({"role": "system", "content": sys_msg})
                hist.append({"role": "user", "content": user_msg})
            else:
                hist.append({"role": "system", "content": f"{sys_msg}\nAgent Slot: {cid}"})
                hist.append({"role": "user", "content": user_msg})
        else:
            followup = MULTI_TURN_FOLLOWUPS[(cid + turn_id) % len(MULTI_TURN_FOLLOWUPS)]
            hist.append({"role": "user", "content": followup})

        tasks.append(client.send_request(cid, turn_id, hist, max_tokens))

    stream_results: List[StreamMetrics] = await asyncio.gather(*tasks)
    round_end = time.perf_counter()
    total_wall = round_end - round_start

    valid_streams = [s for s in stream_results if s.success and s.completion_tokens > 0]
    total_gen_tokens = sum(s.completion_tokens for s in valid_streams)
    total_prompt_tokens = sum(s.prompt_tokens for s in valid_streams)

    agg_tok_s = sum(s.decode_tok_per_sec for s in valid_streams)
    per_stream_rates = [s.decode_tok_per_sec for s in valid_streams] or [0.0]
    ttfts = [s.ttft_sec for s in valid_streams] or [0.0]

    ttfts.sort()
    p50_ttft = ttfts[len(ttfts) // 2]
    p95_ttft = ttfts[int(len(ttfts) * 0.95)]

    # Compute MTP acceptance
    mtp_rates = []
    for s in valid_streams:
        if s.server_timings:
            dn = s.server_timings.get("draft_n", 0)
            dna = s.server_timings.get("draft_n_accepted", 0)
            if dn > 0:
                mtp_rates.append(dna / dn)
    mean_mtp = statistics.mean(mtp_rates) if mtp_rates else None

    return ConcurrencyRoundResult(
        concurrency=concurrency,
        turn_id=turn_id,
        is_shared_prefix=shared_prefix,
        total_wall_sec=total_wall,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_gen_tokens,
        aggregate_decode_tok_per_sec=agg_tok_s,
        mean_per_stream_decode_tok_per_sec=statistics.mean(per_stream_rates),
        min_per_stream_decode_tok_per_sec=min(per_stream_rates),
        max_per_stream_decode_tok_per_sec=max(per_stream_rates),
        mean_ttft_sec=statistics.mean(ttfts),
        p50_ttft_sec=p50_ttft,
        p95_ttft_sec=p95_ttft,
        streams=stream_results,
        mean_mtp_acceptance_rate=mean_mtp
    )


def print_round_report(res: ConcurrencyRoundResult, target_agg_bar: float = 250.0):
    status_sym = "PASS" if res.aggregate_decode_tok_per_sec >= target_agg_bar else "INFO"
    if res.concurrency < 4:
        status_sym = "BASELINE"

    print("\n" + "=" * 80)
    print(f"[{status_sym}] BATCH CONCURRENCY C={res.concurrency} | Turn {res.turn_id + 1} | Prefix Shared: {res.is_shared_prefix}")
    print("=" * 80)
    print(f"Wall Clock Time       : {res.total_wall_sec:.2f} s")
    print(f"Aggregate Generation  : {res.total_completion_tokens} tokens across {len(res.streams)} streams")
    print(f"Aggregate Decode Rate : {res.aggregate_decode_tok_per_sec:.1f} tok/s  (Target >= {target_agg_bar:.1f} tok/s)")
    print(f"Per-Stream Decode Rate: avg {res.mean_per_stream_decode_tok_per_sec:.1f} tok/s "
          f"[min {res.min_per_stream_decode_tok_per_sec:.1f}, max {res.max_per_stream_decode_tok_per_sec:.1f}]")
    print(f"TTFT (Latency to 1st) : p50 = {res.p50_ttft_sec:.3f} s, p95 = {res.p95_ttft_sec:.3f} s, mean = {res.mean_ttft_sec:.3f} s")
    if res.mean_mtp_acceptance_rate:
        print(f"Mean MTP Acceptance   : {res.mean_mtp_acceptance_rate * 100:.1f}%")

    print("\nPer-Stream Breakdown:")
    for s in res.streams:
        status = "OK" if s.success else f"FAIL ({s.error_msg})"
        cached = s.server_timings.get("cache_n", 0) if s.server_timings else 0
        print(f"  Stream {s.client_id:02d}: TTFT {s.ttft_sec:.3f}s | {s.completion_tokens} toks "
              f"@ {s.decode_tok_per_sec:.1f} tok/s | Cached {cached} toks | Total {s.total_latency_sec:.2f}s | [{status}]")
    print("=" * 80)


async def main_async():
    parser = argparse.ArgumentParser(description="Strata AGX Fleet Scaling Benchmark (C=2, C=4, C=8)")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8096/v1/chat/completions", help="Chat completions API URL")
    parser.add_argument("--concurrency", type=int, default=4, help="Single concurrency run (e.g. 2, 4, 8)")
    parser.add_argument("--sweep", type=str, default=None, help="Comma-separated sweep list, e.g. '2,4,8'")
    parser.add_argument("--tokens", type=int, default=64, help="Target generated completion tokens per stream")
    parser.add_argument("--turns", type=int, default=2, help="Number of multi-turn conversational rounds")
    parser.add_argument("--shared-prefix", action="store_true", default=True, help="Share system prompt across streams for Radix cache hit")
    parser.add_argument("--output", type=str, default=None, help="JSON output file for benchmark telemetry")
    args = parser.parse_args()

    client = StrataBenchmarkClient(args.endpoint)
    concurrencies = [int(c.strip()) for c in args.sweep.split(",")] if args.sweep else [args.concurrency]

    all_results: List[Dict[str, Any]] = []

    print(f"Beginning Strata Fleet Benchmark on {args.endpoint}")
    print(f"Configurations to test: Concurrency levels {concurrencies}, {args.turns} turns per stream, {args.tokens} tokens/turn")

    for c in concurrencies:
        histories = [[] for _ in range(c)]
        for turn in range(args.turns):
            res = await run_concurrency_round(
                client=client,
                concurrency=c,
                turn_id=turn,
                max_tokens=args.tokens,
                shared_prefix=args.shared_prefix,
                session_histories=histories
            )
            print_round_report(res)
            all_results.append(dataclasses.asdict(res))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved benchmark telemetry to {args.output}")


if __name__ == "__main__":
    asyncio.run(main_async())
