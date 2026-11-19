#!/usr/bin/env python3
"""
Pre-Optimization 4-Slot 800k Quality & Cross-Session Isolation Baseline Harness
Evaluates 4 independent concurrent sessions across 4 deep context tiers (100k, 250k, 500k, 800k tokens per slot).
Aggregate context up to 3.2M tokens.
"""

import argparse
import ast
import concurrent.futures
import os
import json
from pathlib import Path
import re
import ssl
import subprocess
import sys
import time
import urllib.request

# SSL context for HTTPS endpoints
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

GATEWAY_URL = os.environ.get('STRATA_URL', 'http://127.0.0.1:8080/v1/chat/completions')
METRICS_URL = 'http://127.0.0.1:8096/metrics'
KEY_PATH = Path(os.environ.get('STRATA_API_KEY_FILE', '')) if os.environ.get('STRATA_API_KEY_FILE') else None
RESULTS_PATH = Path('quality_isolation_results.json')
OUTPUT_DIR = Path('.')

# Domain Knowledge Bases for Haystack Generation
DOMAINS = {
    'alpha': {
        'slot': 0,
        'session_id': 'session_alpha',
        'domain_name': 'Linux Kernel & Systems Architecture',
        'code': 'OMEGA-9918-AURORA',
        'depth': 0.75,
        'needle': 'The secret verification code for Session Alpha is OMEGA-9918-AURORA.',
        'cross_target_name': 'Session Beta',
        'foreign_codes': ['FALCON-4421-NEBULA', 'VORTEX-7733-COBALT', 'ZEPHYR-1188-GLACIER'],
        'sentences': [
            'The eBPF virtual machine enforces bytecode verification through abstract interpretation before in-kernel JIT execution.',
            'Page cache readahead dynamically scales read window size according to sequential access pattern heuristics.',
            'The completely fair scheduler utilizes red-black trees to track virtual runtime progression across runnable tasks.',
            'Virtual memory allocation divides the address space into page tables with multi-level translation lookaside buffers.',
            'Cgroup v2 unified hierarchy provides single-tree resource controllers for memory limits and CPU bandwidth throttles.',
            'Direct memory access controllers offload high-throughput packet transfers from ring buffers to host memory regions.',
            'The slab allocator groups small kernel objects into contiguous caches to eliminate internal fragmentation overhead.',
            'Synchronous read operations bypass intermediate buffer copies through zero-copy splice and vmsplice primitives.',
            'Lockless circular ring buffers maintain atomic head and tail pointers to coordinate high-speed tracing streams.',
            'Netfilter hook points inspect network frames at ingress and egress stages before routing decision evaluation.'
        ],
        'reasoning_prompt': (
            'Implement a thread-safe LRU Cache with TTL (Time-To-Live) expiration in Python. '
            'Use collections.OrderedDict and threading.Lock. '
            'Provide methods: get(key), put(key, value, ttl_seconds), and cleanup_expired(). '
            'Write clean, production-grade Python code inside a ```python code block.'
        ),
        'reasoning_type': 'python_code',
        'reasoning_checks': ['get', 'put']
    },
    'beta': {
        'slot': 1,
        'session_id': 'session_beta',
        'domain_name': 'Quantum Information & Physics',
        'code': 'FALCON-4421-NEBULA',
        'depth': 0.25,
        'needle': 'The secret verification code for Session Beta is FALCON-4421-NEBULA.',
        'cross_target_name': 'Session Gamma',
        'foreign_codes': ['OMEGA-9918-AURORA', 'VORTEX-7733-COBALT', 'ZEPHYR-1188-GLACIER'],
        'sentences': [
            'Density matrix formalisms describe statistical mixtures and entangled states in composite Hilbert tensor products.',
            'Fault-tolerant quantum computing relies on topological surface code parity checks to suppress physical bit flips.',
            'Unitary transformation matrices preserve Hermitian inner products across continuous state space evolutions.',
            'Superconducting transmon qubits couple nonlinearly to microwave resonators through Josephson junction inductance.',
            'Quantum error correction syndromes localize phase-flip and bit-flip anomalies without destroying superposition phases.',
            'The Grover diffusion operator amplifies target state amplitudes through iterative inversion about the mean.',
            'Non-demolition measurements detect photon number eigenstates while preserving quantum coherence in optical cavities.',
            'Rydberg atom arrays leverage dipole blockade interactions to implement programmable multi-qubit entangling gates.',
            'Clifford group operations generate stabilizer state subgroups capable of efficient classical polynomial simulation.',
            'Decoherence channels model environmental thermalization via Lindblad master equation generator superoperators.'
        ],
        'reasoning_prompt': (
            'Solve this 5-step workstation assignment logic puzzle:\n'
            'Five workstations (A, B, C, D, E) are assigned to positions 1 to 5 from left to right.\n'
            '1. Workstation D is at position 4.\n'
            '2. Workstation B is immediately to the right of Workstation A.\n'
            '3. Workstation C is at position 1.\n'
            '4. Workstation E is at position 5.\n'
            'Determine the exact workstation at each position 1 to 5. State the final assignment clearly in the format:\n'
            'Position 1: [workstation]\nPosition 2: [workstation]\nPosition 3: [workstation]\nPosition 4: [workstation]\nPosition 5: [workstation]'
        ),
        'reasoning_type': 'logic_puzzle',
        'reasoning_checks': ['Position 1: C', 'Position 2: A', 'Position 3: B', 'Position 4: D', 'Position 5: E']
    },
    'gamma': {
        'slot': 2,
        'session_id': 'session_gamma',
        'domain_name': 'Compiler Engineering & Formal Type Theory',
        'code': 'VORTEX-7733-COBALT',
        'depth': 0.50,
        'needle': 'The secret verification code for Session Gamma is VORTEX-7733-COBALT.',
        'cross_target_name': 'Session Delta',
        'foreign_codes': ['OMEGA-9918-AURORA', 'FALCON-4421-NEBULA', 'ZEPHYR-1188-GLACIER'],
        'sentences': [
            'Dominance frontier algorithms place minimal phi functions at merge nodes in static single assignment representation.',
            'Linear type systems enforce single-use capabilities to guarantee memory safety without garbage collection pauses.',
            'Iterated register coalescing combines live range interference graph coloring with conservative spilling heuristics.',
            'The Hindley-Milner type inference engine unifies polymorphic type variables through Robinson first-order equations.',
            'Continuation-passing style transformations make control flow explicit and turn return jumps into tail calls.',
            'Abstract interpretation approximates program semantics over complete lattices to prove absence of runtime faults.',
            'Instruction scheduling reorganizes DAG operations to minimize pipeline stalls and maximize functional unit occupancy.',
            'Parametric polymorphism in System F models universal quantification over arbitrary concrete data representations.',
            'Loop invariant code motion identifies loop-invariant expressions and hoists them into loop preheader blocks.',
            'Defunctionalization converts higher-order closures into first-order data types with dispatch evaluators.'
        ],
        'reasoning_prompt': (
            'Implement an AVL Tree insertion algorithm with automatic balance factor recalculation, '
            'left rotation, and right rotation in Python. '
            'Write clean, production-grade Python code inside a ```python code block.'
        ),
        'reasoning_type': 'python_code',
        'reasoning_checks': ['insert']
    },
    'delta': {
        'slot': 3,
        'session_id': 'session_delta',
        'domain_name': 'Astrophysics & Relativistic Mechanics',
        'code': 'ZEPHYR-1188-GLACIER',
        'depth': 0.90,
        'needle': 'The secret verification code for Session Delta is ZEPHYR-1188-GLACIER.',
        'cross_target_name': 'Session Alpha',
        'foreign_codes': ['OMEGA-9918-AURORA', 'FALCON-4421-NEBULA', 'VORTEX-7733-COBALT'],
        'sentences': [
            'Kerr black hole spacetime geometry exhibits frame dragging within the ergosphere bounded by the static limit.',
            'Neutron star crust matter undergoes pasta phase transitions at nuclear saturation densities near core interfaces.',
            'Accretion disk magneto-rotational instability drives turbulent angular momentum transport toward the central compact object.',
            'Gravitational wave chirps from binary inspirals encode component masses and effective spin alignment parameters.',
            'Primordial nucleosynthesis limits cosmic baryon density based on primordial deuterium and helium isotope ratios.',
            'Relativistic jet collimation is governed by magnetic poloidal flux threading the event horizon ergosphere.',
            'Chandrasekhar mass boundaries constrain stable white dwarf configurations against relativistic electron degeneracy.',
            'Cosmic microwave background acoustic oscillations reflect sound horizon scales at the epoch of hydrogen recombination.',
            'Tolman-Oppenheimer-Volkoff hydrostatic equations govern internal pressure balance in general relativistic stars.',
            'Synchrotron emission spectra from relativistic electrons reveal magnetic field strengths in supernova remnant shells.'
        ],
        'reasoning_prompt': (
            'Implement Dijkstra shortest-path algorithm in Python using heapq. '
            'The function `dijkstra(graph, start)` should return both the distances dictionary '
            'and previous_nodes dictionary for full path reconstruction. '
            'Write clean, production-grade Python code inside a ```python code block.'
        ),
        'reasoning_type': 'python_code',
        'reasoning_checks': ['dijkstra', 'heapq', 'heappop', 'heappush']
    }
}


def load_api_key(key_file=None):
    if key_file:
        return Path(key_file).read_text().strip()
    if KEY_PATH and KEY_PATH.exists():
        return KEY_PATH.read_text().strip()
    return os.environ.get('STRATA_API_KEY', '')


def query_metrics():
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=5) as r:
            return json.loads(r.read().decode('utf-8'))
    except Exception as e:
        return {'_error': str(e)}


def query_nvidia_smi():
    try:
        res = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
             '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        lines = [line.strip() for line in res.stdout.strip().split('\n') if line.strip()]
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            gpus.append({
                'index': int(parts[0]),
                'name': parts[1],
                'memory_total': parts[2],
                'memory_used': parts[3],
                'memory_free': parts[4],
                'utilization': parts[5]
            })
        return gpus
    except Exception as e:
        return [{'error': str(e)}]


def query_host_mem():
    try:
        res = subprocess.run(['free', '-h'], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        return str(e)


TOKENIZER_PATH = os.environ.get('STRATA_TOKENIZER', '')
MODEL_NAME = os.environ.get('STRATA_MODEL', 'strata')

def calibrate_domains(tok_path_arg=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from conversation_cache_parity import load_tokenizer
    tok_path = Path(tok_path_arg or TOKENIZER_PATH)
    tok = load_tokenizer(tok_path)
    for k, d in DOMAINS.items():
        s = d['sentences']
        lines = [f'Document Section {i // len(s)}.{i % len(s)}: ' + s[i % len(s)] + '\n' for i in range(100)]
        sample = ''.join(lines)
        tok_count = len(tok.encode(sample))
        d['cpt'] = len(sample) / max(tok_count, 1)
        print(f"  {k}: calibrated {d['cpt']:.3f} chars/tok", flush=True)
    return tok


def build_haystack(k, target_tokens):
    d = DOMAINS[k]
    cpt = d['cpt']
    depth = d['depth']
    target_chars = int(target_tokens * cpt)
    chars_before = int(target_chars * depth)
    chars_after = target_chars - chars_before

    s = d['sentences']
    # Build before
    b_parts = []
    curr = 0
    idx = 0
    while curr < chars_before:
        part = f'Document Section {idx // len(s)}.{idx % len(s)}: ' + s[idx % len(s)] + '\n'
        b_parts.append(part)
        curr += len(part)
        idx += 1
    text_before = ''.join(b_parts)[:chars_before]

    # Build after
    a_parts = []
    curr = 0
    while curr < chars_after:
        part = f'Document Section {idx // len(s)}.{idx % len(s)}: ' + s[idx % len(s)] + '\n'
        a_parts.append(part)
        curr += len(part)
        idx += 1
    text_after = ''.join(a_parts)[:chars_after]

    needle_block = f"\n\n{d['needle']}\n\n"
    full_text = text_before + needle_block + text_after
    actual_pos = full_text.find(d['code'])
    actual_depth = actual_pos / max(len(full_text), 1)
    return full_text, actual_depth


def post_completion(api_key, session_id, messages, max_tokens=100, enable_thinking=False, timeout=1200):
    payload = {
        'model': MODEL_NAME,
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': 0.0,
        'chat_template_kwargs': {'enable_thinking': enable_thinking},
        'session_id': session_id,
        'user': session_id
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(GATEWAY_URL, data=data, headers={
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'X-Session-ID': session_id
    })
    t0 = time.time()
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, context=CTX, timeout=timeout) as resp:
                res = json.loads(resp.read().decode('utf-8'))
                t1 = time.time()
                elapsed = round(t1 - t0, 3)
                return res, elapsed
        except urllib.error.HTTPError as e:
            if attempt == 0 and e.code in (502, 503, 504):
                print(f"[RETRY] Session {session_id} received HTTP {e.code}, waiting 5s before retry...", flush=True)
                time.sleep(5)
                continue
            raise


def verify_reasoning_output(domain_info, text):
    # Repetition check (assert no degenerate loops)
    words = text.split()
    repetition_collapsed = False
    if len(words) >= 30:
        from collections import Counter
        ngrams = [tuple(words[i:i+5]) for i in range(len(words)-4)]
        most_common = Counter(ngrams).most_common(1)
        if most_common and most_common[0][1] > 6:
            repetition_collapsed = True

    if repetition_collapsed:
        return False, "Repetition collapse detected (repetitive 5-gram loop)"

    if domain_info['reasoning_type'] == 'python_code':
        code = None
        if '```python' in text:
            code = text.split('```python', 1)[1].split('```', 1)[0].strip()
        elif '```' in text:
            code = text.split('```', 1)[1].split('```', 1)[0].strip()
        elif 'def ' in text or 'class ' in text:
            code = text.strip()

        if not code:
            return False, "No Python code found in output"

        # Try parsing full code, or trim incomplete trailing line if truncated
        lines = code.splitlines()
        parsed = False
        last_err = ""
        while lines:
            try:
                ast.parse('\n'.join(lines))
                parsed = True
                break
            except SyntaxError as e:
                last_err = str(e)
                lines.pop()

        if not parsed:
            return False, f"SyntaxError in code: {last_err}"

        # Check required functions/classes
        missing = [c for c in domain_info['reasoning_checks'] if c not in code]
        if missing:
            return False, f"Code missing required symbols: {missing}"
        return True, "Valid Python syntax & structure"

    elif domain_info['reasoning_type'] == 'logic_puzzle':
        # Check puzzle solutions
        # Position 1: C, Position 2: A, Position 3: B, Position 4: D, Position 5: E
        upper_text = text.upper()
        c1 = ('POSITION 1' in upper_text and 'C' in upper_text) or ('1: C' in upper_text) or ('1=C' in upper_text)
        c2 = ('POSITION 2' in upper_text and 'A' in upper_text) or ('2: A' in upper_text) or ('2=A' in upper_text)
        c3 = ('POSITION 3' in upper_text and 'B' in upper_text) or ('3: B' in upper_text) or ('3=B' in upper_text)
        c4 = ('POSITION 4' in upper_text and 'D' in upper_text) or ('4: D' in upper_text) or ('4=D' in upper_text)
        c5 = ('POSITION 5' in upper_text and 'E' in upper_text) or ('5: E' in upper_text) or ('5=E' in upper_text)
        score = sum([c1, c2, c3, c4, c5])
        if score >= 4:
            return True, f"Logic puzzle solved ({score}/5 assignments verified)"
        return False, f"Logic puzzle failed ({score}/5 assignments verified)"

    return True, "OK"


def run_slot_tier(domain_key, target_tokens, api_key):
    d = DOMAINS[domain_key]
    sid = d['session_id']
    code = d['code']
    foreign_codes = d['foreign_codes']

    print(f"[{domain_key.upper()}] Generating haystack ({target_tokens:,} tokens, depth={d['depth']:.2f})...", flush=True)
    haystack, actual_depth = build_haystack(domain_key, target_tokens)
    q1 = "What is the secret verification code for this session? Answer with the code only."
    prompt1 = f"{haystack}\n\nQuestion: {q1}"

    # Test A: In-Slot Retrieval Fidelity
    print(f"[{domain_key.upper()}] Test A: Sending Turn 1 (Needle Retrieval)...", flush=True)
    messages = [{'role': 'user', 'content': prompt1}]
    res1, dur1 = post_completion(api_key, sid, messages, max_tokens=60, enable_thinking=False)
    ans1 = res1['choices'][0]['message']['content']
    timings1 = res1.get('timings', {})
    usage1 = res1.get('usage', {})
    prompt_tokens1 = usage1.get('prompt_tokens', 0)
    ttft1 = timings1.get('prompt_ms', 0)
    prefill_tok_s1 = timings1.get('prompt_per_second', 0)
    decode_tok_s1 = timings1.get('predicted_per_second', 0)

    needle_match = code in ans1
    print(f"[{domain_key.upper()}] Test A finished in {dur1}s: match={needle_match}, tokens={prompt_tokens1:,}, reply=\"{ans1.strip()}\"", flush=True)

    # Test B: Adversarial Cross-Session Isolation Probe
    q2 = f"What is the secret verification code for {d['cross_target_name']}?"
    print(f"[{domain_key.upper()}] Test B: Sending Turn 2 (Adversarial Query: {q2})...", flush=True)
    messages.append({'role': 'assistant', 'content': ans1})
    messages.append({'role': 'user', 'content': q2})
    res2, dur2 = post_completion(api_key, sid, messages, max_tokens=100, enable_thinking=False)
    ans2 = res2['choices'][0]['message']['content']
    timings2 = res2.get('timings', {})
    usage2 = res2.get('usage', {})

    # Check for leaks of ANY foreign code
    leaked_codes = [fc for fc in foreign_codes if fc in ans2]
    isolation_passed = len(leaked_codes) == 0
    print(f"[{domain_key.upper()}] Test B finished in {dur2}s: isolated={isolation_passed}, leaks={leaked_codes}, reply=\"{ans2.strip()[:100]}...\"", flush=True)

    # Test C: Multi-Turn Reasoning Continuation at Deep Context
    q3 = d['reasoning_prompt']
    print(f"[{domain_key.upper()}] Test C: Sending Turn 3 (Reasoning Continuation)...", flush=True)
    messages.append({'role': 'assistant', 'content': ans2})
    messages.append({'role': 'user', 'content': q3})
    res3, dur3 = post_completion(api_key, sid, messages, max_tokens=550, enable_thinking=False)
    ans3 = res3['choices'][0]['message']['content']
    timings3 = res3.get('timings', {})
    usage3 = res3.get('usage', {})

    reasoning_passed, reasoning_note = verify_reasoning_output(d, ans3)
    print(f"[{domain_key.upper()}] Test C finished in {dur3}s: passed={reasoning_passed} ({reasoning_note}), decode_tok_s={timings3.get('predicted_per_second', 0):.1f}", flush=True)

    return {
        'domain': domain_key,
        'slot': d['slot'],
        'session_id': sid,
        'target_tokens': target_tokens,
        'actual_tokens': prompt_tokens1,
        'needle_code': code,
        'needle_target_depth': d['depth'],
        'needle_actual_depth': round(actual_depth, 3),
        'test_a': {
            'needle_match': needle_match,
            'response': ans1.strip(),
            'ttft_ms': ttft1,
            'prefill_tok_s': prefill_tok_s1,
            'decode_tok_s': decode_tok_s1,
            'duration_s': dur1
        },
        'test_b': {
            'isolation_passed': isolation_passed,
            'leaked_codes': leaked_codes,
            'query': q2,
            'response': ans2.strip()[:200],
            'duration_s': dur2,
            'decode_tok_s': timings2.get('predicted_per_second', 0)
        },
        'test_c': {
            'reasoning_passed': reasoning_passed,
            'reasoning_note': reasoning_note,
            'response_snippet': ans3.strip()[:200],
            'duration_s': dur3,
            'decode_tok_s': timings3.get('predicted_per_second', 0)
        }
    }


def run_tier(tier_name, target_tokens_per_slot, api_key):
    agg_tokens = target_tokens_per_slot * 4
    print(f"\n================================================================================", flush=True)
    print(f"STARTING {tier_name.upper()}: {target_tokens_per_slot:,} tokens/slot (~{agg_tokens:,} aggregate)", flush=True)
    print(f"================================================================================", flush=True)

    vram_before = query_nvidia_smi()
    t_start = time.time()

    keys = ['alpha', 'beta', 'gamma', 'delta']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_slot_tier, k, target_tokens_per_slot, api_key): k for k in keys}
        results = {}
        for f in concurrent.futures.as_completed(futs):
            k = futs[f]
            results[k] = f.result()

    t_end = time.time()
    tier_duration = round(t_end - t_start, 2)
    vram_after = query_nvidia_smi()
    metrics_snapshot = query_metrics()

    # Aggregate summaries
    slot_results = [results[k] for k in keys]
    accuracy_pct = 100.0 * sum(1 for r in slot_results if r['test_a']['needle_match']) / 4.0
    contamination_pct = 100.0 * sum(1 for r in slot_results if not r['test_b']['isolation_passed']) / 4.0
    reasoning_pct = 100.0 * sum(1 for r in slot_results if r['test_c']['reasoning_passed']) / 4.0

    mean_ttft = sum(r['test_a']['ttft_ms'] for r in slot_results) / 4.0
    mean_prefill_tok_s = sum(r['test_a']['prefill_tok_s'] for r in slot_results) / 4.0
    mean_decode_tok_s = sum(r['test_a']['decode_tok_s'] for r in slot_results) / 4.0

    print(f"\n--- {tier_name.upper()} SUMMARY ({tier_duration}s total) ---", flush=True)
    print(f"Needle Retrieval Accuracy: {accuracy_pct:.1f}% ({'PASS' if accuracy_pct == 100.0 else 'FAIL'})", flush=True)
    print(f"Cross-Session Contamination: {contamination_pct:.1f}% ({'PASS' if contamination_pct == 0.0 else 'FAIL - BLEED DETECTED'})", flush=True)
    print(f"Multi-Turn Reasoning Pass: {reasoning_pct:.1f}% ({'PASS' if reasoning_pct == 100.0 else 'FAIL'})", flush=True)
    print(f"Mean TTFT: {mean_ttft:.1f} ms | Mean Prefill: {mean_prefill_tok_s:.1f} tok/s | Mean Decode: {mean_decode_tok_s:.1f} tok/s", flush=True)

    return {
        'tier_name': tier_name,
        'target_tokens_per_slot': target_tokens_per_slot,
        'aggregate_target_tokens': agg_tokens,
        'tier_duration_s': tier_duration,
        'accuracy_pct': accuracy_pct,
        'contamination_pct': contamination_pct,
        'reasoning_pct': reasoning_pct,
        'mean_ttft_ms': round(mean_ttft, 1),
        'mean_prefill_tok_s': round(mean_prefill_tok_s, 1),
        'mean_decode_tok_s': round(mean_decode_tok_s, 1),
        'slots': results,
        'vram_before': vram_before,
        'vram_after': vram_after,
        'slots_metrics': metrics_snapshot.get('live', {}).get('slots', [])
    }


def main():
    parser = argparse.ArgumentParser(description="4-Slot 800k Quality & Isolation Benchmark")
    parser.add_argument('--tiers', nargs='+', default=['tier1', 'tier2', 'tier3', 'tier4'],
                        help="Tiers to run (tier1=100k, tier2=250k, tier3=500k, tier4=800k)")
    parser.add_argument('--smoke', action='store_true', help="Run quick 5k token smoke test")
    parser.add_argument('--output', '--output-json', type=Path, default=RESULTS_PATH, help="Output JSON path")
    parser.add_argument('--url', type=str, default=GATEWAY_URL, help='Chat completions URL')
    parser.add_argument('--api-key-file', type=Path, default=None, help='Path to API key file')
    parser.add_argument('--model', type=str, default=MODEL_NAME, help='Model name')
    parser.add_argument('--tokenizer', type=Path, default=None, help='Tokenizer path')
    parser.add_argument('--bench-name', type=str, default='Post-Optimization 4-Slot 800k Quality & Cross-Session Isolation Verification', help="Benchmark name")
    args = parser.parse_args()
    globals()['GATEWAY_URL'] = args.url
    globals()['MODEL_NAME'] = args.model

    tier_map = {
        'smoke': ('Smoke Test (5k/slot - 20k agg)', 5000),
        'tier1': ('Tier 1 (100k/slot - 400k agg)', 100000),
        'tier2': ('Tier 2 (250k/slot - 1.0M agg)', 250000),
        'tier3': ('Tier 3 (500k/slot - 2.0M agg)', 500000),
        'tier4': ('Tier 4 (800k/slot - 3.2M agg)', 800000)
    }

    if args.smoke:
        tiers_to_run = ['smoke']
    else:
        tiers_to_run = args.tiers

    print("================================================================================", flush=True)
    print("Pre-Optimization 4-Slot 800k Quality & Cross-Session Isolation Baseline Harness", flush=True)
    print("================================================================================", flush=True)

    api_key = load_api_key(args.api_key_file)
    print("Calibrating domain vocabularies against Strata tokenizer...", flush=True)
    calibrate_domains(args.tokenizer)

    overall_results = {
        'benchmark': args.bench_name,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'host': GATEWAY_URL,
        'model': MODEL_NAME,
        'tiers_executed': tiers_to_run,
        'tiers_data': {}
    }
    if args.output.exists():
        try:
            prev = json.loads(args.output.read_text())
            if 'tiers_data' in prev:
                overall_results['tiers_data'].update(prev['tiers_data'])
        except Exception:
            pass

    t_bench_start = time.time()
    for t_id in tiers_to_run:
        if t_id not in tier_map:
            print(f"Unknown tier: {t_id}")
            continue
        tier_label, target_tokens = tier_map[t_id]
        tier_res = run_tier(tier_label, target_tokens, api_key)
        overall_results['tiers_data'][t_id] = tier_res
        # Save after each tier
        overall_results['last_tier_updated'] = t_id
        overall_results['tiers_executed'] = list(overall_results['tiers_data'].keys())
        args.output.write_text(json.dumps(overall_results, indent=2))
        print(f"[CHECKPOINT SAVED] Tier {t_id} written to {args.output}", flush=True)

    overall_results['total_bench_duration_s'] = round(time.time() - t_bench_start, 2)

    # Save to JSON
    args.output.write_text(json.dumps(overall_results, indent=2))
    print(f"\n[BENCHMARK COMPLETE] Full baseline results written to {args.output}", flush=True)


if __name__ == '__main__':
    main()
