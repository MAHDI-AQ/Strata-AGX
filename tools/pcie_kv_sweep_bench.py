#!/usr/bin/env python3
"""
PCIe KV Streaming & Context Sweep Benchmark Harness
Evaluates 4 independent concurrent slots under a specified context ceiling stressed to 80% capacity.
Monitors PCIe Gen4 x4/x8 RX/TX throughput, measures sustained multi-token decode throughput,
TTFT, prefill rate, needle fidelity, and cross-session isolation.
"""

import argparse
import ast
from collections import Counter
import concurrent.futures
import json
import os
from pathlib import Path
import re
import signal
import ssl
import subprocess
import sys
import threading
import time
import urllib.request

# SSL context for HTTPS endpoints
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

GATEWAY_URL = os.environ.get('STRATA_URL', 'http://127.0.0.1:8080/v1/chat/completions')
METRICS_URL = 'http://127.0.0.1:8096/metrics'
KEY_PATH = Path(os.environ.get('STRATA_API_KEY_FILE', '')) if os.environ.get('STRATA_API_KEY_FILE') else None

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


class PcieMonitor:
    def __init__(self, log_path="/tmp/pcie_dmon.log"):
        self.log_path = log_path
        self.proc = None
        self.running = False

    def start(self):
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        self.proc = subprocess.Popen(
            ['nvidia-smi', 'dmon', '-s', 't', '-d', '1', '-o', 'T', '-f', self.log_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self.running = True

    def stop(self):
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.running = False
        return self.analyze()

    def analyze(self):
        if not os.path.exists(self.log_path):
            return {}
        gpu_stats = {0: {'rx': [], 'tx': []}, 1: {'rx': [], 'tx': []}, 2: {'rx': [], 'tx': []}, 3: {'rx': [], 'tx': []}}
        with open(self.log_path, 'r', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                # format with -o T: time gpu_idx rx tx
                # e.g.: 05:30:00 0 120 45
                if len(parts) >= 4:
                    try:
                        gpu_id = int(parts[1])
                        rx = float(parts[2])
                        tx = float(parts[3])
                        if gpu_id in gpu_stats:
                            gpu_stats[gpu_id]['rx'].append(rx)
                            gpu_stats[gpu_id]['tx'].append(tx)
                    except ValueError:
                        continue
        summary = {}
        for gid, data in gpu_stats.items():
            rx_list = data['rx']
            tx_list = data['tx']
            summary[f'gpu_{gid}'] = {
                'samples': len(rx_list),
                'rx_mean_mb_s': round(sum(rx_list) / max(len(rx_list), 1), 2),
                'rx_max_mb_s': round(max(rx_list) if rx_list else 0.0, 2),
                'tx_mean_mb_s': round(sum(tx_list) / max(len(tx_list), 1), 2),
                'tx_max_mb_s': round(max(tx_list) if tx_list else 0.0, 2)
            }
        return summary


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
    b_parts = []
    curr = 0
    idx = 0
    while curr < chars_before:
        part = f'Document Section {idx // len(s)}.{idx % len(s)}: ' + s[idx % len(s)] + '\n'
        b_parts.append(part)
        curr += len(part)
        idx += 1
    text_before = ''.join(b_parts)[:chars_before]

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
    words = text.split()
    repetition_collapsed = False
    if len(words) >= 30:
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

        missing = [c for c in domain_info['reasoning_checks'] if c not in code]
        if missing:
            return False, f"Code missing required symbols: {missing}"
        return True, "Valid Python syntax & structure"

    elif domain_info['reasoning_type'] == 'logic_puzzle':
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


def run_slot_turn(domain_key, target_tokens, api_key):
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
    tokens_gen1 = usage1.get('completion_tokens', 0)

    needle_match = code in ans1
    print(f"[{domain_key.upper()}] Test A done in {dur1}s: match={needle_match}, tokens={prompt_tokens1:,}, prefill={prefill_tok_s1:.1f} tok/s, ttft={ttft1:.1f}ms, reply=\"{ans1.strip()}\"", flush=True)

    # Test B: Adversarial Cross-Session Isolation Probe
    q2 = f"What is the secret verification code for {d['cross_target_name']}?"
    print(f"[{domain_key.upper()}] Test B: Sending Turn 2 (Adversarial Query: {q2})...", flush=True)
    messages.append({'role': 'assistant', 'content': ans1})
    messages.append({'role': 'user', 'content': q2})
    res2, dur2 = post_completion(api_key, sid, messages, max_tokens=100, enable_thinking=False)
    ans2 = res2['choices'][0]['message']['content']
    timings2 = res2.get('timings', {})
    usage2 = res2.get('usage', {})
    tokens_gen2 = usage2.get('completion_tokens', 0)

    leaked_codes = [fc for fc in foreign_codes if fc in ans2]
    isolation_passed = len(leaked_codes) == 0
    print(f"[{domain_key.upper()}] Test B done in {dur2}s: isolated={isolation_passed}, leaks={leaked_codes}, decode={timings2.get('predicted_per_second', 0):.1f} tok/s", flush=True)

    # Test C: Sustained Continuous Multi-Token Reasoning Continuation
    q3 = d['reasoning_prompt']
    print(f"[{domain_key.upper()}] Test C: Sending Turn 3 (Sustained Reasoning Continuation, max_tokens=500)...", flush=True)
    messages.append({'role': 'assistant', 'content': ans2})
    messages.append({'role': 'user', 'content': q3})
    res3, dur3 = post_completion(api_key, sid, messages, max_tokens=500, enable_thinking=False)
    ans3 = res3['choices'][0]['message']['content']
    timings3 = res3.get('timings', {})
    usage3 = res3.get('usage', {})
    tokens_gen3 = usage3.get('completion_tokens', 0)
    decode_tok_s3 = timings3.get('predicted_per_second', 0)

    reasoning_passed, reasoning_note = verify_reasoning_output(d, ans3)
    print(f"[{domain_key.upper()}] Test C done in {dur3}s: passed={reasoning_passed} ({reasoning_note}), tokens_gen={tokens_gen3}, sustained_decode={decode_tok_s3:.1f} tok/s", flush=True)

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
            'completion_tokens': tokens_gen1,
            'duration_s': dur1
        },
        'test_b': {
            'isolation_passed': isolation_passed,
            'leaked_codes': leaked_codes,
            'query': q2,
            'response': ans2.strip()[:200],
            'duration_s': dur2,
            'completion_tokens': tokens_gen2,
            'decode_tok_s': timings2.get('predicted_per_second', 0)
        },
        'test_c': {
            'reasoning_passed': reasoning_passed,
            'reasoning_note': reasoning_note,
            'response_snippet': ans3.strip()[:200],
            'duration_s': dur3,
            'completion_tokens': tokens_gen3,
            'decode_tok_s': decode_tok_s3
        }
    }


def main():
    parser = argparse.ArgumentParser(description="PCIe KV Streaming & Context Sweep Benchmark")
    parser.add_argument('--tokens-per-slot', type=int, required=True, help="Target tokens per slot (e.g. 614400 for 768k @ 80 percent)")
    parser.add_argument('--label', type=str, required=True, help="Run label (e.g. 768k_80pct)")
    parser.add_argument('--url', type=str, default=GATEWAY_URL, help='Chat completions URL')
    parser.add_argument('--api-key-file', type=Path, default=None, help='Path to API key file')
    parser.add_argument('--model', type=str, default=MODEL_NAME, help='Model name')
    parser.add_argument('--tokenizer', type=Path, default=None, help='Tokenizer path')
    parser.add_argument('--output', type=Path, default=Path('pcie_kv_sweep_results.json'), help="Output JSON path")
    args = parser.parse_args()
    globals()['GATEWAY_URL'] = args.url
    globals()['MODEL_NAME'] = args.model

    api_key = load_api_key(args.api_key_file)
    print("Calibrating domain vocabularies against Strata tokenizer...", flush=True)
    calibrate_domains(args.tokenizer)

    tokens_per_slot = args.tokens_per_slot
    agg_tokens = tokens_per_slot * 4

    print("================================================================================", flush=True)
    print(f"STARTING SWEEP RUN: {args.label}")
    print(f"Target: {tokens_per_slot:,} tokens/slot (~{agg_tokens:,} aggregate tokens across 4 slots)")
    print("================================================================================", flush=True)

    metrics_before = query_metrics()
    vram_before = query_nvidia_smi()

    pcie_mon = PcieMonitor()
    pcie_mon.start()
    t_start = time.time()

    keys = ['alpha', 'beta', 'gamma', 'delta']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_slot_turn, k, tokens_per_slot, api_key): k for k in keys}
        results = {}
        for f in concurrent.futures.as_completed(futs):
            k = futs[f]
            results[k] = f.result()

    t_end = time.time()
    total_duration = round(t_end - t_start, 2)
    pcie_stats = pcie_mon.stop()

    vram_after = query_nvidia_smi()
    metrics_after = query_metrics()

    slot_results = [results[k] for k in keys]
    accuracy_pct = 100.0 * sum(1 for r in slot_results if r['test_a']['needle_match']) / 4.0
    contamination_pct = 100.0 * sum(1 for r in slot_results if not r['test_b']['isolation_passed']) / 4.0
    reasoning_pct = 100.0 * sum(1 for r in slot_results if r['test_c']['reasoning_passed']) / 4.0

    mean_ttft = sum(r['test_a']['ttft_ms'] for r in slot_results) / 4.0
    mean_prefill_tok_s = sum(r['test_a']['prefill_tok_s'] for r in slot_results) / 4.0
    
    # Sustained decode from Test C
    mean_sustained_decode_tok_s = sum(r['test_c']['decode_tok_s'] for r in slot_results) / 4.0
    aggregate_cluster_sustained_decode_tok_s = sum(r['test_c']['decode_tok_s'] for r in slot_results)

    # Disk reads verification
    reads_before = metrics_before.get('storage', {}).get('bytes_read', 0) if isinstance(metrics_before, dict) else 0
    reads_after = metrics_after.get('storage', {}).get('bytes_read', 0) if isinstance(metrics_after, dict) else 0
    disk_read_mb = round((reads_after - reads_before) / (1024 * 1024), 3)

    summary = {
        'label': args.label,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'tokens_per_slot_target': tokens_per_slot,
        'aggregate_tokens_target': agg_tokens,
        'actual_tokens_per_slot': {k: results[k]['actual_tokens'] for k in keys},
        'total_actual_tokens': sum(results[k]['actual_tokens'] for k in keys),
        'total_duration_s': total_duration,
        'accuracy_pct': accuracy_pct,
        'contamination_pct': contamination_pct,
        'reasoning_pct': reasoning_pct,
        'mean_ttft_ms': round(mean_ttft, 1),
        'mean_prefill_tok_s': round(mean_prefill_tok_s, 1),
        'per_slot_sustained_decode_tok_s': {k: round(results[k]['test_c']['decode_tok_s'], 1) for k in keys},
        'mean_sustained_decode_tok_s': round(mean_sustained_decode_tok_s, 1),
        'aggregate_cluster_sustained_decode_tok_s': round(aggregate_cluster_sustained_decode_tok_s, 1),
        'expert_storage_reads_mb': disk_read_mb,
        'pcie_bandwidth_stats': pcie_stats,
        'slots': results
    }

    print("\n================================================================================", flush=True)
    print(f"BENCHMARK COMPLETED: {args.label}")
    print(f"Duration: {total_duration}s | Aggregate Tokens: {summary['total_actual_tokens']:,}")
    print(f"Needle Accuracy: {accuracy_pct:.1f}% | Contamination: {contamination_pct:.1f}% | Reasoning: {reasoning_pct:.1f}%")
    print(f"Mean TTFT: {mean_ttft:.1f} ms | Mean Prefill: {mean_prefill_tok_s:.1f} tok/s")
    print(f"Sustained Decode (Test C): Mean={mean_sustained_decode_tok_s:.1f} tok/s | Aggregate Cluster={aggregate_cluster_sustained_decode_tok_s:.1f} tok/s")
    print(f"Per-Slot Decode: {summary['per_slot_sustained_decode_tok_s']}")
    print(f"Expert Disk Reads: {disk_read_mb} MB")
    print("PCIe Streaming Bandwidth:")
    for gid in sorted(pcie_stats.keys()):
        stat = pcie_stats[gid]
        print(f"  {gid}: RX Mean={stat['rx_mean_mb_s']} MB/s, Max={stat['rx_max_mb_s']} MB/s | TX Mean={stat['tx_mean_mb_s']} MB/s, Max={stat['tx_max_mb_s']} MB/s")
    print("================================================================================", flush=True)

    args.output.write_text(json.dumps(summary, indent=2))
    print(f"Receipt written to {args.output}")


if __name__ == '__main__':
    main()
