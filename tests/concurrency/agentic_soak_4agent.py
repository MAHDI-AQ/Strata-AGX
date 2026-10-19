import urllib.request, json, time, concurrent.futures, sys

URL = "http://127.0.0.1:8096/v1/chat/completions"

# 1.5k shared system prompt representing Hermes / agent system doctrine
SYSTEM_PROMPT = (
    "You are a highly capable agentic coding assistant engineered for autonomous software development. "
    "Always follow zero-trust engineering doctrine, maintain strict code quality, eliminate deadlocks, "
    "and rigorously verify all outputs against industrial standards such as vLLM and SGLang. "
) * 20

AGENT_TASKS = [
    "Implement a concurrent lock-free bounded ring buffer in C++20 with atomic head/tail pointers.",
    "Refactor CUDA kernel memory access patterns to eliminate warp divergence and local memory spilling.",
    "Author an io_uring asynchronous DirectStorage engine using O_DIRECT and 4096-byte aligned buffers.",
    "Design a multi-tier RadixTree prefix caching subsystem with dynamic L2 host-RAM eviction policy."
]

def run_agent_turn(agent_id, turn_idx, conversation_history):
    payload = {
        "model": "Qwen3.8-Flash-Next-IQ3_XXS",
        "messages": conversation_history,
        "max_tokens": 48,
        "temperature": 0.6
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(URL, data=data_bytes, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            elapsed = time.time() - t0
            body = json.loads(resp.read().decode("utf-8"))
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = body.get("usage", {})
            cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            prompt_toks = usage.get("prompt_tokens", 0)
            comp_toks = usage.get("completion_tokens", 0)
            timings = body.get("timings", {})
            pred_tps = timings.get("predicted_per_second", 0.0)
            draft_n = timings.get("draft_n", 0)
            draft_acc = timings.get("draft_n_accepted", 0)
            return {
                "agent": agent_id,
                "turn": turn_idx,
                "status": "OK",
                "elapsed": elapsed,
                "cached": cached,
                "prompt": prompt_toks,
                "comp": comp_toks,
                "tps": pred_tps,
                "draft_rate": (draft_acc / draft_n) if draft_n > 0 else 0.0,
                "reply_snippet": content[:40].replace("\n", " ")
            }
    except Exception as e:
        return {
            "agent": agent_id,
            "turn": turn_idx,
            "status": f"ERROR: {e}",
            "elapsed": time.time() - t0
        }

def simulate_agent_lifecycle(agent_id, num_turns=6):
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    results = []
    task = AGENT_TASKS[agent_id % len(AGENT_TASKS)]
    for turn in range(num_turns):
        if turn == 0:
            user_msg = f"Agent {agent_id} Task: {task}"
        elif turn % 2 == 1:
            user_msg = f"Step {turn}: Execute test suite and report memory consumption receipts."
        else:
            user_msg = f"Step {turn}: Patch edge-case buffer alignment and rerun verification."
        
        history.append({"role": "user", "content": user_msg})
        res = run_agent_turn(agent_id, turn, history)
        results.append(res)
        if "ERROR" in res["status"]:
            break
        # Append assistant reply to simulate accumulating multi-turn state
        history.append({"role": "assistant", "content": f"Turn {turn} verified receipt: OK."})
        time.sleep(0.5)
    return results

print("=== STARTING 4-AGENT CONTINUOUS MULTI-TURN SOAK BATTERY ===")
t_start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    futs = [ex.submit(simulate_agent_lifecycle, i, 6) for i in range(4)]
    all_agent_results = [f.result() for f in futs]

total_wall = time.time() - t_start
print(f"\n=== SOAK COMPLETE: Wallclock: {total_wall:.2f}s ===")
all_ok = True
total_queries = 0
total_tokens = 0
for agent_id, res_list in enumerate(all_agent_results):
    print(f"\n--- Agent {agent_id} Summary ({len(res_list)} turns) ---")
    for r in res_list:
        total_queries += 1
        if r.get("status") != "OK":
            all_ok = False
            print(f"  Turn {r.get('turn')}: {r.get('status')} ({r.get('elapsed'):.2f}s)")
        else:
            total_tokens += r.get("comp", 0)
            print(f"  Turn {r['turn']}: {r['elapsed']:.2f}s | Cached: {r['cached']}/{r['prompt']} ({100.0*r['cached']/r['prompt']:.1f}%) | Tokens: {r['comp']} @ {r['tps']:.1f} tok/s | SpecAcc: {r['draft_rate']*100.0:.1f}%")

print(f"\nTOTAL QUERIES: {total_queries} | COMPLETED TOKENS: {total_tokens} | SUCCESS: {all_ok}")
if not all_ok:
    sys.exit(1)
