#!/usr/bin/env python3
import urllib.request
import urllib.error
import json
import time
import concurrent.futures
import sys
import os

SERVER_URL = os.environ.get("STRATA_URL", "http://127.0.0.1:8096/v1/chat/completions")
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "hermes_payload_template.json")
UPSTREAM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class HermesToolExecutor:
    """Simulates realistic agent environment execution for tool calls emitted by the model."""
    def __init__(self, workspace_root):
        self.workspace_root = workspace_root

    def execute(self, tool_name, arguments):
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except Exception:
                args = {}
        else:
            args = arguments or {}

        if tool_name in ("read_file", "view_file"):
            path = args.get("path") or args.get("file_path") or args.get("AbsolutePath") or ""
            target = os.path.join(self.workspace_root, path.lstrip("/"))
            if os.path.exists(target) and os.path.isfile(target):
                try:
                    with open(target, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                    start = max(1, int(args.get("start_line", args.get("StartLine", 1))))
                    end = min(len(lines), int(args.get("end_line", args.get("EndLine", start + 120))))
                    snippet = "".join(f"{i:4d} | {lines[i-1]}" for i in range(start, end + 1))
                    return f"[FILE CONTENT: {path} lines {start}-{end}]\n" + snippet
                except Exception as e:
                    return f"Error reading file {path}: {e}"
            return f"[SIMULATED CODE FILE: {path}]\n// File loaded successfully (3,420 bytes)\n#include <strata/core/layer.hpp>\nvoid kernel_execute() {{ /* optimized path */ }}\n"

        elif tool_name in ("terminal", "execute_command", "run_command"):
            cmd = args.get("command") or args.get("CommandLine") or ""
            return f"[COMMAND EXECUTION OUTPUT: `{cmd}`]\nexit_code: 0\nstdout:\n  -> Verified targets built and linked\n  -> Unit test assertions: 142 passed, 0 failed\n  -> Peak memory delta: +12.4 MiB"

        elif tool_name in ("search_files", "grep_search"):
            query = args.get("query") or args.get("pattern") or ""
            return f"[GREP RESULTS: `{query}`]\nsrc/program/concurrent_serve.cpp:2012: bool admitting = !pending.empty();\nsrc/core/radix_tree.cpp:382: bool RadixNode::hydrate_from_nvme();\ninclude/strata/core/radix_tree.hpp:459: std::recursive_mutex hydrate_mutex_;\n"

        elif tool_name in ("write_file", "patch"):
            return "SUCCESS: File patched cleanly on disk. Backup created. 0 conflicts."

        elif tool_name == "memory":
            return "SUCCESS: Updated Hermes MEMORY.md with new architectural insight."

        return f"Tool `{tool_name}` executed successfully: receipt logged."

def send_chat_completion(messages, tools, max_tokens=128, temperature=0.6, timeout=600):
    payload = {
        "model": "Qwen3.8-Flash-Next-IQ3_XXS",
        "messages": messages,
        "tools": tools,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SERVER_URL,
        data=data_bytes,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.time() - t0
            body = json.loads(resp.read().decode("utf-8"))
            choice = body.get("choices", [{}])[0]
            msg = choice.get("message", {})
            usage = body.get("usage", {})
            timings = body.get("timings", {})
            return {
                "ok": True,
                "elapsed": elapsed,
                "message": msg,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "tps": timings.get("predicted_per_second", 0.0),
                "prefill_tps": timings.get("prompt_per_second", 0.0),
                "draft_n": timings.get("draft_n", 0),
                "draft_acc": timings.get("draft_n_accepted", 0)
            }
    except Exception as e:
        return {
            "ok": False,
            "elapsed": time.time() - t0,
            "error": str(e)
        }

def run_agentic_workflow(agent_id, task_info, template, num_turns=6):
    tool_exec = HermesToolExecutor(UPSTREAM_ROOT)
    system_prompt = template["system_prompt"]
    tools = template["tools"]
    task_prompt = task_info.get("prompt", "")
    task_name = task_info.get("name", f"Agent_{agent_id}")

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: {task_prompt}"}
    ]

    history_receipts = []
    print(f"[Agent {agent_id}] Starting mission: {task_name}", flush=True)

    for turn in range(num_turns):
        res = send_chat_completion(conversation, tools, max_tokens=96)
        if not res["ok"]:
            print(f"[Agent {agent_id} Turn {turn}] FAILED: {res.get('error')} ({res['elapsed']:.2f}s)", flush=True)
            history_receipts.append({"turn": turn, "status": "ERROR", "error": res.get("error"), "elapsed": res["elapsed"]})
            break

        msg = res["message"]
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        prompt_toks = res["prompt_tokens"]
        cached_toks = res["cached_tokens"]
        comp_toks = res["completion_tokens"]
        tps = res["tps"]
        cache_pct = (cached_toks / prompt_toks * 100.0) if prompt_toks > 0 else 0.0
        draft_acc_pct = (res["draft_acc"] / res["draft_n"] * 100.0) if res["draft_n"] > 0 else 0.0

        print(f"[Agent {agent_id} Turn {turn}] {res['elapsed']:.2f}s | Cached: {cached_toks}/{prompt_toks} ({cache_pct:.1f}%) | "
              f"Comp: {comp_toks} @ {tps:.1f} tok/s | SpecAcc: {draft_acc_pct:.1f}% | Tools: {len(tool_calls)}", flush=True)

        history_receipts.append({
            "turn": turn,
            "status": "OK",
            "elapsed": res["elapsed"],
            "prompt_tokens": prompt_toks,
            "cached_tokens": cached_toks,
            "cache_pct": cache_pct,
            "completion_tokens": comp_toks,
            "tps": tps,
            "draft_acc_pct": draft_acc_pct,
            "num_tools": len(tool_calls)
        })

        conversation.append(msg)

        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                t_name = fn.get("name", "")
                t_args = fn.get("arguments", "{}")
                tool_output = tool_exec.execute(t_name, t_args)
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{turn}"),
                    "name": t_name,
                    "content": tool_output
                })
        else:
            next_user_turn = (
                f"Step {turn + 1}: Cross-verify edge case invariants for {task_name}. "
                "Inspect memory consumption, run static analysis checks, and report verified receipts."
            )
            conversation.append({"role": "user", "content": next_user_turn})

        time.sleep(0.5)

    return {
        "agent_id": agent_id,
        "task_name": task_name,
        "receipts": history_receipts
    }

def main():
    print("================================================================================")
    print("   STRATA-AGX INDUSTRIAL-GRADE EXTREME HERMES AGENTIC STRESS BATTERY           ")
    print("================================================================================")
    print(f"Target Server: {SERVER_URL}")
    print(f"Loading Hermes Production Template: {TEMPLATE_PATH}")

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = json.load(f)

    sys_len = len(template["system_prompt"])
    num_tools = len(template["tools"])
    seed_tasks = template["seed_tasks"]

    print(f"System Prompt Size: {sys_len:,} chars (~{sys_len//3.8:.0f} tokens)")
    print(f"Registered Tools:   {num_tools} functions")
    print(f"Concurrent Agents:  {len(seed_tasks)}")
    print("================================================================================\n", flush=True)

    num_turns_per_agent = 6

    print("--- Phase 0: System Prompt Warm-up (Cold Prefill & Radix Seeding) ---")
    warmup_res = send_chat_completion(
        messages=[{"role": "system", "content": template["system_prompt"]}, {"role": "user", "content": "Warmup probe for Radix cache seeding."}],
        tools=template["tools"],
        max_tokens=10,
        timeout=600
    )
    if warmup_res["ok"]:
        cached = warmup_res["cached_tokens"]
        total_p = warmup_res["prompt_tokens"]
        print(f"System Prompt Warm-up COMPLETE in {warmup_res['elapsed']:.2f}s ({cached}/{total_p} cached)")
    else:
        print(f"WARNING: Warm-up failed: {warmup_res.get('error')}")

    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(seed_tasks)) as executor:
        futures = [
            executor.submit(run_agentic_workflow, i, seed_tasks[i], template, num_turns=num_turns_per_agent)
            for i in range(len(seed_tasks))
        ]
        results = [f.result() for f in futures]

    total_wallclock = time.time() - t_start
    print("\n================================================================================")
    print(f"   BATTERY COMPLETE: Wallclock {total_wallclock:.2f}s                           ")
    print("================================================================================")

    all_passed = True
    total_turns_ok = 0
    total_tokens_gen = 0

    for r in results:
        aid = r["agent_id"]
        tname = r["task_name"]
        receipts = r["receipts"]
        print(f"\n--- Agent {aid}: {tname} ({len(receipts)} turns) ---")
        for rec in receipts:
            if rec["status"] == "OK":
                total_turns_ok += 1
                total_tokens_gen += rec["completion_tokens"]
                print(f"  Turn {rec['turn']}: {rec['elapsed']:.2f}s | "
                      f"Cached: {rec['cached_tokens']}/{rec['prompt_tokens']} ({rec['cache_pct']:.1f}%) | "
                      f"Gen: {rec['completion_tokens']} toks @ {rec['tps']:.1f} tok/s | "
                      f"SpecAcc: {rec['draft_acc_pct']:.1f}% | Tools: {rec['num_tools']}")
            else:
                all_passed = False
                print(f"  Turn {rec['turn']}: FAILED - {rec.get('error')} ({rec['elapsed']:.2f}s)")

    print("\n--------------------------------------------------------------------------------")
    print(f"TOTAL TURNS COMPLETED: {total_turns_ok}/{len(seed_tasks) * num_turns_per_agent}")
    print(f"TOTAL TOKENS GENERATED: {total_tokens_gen}")
    print(f"OVERALL STATUS:         {'SUCCESS (100% PASS)' if all_passed else 'FAILURE'}")
    print("--------------------------------------------------------------------------------")

    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
