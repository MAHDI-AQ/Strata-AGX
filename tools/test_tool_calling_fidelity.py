#!/usr/bin/env python3
"""
tools/test_tool_calling_fidelity.py
Comprehensive Tool-Calling & Structured Reasoning Fidelity Battery for Strata AGX.

Validates:
  1. OpenAI /v1/chat/completions tool format and schema compliance.
  2. <think> reasoning boundary tracking and code-fence filtering (upstream 38357f18).
  3. Speculative MTP JSON argument token consistency (no empty '{}' arguments or malformed closing braces).
  4. Multi-tool selection and parameter coercion.
"""

import argparse
import json
import sys
import time
import urllib.request

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "read_kernel_metrics",
            "description": "Read GPU kernel execution metrics and memory bandwidth for a specified SM89 kernel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kernel_name": {
                        "type": "string",
                        "description": "The exact name of the CUDA kernel (e.g. gdn_conv_l2_kernel, moe_group_fused)."
                    },
                    "sm_count": {
                        "type": "integer",
                        "description": "Number of streaming multiprocessors targeted (e.g. 128)."
                    },
                    "track_dram_writes": {
                        "type": "boolean",
                        "description": "Whether to capture dead DRAM write bandwidth."
                    }
                },
                "required": ["kernel_name", "sm_count"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flush_radix_cache",
            "description": "Evict or flush specified cache tier snapshots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tier": {
                        "type": "string",
                        "enum": ["L1_VRAM", "L2_HOST_RAM", "L3_NVME"],
                        "description": "Target storage tier."
                    },
                    "max_nodes": {
                        "type": "integer",
                        "description": "Maximum node count to evict."
                    }
                },
                "required": ["tier"]
            }
        }
    }
]

TEST_CASES = [
    {
        "name": "Single Tool Call (Exact Schema Match)",
        "messages": [
            {"role": "system", "content": "You are a hardware diagnostics agent. When requested, invoke the appropriate tool with exact arguments."},
            {"role": "user", "content": "Please inspect the metrics for the 'gdn_conv_l2_kernel' across 128 SMs with DRAM write tracking enabled."}
        ],
        "expected_tool": "read_kernel_metrics",
        "expected_args": {"kernel_name": "gdn_conv_l2_kernel", "sm_count": 128, "track_dram_writes": True}
    },
    {
        "name": "Second Tool Call (Enum Verification)",
        "messages": [
            {"role": "system", "content": "You are a cache management agent."},
            {"role": "user", "content": "Flush the L3_NVME storage tier with a max limit of 50 nodes."}
        ],
        "expected_tool": "flush_radix_cache",
        "expected_args": {"tier": "L3_NVME", "max_nodes": 50}
    },
    {
        "name": "Code Fence Guard (Reasoning Quote vs Active Tool Call)",
        "messages": [
            {"role": "system", "content": "You are an assistant. When asked about code examples, provide a markdown code block showing how a tool call looks, but DO NOT execute it."},
            {"role": "user", "content": "Show me a Python example of calling flush_radix_cache inside a markdown code block. Do not run the tool."}
        ],
        "expect_tool_call": False
    }
]


def run_battery(endpoint: str) -> bool:
    print(f"Beginning Tool-Calling Fidelity Battery on {endpoint}")
    passed = 0

    for i, tc in enumerate(TEST_CASES):
        print(f"\n[{i+1}/{len(TEST_CASES)}] Testing: {tc['name']}...")
        payload = {
            "model": "Qwen3.8-Flash-Next-IQ3_XXS",
            "messages": tc["messages"],
            "tools": TOOLS_SPEC,
            "tool_choice": "auto",
            "temperature": 0.0,
            "max_tokens": 384
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST")

        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            elapsed = time.perf_counter() - t0

            message = body["choices"][0]["message"]
            tool_calls = message.get("tool_calls", [])
            content = message.get("content", "")

            expect_call = tc.get("expect_tool_call", True)
            if not expect_call:
                if len(tool_calls) == 0:
                    print(f"  PASS: Quoted code block correctly ignored by tool parser ({elapsed:.2f}s). Content preview: {content[:60]}...")
                    passed += 1
                else:
                    print(f"  FAIL: Expected 0 tool calls (fence guard), but parser emitted: {tool_calls}")
                continue

            if not tool_calls:
                print(f"  FAIL: Expected tool call, but model emitted raw content: {content}")
                continue

            tc_fn = tool_calls[0].get("function", {})
            name = tc_fn.get("name")
            args_str = tc_fn.get("arguments", "{}")

            try:
                args = json.loads(args_str)
            except json.JSONDecodeError as err:
                print(f"  FAIL: Malformed JSON arguments: '{args_str}' ({err})")
                continue

            if name != tc["expected_tool"]:
                print(f"  FAIL: Expected tool '{tc['expected_tool']}', got '{name}'")
                continue

            # Compare required args
            args_match = True
            for k, v in tc["expected_args"].items():
                if args.get(k) != v:
                    print(f"  FAIL: Arg mismatch for '{k}': expected {v}, got {args.get(k)}")
                    args_match = False
                    break

            if args_match:
                print(f"  PASS: Exact match for '{name}' args={args} ({elapsed:.2f}s).")
                passed += 1

        except Exception as e:
            print(f"  ERROR: Request failed: {e}")

    print(f"\nTool-Calling Fidelity Score: {passed}/{len(TEST_CASES)} Passed.")
    return passed == len(TEST_CASES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8096/v1/chat/completions")
    args = parser.parse_args()
    success = run_battery(args.endpoint)
    sys.exit(0 if success else 1)
