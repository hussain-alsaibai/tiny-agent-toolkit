"""
reliable_agent.py — a complete, end-to-end agent loop using every
component of tiny-agent-toolkit.

This is a *demo* agent (no real LLM call), but it shows how all the
pieces compose in a production-style agent loop:

    ┌─────────────────────────────────────────────────────┐
    │           User Query                                │
    └──────────────────────┬──────────────────────────────┘
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │   IdempotencyGuard.check("run:<hash>")              │
    │   (skip if already completed this exact request)    │
    └──────────────────────┬──────────────────────────────┘
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │   Scratchpad.load("context")                        │
    │   (restore notes from prior turns/sessions)         │
    └──────────────────────┬──────────────────────────────┘
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │   BudgetGovernor.can_afford(...)                    │
    │   BudgetGovernor.record(...) for every LLM call     │
    └──────────────────────┬──────────────────────────────┘
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │   Agent Loop:                                       │
    │     think → tool.call() → observe → ...            │
    └──────────────────────┬──────────────────────────────┘
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │   IdempotencyGuard.mark_done("action:<hash>")       │
    │   for every external side-effect                    │
    └──────────────────────┬──────────────────────────────┘
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │   Scratchpad.save_text(...) for every turn          │
    │   (so the next session can recall the journey)      │
    └─────────────────────────────────────────────────────┘

Run:
    python examples/reliable_agent.py

No external dependencies. No API keys required.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List

from tiny_agent_toolkit import (
    BudgetExceeded,
    BudgetGovernor,
    DirectoryScratchpad,
    FileIdempotencyStore,
    IdempotencyGuard,
    InMemoryIdempotencyStore,
    ResourceLimits,
    SandboxError,
    Tool,
    ToolError,
    ToolRegistry,
    inspect_source,
    is_safe,
    run_in_subprocess,
    tool,
)


# ═══════════════════════════════════════════════════════════════════
# 1. Tools — math + a sandboxed Python evaluator
# ═══════════════════════════════════════════════════════════════════


def build_tools() -> ToolRegistry:
    """Build a registry of tools the demo agent can use."""
    reg = ToolRegistry()

    @tool(description="Add two numbers")
    def add(a: float, b: float) -> float:
        return a + b

    @tool(description="Multiply two numbers")
    def multiply(a: float, b: float) -> float:
        return a * b

    @tool(description="Square a number")
    def square(x: float) -> float:
        return x * x

    @tool(description="Evaluate a snippet of Python code in a sandbox")
    def eval_python(code: str) -> str:
        """Run `code` in a subprocess with a 2-second timeout and 64MB memory cap.

        Refuses to run code that fails the static safety check.
        """
        issues = inspect_source(code)
        if issues:
            return f"REFUSED: {issues}"
        result = run_in_subprocess(
            code,
            timeout_s=2.0,
            limits=ResourceLimits(cpu_seconds=2, memory_mb=64),
        )
        if result.timed_out:
            return "TIMEOUT after 2s"
        if not result.ok:
            return f"ERROR (exit {result.returncode}): {result.stderr.strip()[:500]}"
        return result.stdout.strip() or "(no output)"

    reg.register(add)
    reg.register(multiply)
    reg.register(square)
    reg.register(eval_python)

    return reg


# ═══════════════════════════════════════════════════════════════════
# 2. Mock LLM — deterministic, no API keys needed
# ═══════════════════════════════════════════════════════════════════


class MockLLM:
    """Deterministic stand-in for an LLM. Follows a hard-coded script.

    Real usage would call OpenAI/Anthropic/Ollama here. The interface
    is `chat(messages) -> dict` returning at minimum:
        {"content": str, "tool_calls": [{"function": {"name", "arguments"}}] | None}
    """

    def __init__(self, scripts: List[Dict[str, Any]]):
        self._scripts = scripts
        self._cursor = 0

    def chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self._cursor >= len(self._scripts):
            return {"content": "Final Answer(done)", "tool_calls": None, "usage": {"input": 50, "output": 10}}
        reply = self._scripts[self._cursor]
        self._cursor += 1
        return reply


def build_demo_script() -> List[Dict[str, Any]]:
    """Build the canned LLM responses for the demo run.

    The agent will:
      1. Plan to square 7 → call `square(x=7)`
      2. Plan to multiply 49 × 3 → call `multiply(a=49, b=3)`
      3. Plan to add 147 + 5 → call `add(a=147, b=5)`
      4. Emit final answer 152
    """
    return [
        {
            "content": "Thought: square 7 first\nAction: square(x=7)",
            "tool_calls": [{"function": {"name": "square", "arguments": json.dumps({"x": 7})}}],
            "usage": {"input_tokens": 100, "output_tokens": 30, "model": "mock-gpt-4o"},
        },
        {
            "content": "Thought: multiply by 3\nAction: multiply(a=49, b=3)",
            "tool_calls": [{"function": {"name": "multiply", "arguments": json.dumps({"a": 49, "b": 3})}}],
            "usage": {"input_tokens": 110, "output_tokens": 30, "model": "mock-gpt-4o"},
        },
        {
            "content": "Thought: add 5\nAction: add(a=147, b=5)",
            "tool_calls": [{"function": {"name": "add", "arguments": json.dumps({"a": 147, "b": 5})}}],
            "usage": {"input_tokens": 120, "output_tokens": 30, "model": "mock-gpt-4o"},
        },
        {
            "content": "Thought: the answer is 152\nAction: Final Answer(152)",
            "tool_calls": None,
            "usage": {"input_tokens": 130, "output_tokens": 25, "model": "mock-gpt-4o"},
        },
    ]


# ═══════════════════════════════════════════════════════════════════
# 3. The agent loop — composes everything
# ═══════════════════════════════════════════════════════════════════


def parse_action(content: str):
    """Parse "Action: tool_name(args)" out of an LLM response.

    Real agents get this for free via tool-calling APIs. We support it
    here so the demo can run without an actual LLM.
    """
    if "Final Answer" in content:
        marker = "Final Answer("
        idx = content.index(marker) + len(marker)
        answer = content[idx:].rstrip(")")
        return ("__final__", answer)
    if "Action:" in content and "(" in content and ")" in content:
        after = content.split("Action:")[-1].strip()
        name = after[: after.index("(")].strip()
        args = after[after.index("(") + 1 : after.rindex(")")].strip()
        return (name, args)
    return (None, None)


def run_agent(
    query: str,
    *,
    registry: ToolRegistry,
    llm: MockLLM,
    budget: BudgetGovernor,
    guard: IdempotencyGuard,
    scratchpad: DirectoryScratchpad,
    max_iterations: int = 8,
) -> Dict[str, Any]:
    """Run the full ReAct-style agent loop.

    Returns a dict with `answer`, `iterations`, `trace`, `budget`, `notes`.
    """
    # Idempotency: skip the entire run if we've already completed this query
    run_key = IdempotencyGuard.make_key("run", query=query)
    if guard.already_done(run_key):
        return {"answer": None, "iterations": 0, "trace": [], "skipped": "duplicate-run"}

    scratchpad.save_text("query", query, role="user")
    messages: List[Dict[str, Any]] = [{"role": "system", "content": "You are a calculator agent."},
                                      {"role": "user", "content": query}]
    trace: List[Dict[str, Any]] = []
    answer: Any = None

    for iteration in range(1, max_iterations + 1):
        # ---- Budget check BEFORE calling LLM ----
        # Rough estimate: assume 200 in + 100 out for the next reply
        if not budget.can_afford(model="mock-gpt-4o", input_tokens=200, output_tokens=100):
            scratchpad.save_text("budget", f"halted at iteration {iteration}: budget exhausted",
                                 iteration=iteration)
            return {"answer": None, "iterations": iteration, "trace": trace, "halted": "budget"}

        # ---- Call LLM ----
        reply = llm.chat(messages)
        content = reply.get("content", "")
        tool_calls = reply.get("tool_calls")
        usage = reply.get("usage") or {}
        budget.record(
            model=usage.get("model", "unknown"),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

        # ---- Branch 1: structured tool calls (OpenAI style) ----
        if tool_calls:
            for tc in tool_calls:
                fn = tc["function"]
                name = fn["name"]
                args_raw = fn["arguments"]

                # Idempotency: skip tool if already done
                action_key = IdempotencyGuard.make_key("action", tool=name, args=args_raw)
                if guard.already_done(action_key):
                    observation = f"(deduped — already ran {name})"
                else:
                    try:
                        observation = registry.call(name, args_raw)
                        observation_str = json.dumps(observation) if not isinstance(observation, str) else observation
                        guard.mark_done(action_key)
                    except ToolError as e:
                        observation_str = f"ERROR: {e}"

                scratchpad.save_text("tool_call", observation_str, tool=name, iter=iteration)
                trace.append({
                    "iter": iteration,
                    "tool": name,
                    "args": args_raw,
                    "observation": observation_str,
                })
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "tool", "name": name, "content": observation_str})
            continue

        # ---- Branch 2: text-based Action: ... parsing ----
        action_name, action_args = parse_action(content)
        if action_name == "__final__":
            answer = action_args
            scratchpad.save_text("final_answer", answer, iter=iteration)
            break
        if action_name:
            action_key = IdempotencyGuard.make_key("action", tool=action_name, args=action_args)
            if guard.already_done(action_key):
                observation = f"(deduped — already ran {action_name})"
            else:
                try:
                    observation = registry.call(action_name, action_args)
                    observation_str = json.dumps(observation) if not isinstance(observation, str) else observation
                    guard.mark_done(action_key)
                except ToolError as e:
                    observation_str = f"ERROR: {e}"

            scratchpad.save_text("tool_call", observation_str, tool=action_name, iter=iteration)
            trace.append({
                "iter": iteration,
                "tool": action_name,
                "args": action_args,
                "observation": observation_str,
            })
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Observation: {observation_str}"})
            continue

        # ---- Branch 3: free-form content treated as final answer ----
        answer = content
        scratchpad.save_text("final_answer", answer, iter=iteration)
        break

    # Mark the whole run as done so a retry doesn't re-execute
    guard.mark_done(run_key)

    return {
        "answer": answer,
        "iterations": len(trace),
        "trace": trace,
        "budget": budget.stats(),
        "notes": len(scratchpad.notes()),
    }


# ═══════════════════════════════════════════════════════════════════
# 4. Demo harness — wires the components and runs the agent
# ═══════════════════════════════════════════════════════════════════


def main() -> int:
    print("=" * 70)
    print("tiny-agent-toolkit — reliable agent demo")
    print("=" * 70)

    # Use a temp workspace so the demo is hermetic.
    tmp = tempfile.mkdtemp(prefix="tiny-agent-demo-")
    try:
        # ---- Compose the toolkit pieces ----
        registry = build_tools()
        llm = MockLLM(build_demo_script())
        budget = BudgetGovernor(
            limit_usd=1.00,
            limit_per_call_usd=0.50,
            limit_daily_usd=5.00,
            warn_at=0.5,
            on_warn=lambda spent, limit: print(f"  ! budget warning: ${spent:.4f} / ${limit:.4f}"),
            persist_path=os.path.join(tmp, "budget.jsonl"),
        )
        guard = IdempotencyGuard(
            store=FileIdempotencyStore(os.path.join(tmp, "dedup")),
            namespace="demo-agent",
        )
        scratchpad = DirectoryScratchpad(os.path.join(tmp, "scratchpad"))

        # ---- Print schemas ----
        print("\n[1] OpenAI schema for the registered tools:")
        for entry in registry.to_openai():
            print(f"    - {entry['function']['name']}: {entry['function']['description']}")
        print("\n[2] MCP schema for the registered tools:")
        for entry in registry.to_mcp():
            print(f"    - {entry['name']}: {entry['description']}")

        # ---- Run the agent ----
        print("\n[3] Running agent on: compute (7^2)*3 + 5")
        result = run_agent(
            "Compute (7^2)*3 + 5 using the available tools.",
            registry=registry,
            llm=llm,
            budget=budget,
            guard=guard,
            scratchpad=scratchpad,
        )
        print(f"\n    Answer: {result['answer']}")
        print(f"    Iterations: {result['iterations']}")
        print(f"    Notes saved: {result['notes']}")
        print(f"    Budget: ${result['budget']['total_cost_usd']:.6f} across {result['budget']['calls']} calls")

        # ---- Replay safety ----
        print("\n[4] Re-running the same query (should be skipped by idempotency):")
        replay = run_agent(
            "Compute (7^2)*3 + 5 using the available tools.",
            registry=registry,
            llm=llm,
            budget=budget,
            guard=guard,
            scratchpad=scratchpad,
        )
        print(f"    Replay result: {replay}")

        # ---- Scratchpad introspection ----
        print("\n[5] Scratchpad notes (last 5):")
        for note in scratchpad.notes()[-5:]:
            print(f"    [{note['ts']:.1f}] {note['heading']}: {str(note['body'])[:60]}")

        # ---- Budget stats ----
        print("\n[6] Budget stats:")
        for k, v in result["budget"].items():
            print(f"    {k}: {v}")

        # ---- Tool-call trace ----
        print("\n[7] Tool-call trace:")
        for step in result["trace"]:
            print(f"    iter={step['iter']} tool={step['tool']}({step['args']}) → {step['observation']}")

        # ---- Sandbox demo ----
        print("\n[8] Sandbox demo:")
        sample = "import os; os.system('echo hi')"
        print(f"    inspect_source({sample!r}) → {inspect_source(sample)}")
        sample2 = "x = sum(range(10))\nprint(x)"
        print(f"    inspect_source({sample2!r}) → {inspect_source(sample2)}")
        result2 = run_in_subprocess(sample2, timeout_s=2.0, limits=ResourceLimits(memory_mb=32))
        print(f"    run_in_subprocess(...) → {result2.stdout.strip()!r}, ok={result2.ok}")

        # ---- Assertions to make CI happy ----
        assert result["answer"] == "152", f"expected 152, got {result['answer']}"
        assert result["iterations"] == 3, f"expected 3 tool calls, got {result['iterations']}"
        assert replay.get("skipped") == "duplicate-run", "replay should be deduped"

        print("\n✓ All assertions passed.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())