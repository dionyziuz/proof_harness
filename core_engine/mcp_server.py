"""Model Context Protocol (MCP) Server for Prime Math.

Exposes tools for Antigravity (AGY) and Gemini:
- solve_math_problem: End-to-end reasoning and execution loop
- run_math_code: Direct execution in the persistent mathematical REPL
- reset_math_session: Resets REPL state
- inspect_math_session: Lists defined symbols, variables, and matrices
- get_recent_thinking_log: Reads real-time thinking tokens and execution logs
- search_math_heuristics: Queries CoALA procedural memory in TurboVec
- record_math_heuristic: Commits a verified mathematical rule into TurboVec
"""

from __future__ import annotations

import logging

from mcp.server.mcpserver import MCPServer

from .kernel import PersistentMathKernel
from .memory import CoALAMemoryManager
from .solver import DEFAULT_LOG_PATH, MathSolver

logger = logging.getLogger("prime_math.mcp")

# Initialize global shared kernel, memory manager, and solver for the MCP session
memory = CoALAMemoryManager()
kernel = PersistentMathKernel()
solver = MathSolver(kernel=kernel, memory=memory)

server = MCPServer(
    name="prime-math",
    description="Agentic mathematical reasoning and persistent Python execution harness",
    version="0.1.0",
)


@server.tool()
async def solve_math_problem(problem: str, max_iterations: int = 6) -> str:
    """Solves a complex math problem autonomously using the thinking model and persistent Python REPL.

    Streams inner reasoning and code execution traces live to logs/thinking.log.

    Args:
        problem: The mathematical problem description, question, or theorem to prove.
        max_iterations: Maximum number of iterative reasoning-code-correction turns (default: 6).

    Returns:
        A structured solution containing the final boxed answer, full reasoning derivation, and executed code.
    """
    res = await solver.asolve(problem=problem, max_iterations=max_iterations, reset_kernel=True)

    lines = []
    lines.append("### Problem Resolution\n")
    if res.boxed_answer:
        lines.append(f"**Final Answer**: \\boxed{{{res.boxed_answer}}}\n")
    elif res.answer:
        lines.append(f"**Final Answer**: {res.answer}\n")
    else:
        lines.append("**Final Answer**: Unable to reach a definitive closed-form answer within iteration limit.\n")

    lines.append(f"- **Verified by Execution**: {res.verified}")
    lines.append(f"- **Iterations Used**: {res.iterations}")
    lines.append(f"- **Live Thinking Log**: `{res.log_file}`")

    if res.retrieved_heuristics:
        lines.append(f"\n#### 🧠 CoALA Procedural Memory Injected ({len(res.retrieved_heuristics)} rules):")
        for h in res.retrieved_heuristics:
            lines.append(f"- {h}")

    if res.learned_heuristics:
        lines.append("\n#### 🎓 New Procedural Heuristics Learned & Committed to TurboVec:")
        for h in res.learned_heuristics:
            lines.append(f"- {h}")

    if res.reasoning_traces:
        lines.append("\n#### 💭 Inner Thinking & Deliberation (Turn-by-Turn):")
        for i, trace in enumerate(res.reasoning_traces, start=1):
            lines.append(f"\n**Turn {i} Deliberation:**\n```text\n{trace}\n```")

    if res.code_history:
        lines.append(f"\n#### 💻 Verification Code Executed ({len(res.code_history)} blocks):")
        for i, (code, exec_res) in enumerate(zip(res.code_history, res.execution_history), start=1):
            status = "✅ Success" if exec_res.success else "❌ Error"
            lines.append(f"```python\n# Block {i} ({status}, {exec_res.execution_time}s)\n{code}\n```")
            if exec_res.stdout:
                lines.append(f"**Output**:\n```\n{exec_res.stdout.strip()}\n```")
            if exec_res.result_repr:
                lines.append(f"**Evaluated Value**: `{exec_res.result_repr}`")
            if exec_res.error:
                lines.append(f"**Error**:\n```\n{exec_res.error.strip()}\n```")

    if res.final_response:
        lines.append(f"\n#### Formal Solution:\n{res.final_response}")

    return "\n".join(lines)


@server.tool()
async def run_math_code(code: str, timeout_seconds: float = 30.0) -> str:
    """Executes Python code in the persistent mathematical REPL.

    Preloaded: SymPy (sp, x, y, z, n, k, solve, Eq), Z3 (Solver, Int, Real), NumPy (np),
    SciPy, mpmath (mp, 50-digit precision), math, itertools, fractions.
    Variables and definitions persist across calls.

    Args:
        code: Python statements or expressions to execute.
        timeout_seconds: Maximum allowed runtime in seconds before aborting (default: 30s).

    Returns:
        Stdout output, evaluated expression value, execution time, and any error message.
    """
    exec_res = kernel.execute(code, timeout_seconds=timeout_seconds)

    out = []
    status = "SUCCESS" if exec_res.success else "FAILED"
    out.append(f"[{status}] Execution finished in {exec_res.execution_time}s")

    if exec_res.stdout:
        out.append(f"Stdout:\n{exec_res.stdout.strip()}")

    if exec_res.result_repr:
        out.append(f"Result: {exec_res.result_repr}")

    if exec_res.error:
        out.append(f"Error:\n{exec_res.error.strip()}")

    user_vars = kernel.get_user_variables()
    if user_vars:
        out.append(f"Active User Variables: {user_vars}")

    return "\n\n".join(out)


@server.tool()
async def reset_math_session() -> str:
    """Clears all user-defined variables and resets the persistent mathematical REPL to clean prelude state."""
    kernel.reset()
    return "Persistent math session has been reset. All user variables cleared; standard preludes reloaded."


@server.tool()
async def inspect_math_session() -> str:
    """Lists all currently defined user variables, symbols, and objects in the persistent REPL."""
    user_vars = kernel.get_user_variables()
    if not user_vars:
        return "No user-defined variables in active session (only standard math preludes are loaded)."

    lines = ["Active Session Variables:"]
    for name, preview in user_vars.items():
        lines.append(f"- `{name}`: {preview}")
    return "\n".join(lines)


@server.tool()
async def get_recent_thinking_log(lines: int = 60) -> str:
    """Reads the most recent lines from the live thinking and execution log.

    Args:
        lines: Number of trailing lines to return from the log (default: 60).

    Returns:
        The recent thinking tokens, code executions, and kernel outputs.
    """
    p = DEFAULT_LOG_PATH
    if not p.exists():
        return f"No thinking log found at {p} yet."

    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
            return "".join(tail).strip()
    except Exception as e:
        return f"Error reading thinking log: {e}"


@server.tool()
async def search_math_heuristics(query: str, limit: int = 3) -> str:
    """Searches TurboVec procedural memory for proven mathematical rules, strategies, or code patterns.

    Args:
        query: Mathematical topic, equation type, or problem domain (e.g. 'Diophantine degree 3').
        limit: Max number of heuristics to retrieve.

    Returns:
        Formatted list of retrieved procedural heuristics.
    """
    heuristics = await memory.aquery_procedural_heuristics(query=query, limit=limit)
    if not heuristics:
        return f"No matching procedural heuristics found in TurboVec for query: '{query}'."

    lines = [f"Retrieved {len(heuristics)} CoALA Procedural Rules from TurboVec:"]
    for i, h in enumerate(heuristics, start=1):
        lines.append(f"{i}. {h}")
    return "\n".join(lines)


@server.tool()
async def record_math_heuristic(rule: str) -> str:
    """Manually stores a new verified mathematical heuristic into TurboVec procedural memory.

    Enforces near-duplicate deduplication (score > 0.90) to prevent rule clutter.

    Args:
        rule: Concise mathematical strategy, invariant observation, or code pattern.

    Returns:
        Confirmation message.
    """
    added = await memory.aadd_procedural_heuristic(rule=rule)
    if added:
        return f"Successfully committed procedural rule to TurboVec ('{memory.agent_id}_procedural'):\n{rule}"
    return f"Rule was skipped as a near-duplicate of an existing procedural memory in TurboVec:\n{rule}"


def main():
    """Entry point for running the MCP server over stdio."""
    import sys
    # Suppress verbose root logs to avoid corrupting stdio MCP communication
    logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
