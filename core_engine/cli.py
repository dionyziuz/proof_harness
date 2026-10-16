"""Command-line interface for Prime Math.

Allows direct interactive problem solving with AGY-style high-level step progress,
REPL interaction, and MCP server launch.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from .config import load_config
from .kernel import PersistentMathKernel
from .solver import MathSolver
from .prover import MathProver
from .falsifier import MathFalsifier
from .discoverer import MathDiscoverer
from .token_shield import extract_high_level_steps
from .mcp_server import main as run_mcp

console = Console()


def cmd_solve(args: argparse.Namespace) -> None:
    """Solves a problem via terminal with clean high-level step-by-step progress."""
    problem = " ".join(args.problem)
    if not problem:
        console.print("[bold red]Error: No problem statement provided.[/bold red]")
        sys.exit(1)

    cfg = load_config()
    console.print(
        Panel(
            f"[bold cyan]Problem:[/bold cyan] {problem}",
            title="Prime Math",
            subtitle=f"Engine: {cfg.base_url} ({cfg.model})",
        )
    )

    solver = MathSolver(default_max_iterations=args.max_iterations)

    def on_step(event: str, data: Any) -> None:
        if event == "turn_start":
            console.print(f"\n[bold blue]─── Turn {data}: Analyzing & Formulating Plan ───[/bold blue]")
        elif event == "turn_done":
            reasoning = data.get("reasoning", "")
            steps = extract_high_level_steps(reasoning, max_points=3)
            if steps:
                step_lines = "\n".join([f"  [cyan]•[/cyan] {s}" for s in steps])
                console.print(
                    Panel(
                        step_lines,
                        title=f"🧠 Step {data.get('turn')}: Reasoning & Strategy",
                        border_style="blue",
                    )
                )
        elif event == "code_start":
            console.print("[bold yellow]⚡ Python Execution:[/bold yellow]")
            console.print(Syntax(data, "python", theme="monokai", line_numbers=False))
        elif event == "code_done":
            status = "[green]✓ Success[/green]" if data.success else "[red]✗ Error[/red]"
            summary = f"[{status}] Runtime: {data.execution_time}s"
            if data.result_repr:
                summary += f" | [bold cyan]=> {data.result_repr}[/bold cyan]"
            elif data.stdout:
                first_line = data.stdout.strip().splitlines()[0]
                summary += f" | [dim]{first_line[:80]}[/dim]"
            console.print(summary)
            if data.error:
                console.print(f"[bold red]Traceback:[/bold red] {data.error.strip()}")

    res = solver.solve(problem, max_iterations=args.max_iterations, on_step=on_step)

    console.print("\n" + "=" * 60 + "\n")
    if res.boxed_answer:
        extra = []
        if res.retrieved_heuristics:
            extra.append(f"Retrieved Rules: {len(res.retrieved_heuristics)}")
        if res.learned_heuristics:
            extra.append(f"Learned Heuristics: {len(res.learned_heuristics)}")
        extra_str = (" | " + " | ".join(extra)) if extra else ""

        console.print(
            Panel(
                f"[bold green]Final Answer:[/bold green] \\boxed{{{res.boxed_answer}}}\n"
                f"[dim]Verified: {res.verified} | Iterations: {res.iterations}{extra_str}[/dim]",
                border_style="green",
            )
        )
    elif res.answer:
        console.print(Panel(f"[bold yellow]Final Answer:[/bold yellow] {res.answer}", border_style="yellow"))
    else:
        console.print(Panel("[bold red]No conclusive answer reached within iteration limit.[/bold red]", border_style="red"))


def cmd_prove(args: argparse.Namespace) -> None:
    """Derives, proves, and computationally verifies a mathematical model or theorem."""
    spec = " ".join(args.spec)
    if not spec:
        console.print("[bold red]Error: No specification or conjecture provided.[/bold red]")
        sys.exit(1)

    cfg = load_config()
    console.print(
        Panel(
            f"[bold cyan]Conjecture / Model Spec:[/bold cyan] {spec}",
            title="Prime Math - Prover & Model Derivation Engine",
            subtitle=f"Engine: {cfg.base_url} ({cfg.model})",
        )
    )

    prover = MathProver(
        default_max_iterations=args.max_iterations,
        enable_literature_search=getattr(args, "literature_search", False),
    )

    def on_step(event: str, data: Any) -> None:
        if event == "turn_start":
            console.print(f"\n[bold blue]─── Turn {data}: Analyzing & Formalizing Invariants ───[/bold blue]")
        elif event == "code_start":
            console.print("[bold yellow]⚡ Python Verification Kernel:[/bold yellow]")
            console.print(Syntax(data, "python", theme="monokai", line_numbers=False))
        elif event == "code_done":
            status = "[green]✓ Success[/green]" if data.success else "[red]✗ Error[/red]"
            summary = f"[{status}] Runtime: {data.execution_time}s"
            if data.result_repr:
                summary += f" | [bold cyan]=> {data.result_repr}[/bold cyan]"
            elif data.stdout:
                first_line = data.stdout.strip().splitlines()[0]
                summary += f" | [dim]{first_line[:80]}[/dim]"
            console.print(summary)
            if data.error:
                console.print(f"[bold red]Traceback:[/bold red] {data.error.strip()}")

    res = prover.prove(spec, max_iterations=args.max_iterations, on_step=on_step)

    console.print("\n" + "=" * 60 + "\n")
    if res.status == "PROVEN":
        invariants_str = "\n".join([f"  • {inv}" for inv in res.verified_invariants]) if res.verified_invariants else "  (None explicitly tagged)"
        console.print(
            Panel(
                f"[bold green]THEOREM / FORMULATION:[/bold green]\n{res.theorem}\n\n"
                f"[bold cyan]VERIFIED INVARIANTS:[/bold cyan]\n"
                f"{invariants_str}\n\n"
                f"[dim]Iterations: {res.iterations} | Log: {res.log_file}[/dim]",
                title="[bold green]✓ Mathematically Proven & Verified[/bold green]",
                border_style="green",
            )
        )
        if res.implementation:
            console.print(
                Panel(
                    Syntax(res.implementation, "python", theme="monokai", line_numbers=True),
                    title="Verified Implementation",
                    border_style="cyan",
                )
            )
    else:
        console.print(
            Panel(
                f"[bold yellow]Proof Status: {res.status}[/bold yellow]\n\n"
                f"{res.final_response[:500]}...",
                border_style="yellow",
            )
        )


def cmd_falsify(args: argparse.Namespace) -> None:
    """Actively stress-tests and searches for counterexamples to a mathematical claim."""
    claim = " ".join(args.claim)
    if not claim:
        console.print("[bold red]Error: No claim or conjecture provided.[/bold red]")
        sys.exit(1)

    cfg = load_config()
    console.print(
        Panel(
            f"[bold cyan]Claim to Falsify:[/bold cyan] {claim}",
            title="Prime Math - Adversarial Falsifier & Counterexample Hunter",
            subtitle=f"Engine: {cfg.base_url} ({cfg.model})",
        )
    )

    falsifier = MathFalsifier(default_max_iterations=args.max_iterations)

    def on_step(event: str, data: Any) -> None:
        if event == "turn_start":
            console.print(f"\n[bold blue]─── Turn {data}: Searching for Edge-Case Counterexamples ───[/bold blue]")
        elif event == "code_start":
            console.print("[bold yellow]⚡ Python SMT / Search Kernel:[/bold yellow]")
            console.print(Syntax(data, "python", theme="monokai", line_numbers=False))
        elif event == "code_done":
            status = "[green]✓ Success[/green]" if data.success else "[red]✗ Error[/red]"
            summary = f"[{status}] Runtime: {data.execution_time}s"
            if data.result_repr:
                summary += f" | [bold cyan]=> {data.result_repr}[/bold cyan]"
            elif data.stdout:
                first_line = data.stdout.strip().splitlines()[0]
                summary += f" | [dim]{first_line[:80]}[/dim]"
            console.print(summary)
            if data.error:
                console.print(f"[bold red]Traceback:[/bold red] {data.error.strip()}")

    res = falsifier.falsify(claim, max_iterations=args.max_iterations, on_step=on_step)

    console.print("\n" + "=" * 60 + "\n")
    if res.status == "FALSIFIED":
        console.print(
            Panel(
                f"[bold red]VERDICT: FALSIFIED (Counterexample Discovered)[/bold red]\n\n"
                f"[bold yellow]COUNTEREXAMPLE:[/bold yellow]\n{res.counterexample or 'See implementation'}\n\n"
                f"[bold white]EXPLANATION:[/bold white]\n{res.explanation or 'Counterexample found violating the claim.'}\n\n"
                f"[dim]Iterations: {res.iterations} | Log: {res.log_file}[/dim]",
                title="[bold red]✗ Claim Disproven[/bold red]",
                border_style="red",
            )
        )
        if res.implementation:
            console.print(
                Panel(
                    Syntax(res.implementation, "python", theme="monokai", line_numbers=True),
                    title="Counterexample Verification Script",
                    border_style="red",
                )
            )
    elif res.status == "UNFALSIFIED":
        console.print(
            Panel(
                f"[bold green]VERDICT: UNFALSIFIED (No Counterexample Found)[/bold green]\n\n"
                f"[bold cyan]PROOF / BOUNDS:[/bold cyan]\n{res.proof or 'Exhaustive SMT search verified claim over domain.'}\n\n"
                f"[dim]Iterations: {res.iterations} | Log: {res.log_file}[/dim]",
                title="[bold green]✓ Claim Stood Up to Adversarial Search[/bold green]",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold yellow]Falsification Verdict: {res.status}[/bold yellow]\n\n"
                f"{res.final_response[:500]}...",
                border_style="yellow",
            )
        )


def cmd_discover(args: argparse.Namespace) -> None:
    """Inductively discovers mathematical laws, closed forms, or recurrences from data."""
    data_input = " ".join(args.data)
    if not data_input:
        console.print("[bold red]Error: No sequence, data points, or system description provided.[/bold red]")
        sys.exit(1)

    cfg = load_config()
    console.print(
        Panel(
            f"[bold cyan]Input Data / Sequence:[/bold cyan] {data_input}",
            title="Prime Math - Mathematical Law & Invariant Discoverer",
            subtitle=f"Engine: {cfg.base_url} ({cfg.model})",
        )
    )

    discoverer = MathDiscoverer(default_max_iterations=args.max_iterations)

    def on_step(event: str, data: Any) -> None:
        if event == "turn_start":
            console.print(f"\n[bold blue]─── Turn {data}: Analyzing Sequences & Probing Transformations ───[/bold blue]")
        elif event == "code_start":
            console.print("[bold yellow]⚡ Python Discovery Kernel:[/bold yellow]")
            console.print(Syntax(data, "python", theme="monokai", line_numbers=False))
        elif event == "code_done":
            status = "[green]✓ Success[/green]" if data.success else "[red]✗ Error[/red]"
            summary = f"[{status}] Runtime: {data.execution_time}s"
            if data.result_repr:
                summary += f" | [bold cyan]=> {data.result_repr}[/bold cyan]"
            elif data.stdout:
                first_line = data.stdout.strip().splitlines()[0]
                summary += f" | [dim]{first_line[:80]}[/dim]"
            console.print(summary)
            if data.error:
                console.print(f"[bold red]Traceback:[/bold red] {data.error.strip()}")

    res = discoverer.discover(data_input, max_iterations=args.max_iterations, on_step=on_step)

    console.print("\n" + "=" * 60 + "\n")
    if res.status == "DISCOVERED":
        console.print(
            Panel(
                f"[bold green]DISCOVERED LAW / INVARIANT:[/bold green]\n{res.law or 'Mathematical Law'}\n\n"
                f"[bold cyan]GOVERNING FORMULA / CONJECTURE:[/bold cyan]\n{res.conjecture or 'See implementation'}\n\n"
                f"[bold white]COMPUTATIONAL EVIDENCE:[/bold white]\n{res.evidence or 'Verified against input data.'}\n\n"
                f"[dim]Iterations: {res.iterations} | Log: {res.log_file}[/dim]",
                title="[bold green]✓ Mathematical Law Discovered & Formulated[/bold green]",
                border_style="green",
            )
        )
        if res.implementation:
            console.print(
                Panel(
                    Syntax(res.implementation, "python", theme="monokai", line_numbers=True),
                    title="Verified Implementation & Generator",
                    border_style="cyan",
                )
            )
    else:
        console.print(
            Panel(
                f"[bold yellow]Discovery Status: {res.status}[/bold yellow]\n\n"
                f"{res.final_response[:500]}...",
                border_style="yellow",
            )
        )


def cmd_eval(args: argparse.Namespace) -> None:
    """Executes code directly in the persistent math kernel."""
    code = " ".join(args.code)
    kernel = PersistentMathKernel()
    res = kernel.execute(code)
    if res.stdout:
        print(res.stdout, end="")
    if res.result_repr:
        console.print(f"[bold cyan]=>[/bold cyan] {res.result_repr}")
    if res.error:
        console.print(f"[bold red]Error:[/bold red] {res.error}")


def cmd_repl(args: argparse.Namespace) -> None:
    """Interactive persistent REPL shell."""
    kernel = PersistentMathKernel()
    console.print(
        Panel(
            "[bold green]Prime Math Persistent REPL[/bold green]\n"
            "SymPy, Z3, mpmath, NumPy, and SciPy preloaded.\n"
            "Type [cyan]exit[/cyan] or [cyan]quit[/cyan] to exit, [cyan]vars[/cyan] to inspect variables, [cyan]reset[/cyan] to clear.",
            title="REPL",
        )
    )

    while True:
        try:
            line = input("prime-math > ")
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting.")
            break

        stripped = line.strip()
        if not stripped:
            continue
        if stripped in ("exit", "quit"):
            break
        if stripped == "reset":
            kernel.reset()
            console.print("[yellow]Session reset.[/yellow]")
            continue
        if stripped == "vars":
            uvars = kernel.get_user_variables()
            if not uvars:
                console.print("[dim]No user variables defined.[/dim]")
            else:
                table = Table(title="Active Variables")
                table.add_column("Name", style="cyan")
                table.add_column("Value / Representation", style="magenta")
                for k, v in uvars.items():
                    table.add_row(k, v)
                console.print(table)
            continue

        res = kernel.execute(stripped)
        if res.stdout:
            print(res.stdout, end="")
        if res.result_repr:
            console.print(f"[bold cyan]=>[/bold cyan] {res.result_repr}")
        if res.error:
            console.print(f"[bold red]Error:[/bold red] {res.error}")


def cmd_step(args: argparse.Namespace) -> None:
    """Executes a single reasoning-and-calculation turn of a session."""
    from .session import SolverSession

    if args.action == "start":
        problem = " ".join(args.problem)
        if not problem:
            console.print("[bold red]Error: No problem provided for step start.[/bold red]")
            sys.exit(1)
        console.print(Panel(f"[bold cyan]Problem:[/bold cyan] {problem}", title="Prime Math (Step Session)"))
        session = SolverSession(problem=problem, max_iterations=args.max_iterations)
        report = session.step()
        session.save()
    elif args.action == "next":
        try:
            session = SolverSession.load()
        except Exception as e:
            console.print(f"[bold red]Error loading active session:[/bold red] {e}")
            sys.exit(1)
        report = session.step()
        session.save()
    else:
        console.print("[bold red]Action must be 'start' or 'next'[/bold red]")
        return

    # Render turn report
    if report.reasoning_steps:
        bullets = "\n".join([f"  [cyan]•[/cyan] {s}" for s in report.reasoning_steps])
        console.print(Panel(bullets, title=f"🧠 Step {report.turn}: Reasoning & Strategy", border_style="blue"))

    if report.code_blocks:
        for i, (code, exec_res) in enumerate(zip(report.code_blocks, report.exec_results), start=1):
            console.print(f"[bold yellow]⚡ Python Execution (Block {i}):[/bold yellow]")
            console.print(Syntax(code, "python", theme="monokai", line_numbers=False))
            status = "[green]✓ Success[/green]" if exec_res.success else "[red]✗ Error[/red]"
            summary = f"[{status}] Runtime: {exec_res.execution_time}s"
            if exec_res.result_repr:
                summary += f" | [bold cyan]=> {exec_res.result_repr}[/bold cyan]"
            elif exec_res.stdout:
                first_line = exec_res.stdout.strip().splitlines()[0]
                summary += f" | [dim]{first_line[:80]}[/dim]"
            console.print(summary)
            if exec_res.error:
                console.print(f"[bold red]Traceback:[/bold red] {exec_res.error.strip()}")

    if report.boxed_answer:
        console.print(
            Panel(
                f"[bold green]Final Answer Verified:[/bold green] \\boxed{{{report.boxed_answer}}}\n"
                f"[dim]Finished in {report.turn} turns.[/dim]",
                border_style="green",
            )
        )
    elif report.finished:
        console.print(Panel("[bold red]Session finished without conclusive boxed answer.[/bold red]", border_style="red"))
    else:
        console.print(f"[dim]Turn {report.turn} complete. Ready for next step.[/dim]")


def cmd_bench(args: argparse.Namespace) -> None:
    """Runs automated benchmark against Project Euler dataset."""
    from .evaluator import EulerEvaluator
    kwargs = {
        "num_workers": args.workers,
        "max_iterations": args.max_iterations,
    }
    if getattr(args, "dataset", None):
        kwargs["dataset_path"] = Path(args.dataset)
    evaluator = EulerEvaluator(**kwargs)
    
    if args.problems:
        prob_ids = [int(p.strip()) for p in args.problems.split(",") if p.strip()]
    else:
        prob_ids = list(range(args.start, args.end + 1))
        
    summary = evaluator.run_benchmark(prob_ids, skip_passed=not args.force)
    
    # Render rich summary table
    table = Table(title="Benchmark Results Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Total Problems Tested", str(summary.total_problems))
    table.add_row("Passed", f"[green]{summary.passed}[/green]")
    table.add_row("Failed", f"[red]{summary.failed}[/red]")
    table.add_row("Pass Rate", f"[bold green]{summary.pass_rate_percent}%[/bold green]")
    table.add_row("Total Runtime", f"{summary.total_runtime_seconds}s")
    table.add_row("Avg Time / Problem", f"{summary.average_runtime_seconds}s")
    console.print(table)
    
    if summary.failure_taxonomy:
        f_table = Table(title="Failure Taxonomy Breakdown")
        f_table.add_column("Failure Category", style="red")
        f_table.add_column("Count", style="yellow")
        for cat, count in summary.failure_taxonomy.items():
            f_table.add_row(cat, str(count))
        console.print(f_table)


def cmd_frontier(args: argparse.Namespace) -> None:
    """Runs automated benchmark against FrontierMath dataset."""
    if getattr(args, "no_fast_composer", False):
        os.environ["PRIME_MATH_FAST_ENABLED"] = "false"

    from .evaluator_frontier import FrontierMathEvaluator
    evaluator = FrontierMathEvaluator(
        num_workers=args.workers,
        max_iterations=args.max_iterations,
        max_problem_timeout_seconds=args.timeout,
    )
    prob_ids = [int(p.strip()) for p in args.problems.split(",") if p.strip()] if args.problems else None
    summary = evaluator.run_benchmark(tier=args.tier, problem_ids=prob_ids)

    # Render rich summary table
    table = Table(title="FrontierMath Benchmark Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Total Problems", str(summary.total_problems))
    table.add_row("Passed", f"[green]{summary.passed}[/green]")
    table.add_row("Failed", f"[red]{summary.failed}[/red]")
    table.add_row("Pass Rate", f"[bold green]{summary.pass_rate_percent}%[/bold green]")
    table.add_row("Total Runtime", f"{summary.total_runtime_seconds}s")
    for tier, acc in summary.tier_accuracy.items():
        table.add_row(f"Accuracy ({tier})", acc)
    console.print(table)


def cmd_remediate(args: argparse.Namespace) -> None:
    """Runs automated 3-tier post-evaluation remediation on failed problems."""
    from .remediation import RemediationEngine

    prob_ids = None
    if args.problems:
        prob_ids = [int(p.strip()) for p in args.problems.split(",") if p.strip()]

    max_lines = getattr(args, "max_tier1_lines", 10)
    engine = RemediationEngine(
        max_tier1_lines_target=max_lines,
        max_tier1_lines_replacement=int(max_lines * 1.5),
    )
    engine.run_remediation(
        problem_ids=prob_ids,
        latest_n_evals=args.latest,
        auto_apply_tier3=not args.no_apply,
        auto_apply_tier1=not args.no_apply_tier1 and not args.no_apply,
        dry_run=args.dry_run,
        limit=args.limit,
    )


def cmd_heuristic(args: argparse.Namespace) -> None:
    """Manages CoALA procedural heuristics in TurboVec."""
    if not getattr(args, "action", None):
        console.print("[yellow]Usage: prime-math heuristic [add|search] ...[/yellow]")
        console.print("Run 'prime-math heuristic --help' for available actions.")
        return

    from .memory import CoALAMemoryManager
    from .remediation import is_valid_universal_heuristic

    memory = CoALAMemoryManager()
    action = args.action

    if action == "add":
        rule = " ".join(args.rule).strip()
        is_valid, reason = is_valid_universal_heuristic(rule)
        if not is_valid:
            console.print(f"[bold red]❌ Rejected by 4-Gate Universality Rubric:[/bold red] {reason}")
            sys.exit(1)

        existing = memory.query_procedural_heuristics(rule, limit=1, min_score=0.90)
        if existing:
            console.print(f"[yellow]⚠️ Skipped as near-duplicate (>0.90 match):[/yellow]\n{existing[0]}")
            return

        success = memory.add_procedural_heuristic(rule)
        if success:
            console.print(f"[bold green]✅ Successfully committed procedural heuristic to TurboVec:[/bold green]\n{rule}")
        else:
            console.print("[red]Failed to commit heuristic.[/red]")
            sys.exit(1)

    elif action == "search":
        query = " ".join(args.query).strip()
        limit = getattr(args, "limit", 5)
        min_score = getattr(args, "min_score", 0.50)

        col = f"{memory.agent_id}_procedural"
        url = f"{memory.base_url}/v1/memory/{col}/search"
        with memory._client() as client:
            resp = client.get(url, params={"query": query, "limit": limit * 2})
            if resp.status_code != 200:
                console.print(f"[red]TurboVec query failed ({resp.status_code}): {resp.text}[/red]")
                return
            data = resp.json()
            results = data.get("results", [])

        valid_results = []
        for r in results:
            score = r.get("score", r.get("similarity", 0.0))
            if score >= min_score:
                text = r.get("payload", {}).get("text", "").strip()
                if text:
                    valid_results.append((score, text))

        if not valid_results:
            console.print(f"[yellow]No matching procedural heuristics found for query: '{query}' (min_score={min_score})[/yellow]")
            return

        table = Table(title=f"CoALA Procedural Heuristics (Query: '{query}')")
        table.add_column("Score", style="cyan", width=8)
        table.add_column("Heuristic Rule", style="white")
        for score, text in valid_results[:limit]:
            table.add_row(f"{score:.3f}", text)
        console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(prog="prime-math", description="Prime Math CLI & Agentic Harness")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # heuristic
    heur_p = subparsers.add_parser("heuristic", help="Manage CoALA procedural heuristics in TurboVec")
    heur_sub = heur_p.add_subparsers(dest="action", help="Heuristic action (add, search)")

    h_add_p = heur_sub.add_parser("add", help="Validate and commit a universal mathematical heuristic")
    h_add_p.add_argument("rule", nargs="+", help="Heuristic rule text")

    h_search_p = heur_sub.add_parser("search", help="Search procedural heuristics via TurboVec vector similarity")
    h_search_p.add_argument("query", nargs="+", help="Mathematical topic or concept query")
    h_search_p.add_argument("--limit", type=int, default=5, help="Maximum number of heuristics to return (default: 5)")
    h_search_p.add_argument("--min-score", type=float, default=0.50, help="Minimum cosine similarity score (default: 0.50)")
    heur_p.set_defaults(func=cmd_heuristic)

    # remediate
    remed_p = subparsers.add_parser("remediate", help="Run automated 3-tier post-evaluation failure remediation")
    remed_p.add_argument("--problems", type=str, default=None, help="Comma-separated problem IDs to remediate (e.g. 850,851,854)")
    remed_p.add_argument("--latest", type=int, default=1, help="Number of latest evaluation run files to scan (default: 1)")
    remed_p.add_argument("--limit", type=int, default=None, help="Maximum number of failed problems to diagnose")
    remed_p.add_argument("--dry-run", action="store_true", help="Diagnose failures and evaluate Tier 2 rubric without committing to TurboVec or modifying code")
    remed_p.add_argument("--no-apply", action="store_true", help="Do not auto-apply Tier 3 procedural heuristics or Tier 1 harness patches")
    remed_p.add_argument("--no-apply-tier1", action="store_true", help="Do not auto-apply Tier 1 harness code patches (leaves staged only)")
    remed_p.add_argument("--max-tier1-lines", type=int, default=10, help="Maximum lines allowed for automated Tier 1 patch before parking for paired AGY review (default: 10)")
    remed_p.set_defaults(func=cmd_remediate)

    # bench
    bench_p = subparsers.add_parser("bench", help="Run automated benchmark on Project Euler problems")
    bench_p.add_argument("--start", type=int, default=1, help="Starting problem ID")
    bench_p.add_argument("--end", type=int, default=10, help="Ending problem ID")
    bench_p.add_argument("--problems", type=str, default=None, help="Comma-separated problem IDs (e.g. 1,2,3,78)")
    bench_p.add_argument("--workers", type=int, default=2, help="Number of concurrent workers (default: 2)")
    bench_p.add_argument("--max-iterations", type=int, default=12, help="Max iterations per problem (default: 12)")
    bench_p.add_argument("--dataset", type=str, default=None, help="Custom dataset JSON path")
    bench_p.add_argument("--force", action="store_true", help="Force re-running already solved problems (default skips passed)")
    bench_p.set_defaults(func=cmd_bench)

    # frontier
    frontier_p = subparsers.add_parser("frontier", help="Run automated benchmark on FrontierMath problems (Tiers 1-4)")
    frontier_p.add_argument("--tier", type=int, default=None, choices=[1, 2, 3, 4], help="Evaluate specific Tier (1-4)")
    frontier_p.add_argument("--problems", type=str, default=None, help="Comma-separated problem IDs (e.g. 1,2,5)")
    frontier_p.add_argument("--workers", type=int, default=2, help="Number of concurrent workers (default: 2)")
    frontier_p.add_argument("--max-iterations", type=int, default=60, help="Max iterations per problem (default: 60)")
    frontier_p.add_argument("--timeout", type=float, default=3600.0, help="Max wall-clock seconds per problem before moving on (default: 3600s / 1h)")
    frontier_p.add_argument("--no-fast-composer", action="store_true", help="Bypass Tier 1 Fast Composer and start directly with Deep Orchestrator + Stage 0")
    frontier_p.set_defaults(func=cmd_frontier)

    # step
    step_p = subparsers.add_parser("step", help="Execute problem solving step-by-step")
    step_sub = step_p.add_subparsers(dest="action", help="Step action")
    
    start_p = step_sub.add_parser("start", help="Start new step session")
    start_p.add_argument("problem", nargs="+", help="Problem statement")
    start_p.add_argument("--max-iterations", type=int, default=12, help="Max reasoning iterations (default: 12)")

    step_sub.add_parser("next", help="Execute next turn in active session")

    step_p.set_defaults(func=cmd_step)

    # prove
    prove_p = subparsers.add_parser("prove", help="Derive and computationally verify a generic mathematical model or theorem")
    prove_p.add_argument("spec", nargs="+", help="Conjecture, dynamical system, or model specification")
    prove_p.add_argument("--max-iterations", type=int, default=8, help="Max derivation/verification iterations (default: 8)")
    prove_p.add_argument("--literature-search", action="store_true", default=False, help="Enable pre-proof academic literature harvesting (arXiv / MathWorld)")
    prove_p.set_defaults(func=cmd_prove)

    # falsify
    falsify_p = subparsers.add_parser("falsify", help="Actively stress-test and search for counterexamples to a mathematical claim")
    falsify_p.add_argument("claim", nargs="+", help="Mathematical claim or conjecture to falsify")
    falsify_p.add_argument("--max-iterations", type=int, default=6, help="Max adversarial search iterations (default: 6)")
    falsify_p.set_defaults(func=cmd_falsify)

    # discover
    discover_p = subparsers.add_parser("discover", help="Inductively discover mathematical laws, closed forms, or recurrences from data")
    discover_p.add_argument("data", nargs="+", help="Sequence, data points, or mathematical system description to analyze")
    discover_p.add_argument("--max-iterations", type=int, default=8, help="Max discovery iterations (default: 8)")
    discover_p.set_defaults(func=cmd_discover)

    # solve
    solve_p = subparsers.add_parser("solve", help="Solve a math problem autonomously")
    solve_p.add_argument("problem", nargs="+", help="Problem statement")
    solve_p.add_argument("--max-iterations", type=int, default=12, help="Max reasoning iterations (default: 12)")
    solve_p.set_defaults(func=cmd_solve)

    # eval
    eval_p = subparsers.add_parser("eval", help="Execute math code directly in the preloaded kernel")
    eval_p.add_argument("code", nargs="+", help="Python code to evaluate")
    eval_p.set_defaults(func=cmd_eval)

    # repl
    repl_p = subparsers.add_parser("repl", help="Start interactive persistent math REPL")
    repl_p.set_defaults(func=cmd_repl)

    # mcp
    mcp_p = subparsers.add_parser("mcp", help="Start MCP server over stdio")
    mcp_p.set_defaults(func=lambda _: run_mcp())

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
