#!/usr/bin/env python3
"""run_lean_prover.py — Root CLI Runner for Prime Math's Formal Lean 4 Prover (`LeanProver`).

Features:
- Deterministic 5-Gate Lean 4 Compiler & `#print axioms` Kernel Audit (100% objective hit vs fail)
- Interactive `LeanKernel` with `verify_lean4_kernel`, `run_lean`, `sympy`, `mpmath`
- Supports `--verify-only <file.lean>` to audit any existing `.lean` file with `#print axioms`
- Saves certified `.lean` artifact directly alongside a structured JSON verification report
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core_engine import (
    ALLOWED_CORE_AXIOMS,
    CoALAMemoryManager,
    LeanKernel,
    LeanKernelValidationResult,
    LeanProver,
    TunnelThinkingClient,
    load_config,
    verify_lean4_kernel,
)
from core_engine.config import configure_env_file

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def print_lean_validation_table(title: str, val_res: LeanKernelValidationResult) -> None:
    """Renders a rich terminal scorecard for a LeanKernelValidationResult."""
    table = Table(title=title)
    table.add_column("Gate / Property", style="cyan")
    table.add_column("Result", style="magenta")

    hit_badge = (
        "[bold green]100% CERTIFIED HIT (PASS)[/bold green]"
        if val_res.passed
        else f"[bold red]REJECT ({val_res.status})[/bold red]"
    )
    table.add_row("Overall LeanProver Verdict", hit_badge)
    table.add_row("Lean Compiler Version", str(val_res.lean_version or "Lean 4"))
    table.add_row("Execution Time", f"{val_res.execution_time:.3f}s")
    table.add_row(
        "Gate 1: Compiler & Goals (`0 errors, 0 unsolved`)",
        "PASS"
        if (not val_res.compiler_errors and not val_res.unsolved_goals)
        else f"FAIL ({len(val_res.compiler_errors)} errors)",
    )
    table.add_row(
        "Gate 2: Zero `sorry` & Zero Warnings",
        "PASS"
        if (not val_res.has_sorry and not val_res.compiler_warnings)
        else f"FAIL (sorry={val_res.has_sorry}, warnings={len(val_res.compiler_warnings)})",
    )
    table.add_row(
        "Gate 3: `#print axioms` Purity (`0 custom axioms`)",
        "PASS" if not val_res.unauthorized_axioms else f"FAIL ({val_res.unauthorized_axioms})",
    )
    table.add_row(
        "Gate 4: Semantic Depth (`audit_lean_semantic_depth`)",
        "PASS" if not val_res.semantic_violations else f"FAIL ({val_res.semantic_violations})",
    )
    table.add_row(
        "Gate 5: Required Theorems Present",
        "PASS"
        if not val_res.missing_required_theorems
        else f"FAIL (missing: {val_res.missing_required_theorems})",
    )
    table.add_row(
        "Theorems Certified",
        f"{len(val_res.theorems_found)}: {', '.join(val_res.theorems_found)}",
    )
    table.add_row(
        "Definitions / Structures",
        f"{len(val_res.definitions_found)}: {', '.join(val_res.definitions_found)}",
    )

    if val_res.axiom_report:
        ax_lines = [
            f"{thm}: {deps if deps else '(none — pure kernel reduction)'}"
            for thm, deps in val_res.axiom_report.items()
        ]
        table.add_row(
            "Kernel `#print axioms` Audit",
            "\n".join(ax_lines[:15])
            + (f"\n... (+{len(ax_lines) - 15} more)" if len(ax_lines) > 15 else ""),
        )

    if not val_res.passed:
        table.add_row("Issues Summary", val_res.issues_summary)

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prime Math's Formal Lean 4 Prover (`LeanProver`).")
    parser.add_argument(
        "--verify-only",
        type=Path,
        default=None,
        help="Directly validate an existing .lean file through the 5-gate Kernel Judge without LLM calls.",
    )
    parser.add_argument(
        "--task-file",
        type=Path,
        default=None,
        help="Path to JSON task specification.",
    )
    parser.add_argument(
        "--output-lean",
        type=Path,
        default=None,
        help="Override output .lean file path (defaults to task JSON 'output_lean_file' or <task_id>.lean).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="Maximum interactive prover turns (default: 8).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Total solve timeout in seconds (default: 3600).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override PRIME_MATH_MODEL (defaults to .env PRIME_MATH_MODEL).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=PROJECT_ROOT / ".env",
        help="Path to .env configuration.",
    )
    parser.add_argument(
        "--lake-dir",
        type=Path,
        default=None,
        help="Optional Lake/Mathlib project directory for `lake env lean`.",
    )
    parser.add_argument(
        "--skip-stage0",
        action="store_true",
        default=False,
        help="Skip Stage-0 LLM Problem-Statement Audit.",
    )

    args = parser.parse_args()

    # Mode 1: Direct deterministic verification of a .lean file
    if args.verify_only is not None:
        lean_path = args.verify_only.resolve()
        if not lean_path.is_file():
            console.print(f"[bold red]Lean file not found:[/bold red] {lean_path}")
            sys.exit(1)
        code = lean_path.read_text(encoding="utf-8")
        val_res = verify_lean4_kernel(code, lake_dir=args.lake_dir)
        print_lean_validation_table(f"Lean 4 Kernel Audit: {lean_path.name}", val_res)
        sys.exit(0 if val_res.passed else 1)

    if args.task_file is None:
        console.print("[bold red]Please specify --task-file <task.json> or --verify-only <file.lean>[/bold red]")
        sys.exit(2)

    task_file = args.task_file.resolve()
    if not task_file.is_file():
        console.print(f"[bold red]Task file not found:[/bold red] {task_file}")
        sys.exit(1)

    with open(task_file, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    task_id = task_data.get("task_id", task_file.stem)
    task_prompt = task_data.get("prompt", "")
    required_theorems: List[str] = task_data.get("required_theorems", [])
    default_lean_name = task_data.get("output_lean_file", f"{task_id}.lean")
    out_lean_path = (args.output_lean or (task_file.parent / default_lean_name)).resolve()

    if args.env_file.is_file():
        configure_env_file(str(args.env_file))
    if args.model:
        os.environ["PRIME_MATH_MODEL"] = args.model

    cfg = load_config()
    client = TunnelThinkingClient(config=cfg)
    memory = CoALAMemoryManager(agent_id="lean_prover")

    log_dir = task_file.parent / "archive" / "logs"
    results_dir = task_file.parent / "archive" / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{task_id}.log"
    out_result_path = results_dir / f"{task_id}_lean_result.json"

    console.print(
        Panel(
            f"[bold blue]LeanProver Engine — 100% Deterministic Kernel Verification[/bold blue]\n"
            f"• Task ID: [bold]{task_id}[/bold]\n"
            f"• Model: [bold]{cfg.model}[/bold]\n"
            f"• Required Theorems ({len(required_theorems)}): [bold]{', '.join(required_theorems) or '(any)'}[/bold]\n"
            f"• Allowed Kernel Axioms: [bold]{sorted(ALLOWED_CORE_AXIOMS)}[/bold]\n"
            f"• Output Lean Artifact: [bold green]{out_lean_path}[/bold green]\n"
            f"• Log File: {log_path}",
            title="LeanProver Session",
            border_style="cyan",
        )
    )

    kernel = LeanKernel(required_theorems=required_theorems, lake_dir=args.lake_dir)
    prover = LeanProver(
        client=client,
        kernel=kernel,
        memory=memory,
        required_theorems=required_theorems,
        lake_dir=args.lake_dir,
        default_max_iterations=args.max_iterations,
        log_path=log_path,
        enable_thinking=True,
    )

    prob_audit = {"verdict": "VALID", "skipped": True}
    if not args.skip_stage0:
        console.print("[bold magenta]Running Stage-0 Problem-Formulation & Axiom Purity Audit...[/bold magenta]")
        prob_audit = prover.audit_problem_statement(task_prompt)
        if prob_audit.get("verdict") != "VALID":
            console.print(
                Panel(
                    f"[bold red]STAGE-0 AUDIT REJECTED[/bold red]\n"
                    f"• Issues: {prob_audit.get('issues')}",
                    title="Stage-0 Failure",
                )
            )
            sys.exit(2)
        console.print("[bold green]PASS — Stage-0 Specification Certified Axiom-Clean[/bold green]")

    def on_step(event: str, data: Any) -> None:
        if event == "turn_start":
            console.print(f"\n[bold cyan]─── LeanProver Turn {data}/{args.max_iterations} ───[/bold cyan]")
        elif event == "code_start":
            console.print(f"[yellow]⚡ Executing Python/Lean REPL block {data.get('block')}...[/yellow]")
        elif event == "audit_start":
            console.print("[magenta]>>> Running Deterministic 5-Gate Lean 4 Compiler & `#print axioms` Judge...[/magenta]")
        elif event == "audit_result":
            verdict = data.get("verdict", "")
            status = data.get("status", "")
            if verdict == "VALID":
                console.print(f"[bold green]>>> LEAN 4 ENGINE HIT: {status} (validity=1.00)[/bold green]")
            else:
                console.print(f"[bold red]>>> LEAN 4 ENGINE REJECT ({status}):[/bold red] {str(data.get('issues', ''))[:200]}")
        elif event == "proof_accepted":
            console.print("[bold green]*** 100% CERTIFIED LEAN 4 PROOF LOCKED IN ***[/bold green]")

    solve_res, val_res = prover.prove_lean(
        task_id=task_id,
        task_prompt=task_prompt,
        max_iterations=args.max_iterations,
        max_time_seconds=args.timeout,
        on_step=on_step,
    )

    print_lean_validation_table(f"LeanProver Final Verification: {task_id}", val_res)

    if val_res.lean_code.strip():
        out_lean_path.write_text(val_res.lean_code.strip() + "\n", encoding="utf-8")
        console.print(f"[bold green]Saved Lean 4 proof artifact to:[/bold green] {out_lean_path}")

    payload = {
        "task_id": task_id,
        "certified_hit": val_res.passed,
        "status": val_res.status,
        "verdict": val_res.verdict,
        "iterations": solve_res.iterations,
        "duration_s": solve_res.duration_s,
        "output_lean_file": str(out_lean_path),
        "stage0_audit": prob_audit,
        "lean_validation": val_res.to_dict(),
    }
    with open(out_result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    console.print(f"[bold green]Saved LeanProver verification report to:[/bold green] {out_result_path}")

    sys.exit(0 if val_res.passed else 1)


if __name__ == "__main__":
    main()
