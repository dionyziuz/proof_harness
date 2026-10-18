#!/usr/bin/env python3
"""TCS-Bench Stage 1 -- solve every task with the prime-math harness (solver script untouched).

Protocol (TCS-Bench, arXiv 2608.09538):
  * The solver receives ONLY the self-contained task prompt (context + target statement).
  * The ground-truth proof is withheld from the solver; it is used only at scoring time.
  * "Tasks are designed to be solved WITHOUT internet access. The solver is PROHIBITED
     from accessing the original paper or external sources."
    => Stage 0 (pre-solve academic web research) is DISABLED by default.
       (--enable-research exists only for ablation runs; it violates the TCS protocol.)

Anti-cheating, without any harness edits (solver.py / kernel.py / prompts.py untouched):
  1. The `ground_truth_solution` field is stripped from the task dict BEFORE the solver
     ever touches it -- the GT is physically absent from the solver's input.
  2. MathSolver(enable_pre_solve_research=False) -- Stage 0 performs no web lookups.
     (This is the existing constructor flag; the harness runs exactly as-is.)
  3. TCSCompliantKernel (below) -- the harness ALSO injects live web tools
     (search_math / read_paper / oeis_lookup) into the model's REPL namespace
     UNCONDITIONALLY (kernel.py _load_prelude, independent of the Stage 0 flag), and the
     orchestrator prompt advertises them. A 3-task smoke run (2026-09-16) observed the
     model using search_math mid-solve and retrieving the task's own source paper.
     The subclass replaces the three tools with protocol stubs and AST-scans every
     model code block for network-module imports, so any attempt (advertised tool or
     raw `import urllib.request` etc.) yields an explicit protocol message instead of
     a live fetch.
  4. Post-run evidence: scripts/tcs_bench_leakage_audit.py post --solve-dir <this run>
     checks candidate-vs-GT n-gram overlap, arXiv-id/arXiv-URL mentions in the task
     log, and counts web-tool call attempts (attempts that hit the stub are safe).

Fresh solver per task (same pattern as evaluator_frontier.py): own
TunnelThinkingClient + TCSCompliantKernel + CoALAMemoryManager + per-task log file.
Note: solve() always consolidates learned heuristics back to the SHARED TurboVec
collection (the harness has no disable flag) -- tasks from the same source paper can
influence each other through memory within a run; the audit script reports that grouping.

Output layout: data/eval_results/tcs_prove_<ts>/
  manifest.json           run args + task list
  tasks/<task_id>.json    candidate proof (SolveResult.final_response) + metadata
  logs/<task_id>.log      per-task harness log (the prover's own logging)
  summary.json            final counts

Usage:
  .venv/bin/python scripts/run_tcs_bench_prove.py --limit 3     # smoke test
  .venv/bin/python scripts/run_tcs_bench_prove.py               # all tasks, 2 workers
  .venv/bin/python scripts/run_tcs_bench_prove.py --force       # re-prove existing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

import httpx

from core_engine.kernel import PersistentMathKernel
from core_engine.llm_client import TunnelThinkingClient
from core_engine.memory import CoALAMemoryManager
from core_engine.prover import MathProver
from core_engine.prover_kernel import TCSCompliantKernel
from core_engine.tcs_config import load_tcs_config

_LOCAL_TCS_TASKS = PROJECT_ROOT / "data" / "tcs" / "tasks"
_EXTERNAL_TCS_TASKS = PROJECT_ROOT.parent / "TCSBench" / "tasks"
DEFAULT_TASKS_DIR = _LOCAL_TCS_TASKS if _LOCAL_TCS_TASKS.is_dir() else _EXTERNAL_TCS_TASKS
EVAL_DIR = PROJECT_ROOT / "data" / "eval_results"

console = Console()
print_lock = threading.Lock()


def log(msg: str) -> None:
    with print_lock:
        console.print(msg)


_NETWORK_MODULES = frozenset({
    "http", "httpx", "requests", "urllib", "socket", "socketserver", "ssl",
    "ftplib", "smtplib", "telnetlib", "imaplib", "poplib", "nntplib", "ldap",
    "websockets", "aiohttp",
})

_TCS_PROTOCOL_MSG = (
    "[TCS-BENCH PROTOCOL: internet access DISABLED for this run] "
    "This benchmark prohibits all external sources (the task's source paper included). "
    "Work only from the problem statement and context given in the task prompt."
)



def load_tasks(tasks_dir: Path, limit: Optional[int], task_id: Optional[str] = None, task_range: Optional[str] = None, skip_tasks: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load task prompts. The ground_truth_solution field is deliberately NOT copied
    into the task dicts -- it never enters the solver's path in this process."""
    tasks: List[Dict[str, Any]] = []
    all_paths = sorted(tasks_dir.glob("task_*.json"))
    indexed_paths = list(enumerate(all_paths, 1))

    if task_range:
        parts = task_range.replace(":", "-").split("-")
        if len(parts) == 2:
            start_idx = max(1, int(parts[0]))
            end_idx = int(parts[1])
            indexed_paths = [ip for ip in indexed_paths if start_idx <= ip[0] <= end_idx]

    skip_set = set()
    skip_substrings = []
    if skip_tasks:
        for item in skip_tasks.split(","):
            item = item.strip()
            if not item:
                continue
            if item.isdigit():
                skip_set.add(int(item))
            elif "-" in item and all(p.strip().isdigit() for p in item.split("-", 1)):
                p1, p2 = item.split("-", 1)
                for i in range(int(p1), int(p2) + 1):
                    skip_set.add(i)
            else:
                skip_substrings.append(item)

    for orig_idx, path in indexed_paths:
        if orig_idx in skip_set:
            continue
        if any(sub in path.stem for sub in skip_substrings):
            continue
        if task_id:
            wanted = [w.strip() for w in task_id.split(",") if w.strip()]
            if not any(w == path.stem or w in path.stem for w in wanted):
                continue
        data = json.loads(path.read_text(encoding="utf-8"))
        prompt = data.get("prompt", "")
        if not prompt or not prompt.strip():
            continue
        tasks.append({
            "task_id": path.stem,
            "arxiv_id": data.get("arxiv_id"),
            "statement_id": data.get("statement_id"),
            "prompt": prompt,
            "index": orig_idx,
        })
    if limit:
        tasks = tasks[:limit]
    return tasks


def solve_one(task: Dict[str, Any], args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    out_path = run_dir / "tasks" / f"{task['task_id']}.json"
    ckpt_path = run_dir / "checkpoints" / f"{task['task_id']}.ckpt.json"
    if args.force and ckpt_path.exists():
        try:
            ckpt_path.unlink()
        except Exception:
            pass
    if out_path.exists() and not args.force:
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            if prev.get("candidate_proof"):
                prev_copy = dict(prev)
                prev_copy["skipped"] = True
                return prev_copy
        except Exception:
            pass

    task_log = run_dir / "logs" / f"{task['task_id']}.log"
    attempts = args.retries + 1
    for attempt in range(1, attempts + 1):
        client = TunnelThinkingClient()
        try:
            # TCS protocol: REPL web tools stay disabled unless this is an explicit
            # ablation run (--enable-research).
            kernel = (PersistentMathKernel() if args.enable_research
                      else TCSCompliantKernel())
            memory = CoALAMemoryManager()
            solver = MathProver(
                client=client,
                kernel=kernel,
                memory=memory,
                default_max_iterations=args.max_iterations,
                log_path=task_log,
                enable_thinking=getattr(args, "enable_thinking", True),
                enable_literature_search=getattr(args, "enable_research", False),
            )
            def on_step(event: str, data: Any) -> None:
                tid = task["task_id"]
                if event == "checkpoint_resume":
                    log(
                        f"[{tid}] [bold yellow]>>> Resumed from checkpoint at stage='{data.get('stage')}', "
                        f"turn={data.get('completed_turn')}/{args.max_iterations}[/bold yellow]"
                    )
                elif event == "turn_start":
                    log(f"[{tid}] [bold cyan]>>> Turn {data}/{args.max_iterations} started[/bold cyan]")
                elif event == "code_start":
                    first_line = data.strip().split("\n")[0] if data else ""
                    log(f"[{tid}] [yellow]REPL executing:[/yellow] {first_line[:80]}")
                elif event == "code_done":
                    if data.success:
                        out_snip = (data.stdout.strip() or data.result_repr or "")[:100]
                        log(f"[{tid}] [green]REPL success ({data.execution_time:.2f}s):[/green] {out_snip}")
                    else:
                        err_snip = (data.error or "")[:120].replace("\n", " ")
                        log(f"[{tid}] [red]REPL error ({data.execution_time:.2f}s):[/red] {err_snip}")
                elif event == "turn_done":
                    t_num = data.get("turn", "?")
                    c_len = len(data.get("content", "") or "")
                    log(f"[{tid}] [dim]Turn {t_num} finished ({c_len} chars generated)[/dim]")
                elif event == "architect_start":
                    log(f"[{tid}] [bold magenta]>>> Context Architect Subagent Analyzing Reduction...[/bold magenta]")
                elif event == "architect_done":
                    log(f"[{tid}] [green]Context Architect Blueprint derived ({len(data)} chars)[/green]")
                elif event == "falsifier_start":
                    log(f"[{tid}] [bold magenta]>>> Computational Falsifier Subagent Testing Claims in REPL...[/bold magenta]")
                elif event == "falsifier_done":
                    v_cnt = len(data.get("verified", []))
                    f_cnt = len(data.get("falsified", []))
                    log(f"[{tid}] [yellow]Falsifier finished: {v_cnt} verified, {f_cnt} alerts[/yellow]")
                elif event == "audit_start":
                    log(f"[{tid}] [magenta]Auditing proof draft ({len(data)} chars)...[/magenta]")
                elif event == "audit_result":
                    verdict = data.get("verdict", "")
                    issues = data.get("issues", "")
                    if verdict == "VALID":
                        log(f"[{tid}] [bold green]Auditor VERDICT: VALID (Proof verified!)[/bold green]")
                    else:
                        issue_snip = issues.strip().replace("\n", " ") if issues else ""
                        log(f"[{tid}] [bold yellow]Auditor VERDICT: REJECT:[/bold yellow] {issue_snip[:200]}")
                elif event == "proof_accepted":
                    log(f"[{tid}] [bold green]*** PROOF ACCEPTED & RECORDED ***[/bold green]")
                elif event == "skeptic_start":
                    log(f"[{tid}] [bold red]>>> Adversarial Skeptic Subagent Attacking Claim...[/bold red]")
                elif event == "skeptic_done":
                    v_status = "[green]CERTIFIED[/green]" if data.get("verified") else "[red]REFUTED[/red]"
                    r_snip = data.get("reason", "")[:120].replace("\n", " ")
                    log(f"[{tid}] Skeptic result: {v_status} - {r_snip}")
                elif event == "scribe_start":
                    log(f"[{tid}] [bold blue]>>> Publication Scribe Subagent Typesetting Proof...[/bold blue]")
                elif event == "scribe_done":
                    v_status = "[green]VALID[/green]" if data.get("valid") else "[yellow]NEEDS_TYPESET[/yellow]"
                    p_len = len(data.get("proof", "") or "")
                    log(f"[{tid}] Scribe result: {v_status} ({p_len} chars)")
                elif event == "synthesis_start":
                    log(f"[{tid}] [bold magenta]>>> Dedicated Publication Proof Synthesis Started[/bold magenta]")
                elif event == "synthesis_done":
                    p_len = len(data.get("proof", "") or "")
                    v_str = "VALID" if data.get("valid") else "INVALID"
                    log(f"[{tid}] [cyan]Synthesis pass finished: proof={p_len} chars ({v_str})[/cyan]")
                elif event == "revision_start":
                    log(f"[{tid}] [bold cyan]Initiating revision turn with auditor critique...[/bold cyan]")

            res = solver.solve(
                task_id=task["task_id"],
                task_prompt=task["prompt"],
                max_iterations=args.max_iterations,
                max_time_seconds=args.timeout,
                on_step=on_step,
                checkpoint_path=ckpt_path,
            )
            rec: Dict[str, Any] = {
                "task_id": task["task_id"],
                "arxiv_id": task["arxiv_id"],
                "statement_id": task["statement_id"],
                "status": "completed" if res.candidate_proof else "no_proof",
                "candidate_proof": res.candidate_proof,
                "verified": res.verified,
                "iterations": res.iterations,
                "reasoning_traces": res.reasoning_traces,
                "code_history": res.code_history,
                "retrieved_heuristics": res.retrieved_heuristics,
                "learned_heuristics": res.learned_heuristics,
                "blueprint": getattr(res, "blueprint", None),
                "verified_facts": getattr(res, "verified_facts", []),
                "falsified_alerts": getattr(res, "falsified_alerts", []),
                "attempts": attempt,
                "duration_s": res.duration_s,
                "log_file": str(task_log),
            }
            out_path.write_text(
                json.dumps(rec, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
            if getattr(args, "telegram", False):
                try:
                    from core_engine.telegram import load_telegram_credentials, send_telegram_message
                    bot_token, chat_id = load_telegram_credentials()
                    v_str = "VERIFIED" if res.verified else "UNVERIFIED"
                    p_len = len(res.candidate_proof or "")
                    tg_msg = (
                        f"<b>[TCS-Bench Prover] Task Complete</b>\n"
                        f"<b>Task:</b> <code>{task['task_id']}</code>\n"
                        f"<b>Status:</b> {rec['status']} ({v_str})\n"
                        f"<b>Proof Length:</b> {p_len} chars\n"
                        f"<b>Iterations:</b> {res.iterations}\n"
                        f"<b>Duration:</b> {res.duration_s:.1f}s"
                    )
                    send_telegram_message(bot_token, chat_id, tg_msg)
                except Exception as tg_err:
                    log(f"[yellow]Telegram notification error for {task['task_id']}: {tg_err}[/yellow]")
            return rec
        except Exception as exc:
            retriable = isinstance(exc, (httpx.HTTPError, OSError))
            if not retriable or attempt == attempts:
                rec = {
                    "task_id": task["task_id"],
                    "arxiv_id": task["arxiv_id"],
                    "statement_id": task["statement_id"],
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "candidate_proof": None,
                    "attempts": attempt,
                    "duration_s": None,
                    "log_file": str(task_log),
                }
                out_path.write_text(
                    json.dumps(rec, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
                return rec
            log(f"[yellow]retry {task['task_id']} "
                f"({attempt}/{attempts - 1}): {type(exc).__name__}: {exc}[/yellow]")
            time.sleep(30 * attempt)
    raise RuntimeError("unreachable")  # pragma: no cover


def main() -> None:
    tcs_cfg = load_tcs_config()
    parser = argparse.ArgumentParser(
        description="TCS-Bench Stage 1: solve all tasks (MathProver, Stage 0 OFF)")
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    parser.add_argument("--task-id", type=str, default=None,
                        help="Solve a single specific task ID (e.g. task_1907.04585_statement_93_142)")
    parser.add_argument("--task-range", type=str, default=None,
                        help="Solve a 1-indexed range of tasks (e.g. 4-10)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=tcs_cfg.prover_max_iterations,
                        help=f"Maximum prover turns per task (default: {tcs_cfg.prover_max_iterations} from tcs_config)")
    parser.add_argument("--telegram", action="store_true", default=False,
                        help="Send live problem completion alerts directly to Telegram")
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="per-task time budget in seconds (default 3600)")
    parser.add_argument("--retries", type=int, default=2,
                        help="extra attempts per task on transient network errors (default 2)")
    parser.add_argument("--limit", type=int, default=None,
                        help="solve only the first N tasks (smoke test)")
    parser.add_argument("--force", action="store_true",
                        help="re-solve tasks that already have an output file")
    parser.add_argument("--prove-dir", "--solve-dir", dest="prove_dir", type=Path, default=None,
                        help="Resume or populate an existing prove directory (e.g. data/eval_results/tcs_prove_<ts>)")
    parser.add_argument("--seed-from", type=Path, default=None,
                        help="Path to previous prove run directory to seed already-solved tasks from")
    parser.add_argument("--enable-research", action="store_true",
                        help="OPT-IN: enable ALL web research (Stage 0 pre-solve AND the "
                             "REPL web tools, by using the plain kernel instead of "
                             "TCSCompliantKernel). Violates the TCS-Bench no-internet "
                             "rule; ablation only. Default: web access fully disabled.")
    parser.add_argument("--skip-tasks", type=str, default=None,
                        help="Comma-separated 1-based task indices or task ID substrings to skip (e.g. 1,5,10,18,19)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable internal thinking tokens on Qwen (default: thinking enabled).")
    args = parser.parse_args()
    args.enable_thinking = not getattr(args, "no_thinking", False)

    if not args.tasks_dir.is_dir():
        sys.exit(f"tasks dir not found: {args.tasks_dir}\n"
                 f"  clone it with: git clone --depth 1 https://github.com/TCSBench/TCSBench")

    tasks = load_tasks(args.tasks_dir, args.limit, args.task_id, args.task_range, args.skip_tasks)
    if not tasks:
        sys.exit("no task_*.json files found")

    # Ensure background solvers yield CPU priority to interactive SSH sessions
    try:
        os.nice(10)
    except Exception:
        pass

    if args.prove_dir:
        run_dir = args.prove_dir
        (run_dir / "tasks").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = EVAL_DIR / f"tcs_prove_{stamp}"
        (run_dir / "tasks").mkdir(parents=True)
        (run_dir / "logs").mkdir(parents=True)
    if args.seed_from and (args.seed_from / "tasks").is_dir():
        import shutil
        for p in (args.seed_from / "tasks").glob("*.json"):
            dst = run_dir / "tasks" / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
        if (args.seed_from / "logs").is_dir():
            for p in (args.seed_from / "logs").glob("*.log"):
                dst = run_dir / "logs" / p.name
                if not dst.exists():
                    shutil.copy2(p, dst)
        log(f"[green]Seeded existing task outputs from {args.seed_from}[/green]")
    (run_dir / "manifest.json").write_text(json.dumps({
        "stage": "tcs_bench_prove",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "tasks_dir": str(args.tasks_dir),
        "args": vars(args),
        "stage0_research": ("ON (protocol violation -- ablation only)"
                            if args.enable_research else
                            "OFF (TCS-Bench no-internet rule)"),
        "repl_web_tools": ("ON (plain kernel, ablation only)"
                           if args.enable_research else
                           "OFF (TCSCompliantKernel: stubs + network-import AST block)"),
        "tasks": [t["task_id"] for t in tasks],
    }, indent=2, default=str), encoding="utf-8")

    # Fail fast if the shared memory backend is down (solve() would raise per task)
    try:
        CoALAMemoryManager().check_health()
        log("[green]TurboVec memory health: OK[/green]")
    except Exception as exc:
        sys.exit(f"TurboVec memory unavailable ({exc}) -- start TurboVec (port 18792) and retry.")

    console.print(f"[bold blue]=== TCS-Bench Stage 1: {len(tasks)} task(s), "
                  f"{min(args.workers, len(tasks))} worker(s) ===[/bold blue]")
    console.print(f"[bold]Run dir:[/bold] {run_dir}")
    res_str = (
        "[red]ON -- PROTOCOL VIOLATION (ablation only)[/red]"
        if args.enable_research else
        "[green]OFF (no web lookups, per TCS-Bench rule)[/green]"
    )
    console.print(f"[bold]Stage 0 research:[/bold] {res_str}")
    repl_str = (
        "[red]ON -- plain kernel (ablation only)[/red]"
        if args.enable_research else
        "[green]OFF (TCSCompliantKernel: stubs + import block)[/green]"
    )
    console.print(f"[bold]REPL web tools:[/bold] {repl_str}")
    console.print("[dim]Anti-cheat: GT field stripped from solver input; web tools "
                  "disabled; harness scripts unmodified.[/dim]")

    t0 = time.perf_counter()
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
        futures = {ex.submit(solve_one, t, args, run_dir): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # worker crash safety net
                rec = {"task_id": task["task_id"], "arxiv_id": task["arxiv_id"],
                       "status": "error", "error": f"worker crash: {exc}"}
            results.append(rec)
            if rec.get("skipped"):
                n_proof = len(rec.get("candidate_proof") or "")
                log(f"[yellow]skip  {rec['task_id']} (already solved: proof={n_proof} chars)[/yellow]")
            elif rec.get("status") == "completed":
                n_proof = len(rec.get("candidate_proof") or "")
                log(f"[green]done  {rec['task_id']}[/green]  "
                    f"iters={rec.get('iterations')}  {rec.get('duration_s')}s  "
                    f"proof={n_proof} chars")
            else:
                log(f"[red]fail  {rec['task_id']}: {rec.get('error', '')[:140]}[/red]")

    elapsed = round(time.perf_counter() - t0, 2)
    completed = [r for r in results if r["status"] == "completed"]
    with_proof = [r for r in completed if r.get("candidate_proof")]
    errors = [r for r in results if r["status"] == "error"]
    summary = {
        "run_dir": str(run_dir),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": elapsed,
        "total": len(tasks),
        "completed": len(completed),
        "with_proof": len(with_proof),
        "errors": len(errors),
        "error_tasks": [r["task_id"] for r in errors],
        "error_details": {r["task_id"]: r.get("error") for r in errors},
        "results": results,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    table = Table(title="TCS-Bench Stage 1 summary")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("completed", str(len(completed)))
    table.add_row("with non-empty proof", str(len(with_proof)))
    table.add_row("errors", str(len(errors)))
    table.add_row("elapsed (s)", str(elapsed))
    console.print(table)
    if errors:
        console.print(f"[yellow]Error tasks: {', '.join(r['task_id'] for r in errors)}[/yellow]")
    console.print(f"\n[bold]Next stage:[/bold] "
                 f".venv/bin/python scripts/run_tcs_bench_judge.py --solve-dir {run_dir}")
    console.print(f"[bold]Leakage audit:[/bold] "
                 f".venv/bin/python scripts/tcs_bench_leakage_audit.py post --solve-dir {run_dir}")


if __name__ == "__main__":
    main()
