#!/usr/bin/env python3
"""Autonomous Continuous TCS-Bench Pipeline (Solve + Immediate Judge + Live Scoreboard).

Pipeline Lifecycle per Task:
1. Loads task prompt (airgapped, ground truth stripped).
2. Solves problem using MathProver (up to 8 turns, persistent REPL, CoALA memory).
3. Saves solver record to <solve_dir>/tasks/<task_id>.json.
4. Immediately invokes the judge harness (core_engine.judging.judge_tcs_proof):
   - Config B (headline): k independent referee replicates (default 3); pass iff the MEDIAN
     validity >= TCS_PASS_THRESHOLD (core_engine/tcs_config.py, env TCS_PIPELINE_JUDGE_THRESHOLD).
   - Config A (optional, arXiv:2608.09538 consensus: 4 calls, >=3/4 votes) -> secondary metrics only.
5. Saves judgment record to <solve_dir>/judgments/judge_<task_id>.json.
6. Atomically updates running metrics and displays the live scorecard.

Scoring rules:
- A judge reply whose validity cannot be parsed in any replicate (status "judge_parse_error") and
  judge transport failures (status "error") are NOT fails: they are excluded from every pass-rate
  denominator and reported as separate counts. A missing validity is never converted to 0.0.
- Lean 4 compilation is reported as a separate EXPERIMENTAL metric only; it is not a soundness
  certificate (axioms are not blocked and the formal statement is not checked against the task).

Usage:
  python3 scripts/run_tcs_continuous_pipeline.py --limit 50 --workers 2
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core_engine.tcs_config import TCS_JUDGE_REPLICATES, TCS_PASS_THRESHOLD, load_tcs_config

from run_tcs_bench_judge import judge_with_retry, load_task_map

_LOCAL_TCS_TASKS = PROJECT_ROOT / "data" / "tcs" / "tasks"
_EXTERNAL_TCS_TASKS = PROJECT_ROOT.parent / "TCSBench" / "tasks"
DEFAULT_TASKS_DIR = _LOCAL_TCS_TASKS if _LOCAL_TCS_TASKS.is_dir() else _EXTERNAL_TCS_TASKS
DEFAULT_EVAL_DIR = PROJECT_ROOT / "data" / "eval_results"

# Secondary "directionally sound" bar on TCS Judge median validity (not a pass).
DIRECTIONAL_THRESHOLD = 0.40

# Judge statuses that carry no verdict about the proof; excluded from pass-rate denominators.
UNSCORED_JUDGE_STATUSES = ("judge_parse_error", "error")

console = Console()
print_lock = threading.Lock()
metrics_lock = threading.Lock()


def log(msg: str) -> None:
    with print_lock:
        console.print(msg)


def headline_label(threshold: float) -> str:
    return f"TCS Judge pass (median validity >= {threshold:.2f})"


def directional_label() -> str:
    return f"Directionally sound (TCS Judge median validity >= {DIRECTIONAL_THRESHOLD:.2f})"


def build_tracker_entry(task_idx: int, task_id: str, solve_rec: Dict[str, Any],
                        judge_rec: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """Builds one live-tracker row from a solve record and a judge record (pure function)."""
    a_rec = judge_rec.get("a") or {}
    b_rec = judge_rec.get("b") or {}
    judge_status = judge_rec.get("status", "unknown")

    a_verdict = a_rec.get("verdict", "0") if a_rec else "0"
    pass_a = (a_verdict == "1")

    # TCS Judge: keep None as None. A missing validity is a parse failure, never a 0.0 score.
    b_val = b_rec.get("validity") if b_rec else None
    judge_parse_error = judge_status == "judge_parse_error" or (
        bool(b_rec) and b_val is None and judge_status == "judged"
    )
    if judge_parse_error:
        judge_status = "judge_parse_error"
    pass_b = b_val is not None and b_val >= threshold
    scored = judge_status not in UNSCORED_JUDGE_STATUSES

    lean_rec = judge_rec.get("lean")
    pass_lean = bool(lean_rec.get("passed", False)) if lean_rec else False
    lean_verdict = lean_rec.get("verdict", "0") if lean_rec else "0"

    return {
        "task_index": task_idx,
        "task_id": task_id,
        "status": solve_rec.get("status", "unknown"),
        "judge_status": judge_status,
        "scored": scored,
        "judge_parse_error": judge_parse_error,
        "duration_s": solve_rec.get("duration_s", 0.0),
        "iterations": solve_rec.get("iterations", 0),
        "candidate_chars": len(solve_rec.get("candidate_proof") or ""),
        "judge_a_verdict": a_verdict,
        "judge_a_ones": a_rec.get("ones", 0) if a_rec else 0,
        "judge_b_validity": b_val,
        "judge_b_replicates": b_rec.get("replicate_validities") if b_rec else None,
        "judge_b_pass_probability": b_rec.get("pass_probability") if b_rec else None,
        "judge_b_spread": b_rec.get("validity_spread") if b_rec else None,
        "judge_issues": (b_rec.get("scores") or {}).get("issues") if b_rec else None,
        "pass_a": pass_a,
        "pass_b": pass_b,
        # Headline = TCS Judge only (no OR-aggregation). Key kept for backward compatibility.
        "pass_proven": pass_b,
        "pass_lean_experimental": pass_lean,
        "pass_lean": pass_lean,
        "lean_verdict": lean_verdict,
        # Secondary metrics (A and B only; Lean is never folded in).
        "pass_either": pass_a or pass_b,
        "pass_both": pass_a and pass_b,
        "pass_directional": b_val is not None and b_val >= DIRECTIONAL_THRESHOLD,
        "threshold": threshold,
        "timestamp": datetime.now().isoformat(),
    }


def _is_scored(r: Dict[str, Any]) -> bool:
    if "scored" in r:
        return bool(r["scored"])
    return r.get("judge_status") not in UNSCORED_JUDGE_STATUSES  # legacy rows: scored


def summarize_records(records: List[Dict[str, Any]], total_target: int, threshold: float) -> Dict[str, Any]:
    """Aggregates tracker rows into the scoreboard (pure function).

    Denominator for every pass rate = rows with a real verdict (judge_parse_error / judge errors
    excluded and reported separately).
    """
    checked = len(records)
    scored_rows = [r for r in records if _is_scored(r)]
    scored = len(scored_rows)
    n_parse = sum(1 for r in records if r.get("judge_status") == "judge_parse_error" or r.get("judge_parse_error"))
    n_err = sum(1 for r in records if r.get("judge_status") == "error")

    def block(key: str, rule: str, **extra: Any) -> Dict[str, Any]:
        passes = sum(1 for r in scored_rows if r.get(key, False))
        d = {
            "passes": passes,
            "scored": scored,
            "pass_percentage": round(passes / scored * 100.0, 2) if scored else 0.0,
            "rule": rule,
        }
        d.update(extra)
        return d

    tcs_judge = block("pass_b", f"TCS Judge median validity >= {threshold:.2f}")
    pps = [r["judge_b_pass_probability"] for r in scored_rows if r.get("judge_b_pass_probability") is not None]
    return {
        "target_total_problems": total_target,
        "total_checked": checked,
        "total_scored": scored,
        "judge_parse_errors": n_parse,
        "judge_errors": n_err,
        "threshold": threshold,
        "progress_percent": round(checked / total_target * 100.0, 1) if total_target else 0.0,
        "headline_metric": "tcs_judge",
        "headline_label": headline_label(threshold),
        "tcs_judge": tcs_judge,
        "judge_b": tcs_judge,  # backward compatibility
        # Mean per-task P(pass) across replicates: a smoother estimate of the expected pass rate.
        "tcs_judge_expected_pass_rate": round(sum(pps) / len(pps) * 100.0, 2) if pps else None,
        "judge_b_expected_pass_rate": round(sum(pps) / len(pps) * 100.0, 2) if pps else None,
        # Backward-compatible alias of the headline
        "proven": dict(tcs_judge, rule=f"alias of tcs_judge: TCS Judge median validity >= {threshold:.2f}"),
        "directionally_sound": block("pass_directional", directional_label()),
        "combined_either": block("pass_either", "TCS Judge pass OR consensus pass (secondary)"),
        "strict_consensus_both": block("pass_both", "TCS Judge pass AND consensus pass (secondary)"),
        "lean": block(
            "pass_lean", "EXPERIMENTAL: Lean 4 compiles with no sorry; NOT a soundness certificate "
                         "(axioms not blocked, statement not checked)", experimental=True),
        "recent_tasks": records[-10:],
        "updated_at": datetime.now().isoformat(),
    }


class PipelineScorecard:
    """Thread-safe atomic metrics tracker and live scorecard for TCS-Bench."""

    def __init__(self, total_target: int, solve_dir: Path, threshold: Optional[float] = None):
        self.total_target = total_target
        self.solve_dir = solve_dir
        self.threshold = TCS_PASS_THRESHOLD if threshold is None else threshold
        self.metrics_file = solve_dir / "pipeline_metrics.json"
        self.tracker_file = solve_dir / "live_tracker.jsonl"
        self.records: List[Dict[str, Any]] = []

        # Preload existing records from live_tracker if present to support resuming/range execution
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, "r", encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if line:
                            self.records.append(json.loads(line))
                self.records.sort(key=lambda r: r.get("task_index", 0))
            except Exception as e:
                log(f"[yellow]Warning: Could not preload existing tracker records: {e}[/yellow]")


    def record_judgment(self, task_idx: int, task_id: str, solve_rec: Dict[str, Any], judge_rec: Dict[str, Any]) -> Dict[str, Any]:
        with metrics_lock:
            entry = build_tracker_entry(task_idx, task_id, solve_rec, judge_rec, self.threshold)

            # Update or append
            existing_idx = next((i for i, r in enumerate(self.records) if r["task_id"] == task_id), None)
            if existing_idx is not None:
                self.records[existing_idx] = entry
            else:
                self.records.append(entry)

            # Sort records by task index
            self.records.sort(key=lambda r: r["task_index"])

            summary = summarize_records(self.records, self.total_target, self.threshold)

            # Atomic save to disk
            tmp_metrics = self.metrics_file.with_suffix(".tmp")
            tmp_metrics.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            tmp_metrics.replace(self.metrics_file)

            tmp_tracker = self.tracker_file.with_suffix(".tmp")
            with open(tmp_tracker, "w", encoding="utf-8") as fp:
                for r in self.records:
                    fp.write(json.dumps(r) + "\n")
            tmp_tracker.replace(self.tracker_file)

            return summary


def display_task_update(summary: Dict[str, Any], task_idx: int, task_id: str, solve_rec: Dict[str, Any], judge_rec: Dict[str, Any]) -> None:
    checked = summary["total_checked"]
    scored = summary.get("total_scored", checked)
    target = summary["target_total_problems"]
    pct_checked = summary["progress_percent"]
    threshold = summary.get("threshold", TCS_PASS_THRESHOLD)

    b_rec = judge_rec.get("b") or {}
    b_val = b_rec.get("validity") if b_rec else None
    if judge_rec.get("status") == "judge_parse_error" or (b_rec and b_val is None):
        b_str = "[bold magenta]PARSE ERROR (excluded from pass rate)[/bold magenta]"
        overall = "[bold magenta]UNSCORED[/bold magenta]"
    elif b_val is None:
        b_str = f"[dim]n/a ({judge_rec.get('status', '?')})[/dim]"
        overall = "[bold red]FAIL[/bold red]" if judge_rec.get("status") != "error" else "[bold magenta]UNSCORED[/bold magenta]"
    else:
        b_pass = b_val >= threshold
        reps = b_rec.get("replicate_validities")
        pp = b_rec.get("pass_probability")
        b_str = (f"[bold cyan]{b_val:.2f} ({'PASS' if b_pass else 'FAIL'})[/bold cyan]"
                 + (f" reps={reps} P(pass)={pp}" if reps else ""))
        overall = "[bold green]PASS[/bold green]" if b_pass else "[bold red]FAIL[/bold red]"

    lean_rec = judge_rec.get("lean")
    p_len = len(solve_rec.get("candidate_proof") or "")
    dur = solve_rec.get("duration_s") or 0.0

    log("\n" + "=" * 80)
    log(f"[bold cyan]>>> COMPLETED TASK {task_idx}/{target}:[/bold cyan] [white]{task_id}[/white]")
    log(f"    Solve Time: [yellow]{dur:.1f}s[/yellow] | Iterations: [yellow]{solve_rec.get('iterations', '?')}[/yellow] | Candidate Proof: [yellow]{p_len} chars[/yellow]")
    if lean_rec:
        lean_status = "PASS" if lean_rec.get("passed", False) else "FAIL"
        log(f"    Lean 4 (EXPERIMENTAL, not a soundness certificate): {lean_status} (Status: {lean_rec.get('status', '?')})")
    log(f"    TCS Judge (median validity): {b_str} | Headline: {overall}")
    log("-" * 80)
    log(f"[bold white]LIVE METRICS AFTER {checked}/{target} PROBLEMS ({pct_checked}% Checked, {scored} scored):[/bold white]")
    tj = summary.get("tcs_judge") or summary["judge_b"]
    log(f"  • [cyan]{headline_label(threshold)}[/cyan]:  [bold cyan]{tj['passes']} / {scored} ({tj['pass_percentage']}%) [/bold cyan]")
    ds = summary["directionally_sound"]
    log(f"  • [yellow]{directional_label()}[/yellow]: [bold yellow]{ds['passes']} / {scored} ({ds['pass_percentage']}%) [/bold yellow]")
    if summary.get("judge_parse_errors") or summary.get("judge_errors"):
        log(f"  • [magenta]Unscored (excluded): {summary.get('judge_parse_errors', 0)} judge parse errors, "
            f"{summary.get('judge_errors', 0)} judge errors[/magenta]")
    if summary.get("lean") and summary["lean"]["passes"] > 0:
        log(f"  • [dim]Lean 4 compile (EXPERIMENTAL): {summary['lean']['passes']} / {scored} ({summary['lean']['pass_percentage']}%)[/dim]")
    log("=" * 80 + "\n")


def process_single_task(
    orig_idx: int,
    task_path: Path,
    args: argparse.Namespace,
    run_dir: Path,
    task_map: Dict[str, Dict[str, Any]],
    scorecard: PipelineScorecard,
) -> None:
    task_id = task_path.stem
    task_data = task_map.get(task_id, {})
    prompt = task_data.get("prompt", "")

    # Edge case: Empty benchmark prompt (e.g. task 10)
    if not prompt or not prompt.strip():
        log(f"[yellow]Skipping Task {orig_idx} ({task_id}): Empty benchmark prompt in dataset.[/yellow]")
        empty_solve = {
            "task_id": task_id,
            "status": "skipped_empty_prompt",
            "candidate_proof": None,
            "duration_s": 0.0,
            "iterations": 0,
        }
        empty_judge = {
            "task_id": task_id,
            "status": "skipped_empty_prompt",
            "a": {"ones": 0, "verdict": "0"},
            "b": {"validity": 0.0, "verdict": "0"},
        }
        summary = scorecard.record_judgment(orig_idx, task_id, empty_solve, empty_judge)
        display_task_update(summary, orig_idx, task_id, empty_solve, empty_judge)
        return

    solve_out = run_dir / "tasks" / f"{task_id}.json"
    judge_out = run_dir / "judgments" / f"judge_{task_id}.json"
    task_log = run_dir / "logs" / f"{task_id}.log"
    ckpt_path = run_dir / "checkpoints" / f"{task_id}.ckpt.json"

    if args.force:
        for stale_file in (ckpt_path, task_log):
            if stale_file.exists():
                try:
                    stale_file.unlink()
                except Exception:
                    pass

    # Step 1: Check if already judged and valid
    if judge_out.exists() and solve_out.exists() and not args.force:
        try:
            prev_solve = json.loads(solve_out.read_text(encoding="utf-8"))
            prev_judge = json.loads(judge_out.read_text(encoding="utf-8"))
            summary = scorecard.record_judgment(orig_idx, task_id, prev_solve, prev_judge)
            log(f"[dim]Loaded existing judgment for Task {orig_idx} ({task_id})[/dim]")
            display_task_update(summary, orig_idx, task_id, prev_solve, prev_judge)
            return
        except Exception as e:
            log(f"[yellow]Failed to load existing record for Task {orig_idx} ({task_id}): {e}[/yellow]")

    # Step 2: Seed from existing portfolio if explicitly provided and not no-seed
    if not solve_out.exists() and args.seed_from and not getattr(args, "no_seed", False) and (args.seed_from / "tasks" / f"{task_id}.json").exists() and not args.force:
        seed_path = args.seed_from / "tasks" / f"{task_id}.json"
        try:
            seed_data = json.loads(seed_path.read_text(encoding="utf-8"))
            if seed_data.get("candidate_proof"):
                solve_out.write_text(json.dumps(seed_data, indent=2), encoding="utf-8")
                log(f"[cyan]Seeded candidate proof for Task {orig_idx} ({task_id}) from calibration best[/cyan]")
        except Exception:
            pass

    # Step 3: Solve task if not solved
    solve_rec = None
    if solve_out.exists() and not args.force:
        try:
            solve_rec = json.loads(solve_out.read_text(encoding="utf-8"))
        except Exception:
            solve_rec = None

    if solve_rec is None or not solve_rec.get("candidate_proof"):
        log(f"[{task_id}] [bold cyan]Starting Solver (Turns 1..{args.max_iterations})...[/bold cyan]")
        # Heavy solver stack imported lazily so the scoring helpers stay importable in isolation.
        from core_engine.llm_client import TunnelThinkingClient
        from core_engine.memory import CoALAMemoryManager
        from core_engine.prover import MathProver
        from core_engine.prover_kernel import TCSCompliantKernel

        client = TunnelThinkingClient()
        kernel = TCSCompliantKernel()
        memory = CoALAMemoryManager()

        solver = MathProver(
            client=client,
            kernel=kernel,
            memory=memory,
            default_max_iterations=args.max_iterations,
            log_path=task_log,
            enable_thinking=not args.no_thinking,
            enable_literature_search=False,
        )

        def on_step(event: str, data: Any) -> None:
            if event == "checkpoint_resume":
                log(
                    f"[{task_id}] [bold yellow]>>> Resumed from checkpoint at stage='{data.get('stage')}', "
                    f"turn={data.get('completed_turn')}/{args.max_iterations}[/bold yellow]"
                )
            elif event == "turn_start":
                log(f"[{task_id}] Turn {data}/{args.max_iterations} started")
            elif event == "audit_result":
                verdict = data.get("verdict", "")
                if verdict == "VALID":
                    log(f"[{task_id}] [bold green]Auditor: VALID[/bold green]")
            elif event == "proof_accepted":
                log(f"[{task_id}] [bold green]*** Proof Accepted & Recorded ***[/bold green]")

        res = None
        last_solve_err: Optional[Exception] = None
        for solve_attempt in range(1, 4):
            try:
                res = solver.solve(
                    task_id=task_id,
                    task_prompt=prompt,
                    max_iterations=args.max_iterations,
                    max_time_seconds=args.timeout,
                    on_step=on_step,
                    checkpoint_path=ckpt_path,
                )
                break
            except Exception as exc:
                last_solve_err = exc
                backoff_s = 20.0 * solve_attempt
                log(
                    f"[yellow][{task_id}] Solver attempt {solve_attempt}/3 encountered transport/runtime error: "
                    f"{exc}. Retrying in {backoff_s:.0f}s...[/yellow]"
                )
                if solve_attempt < 3:
                    time.sleep(backoff_s)
        if res is None:
            raise RuntimeError(f"Solver failed after 3 task-level attempts: {last_solve_err}") from last_solve_err

        solve_rec = {
            "task_id": task_id,
            "arxiv_id": task_data.get("arxiv_id"),
            "statement_id": task_data.get("statement_id"),
            "status": "completed" if res.candidate_proof else "no_proof",
            "candidate_proof": res.candidate_proof,
            "verified": res.verified,
            "iterations": res.iterations,
            "reasoning_traces": res.reasoning_traces,
            "code_history": res.code_history,
            "retrieved_heuristics": res.retrieved_heuristics,
            "learned_heuristics": res.learned_heuristics,
            "blueprint": getattr(res, "blueprint", None),
            "duration_s": res.duration_s,
            "log_file": str(task_log),
        }
        solve_out.write_text(json.dumps(solve_rec, indent=2, ensure_ascii=False), encoding="utf-8")

    # Step 4: Immediately Judge
    log(f"[{task_id}] [bold magenta]Solving finished. Invoking Judge Harness immediately...[/bold magenta]")
    from core_engine.llm_client import TunnelThinkingClient
    judge_client = TunnelThinkingClient()
    judge_args = argparse.Namespace(
        config=args.config,
        threshold=args.judge_threshold,
        temp_a=0.3,
        temp_b=0.1,
        max_tokens=16384,
        replicates=getattr(args, "judge_replicates", TCS_JUDGE_REPLICATES),
        adaptive=getattr(args, "judge_adaptive", False),
    )

    judge_rec = judge_with_retry(
        rec=solve_rec,
        task_data=task_data,
        args=judge_args,
        client=judge_client,
        thinking=args.judge_thinking,
        timeout_seconds=args.judge_timeout,
        retries=2,
    )
    judge_out.write_text(json.dumps(judge_rec, indent=2, ensure_ascii=False), encoding="utf-8")

    # Step 5: Update atomic scorecard and display
    summary = scorecard.record_judgment(orig_idx, task_id, solve_rec, judge_rec)
    display_task_update(summary, orig_idx, task_id, solve_rec, judge_rec)

    # Step 5.5: Automatic Procedural Memory Hygiene (prune unconfirmed heuristics if Official TCS Judge rejects)
    try:
        b_val = (judge_rec.get("b") or {}).get("validity")
        threshold = summary.get("threshold", TCS_PASS_THRESHOLD)
        if (b_val is None or b_val < threshold) and solve_rec.get("learned_heuristics"):
            from core_engine.memory import CoALAMemoryManager
            pruned_n = CoALAMemoryManager().remove_by_task_id("prime_math_procedural", task_id)
            if pruned_n:
                log(f"[yellow][{task_id}] Pruned {pruned_n} unconfirmed heuristic(s) after Official TCS Judge rejection.[/yellow]")
    except Exception as e:
        log(f"[yellow]Procedural memory hygiene check warning for {task_id}: {e}[/yellow]")

    # Step 6: Live Telegram notification
    if getattr(args, "telegram", False):
        try:
            from core_engine.telegram import load_telegram_credentials, format_task_report, send_telegram_message
            bot_token, chat_id = load_telegram_credentials()
            tracker_entry = next((r for r in scorecard.records if r["task_id"] == task_id), None)
            if tracker_entry:
                msg = format_task_report(tracker_entry, summary)
                send_telegram_message(bot_token, chat_id, msg)
        except Exception as e:
            log(f"[yellow]Telegram notification error for {task_id}: {e}[/yellow]")


def main() -> None:
    tcs_cfg = load_tcs_config()

    parser = argparse.ArgumentParser(
        description="Continuous TCS-Bench Pipeline: Solve + Immediate Judge + Live Scoreboard")
    parser.add_argument("--limit", type=int, default=50,
                        help="Number of problems to evaluate from start of dataset (default: 50)")
    parser.add_argument("--task-range", type=str, default=None,
                        help="1-indexed range of tasks to run (e.g. 1-50)")
    parser.add_argument("--workers", type=int, default=tcs_cfg.pipeline_workers,
                        help=f"Number of concurrent pipeline workers (default: {tcs_cfg.pipeline_workers} from .env)")
    parser.add_argument("--max-iterations", type=int, default=tcs_cfg.prover_max_iterations,
                        help=f"Maximum prover turns per task (default: {tcs_cfg.prover_max_iterations} from .env)")
    parser.add_argument("--timeout", type=float, default=tcs_cfg.solver_timeout_seconds,
                        help=f"Per-task solve timeout budget in seconds (default: {tcs_cfg.solver_timeout_seconds})")
    parser.add_argument("--judge-timeout", type=float, default=120,
                        help="Per-judge-call timeout in seconds (default: 120)")
    parser.add_argument("--prove-dir", "--solve-dir", dest="prove_dir", type=Path, default=DEFAULT_EVAL_DIR / "tcs_pipeline_50",
                        help="Directory to save tasks, logs, and judgments")
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR,
                        help="Path to TCSBench tasks directory")
    parser.add_argument("--seed-from", type=Path, default=None,
                        help="Optional: Seed candidate proofs from a previous run dir (default: None for clean slate)")
    parser.add_argument("--no-seed", action="store_true", default=False,
                        help="Disable all seeding (clean slate)")
    parser.add_argument("--config", choices=["b", "tcs", "a", "ab", "lean"], default=tcs_cfg.pipeline_judge_config,
                        help=f"Judge configuration (primary referee 'b'/'tcs', dual 'ab', legacy 'a', or 'lean'; default: {tcs_cfg.pipeline_judge_config} from .env)")
    parser.add_argument("--judge-threshold", type=float, default=tcs_cfg.pipeline_judge_threshold,
                        help=f"Config B passing validity threshold (default: {tcs_cfg.pipeline_judge_threshold} from .env)")
    parser.add_argument("--judge-replicates", type=int, default=TCS_JUDGE_REPLICATES,
                        help=f"Independent Judge-B replicates per task; median is the reported validity "
                             f"(default: {TCS_JUDGE_REPLICATES})")
    parser.add_argument("--judge-adaptive", action="store_true", default=tcs_cfg.pipeline_judge_adaptive,
                        help=f"Adaptive replicates: stop after 1 judge call if it is outside threshold +/- 0.20 (default: {tcs_cfg.pipeline_judge_adaptive})")
    parser.add_argument("--no-judge-adaptive", action="store_false", dest="judge_adaptive",
                        help="Disable adaptive replicates and always run all replicates")
    parser.add_argument("--judge-thinking", action="store_true", default=tcs_cfg.pipeline_judge_thinking,
                        help=f"Enable thinking tokens on judge calls (default: {tcs_cfg.pipeline_judge_thinking} from .env)")

    parser.add_argument("--no-judge-thinking", action="store_false", dest="judge_thinking",
                        help="Disable thinking tokens on judge calls")
    parser.add_argument("--tcs-judge", "--tcs_judge", dest="tcs_judge", action="store_true", default=False,
                        help="Route proof evaluation to the unified Thinking TCS Judge")
    parser.add_argument("--telegram", action="store_true", default=False,
                        help="Send live problem updates and final summary directly to Telegram")
    parser.add_argument("--no-thinking", action="store_true", default=False,
                        help="Disable thinking tokens on solver Qwen (default: False)")
    parser.add_argument("--task-ids", type=str, default=None,
                        help="Comma-separated list of specific task indices (e.g. 6,18,50,51) or task_ids to run")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Force re-solving and re-judging existing tasks")

    args = parser.parse_args()
    if args.tcs_judge:
        args.config = "tcs"
        args.judge_thinking = True
    args.solve_dir = args.prove_dir

    if not args.tasks_dir.is_dir():
        sys.exit(f"Tasks directory not found: {args.tasks_dir}")

    # Create run directories
    args.solve_dir.mkdir(parents=True, exist_ok=True)
    (args.solve_dir / "tasks").mkdir(parents=True, exist_ok=True)
    (args.solve_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.solve_dir / "judgments").mkdir(parents=True, exist_ok=True)

    def _has_prompt(p: Path) -> bool:
        try:
            with open(p, "r", encoding="utf-8") as fp:
                d = json.load(fp)
                return bool(d.get("prompt", "").strip())
        except Exception:
            return False

    all_task_paths = [p for p in sorted(args.tasks_dir.glob("task_*.json")) if _has_prompt(p)]
    if args.task_ids:
        raw_ids = [x.strip() for x in args.task_ids.split(",") if x.strip()]
        numeric_ids = {int(x) for x in raw_ids if x.isdigit()}
        str_ids = {x for x in raw_ids if not x.isdigit()}
        target_indexed_tasks = [
            (i, p) for i, p in enumerate(all_task_paths, 1)
            if i in numeric_ids or p.stem in str_ids or p.name in str_ids
        ]
        total_target = max(len(target_indexed_tasks), args.limit)
        if (args.solve_dir / "pipeline_metrics.json").exists():
            try:
                prev = json.loads((args.solve_dir / "pipeline_metrics.json").read_text(encoding="utf-8"))
                if prev.get("total_target"):
                    total_target = prev["total_target"]
            except Exception:
                pass
    elif args.task_range:
        parts = args.task_range.replace(":", "-").split("-")
        start_idx = max(1, int(parts[0]))
        end_idx = int(parts[1])
        target_indexed_tasks = [(i, p) for i, p in enumerate(all_task_paths, 1) if start_idx <= i <= end_idx]
        total_target = max(end_idx, args.limit)
    else:
        limit = min(args.limit, len(all_task_paths))
        target_indexed_tasks = list(enumerate(all_task_paths[:limit], 1))
        total_target = limit

    total_tasks = len(target_indexed_tasks)
    scorecard = PipelineScorecard(total_target, args.solve_dir, threshold=args.judge_threshold)

    if args.telegram:
        try:
            from core_engine.telegram import load_telegram_credentials, send_telegram_message
            bot_token, chat_id = load_telegram_credentials()
            checked_initial = len(scorecard.records)
            passes_initial = sum(1 for r in scorecard.records if r.get("pass_b"))
            launch_msg = (
                f"🚀 <b>TCS-Bench Continuous Pipeline ({args.solve_dir.name}) Launched!</b>\n\n"
                f"• <b>Target Problems:</b> {total_target}\n"
                f"• <b>Workers:</b> {args.workers} concurrent | <b>Max Iterations:</b> {args.max_iterations} turns\n"
                f"• <b>Completed/Skipped Tasks:</b> {checked_initial} ({passes_initial} passed, {checked_initial - passes_initial} not passed)\n"
                f"• <b>Remaining Tasks to Run:</b> {total_target - checked_initial}\n"
                f"• <b>Judge:</b> TCS Judge, median of {args.judge_replicates} (pass ≥ {args.judge_threshold:.2f})"
            )
            send_telegram_message(bot_token, chat_id, launch_msg)
        except Exception as e:
            log(f"[yellow]Could not send Telegram launch notification: {e}[/yellow]")

    start_display = target_indexed_tasks[0][0] if target_indexed_tasks else 1
    end_display = target_indexed_tasks[-1][0] if target_indexed_tasks else total_tasks

    log(Panel(
        f"[bold green]Starting Autonomous TCS-Bench Continuous Pipeline[/bold green]\n"
        f"Target Problems: [bold cyan]{start_display} to {end_display}[/bold cyan] ({total_tasks} active tasks, target total: {total_target})\n"
        f"Workers: [bold yellow]{args.workers}[/bold yellow] | Max Solver Iterations: [bold yellow]{args.max_iterations}[/bold yellow]\n"
        f"Referee: [bold yellow]TCS Judge[/bold yellow] (median of {args.judge_replicates} replicates; pass >= {args.judge_threshold:.2f})\n"
        f"Output Directory: [dim]{args.solve_dir}[/dim]\n"
        f"Seeding: [dim]{args.seed_from}[/dim]",
        title="Prime Math: Continuous Proof Pipeline",
    ))

    # Preload task map (with ground truth solutions strictly isolated for judge use)
    task_map = load_task_map(args.tasks_dir)

    # Set process niceness to ensure interactive SSH responsiveness
    try:
        os.nice(tcs_cfg.pipeline_niceness)
    except Exception:
        pass

    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_single_task,
                idx,
                path,
                args,
                args.solve_dir,
                task_map,
                scorecard,
            ): (idx, path.stem)
            for idx, path in target_indexed_tasks
        }
        for future in as_completed(futures):
            idx, tid = futures[future]
            try:
                future.result()
            except Exception as e:
                log(f"[bold red]Pipeline exception on Task {idx} ({tid}): {e}[/bold red]")

    elapsed = time.perf_counter() - start_time
    final_summary = json.loads((args.solve_dir / "pipeline_metrics.json").read_text(encoding="utf-8"))

    # Final summary display
    table = Table(title="TCS-Bench Pipeline Final Scorecard")
    table.add_column("Metric", style="cyan")
    table.add_column("Passed", style="green")
    table.add_column("Checked", style="white")
    table.add_column("Pass Rate (%)", style="bold yellow")

    dir_data = final_summary.get("directionally_sound", {})
    tcs_data = final_summary.get("tcs_judge") or final_summary.get("judge_b", {})
    lean_data = final_summary.get("lean", {})
    tcs_thresh = final_summary.get("threshold", args.judge_threshold)
    scored = str(final_summary.get("total_scored", final_summary["total_checked"]))
    table.add_row(f"[bold]HEADLINE: {headline_label(tcs_thresh)}[/bold]", str(tcs_data.get("passes", 0)), scored, f"{tcs_data.get('pass_percentage', 0.0)}%")
    table.add_row(directional_label(), str(dir_data.get("passes", 0)), scored, f"{dir_data.get('pass_percentage', 0.0)}%")
    if lean_data.get("passes"):
        table.add_row("EXPERIMENTAL: Lean 4 compile (not a soundness certificate)", str(lean_data.get("passes", 0)), scored, f"{lean_data.get('pass_percentage', 0.0)}%")
    table.add_row("Judge parse errors (excluded)", str(final_summary.get("judge_parse_errors", 0)), str(final_summary["total_checked"]), "-")
    table.add_row("Judge errors (excluded)", str(final_summary.get("judge_errors", 0)), str(final_summary["total_checked"]), "-")

    log("\n" + "=" * 80)
    log(f"[bold green]PIPELINE COMPLETE IN {elapsed:.1f}s ({elapsed/60:.1f} minutes)[/bold green]")
    log(table)
    log(f"Final metrics saved to: [cyan]{args.solve_dir / 'pipeline_metrics.json'}[/cyan]")
    log("=" * 80 + "\n")

    if getattr(args, "telegram", False):
        try:
            from core_engine.telegram import load_telegram_credentials, format_final_summary, send_telegram_message
            bot_token, chat_id = load_telegram_credentials()
            send_telegram_message(bot_token, chat_id, format_final_summary(final_summary))
        except Exception as e:
            log(f"[yellow]Telegram final summary error: {e}[/yellow]")


if __name__ == "__main__":
    main()
