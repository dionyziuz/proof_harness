#!/usr/bin/env python3
"""Standalone Unassisted One-Pass Baseline for TCS-Bench (Solve in 1 Pass + Official TCS Judge).

Designed for rigorous, reviewer-proof academic benchmarking:
1. Loads the exact task prompt from TCS-Bench (airgapped, ground truth stripped).
2. Queries the base model in a single unassisted pass (T = 1):
   - Zero multi-turn revision loops.
   - Zero Python/SMT REPL execution kernels.
   - Zero CoALA vector memory injections or Context Architect blueprints.
   - Full native thinking tokens enabled (fair test-time reasoning compute).
3. Extracts the candidate LaTeX proof with generous formatting fallback so no attempt
   is ever disqualified on LaTeX formatting quirks.
4. Saves the solver record to <solve_dir>/tasks/<task_id>.json and full trace to <solve_dir>/logs/<task_id>.log.
5. Immediately invokes the official calibrated referee (core_engine.judging.judge_tcs_proof,
   Config B: 3 independent replicates, median validity >= 0.50).
6. Atomically updates <solve_dir>/pipeline_metrics.json and <solve_dir>/live_tracker.jsonl.

Usage:
  # Run on first 34 problems with 4 workers (using active .env):
  python3 scripts/run_tcs_onepass_baseline.py --prove-dir data/eval_results/tcs_onepass_muse_v1 --limit 34 --workers 4

  # Run full 300-problem baseline with custom .env file:
  python3 scripts/run_tcs_onepass_baseline.py --env-file .env.qwen --prove-dir data/eval_results/tcs_onepass_qwen_v1 --limit 300 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

from core_engine.proof_extractor import extract_latex_proof
from core_engine.tcs_config import TCS_JUDGE_REPLICATES, TCS_PASS_THRESHOLD, load_tcs_config
from run_tcs_bench_judge import judge_with_retry, load_task_map
from run_tcs_continuous_pipeline import (
    DEFAULT_EVAL_DIR,
    DEFAULT_TASKS_DIR,
    DIRECTIONAL_THRESHOLD,
    PipelineScorecard,
    directional_label,
    display_task_update,
    headline_label,
    log,
)

ONEPASS_SYSTEM_PROMPT = (
    "You are an expert researcher in Theoretical Computer Science and formal mathematics.\n"
    "Your task is to write a complete, rigorous, and self-contained formal LaTeX proof "
    "for the target theorem, lemma, proposition, or claim presented in the problem prompt.\n\n"
    "Guidelines:\n"
    "1. Use the definitions, notation, and context statements provided in the prompt.\n"
    "2. Justify every algebraic step, reduction, inequality direction, and asymptotic bound from first principles.\n"
    "3. Explicitly discharge all boundary cases, degenerate regimes, and theorem preconditions.\n"
    "4. Enclose your final complete proof strictly inside \\begin{proof} ... \\end{proof}."
)


def extract_onepass_candidate(content: str, reasoning: Optional[str] = None) -> str:
    """Extracts the candidate LaTeX proof from a single-pass model response.

    Uses core_engine's structured extractor first, then falls back to outer
    \\begin{proof}...\\end{proof} matching or wrapping the full response so that
    no baseline model completion is ever penalized or dropped on formatting.
    """
    text = (content or "").strip()

    # 1. Try canonical extractor on content
    if text:
        cand = extract_latex_proof(text)
        if cand:
            return cand

        # 2. Generous outer \begin{proof} ... \end{proof} span
        idx_start = text.find(r"\begin{proof}")
        idx_end = text.rfind(r"\end{proof}")
        if idx_start != -1 and idx_end != -1 and idx_end > idx_start:
            return text[idx_start : idx_end + len(r"\end{proof}")].strip()
        if idx_start != -1:
            return (text[idx_start:].strip() + "\n\\end{proof}").strip()

        # 3. If the model wrote substantive math without \begin{proof} tags, wrap it
        return f"\\begin{{proof}}\n{text}\n\\end{{proof}}"

    # 4. Fallback: check reasoning trace if content was empty (e.g. token budget hit inside <think>)
    r_text = (reasoning or "").strip()
    if r_text:
        idx_start = r_text.rfind(r"\begin{proof}")
        if idx_start != -1:
            idx_end = r_text.rfind(r"\end{proof}")
            if idx_end != -1 and idx_end > idx_start:
                return r_text[idx_start : idx_end + len(r"\end{proof}")].strip()
            return (r_text[idx_start:].strip() + "\n\\end{proof}").strip()

    return ""


def process_onepass_task(
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

    if not prompt or not prompt.strip():
        log(f"[yellow]Skipping Task {orig_idx} ({task_id}): Empty benchmark prompt.[/yellow]")
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

    if args.force:
        if task_log.exists():
            try:
                task_log.unlink()
            except Exception:
                pass

    # Step 1: Resume if already solved and judged
    if judge_out.exists() and solve_out.exists() and not args.force:
        try:
            prev_solve = json.loads(solve_out.read_text(encoding="utf-8"))
            prev_judge = json.loads(judge_out.read_text(encoding="utf-8"))
            summary = scorecard.record_judgment(orig_idx, task_id, prev_solve, prev_judge)
            log(f"[dim]Loaded existing one-pass judgment for Task {orig_idx} ({task_id})[/dim]")
            display_task_update(summary, orig_idx, task_id, prev_solve, prev_judge)
            return
        except Exception as e:
            log(f"[yellow]Failed to load existing record for Task {orig_idx} ({task_id}): {e}[/yellow]")

    # Step 2: Run single-pass completion if not yet solved
    solve_rec = None
    if solve_out.exists() and not args.force:
        try:
            solve_rec = json.loads(solve_out.read_text(encoding="utf-8"))
        except Exception:
            solve_rec = None

    if solve_rec is None or not solve_rec.get("candidate_proof"):
        from core_engine.llm_client import TunnelThinkingClient

        client = TunnelThinkingClient()
        log(f"[{task_id}] [bold cyan]Running Unassisted One-Pass Generation (Model: {client.config.model})...[/bold cyan]")

        messages = [
            {"role": "system", "content": ONEPASS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    "---\n"
                    "Write a complete, rigorous, and self-contained formal LaTeX proof for the target statement above, "
                    "enclosed in \\begin{proof} ... \\end{proof}."
                ),
            },
        ]

        t0 = time.perf_counter()
        resp = None
        last_err: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                resp = client.complete(
                    messages=messages,
                    timeout_seconds=args.timeout,
                    enable_thinking=not args.no_thinking,
                    stream_collect=True,
                )
                break
            except Exception as exc:
                last_err = exc
                backoff_s = 15.0 * attempt
                log(
                    f"[yellow][{task_id}] One-pass attempt {attempt}/3 encountered error: "
                    f"{exc}. Retrying in {backoff_s:.0f}s...[/yellow]"
                )
                if attempt < 3:
                    time.sleep(backoff_s)

        if resp is None:
            raise RuntimeError(f"One-pass generation failed after 3 attempts: {last_err}") from last_err

        duration_s = round(time.perf_counter() - t0, 2)
        candidate_proof = extract_onepass_candidate(resp.content, resp.reasoning)

        # Write detailed log trace
        log_lines = [
            "=" * 80,
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TCS-BENCH UNASSISTED ONE-PASS SESSION: {task_id}",
            f"Model: {resp.model} | Duration: {duration_s}s | Prompt Tokens: {resp.prompt_tokens} | Completion Tokens: {resp.completion_tokens}",
            "=" * 80,
            "\n--- [TURN 1/1] INNER THINKING ---",
            resp.reasoning or "(none)",
            "\n--- [TURN 1/1] MODEL GENERATED OUTPUT ---",
            resp.content or "(empty)",
            "\n--- [EXTRACTED CANDIDATE PROOF] ---",
            f"Length: {len(candidate_proof)} chars",
            candidate_proof,
            "=" * 80 + "\n",
        ]
        task_log.write_text("\n".join(log_lines), encoding="utf-8")

        solve_rec = {
            "task_id": task_id,
            "arxiv_id": task_data.get("arxiv_id"),
            "statement_id": task_data.get("statement_id"),
            "mode": "onepass_unassisted",
            "model": resp.model,
            "status": "completed" if candidate_proof else "no_proof",
            "candidate_proof": candidate_proof,
            "verified": False,
            "iterations": 1,
            "reasoning_traces": [resp.reasoning] if resp.reasoning else [],
            "code_history": [],
            "retrieved_heuristics": [],
            "learned_heuristics": [],
            "blueprint": None,
            "duration_s": duration_s,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "log_file": str(task_log),
        }
        solve_out.write_text(json.dumps(solve_rec, indent=2, ensure_ascii=False), encoding="utf-8")

    # Step 3: Immediately Judge with Official Calibrated Referee
    log(f"[{task_id}] [bold magenta]One-pass complete ({len(solve_rec.get('candidate_proof') or '')} chars). Invoking Official TCS Judge...[/bold magenta]")
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

    # Step 4: Update atomic scorecard and display
    summary = scorecard.record_judgment(orig_idx, task_id, solve_rec, judge_rec)
    display_task_update(summary, orig_idx, task_id, solve_rec, judge_rec)

    if getattr(args, "telegram", False):
        try:
            from core_engine.telegram import format_task_report, load_telegram_credentials, send_telegram_message

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
        description="Standalone Unassisted One-Pass Baseline for TCS-Bench (T=1 + Calibrated TCS Judge)"
    )
    parser.add_argument("--env-file", "-e", type=Path, default=None,
                        help="Path to .env configuration file to load (default: .env)")
    parser.add_argument("--limit", type=int, default=300,
                        help="Number of problems to evaluate from start of dataset (default: 300)")
    parser.add_argument("--task-range", type=str, default=None,
                        help="1-indexed range of tasks to run (e.g. 1-34 or 1-300)")
    parser.add_argument("--task-ids", type=str, default=None,
                        help="Comma-separated list of specific task indices or task_ids to run")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of concurrent workers (default: 4)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="Per-task single-pass timeout in seconds (default: 600)")
    parser.add_argument("--judge-timeout", type=float, default=tcs_cfg.judge_timeout_seconds,
                        help=f"Per-judge-call timeout in seconds (default: {tcs_cfg.judge_timeout_seconds})")
    parser.add_argument("--prove-dir", "--solve-dir", dest="prove_dir", type=Path,
                        default=DEFAULT_EVAL_DIR / "tcs_onepass_baseline",
                        help="Directory to save one-pass tasks, logs, and judgments")
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR,
                        help="Path to TCSBench tasks directory")
    parser.add_argument("--config", choices=["b", "tcs", "a", "ab", "lean"], default=tcs_cfg.pipeline_judge_config,
                        help=f"Judge configuration (default: {tcs_cfg.pipeline_judge_config})")
    parser.add_argument("--judge-threshold", type=float, default=tcs_cfg.pipeline_judge_threshold,
                        help=f"Config B passing validity threshold (default: {tcs_cfg.pipeline_judge_threshold})")
    parser.add_argument("--judge-replicates", type=int, default=TCS_JUDGE_REPLICATES,
                        help=f"Independent Judge-B replicates per task (default: {TCS_JUDGE_REPLICATES})")
    parser.add_argument("--judge-adaptive", action="store_true", default=tcs_cfg.pipeline_judge_adaptive,
                        help="Adaptive replicates: stop after 1 call if outside threshold +/- 0.20")
    parser.add_argument("--no-judge-adaptive", action="store_false", dest="judge_adaptive",
                        help="Disable adaptive replicates and always run all replicates")
    parser.add_argument("--judge-thinking", action="store_true", default=tcs_cfg.pipeline_judge_thinking,
                        help="Enable thinking tokens on judge calls")
    parser.add_argument("--no-judge-thinking", action="store_false", dest="judge_thinking",
                        help="Disable thinking tokens on judge calls")
    parser.add_argument("--no-thinking", action="store_true", default=False,
                        help="Disable thinking tokens on one-pass solver")
    parser.add_argument("--telegram", action="store_true", default=False,
                        help="Send live updates to Telegram")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Force re-running and re-judging existing tasks")

    args = parser.parse_args()
    if args.env_file:
        from core_engine.config import configure_env_file

        configure_env_file(args.env_file)
        tcs_cfg = load_tcs_config(env_path=args.env_file, reload=True)

    args.solve_dir = args.prove_dir
    if not args.tasks_dir.is_dir():
        sys.exit(f"Tasks directory not found: {args.tasks_dir}")

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
    elif args.task_range:
        parts = args.task_range.replace(":", "-").split("-")
        start_idx = max(1, int(parts[0]))
        end_idx = int(parts[1])
        target_indexed_tasks = [(i, p) for i, p in enumerate(all_task_paths, 1) if start_idx <= i <= end_idx]
        total_target = max(end_idx, len(target_indexed_tasks))
    else:
        limit = min(args.limit, len(all_task_paths))
        target_indexed_tasks = list(enumerate(all_task_paths[:limit], 1))
        total_target = limit

    total_tasks = len(target_indexed_tasks)
    scorecard = PipelineScorecard(total_target, args.solve_dir, threshold=args.judge_threshold)

    start_display = target_indexed_tasks[0][0] if target_indexed_tasks else 1
    end_display = target_indexed_tasks[-1][0] if target_indexed_tasks else total_tasks

    log(Panel(
        f"[bold green]Starting Unassisted One-Pass Baseline (T = 1)[/bold green]\n"
        f"Target Problems: [bold cyan]{start_display} to {end_display}[/bold cyan] ({total_tasks} active tasks, target total: {total_target})\n"
        f"Workers: [bold yellow]{args.workers}[/bold yellow] | Mode: [bold yellow]Pure Single-Pass (Zero Harness Scaffolding)[/bold yellow]\n"
        f"Referee: [bold yellow]TCS Judge[/bold yellow] (median of {args.judge_replicates} replicates; pass >= {args.judge_threshold:.2f})\n"
        f"Output Directory: [dim]{args.solve_dir}[/dim]",
        title="Proof Harness: Unassisted One-Pass Baseline",
    ))

    task_map = load_task_map(args.tasks_dir)

    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_onepass_task,
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
                log(f"[bold red]One-pass exception on Task {idx} ({tid}): {e}[/bold red]")

    elapsed = time.perf_counter() - start_time
    final_summary = json.loads((args.solve_dir / "pipeline_metrics.json").read_text(encoding="utf-8"))

    table = Table(title="TCS-Bench One-Pass Baseline Final Scorecard")
    table.add_column("Metric", style="cyan")
    table.add_column("Passed", style="green")
    table.add_column("Checked", style="white")
    table.add_column("Pass Rate (%)", style="bold yellow")

    dir_data = final_summary.get("directionally_sound", {})
    tcs_data = final_summary.get("tcs_judge") or final_summary.get("judge_b", {})
    tcs_thresh = final_summary.get("threshold", args.judge_threshold)
    scored = str(final_summary.get("total_scored", final_summary["total_checked"]))
    table.add_row(f"[bold]HEADLINE: {headline_label(tcs_thresh)}[/bold]", str(tcs_data.get("passes", 0)), scored, f"{tcs_data.get('pass_percentage', 0.0)}%")
    table.add_row(directional_label(), str(dir_data.get("passes", 0)), scored, f"{dir_data.get('pass_percentage', 0.0)}%")
    table.add_row("Judge parse errors (excluded)", str(final_summary.get("judge_parse_errors", 0)), str(final_summary["total_checked"]), "-")
    table.add_row("Judge errors (excluded)", str(final_summary.get("judge_errors", 0)), str(final_summary["total_checked"]), "-")

    log("\n" + "=" * 80)
    log(f"[bold green]ONE-PASS BASELINE COMPLETE IN {elapsed:.1f}s ({elapsed/60:.1f} minutes)[/bold green]")
    log(table)
    log(f"Final metrics saved to: [cyan]{args.solve_dir / 'pipeline_metrics.json'}[/cyan]")
    log("=" * 80 + "\n")


if __name__ == "__main__":
    main()
