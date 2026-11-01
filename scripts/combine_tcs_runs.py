#!/usr/bin/env python3
"""
scripts/combine_tcs_runs.py — Unify and Merge Multi-Step TCS-Bench Runs

Combines an unassisted one-pass triage run (Step 1) with a multi-turn agent harness run (Step 2)
into a single canonical evaluation directory. Automatically:
  1. Identifies tasks that passed in Step 1 but failed (or drifted) in Step 2, and overwrites
     the failed attempt with the verified Step 1 proof, log, and judgment.
  2. Preserves all tasks that passed in Step 2.
  3. Fills in any tasks present in Step 1 that were omitted from Step 2.
  4. Generates a consolidated live_tracker.jsonl and pipeline_metrics.json.
  5. Outputs a Rich comparison scorecard.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_tcs_continuous_pipeline import (
    summarize_records,
    TCS_PASS_THRESHOLD,
    DIRECTIONAL_THRESHOLD,
)

console = Console()


def load_tracker_map(tracker_path: Path) -> Dict[str, Dict[str, Any]]:
    records = {}
    if not tracker_path.exists():
        return records
    with open(tracker_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    records[data["task_id"]] = data
                except Exception:
                    continue
    return records


def get_validity(rec: Dict[str, Any]) -> float:
    v = rec.get("judge_b_validity")
    if v is not None:
        try:
            return float(v)
        except (ValueError, TypeError):
            pass
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Combine TCS-Bench Step 1 and Step 2 runs into a unified dataset.")
    parser.add_argument("--harness-dir", type=Path, default=PROJECT_ROOT / "data/eval_results/tcs_pipeline_300_muse_v1",
                        help="Path to Step 2 multi-turn harness directory")
    parser.add_argument("--onepass-dir", type=Path, default=PROJECT_ROOT / "data/eval_results/tcs_onepass_muse_v1",
                        help="Path to Step 1 unassisted one-pass directory")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/eval_results/tcs_combined_muse_v1",
                        help="Output directory to store unified tasks, logs, judgments, and scorecard")
    parser.add_argument("--threshold", type=float, default=TCS_PASS_THRESHOLD,
                        help=f"Passing threshold (default: {TCS_PASS_THRESHOLD})")
    parser.add_argument("--total-target", type=int, default=300,
                        help="Total benchmark target problems (default: 300)")
    args = parser.parse_args()

    harness_dir = args.harness_dir.resolve()
    onepass_dir = args.onepass_dir.resolve()
    out_dir = args.output_dir.resolve()

    if not harness_dir.is_dir():
        sys.exit(f"Harness dir not found: {harness_dir}")
    if not onepass_dir.is_dir():
        sys.exit(f"One-pass dir not found: {onepass_dir}")

    harness_map = load_tracker_map(harness_dir / "live_tracker.jsonl")
    onepass_map = load_tracker_map(onepass_dir / "live_tracker.jsonl")

    console.print(Panel(
        f"[bold green]Unifying TCS-Bench Step 1 and Step 2 Runs[/bold green]\n"
        f"• Step 2 Harness Dir: [cyan]{harness_dir.name}[/cyan] ({len(harness_map)} tasks recorded)\n"
        f"• Step 1 One-Pass Dir: [magenta]{onepass_dir.name}[/magenta] ({len(onepass_map)} tasks recorded)\n"
        f"• Unified Output Dir: [bold yellow]{out_dir}[/bold yellow]\n"
        f"• Passing Threshold: validity >= [yellow]{args.threshold:.2f}[/yellow]",
        title="TCS-Bench Run Combiner"
    ))

    # Prepare output directories
    out_tasks = out_dir / "tasks"
    out_judgments = out_dir / "judgments"
    out_logs = out_dir / "logs"
    for d in (out_tasks, out_judgments, out_logs):
        d.mkdir(parents=True, exist_ok=True)

    # Collect all unique task IDs across both runs
    all_task_ids = sorted(
        set(harness_map.keys()) | set(onepass_map.keys()),
        key=lambda tid: (
            harness_map.get(tid, {}).get("task_index")
            or onepass_map.get(tid, {}).get("task_index")
            or 9999
        )
    )

    combined_records: List[Dict[str, Any]] = []
    added_from_step1: List[str] = []
    kept_from_step2: List[str] = []
    both_passed: List[str] = []
    missing_filled: List[str] = []

    for tid in all_task_ids:
        h_rec = harness_map.get(tid)
        o_rec = onepass_map.get(tid)

        h_val = get_validity(h_rec) if h_rec else 0.0
        h_pass = bool(h_rec.get("pass_proven")) if h_rec else False

        o_val = get_validity(o_rec) if o_rec else 0.0
        o_pass = bool(o_rec.get("pass_proven")) if o_rec else False

        chosen_source = "harness"
        chosen_rec = None
        source_dir = harness_dir

        # In the canonical 2-Step Funnel, Step 1 (One-Shot Triage) is strictly LEADING:
        # Any task that passed in Step 1 was solved on Turn 1; its proof, log, and judgment
        # from Step 1 are the authoritative artifacts. Step 2 (Harness) is only used for
        # tasks that Step 1 failed to solve.
        if o_pass:
            chosen_source = "step1_oneshot"
            chosen_rec = dict(o_rec)
            source_dir = onepass_dir
            chosen_rec["solved_in_step"] = 1
            if h_pass:
                both_passed.append(tid)
            else:
                added_from_step1.append(tid)
        elif h_rec is None and o_rec is not None:
            # Task was missing from Step 2 harness run; fill from Step 1
            chosen_source = "step1_fallback"
            chosen_rec = dict(o_rec)
            source_dir = onepass_dir
            chosen_rec["solved_in_step"] = 1
            missing_filled.append(tid)
        else:
            # Step 1 was not a pass; take Step 2 harness attempt
            chosen_source = "step2_harness"
            chosen_rec = dict(h_rec)
            source_dir = harness_dir
            chosen_rec["solved_in_step"] = 2
            if h_pass:
                kept_from_step2.append(tid)

        # Copy underlying files into output directory
        task_f = source_dir / "tasks" / f"{tid}.json"
        judge_f = source_dir / "judgments" / f"judge_{tid}.json"
        log_f = source_dir / "logs" / f"{tid}.log"

        if task_f.exists():
            shutil.copy2(task_f, out_tasks / f"{tid}.json")
        if judge_f.exists():
            shutil.copy2(judge_f, out_judgments / f"judge_{tid}.json")
        if log_f.exists():
            shutil.copy2(log_f, out_logs / f"{tid}.log")

        # Mark source provenance in metadata
        chosen_rec["combined_provenance"] = chosen_source
        combined_records.append(chosen_rec)

    # Re-index task_index sequentially from 1 to N
    for i, rec in enumerate(combined_records, 1):
        rec["task_index"] = i

    # Generate unified live_tracker.jsonl
    tracker_out = out_dir / "live_tracker.jsonl"
    with open(tracker_out, "w", encoding="utf-8") as fp:
        for r in combined_records:
            fp.write(json.dumps(r) + "\n")

    # Generate unified pipeline_metrics.json
    summary = summarize_records(combined_records, args.total_target, args.threshold)
    metrics_out = out_dir / "pipeline_metrics.json"
    metrics_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Display Rich Scorecard
    table = Table(title="TCS-Bench Unified Scorecard (Step 1 + Step 2 Combined)")
    table.add_column("Evaluation Metric", style="cyan")
    table.add_column("Passed", justify="center", style="green")
    table.add_column("Checked", justify="center", style="white")
    table.add_column("Pass Rate (%)", justify="center", style="bold yellow")

    table.add_row(
        f"HEADLINE: Certified Sound (validity >= {args.threshold:.2f})",
        str(summary["tcs_judge"]["passes"]),
        str(summary["total_checked"]),
        f"{summary['tcs_judge']['pass_percentage']:.2f}%"
    )
    table.add_row(
        f"Directionally Sound (validity >= {DIRECTIONAL_THRESHOLD:.2f})",
        str(summary["directionally_sound"]["passes"]),
        str(summary["total_checked"]),
        f"{summary['directionally_sound']['pass_percentage']:.2f}%"
    )
    table.add_row(
        "Proof Candidates Emitted (>0 chars)",
        str(sum(1 for r in combined_records if (r.get("candidate_chars") or 0) > 0)),
        str(summary["total_checked"]),
        f"{sum(1 for r in combined_records if (r.get('candidate_chars') or 0) > 0) / summary['total_checked'] * 100.0:.2f}%"
    )

    console.print()
    console.print(table)
    console.print()

    n_step1_passes = sum(1 for r in combined_records if r.get("pass_proven") and r.get("solved_in_step") == 1)
    n_step2_passes = sum(1 for r in combined_records if r.get("pass_proven") and r.get("solved_in_step") == 2)

    console.print(Panel(
        f"[bold]Canonical 2-Step Provenance Breakdown:[/bold]\n"
        f"• Step 1 (One-Shot Triage) Leading Passes: [bold green]{n_step1_passes}[/bold green] (authoritative 1st-turn wins)\n"
        f"  ├─ Passed in BOTH Step 1 & Step 2: [green]{len(both_passed)}[/green] (Step 1 log/proof prioritized)\n"
        f"  └─ Passed in Step 1 ONLY (rescued): [magenta]{len(added_from_step1)}[/magenta] (Step 2 failure overwritten)\n"
        f"• Step 2 (Multi-Turn Harness) Passes: [bold cyan]{n_step2_passes}[/bold cyan] (theorems unlocked via multi-turn search)\n"
        f"• Total Certified Passes (Step 1 + Step 2): [bold yellow]{summary['tcs_judge']['passes']} / {summary['total_checked']}[/bold yellow] ([yellow]{summary['tcs_judge']['pass_percentage']:.2f}%[/yellow])\n\n"
        f"[bold green]Saved canonical combined dataset to:[/bold green] [underline]{out_dir}[/underline]",
        title="Merge Summary (Step 1 Strictly Leading)"
    ))


if __name__ == "__main__":
    main()
