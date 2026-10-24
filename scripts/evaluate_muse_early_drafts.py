#!/usr/bin/env python3
"""
Evaluate Early (Turn 2) Proof Drafts from Meta Muse Logs
---------------------------------------------------------
Extracts candidate proofs generated on Turn 2 (or Turn 1) from the raw logs of
data/eval_results/tcs_pipeline_300_muse_v1/logs/, structures them as benchmark
records, and optionally judges them with the official external calibrated TCS Judge.

Usage:
  # Extract drafts and show summary:
  python3 scripts/evaluate_muse_early_drafts.py --extract-only

  # Judge extracted drafts with external TCS Judge:
  python3 scripts/evaluate_muse_early_drafts.py --judge --workers 2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from core_engine.tcs_config import TCS_JUDGE_REPLICATES, TCS_PASS_THRESHOLD, load_tcs_config
from core_engine.judging.tcs_judge import judge_tcs_proof
from core_engine.llm_client import TunnelThinkingClient
from run_tcs_bench_judge import load_task_map


def extract_turn2_drafts(logs_dir: Path, out_dir: Path) -> List[Dict[str, Any]]:
    """Scan all log files, extract Turn 2 LaTeX candidate proofs, and save them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    log_files = sorted(logs_dir.glob("*.log"))
    for lf in log_files:
        tid = lf.stem
        text = lf.read_text(encoding="utf-8", errors="ignore")

        # Check for Turn 2 proof draft marker
        idx = text.find("[Proof Draft Generated] Turn 2:")
        if idx == -1:
            continue

        start_turn = text.rfind("--- [TURN 2", 0, idx)
        if start_turn == -1:
            start_turn = max(0, idx - 15000)
        turn_chunk = text[start_turn:idx]

        p_start = turn_chunk.find(r"\begin{proof}")
        if p_start == -1:
            # fallback to begin{markdown} or proposal
            p_start = turn_chunk.find(r"**Proof:**")
        if p_start == -1:
            p_start = 0

        proof = turn_chunk[p_start:].strip()
        if not proof:
            continue

        # Extract internal auditor score from log if present
        m_aud = re.search(
            r"\[Proof Draft Generated\] Turn 2:.*?\"validity\":\s*([0-9\.]+).*?\"verdict\":\s*\"([^\"]+)\"",
            text[idx:idx + 3000],
            re.DOTALL,
        )
        int_val = float(m_aud.group(1)) if m_aud else None
        int_verd = m_aud.group(2) if m_aud else None

        rec = {
            "task_id": tid,
            "turn": 2,
            "candidate_proof": proof,
            "candidate_chars": len(proof),
            "internal_auditor_validity": int_val,
            "internal_auditor_verdict": int_verd,
            "source_log": str(lf),
        }
        records.append(rec)

        # Save individual draft JSON
        draft_file = out_dir / f"{tid}.json"
        draft_file.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    return records


def run_judging(
    drafts: List[Dict[str, Any]],
    tasks_dir: Path,
    judgments_dir: Path,
    replicates: int = TCS_JUDGE_REPLICATES,
    threshold: float = TCS_PASS_THRESHOLD,
    env_file: str = ".env",
) -> Dict[str, Any]:
    """Judge extracted candidate proofs with the official TCS Judge."""
    judgments_dir.mkdir(parents=True, exist_ok=True)
    task_map = load_task_map(tasks_dir)
    tcs_cfg = load_tcs_config(env_path=env_file)
    client = TunnelThinkingClient(timeout_seconds=tcs_cfg.judge_timeout_seconds)

    results: List[Dict[str, Any]] = []

    for d in drafts:
        tid = d["task_id"]
        judge_out_file = judgments_dir / f"judge_{tid}.json"

        # Check existing judgment
        if judge_out_file.exists():
            try:
                prev_j = json.loads(judge_out_file.read_text(encoding="utf-8"))
                results.append(prev_j)
                continue
            except Exception:
                pass

        if tid not in task_map:
            print(f"[WARN] Task {tid} not found in task map, skipping.")
            continue

        t_data = task_map[tid]
        prompt = t_data["prompt"]
        gt = t_data.get("ground_truth_solution") or ""
        candidate = d["candidate_proof"]

        print(f"[JUDGE] Judging Turn 2 candidate for {tid} ({len(candidate)} chars)...")
        res = judge_tcs_proof(
            candidate=candidate,
            problem_prompt=prompt,
            gt_solution=gt,
            client=client,
            config="tcs",
            replicates=replicates,
            threshold=threshold,
        )

        res_record = {
            "task_id": tid,
            "turn": 2,
            "candidate_chars": len(candidate),
            "internal_auditor_validity": d.get("internal_auditor_validity"),
            "judge_b_validity": res.get("validity"),
            "replicates": res.get("replicate_validities"),
            "pass_probability": res.get("pass_probability"),
            "pass_b": res.get("verdict") == "PASS",
            "issues": res.get("issues"),
            "status": res.get("status", "judged"),
        }
        judge_out_file.write_text(json.dumps(res_record, indent=2), encoding="utf-8")
        results.append(res_record)
        print(f"  --> {tid} Turn 2 External Validity: {res.get('validity')} | Verdict: {res.get('verdict')}")

    # Aggregate metrics
    scored = [r for r in results if r.get("judge_b_validity") is not None]
    passes = [r for r in scored if r.get("pass_b")]
    directional = [r for r in scored if (r.get("judge_b_validity") or 0) >= 0.40]

    summary = {
        "benchmark": "TCS-Bench Muse Turn-2 Early Drafts",
        "total_drafts": len(drafts),
        "total_judged": len(scored),
        "certified_passes": len(passes),
        "pass_rate_over_drafts": (len(passes) / len(drafts) * 100.0) if drafts else 0.0,
        "directionally_sound": len(directional),
        "directional_rate_over_drafts": (len(directional) / len(drafts) * 100.0) if drafts else 0.0,
        "mean_validity": (sum(r["judge_b_validity"] for r in scored) / len(scored)) if scored else 0.0,
        "results": results,
    }
    (judgments_dir / "turn2_summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Turn 2 Proof Drafts from Muse Logs")
    parser.add_argument("--logs-dir", type=Path, default=PROJECT_ROOT / "data" / "eval_results" / "tcs_pipeline_300_muse_v1" / "logs")
    parser.add_argument("--tasks-dir", type=Path, default=PROJECT_ROOT / "data" / "tcs" / "tasks")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "eval_results" / "tcs_pipeline_300_muse_v1" / "early_drafts" / "turn2")
    parser.add_argument("--extract-only", action="store_true", default=False)
    parser.add_argument("--judge", action="store_true", default=False)
    parser.add_argument("--env-file", type=str, default=".env")
    args = parser.parse_args()

    print(f"Scanning logs in {args.logs_dir}...")
    drafts = extract_turn2_drafts(args.logs_dir, args.out_dir / "tasks")
    print(f"Found and extracted {len(drafts)} Turn 2 candidate proofs into {args.out_dir / 'tasks'}")

    for d in drafts:
        print(f"  {d['task_id']} | Chars: {d['candidate_chars']} | Internal Audit: {d['internal_auditor_validity']} ({d['internal_auditor_verdict']})")

    if args.judge:
        print("\nStarting independent external judging of Turn 2 drafts...")
        summary = run_judging(
            drafts=drafts,
            tasks_dir=args.tasks_dir,
            judgments_dir=args.out_dir / "judgments",
            env_file=args.env_file,
        )
        print("\n=== JUDGE SUMMARY ===")
        print(f"Total Judged: {summary['total_judged']}")
        print(f"Certified Passes (>=0.50): {summary['certified_passes']} ({summary['pass_rate_over_drafts']:.1f}%)")
        print(f"Directional (>=0.40): {summary['directionally_sound']} ({summary['directional_rate_over_drafts']:.1f}%)")
        print(f"Mean Validity: {summary['mean_validity']:.3f}")


if __name__ == "__main__":
    main()
