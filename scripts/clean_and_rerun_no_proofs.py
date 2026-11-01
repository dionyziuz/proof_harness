#!/usr/bin/env python3
"""
scripts/clean_and_rerun_no_proofs.py

Identifies all tasks in a given evaluation run that ended with 0 candidate proof characters
(typically due to client read timeouts under old timeout limits), archives their stale logs,
removes their failed task/judgment records, and prints the exact command to re-run them
through the continuous pipeline.
"""

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser(description="Prepare and clean no-proof tasks for continuous re-run.")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "data/eval_results/tcs_pipeline_300_muse_v1",
                        help="Target evaluation directory")
    parser.add_argument("--execute", action="store_true", default=False,
                        help="Execute cleanup immediately (default: dry run)")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    tracker_f = run_dir / "live_tracker.jsonl"
    if not tracker_f.exists():
        raise SystemExit(f"Tracker file not found: {tracker_f}")

    no_proofs = []
    with open(tracker_f) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if (rec.get("candidate_chars") or 0) == 0:
                    no_proofs.append(rec)

    no_proofs.sort(key=lambda x: x["task_index"])
    indices = [str(r["task_index"]) for r in no_proofs]
    indices_str = ",".join(indices)

    print(f"Found {len(no_proofs)} tasks with 0 candidate proof characters in {run_dir.name}:")
    print(f"Task Indices ({len(indices)}): {indices_str}\n")

    archive_dir = run_dir / "archive_no_proof_timeouts"

    if args.execute:
        archive_dir.mkdir(parents=True, exist_ok=True)
        for r in no_proofs:
            tid = r["task_id"]
            for subdir in ["tasks", "judgments", "logs", "checkpoints"]:
                prefix = "judge_" if subdir == "judgments" else ""
                suffix = ".ckpt.json" if subdir == "checkpoints" else (".log" if subdir == "logs" else ".json")
                fpath = run_dir / subdir / f"{prefix}{tid}{suffix}"
                if fpath.exists():
                    dest = archive_dir / subdir
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.move(fpath, dest / fpath.name)

        print(f"✓ Archived stale files to {archive_dir}")
        print("✓ Cleaned failed records from active run directory.")
    else:
        print("[DRY RUN] Run with --execute to archive and clean stale files.")

    print("\nCanonical Launch Command for tmux / background:")
    print("=" * 80)
    print(
        f".venv/bin/python scripts/run_tcs_continuous_pipeline.py \\\n"
        f"  --env-file .env \\\n"
        f"  --domain tcs \\\n"
        f"  --prove-dir {run_dir} \\\n"
        f"  --task-ids {indices_str} \\\n"
        f"  --workers 4 \\\n"
        f"  --max-iterations 8 \\\n"
        f"  --tcs-judge \\\n"
        f"  --telegram"
    )
    print("=" * 80)

if __name__ == "__main__":
    main()
