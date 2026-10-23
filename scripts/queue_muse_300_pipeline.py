#!/usr/bin/env python3
"""Queue and automatically launch the full 300-problem TCS-Bench Muse pipeline with 3 workers.

Waits for the current initial batch (tasks 1-5) to finish, then automatically launches
the complete 300-problem pipeline with 3 concurrent workers.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROVE_DIR = PROJECT_ROOT / "data" / "eval_results" / "tcs_pipeline_300_muse_v1"
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "run_tcs_continuous_pipeline.py"
LOG_FILE = PROVE_DIR / "pipeline_300_continuous.log"


def find_initial_batch_pids():
    """Find PIDs of any running continuous pipeline on task-range 1-5."""
    try:
        out = subprocess.run(["pgrep", "-f", "task-range 1-5"], capture_output=True, text=True).stdout
        pids = [int(p.strip()) for p in out.splitlines() if p.strip()]
        return pids
    except Exception:
        return []


def main():
    initial_pids = find_initial_batch_pids()
    if initial_pids:
        print(f"Waiting for initial batch (PIDs {initial_pids}) to complete...")
        while True:
            active = [p for p in initial_pids if Path(f"/proc/{p}").exists()]
            if not active:
                print("Initial batch has completed.")
                break
            time.sleep(10)
    else:
        print("No active initial batch process found; proceeding directly to full 300 launch.")

    print(f"Launching full 300-problem pipeline with 3 workers into {PROVE_DIR}...")
    cmd = [
        str(PYTHON_BIN),
        str(PIPELINE_SCRIPT),
        "--prove-dir", str(PROVE_DIR),
        "--limit", "300",
        "--task-range", "1-300",
        "--workers", "3",
        "--telegram",
    ]

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n\n=== Auto-launched Full 300 Pipeline with 3 Workers at {time.ctime()} ===\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=f, stderr=subprocess.STDOUT)

    print(f"Full 300 pipeline successfully started (PID: {proc.pid}). Logging to {LOG_FILE}")


if __name__ == "__main__":
    main()
