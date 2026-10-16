# TCS-Bench Dataset

This directory links to the Theoretical Computer Science Benchmark (TCS-Bench) tasks.

## Tasks
- **Location**: `data/tcs/tasks` (or `../TCSBench/tasks` if cloned alongside `prime-math`)
- **Total Problems**: Exactly 300 complete research tasks (`task_001.json` through `task_300.json`).
- **Structure**: Each task contains:
  - `task_id`: Identifier (e.g. `task_001`)
  - `prompt`: Formal research statement and context extracted from peer-reviewed TCS literature.
  - `reference_proof`: Ground truth proof from the publication used by the judge.
  - `source`: arXiv / journal reference details.

## Proving & Evaluation
Run our unified `MathProver` against TCS-Bench:
```bash
.venv/bin/python scripts/run_tcs_bench_prove.py --workers 2
```
Judge candidate proofs against ground-truth reference proofs:
```bash
.venv/bin/python scripts/run_tcs_bench_judge.py --run-dir runs/<run_timestamp> --workers 2
```
