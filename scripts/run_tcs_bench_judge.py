#!/usr/bin/env python3
"""TCS-Bench Stage 2 -- judge every (problem, GT proof, candidate proof) triple with the
validated A.1 verifier (arXiv 2608.09538 prompt A; 4 calls; verdict = >=3/4 agreement).

The solver is NOT invoked here. Candidate proofs are read from the Stage 1 run dir
(data/eval_results/tcs_solve_<ts>/tasks/*.json); ground-truth proofs are loaded in
THIS process only -- the two stages are joined solely by the saved data.

Reuses (imports) the prompt builders, parsers, config loader, and judge-call helper
from scripts/run_judge_validation.py -- the same code that was calibrated on the 12
labeled cases (Config A: 12/12 bimodal; Config B: AUC 0.944). Judge model = Qwen 3.8
via the tunnel (the same model that solved -- self-judging, mirroring how the
benchmark paper scores; the local-Gemma-judge experiment is intentionally not here).

Candidate proof source: SolveResult.final_response (the model's final turn). If a
task has no final_response (error/timeout edge), the joined reasoning traces are
used and the record is marked candidate_source="reasoning_traces_joined". The
candidate is capped at MAX_CANDIDATE_CHARS (tail kept) so an unbounded trace log
cannot overflow the judge's input limit; capped records carry
candidate_truncated=true.

Judging is delegated to core_engine.judging.judge_tcs_proof (the single implementation):
TCS Judge is run as k independent replicates (--replicates, default 3); the reported validity
is the median, plus mean/spread and pass_probability (fraction of replicates >= threshold).
A reply whose validity cannot be recovered in ANY replicate yields status="judge_parse_error"
and is excluded from the pass-rate denominator (it is never recorded as validity 0.0).

Usage:
  .venv/bin/python scripts/run_tcs_bench_judge.py --solve-dir data/eval_results/tcs_solve_<ts>   # TCS Judge, k=3
  .venv/bin/python scripts/run_tcs_bench_judge.py --solve-dir ... --replicates 1  # legacy single call
  .venv/bin/python scripts/run_tcs_bench_judge.py --solve-dir ... --config a   # A.1 only
  .venv/bin/python scripts/run_tcs_bench_judge.py --solve-dir ... --config ab  # dual (secondary metric)
"""

from __future__ import annotations

import argparse
import json
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
from rich.table import Table

import httpx

from core_engine.config import load_config
from core_engine.llm_client import TunnelThinkingClient
from core_engine.judging import (
    TARGET_RE,  # noqa: F401  (re-exported for legacy importers of this script)
    extract_target,  # noqa: F401
    judge_lean_proof,
    judge_tcs_proof,
)
from core_engine.tcs_config import TCS_JUDGE_REPLICATES, TCS_PASS_THRESHOLD

_LOCAL_TCS_TASKS = PROJECT_ROOT / "data" / "tcs" / "tasks"
_EXTERNAL_TCS_TASKS = PROJECT_ROOT.parent / "TCSBench" / "tasks"
DEFAULT_TASKS_DIR = _LOCAL_TCS_TASKS if _LOCAL_TCS_TASKS.is_dir() else _EXTERNAL_TCS_TASKS
EVAL_DIR = PROJECT_ROOT / "data" / "eval_results"

# Hard cap on the judge candidate:
# For publication proofs (candidate_proof), allow up to 100,000 chars and preserve from head.
# For trace-join fallback, keep up to 32,000 chars from the tail.
MAX_PROOF_CHARS = 100000
MAX_TRACE_CHARS = 32000

console = Console()
print_lock = threading.Lock()


def log(msg: str) -> None:
    with print_lock:
        console.print(msg)



def load_task_map(tasks_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(tasks_dir.glob("task_*.json")):
        out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return out


def judge_one(rec: Dict[str, Any], task_data: Dict[str, Any],
              args: argparse.Namespace, client: TunnelThinkingClient,
              thinking: Optional[bool], timeout_seconds: float) -> Dict[str, Any]:
    prompt = task_data["prompt"]
    gt = task_data.get("ground_truth_solution") or ""
    final = rec.get("candidate_proof")
    if final:
        truncated = len(final) > MAX_PROOF_CHARS
        candidate = final[:MAX_PROOF_CHARS] if truncated else final
    else:
        joined = "\n\n".join(rec.get("reasoning_traces") or [])
        truncated = len(joined) > MAX_TRACE_CHARS
        candidate = joined[-MAX_TRACE_CHARS:] if truncated else joined
    out: Dict[str, Any] = {
        "task_id": rec["task_id"],
        "arxiv_id": rec.get("arxiv_id"),
        "statement_id": rec.get("statement_id"),
        "candidate_source": "final_response" if final else "reasoning_traces_joined",
        "candidate_chars": len(candidate),
        "candidate_truncated": truncated,
        "status": "judged",
    }
    if not candidate.strip():
        out["status"] = "no_proof"
        out["a"] = {"verdict": "NO_PROOF"}
        out["b"] = None
        return out

    if args.config == "lean":
        lean_rec = judge_lean_proof(candidate, problem_statement=prompt, timeout=timeout_seconds)
        out["lean"] = lean_rec
        out["verdict"] = lean_rec["verdict"]
        out["status"] = "judged" if lean_rec["passed"] else ("sorry_detected" if lean_rec["has_sorry"] else "fail")
        return out

    # Config A / B / AB: single implementation shared with the library (replicated Config B,
    # median validity, parse-error status, repair retry on truncation).
    res = judge_tcs_proof(
        candidate=candidate,
        problem_prompt=prompt,
        gt_solution=gt,
        client=client,
        config=args.config,
        temp_a=getattr(args, "temp_a", 0.3),
        temp_b=getattr(args, "temp_b", 0.1),
        threshold=getattr(args, "threshold", None),
        timeout_seconds=timeout_seconds,
        thinking=thinking,
        max_tokens=getattr(args, "max_tokens", 16384),
        replicates=getattr(args, "replicates", TCS_JUDGE_REPLICATES),
        borderline_replicates=False,  # --replicates 1 means exactly one call (legacy baseline)
        adaptive=getattr(args, "adaptive", False),
    )
    for key in ("status", "verdict", "threshold", "a", "b", "validity", "pass_probability",
                "replicate_validities", "validity_mean", "validity_spread", "scores", "issues"):
        if key in res:
            out[key] = res[key]
    if args.config in ("b", "tcs", "ab"):
        b = out.get("b") or {}
        if b.get("validity") is None and out["status"] == "judged":
            # Defensive: a missing validity is a parse failure, never a judged 0.0.
            out["status"] = "judge_parse_error"
    return out


def judge_with_retry(rec: Dict[str, Any], task_data: Dict[str, Any],
                     args: argparse.Namespace, client: TunnelThinkingClient,
                     thinking: Optional[bool], timeout_seconds: float,
                     retries: int) -> Dict[str, Any]:
    """judge_one + runner-level retry on transient tunnel errors (HTTP 524 etc.).
    The client's own internal retry is exhausted before an exception reaches here;
    a retried task gets a FRESH full set of calls (no partial-call stitching)."""
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            return judge_one(rec, task_data, args, client, thinking, timeout_seconds)
        except Exception as exc:
            retriable = isinstance(exc, (httpx.HTTPError, OSError))
            if not retriable or attempt == attempts:
                return {"task_id": rec["task_id"], "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "attempts": attempt}
            log(f"[yellow]retry {rec['task_id']} ({attempt}/{attempts - 1}): "
                f"{type(exc).__name__}: {exc}[/yellow]")
            time.sleep(30 * attempt)
    raise RuntimeError("unreachable")  # pragma: no cover


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TCS-Bench Stage 2: judge all saved candidate proofs (A.1, 4 calls, >=3/4)")
    parser.add_argument("--env-file", "-e", type=Path, default=None,
                        help="Path to .env configuration file to load (default: .env)")
    parser.add_argument("--prove-dir", "--solve-dir", dest="solve_dir", type=Path, required=True,
                        help="Stage 1 prove/solve run dir (data/eval_results/tcs_prove_<ts> or tcs_solve_<ts>)")
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=2,
                        help="extra attempts per task on transient network errors (default 2)")
    parser.add_argument("--config", choices=["b", "tcs", "a", "ab", "lean"], default="b",
                        help="TCS Judge config: primary referee ('b' or 'tcs', default), dual ('ab'), legacy A.1 ('a'), or Lean 4 ('lean')")
    parser.add_argument("--threshold", type=float, default=TCS_PASS_THRESHOLD,
                        help=f"TCS Judge binary threshold on continuous validity "
                             f"(default: TCS_PASS_THRESHOLD={TCS_PASS_THRESHOLD:.2f})")
    parser.add_argument("--replicates", type=int, default=TCS_JUDGE_REPLICATES,
                        help=f"Independent Config-B judge calls per task; median is reported "
                             f"(default: {TCS_JUDGE_REPLICATES}). Use 1 to reproduce the legacy single-call judge.")
    parser.add_argument("--adaptive", action="store_true", default=True,
                        help="Adaptive replicates: stop after 1 call if it is outside threshold +/- 0.20 (default: True)")
    parser.add_argument("--no-adaptive", action="store_false", dest="adaptive",
                        help="Disable adaptive replicates and always run all replicates")
    parser.add_argument("--no-thinking", action="store_true", default=False,
                        help="Disable judge thinking (default: thinking is ON)")
    parser.add_argument("--enable-thinking", action="store_true", default=True,
                        help="Enable judge thinking (default: True)")
    parser.add_argument("--temp-a", type=float, default=0.3)
    parser.add_argument("--temp-b", type=float, default=0.1)
    parser.add_argument("--task-id", type=str, default=None,
                        help="Judge a single specific task ID")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=None,
                        help="per-judge-call timeout (default: model config timeout)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output JSON path (default data/eval_results/tcs_judge_<ts>.json)")
    args = parser.parse_args()
    if args.env_file:
        from core_engine.config import configure_env_file
        configure_env_file(args.env_file)

    if not (args.solve_dir / "tasks").is_dir():
        sys.exit(f"no tasks/ dir under --solve-dir: {args.solve_dir}")
    if not args.tasks_dir.is_dir():
        sys.exit(f"tasks dir not found: {args.tasks_dir}")

    cfg = load_config()
    timeout_seconds = args.timeout if args.timeout is not None else cfg.timeout_seconds
    thinking = not args.no_thinking

    candidates = [json.loads(p.read_text(encoding="utf-8"))
                  for p in sorted((args.solve_dir / "tasks").glob("*.json"))]
    # only judged what Stage 1 actually solved
    candidates = [c for c in candidates if c.get("status") == "completed"]
    if args.task_id:
        candidates = [c for c in candidates if c.get("task_id") == args.task_id or args.task_id in c.get("task_id", "")]
        if not candidates:
            sys.exit(f"task {args.task_id} not found among completed tasks in {args.solve_dir}")
    if not candidates:
        sys.exit("no completed candidate proofs found in the solve dir")
    task_map = load_task_map(args.tasks_dir)

    console.print(f"[bold blue]=== TCS-Bench Stage 2: judging {len(candidates)} "
                  f"candidate proof(s), {min(args.workers, len(candidates))} worker(s) ===[/bold blue]")
    console.print(f"Model: [bold]{cfg.model}[/bold] | thinking: [bold]{'ON' if thinking else 'OFF'}[/bold] | "
                  f"config={args.config} | temps A={args.temp_a} B={args.temp_b} | "
                  f"max_tokens={args.max_tokens} | threshold={args.threshold} | "
                  f"replicates={args.replicates}{' (adaptive)' if args.adaptive else ''}")

    client = TunnelThinkingClient()
    t0 = time.perf_counter()
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(candidates))) as ex:
        futures = {ex.submit(judge_with_retry, rec, task_map[rec["task_id"]], args, client,
                             thinking, timeout_seconds, args.retries): rec["task_id"]
                   for rec in candidates
                   if rec["task_id"] in task_map}
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                out = fut.result()
            except Exception as exc:
                out = {"task_id": tid, "status": "error",
                       "error": f"{type(exc).__name__}: {exc}"}
            results.append(out)
            a = out.get("a") or {}
            b = out.get("b")
            if out["status"] == "judged":
                a_str = f"A={a.get('ones', '?')}/4 -> {a.get('verdict', '?')}" \
                    if a else "A=-"
                b_str = (f"B={b['validity']:.2f} -> {b['verdict']} "
                         f"(reps={b.get('replicate_validities')}, P(pass)={b.get('pass_probability')})") \
                    if b and b.get("validity") is not None else "B=-"
                log(f"[green]done  {tid}[/green]  {a_str}  {b_str}")
            elif out["status"] == "judge_parse_error":
                log(f"[magenta]parse {tid}: judge reply unparseable in all replicates "
                    f"(excluded from pass-rate denominator)[/magenta]")
            else:
                log(f"[yellow]skip  {tid}: {out.get('error') or out['status']}[/yellow]")
    elapsed = round(time.perf_counter() - t0, 2)

    judged = [r for r in results if r["status"] == "judged"]
    parse_errors = [r for r in results if r["status"] == "judge_parse_error"]
    no_proof = [r for r in results if r["status"] == "no_proof"]
    errors = [r for r in results if r["status"] == "error"]
    a1 = sum(1 for r in judged if (r.get("a") or {}).get("verdict") == "1")
    a0 = sum(1 for r in judged if (r.get("a") or {}).get("verdict") == "0")
    a_inc = sum(1 for r in judged if (r.get("a") or {}).get("verdict") == "INCONCLUSIVE")
    a_denom = a1 + a0
    b_scores = [r["b"]["validity"] for r in judged
                if r.get("b") and r["b"].get("validity") is not None]
    b1 = sum(1 for r in judged
             if r.get("b") and r["b"].get("verdict") == "1")
    b_denom = len(b_scores)  # judge_parse_error records are excluded, not counted as fails
    b_pp = [r["b"]["pass_probability"] for r in judged
            if r.get("b") and r["b"].get("pass_probability") is not None]
    agreement = sum(1 for r in judged
                    if (r.get("a") or {}).get("verdict") in ("0", "1")
                    and r.get("b") and r["b"].get("verdict") in ("0", "1")
                    and (r.get("a") or {})["verdict"] == r["b"]["verdict"])
    a_votes_hist: Dict[str, int] = {}
    for r in judged:
        v = str((r.get("a") or {}).get("ones", "?"))
        a_votes_hist[v] = a_votes_hist.get(v, 0) + 1

    summary = {
        "judged": len(judged),
        "a_pass": a1,
        "a_fail": a0,
        "a_inconclusive": a_inc,
        "a_pass_rate": round(a1 / a_denom, 4) if a_denom else None,
        "a_votes_histogram": a_votes_hist,
        "b_pass": b1,
        "b_judged": b_denom,
        "b_pass_rate": round(b1 / b_denom, 4) if b_denom else None,
        "b_threshold": args.threshold,
        "b_replicates": args.replicates,
        "b_mean_validity": round(sum(b_scores) / len(b_scores), 4) if b_scores else None,
        "b_expected_pass_rate": round(sum(b_pp) / len(b_pp), 4) if b_pp else None,
        "judge_parse_errors": len(parse_errors),
        "a_b_agreement": agreement,
        "no_proof": len(no_proof),
        "errors": len(errors),
    }

    table = Table(title="TCS-Bench Stage 2 (A.1 >=3/4) aggregate")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("judged", str(len(judged)))
    table.add_row("A pass (>=3/4)", f"{a1} / {a_denom}" +
                  (f"  ({a1 / a_denom:.1%})" if a_denom else ""))
    table.add_row("A inconclusive", str(a_inc))
    if b_scores:
        table.add_row(f"B pass (median of {args.replicates} >= {args.threshold:.2f})",
                      f"{b1} / {b_denom}" + (f"  ({b1 / b_denom:.1%})" if b_denom else ""))
        table.add_row("B mean validity", str(summary["b_mean_validity"]))
        table.add_row("B expected pass rate (mean P(pass))", str(summary["b_expected_pass_rate"]))
    table.add_row("judge parse errors (excluded)", str(len(parse_errors)))
    if a_denom:
        table.add_row("A/B agreement", str(agreement))
    table.add_row("no proof produced", str(len(no_proof)))
    table.add_row("errors", str(len(errors)))
    table.add_row("elapsed (s)", str(elapsed))
    console.print(table)

    out_file = args.out or (EVAL_DIR / f"tcs_judge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    payload = {
        "meta": {
            "stage": "tcs_bench_judge",
            "solve_dir": str(args.solve_dir),
            "tasks_dir": str(args.tasks_dir),
            "args": vars(args),
            "judge_model": cfg.model,
            "thinking": thinking,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": elapsed,
        },
        "summary": summary,
        "results": results,
    }
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    console.print(f"\n[green]Saved full results (incl. all raw judge outputs) to {out_file}[/green]")


if __name__ == "__main__":
    main()
