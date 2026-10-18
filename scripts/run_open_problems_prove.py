#!/usr/bin/env python3
"""Autonomous Prover and Judge Pipeline for the 50 Open Mathematical Problems.

Executes autonomous mathematical proving on open research problems (from Epoch AI
and Scientific American, both FrontierMath 50 and Erdős Lean 4 formal conjectures).
Unlike closed benchmarks, literature search is ENABLED by default.
Already-solved problems are EXCLUDED by default to focus strictly on open mathematics.
Includes real-time Telegram notifications per problem after evaluation.

Usage:
  # Prove open FrontierMath problems (excludes 9 solved, 41 open, 8 iterations, max 1 hr):
  .venv/bin/python scripts/run_open_problems_prove.py --limit 5 --workers 2

  # Prove a specific problem with literature search:
  .venv/bin/python scripts/run_open_problems_prove.py --problem-id baillie-psw

  # Prove Lean 4 formal conjectures:
  .venv/bin/python scripts/run_open_problems_prove.py --domain lean --limit 5 --workers 2

  # Quick smoke test to verify model + Lean 4 compiler judge + Telegram:
  .venv/bin/python scripts/run_open_problems_prove.py --smoke-test
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

# Setup Prime Math path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_engine.judging import extract_lean_code, judge_lean_proof
from core_engine.llm_client import TunnelThinkingClient
from core_engine.prover import MathProver
from core_engine.telegram import load_telegram_credentials, send_telegram_message

console = Console()
print_lock = threading.Lock()
stats_lock = threading.Lock()

DEFAULT_OPEN_PROBLEMS_DIR = PROJECT_ROOT / "data" / "open_problems"
DEFAULT_EVAL_DIR = PROJECT_ROOT / "data" / "eval_results"


def log(msg: str) -> None:
    with print_lock:
        console.print(msg)


def load_frontiermath50_problems(
    problems_dir: Path,
    limit: Optional[int] = None,
    problem_id: Optional[str] = None,
    problem_range: Optional[str] = None,
    exclude_solved: bool = True,
) -> List[Dict[str, Any]]:
    """Loads FrontierMath 50 open problems, excluding solved ones by default."""
    json_path = problems_dir / "problems.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"FrontierMath 50 JSON not found at {json_path}")

    data: List[Dict[str, Any]] = json.loads(json_path.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for entry in data:
        is_solved = bool(entry.get("solved", False))
        pid = entry.get("problem_id", str(entry.get("id")))

        if exclude_solved and is_solved:
            skipped.append(f"{pid} ({entry.get('title')})")
            continue

        prompt = entry.get("prompt", "")
        # Fallback to tasks/ markdown file if prompt in JSON is empty
        if not prompt.strip():
            idx = entry.get("id", 1)
            task_files = list((problems_dir / "tasks").glob(f"{idx:02d}_*.md"))
            if task_files:
                prompt = task_files[0].read_text(encoding="utf-8")

        items.append({
            "index": entry.get("id"),
            "id": pid,
            "title": entry.get("title", pid),
            "field": entry.get("field", ""),
            "prompt": prompt,
            "solved": is_solved,
            "solved_date": entry.get("solved_date"),
            "type": "frontiermath50",
            "notability": entry.get("notability", ""),
            "tags": entry.get("tags", []),
        })

    if exclude_solved and skipped:
        log(f"[dim]Excluded {len(skipped)} already-solved problems from open run ({len(items)} unsolved remain).[/dim]")

    if problem_id:
        items = [p for p in items if p["id"] == problem_id or str(p["index"]) == problem_id]
        if not items:
            raise ValueError(f"Problem ID {problem_id} not found in FrontierMath 50 (check if excluded via --exclude-solved).")
        return items

    if problem_range:
        parts = problem_range.split("-")
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            items = [p for p in items if start <= p["index"] <= end]

    if limit is not None:
        items = items[:limit]

    return items


def load_lean_problems(
    lean_dir: Path,
    limit: Optional[int] = None,
    problem_id: Optional[str] = None,
    problem_range: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Loads Lean 4 open problem items from manifest.json and .lean files."""
    manifest_path = lean_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Lean open problems manifest not found at {manifest_path}")

    manifest: List[Dict[str, Any]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = []

    for entry in manifest:
        pid = entry["id"]
        file_path = lean_dir / entry["file"]
        if not file_path.is_file():
            continue

        spec_content = file_path.read_text(encoding="utf-8")
        items.append({
            "index": entry["index"],
            "id": pid,
            "title": f"Erdős Problem {pid.split('.')[0].replace('Erdos', '')}",
            "file": str(file_path),
            "disproof": entry.get("disproof", f"{pid}.disproof"),
            "spec": spec_content,
            "type": "lean",
        })

    if problem_id:
        items = [p for p in items if p["id"] == problem_id or str(p["index"]) == problem_id]
        if not items:
            raise ValueError(f"Problem ID {problem_id} not found in Lean manifest.")
        return items

    if problem_range:
        parts = problem_range.split("-")
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            items = [p for p in items if start <= p["index"] <= end]

    if limit is not None:
        items = items[:limit]

    return items


def format_telegram_alert(
    task_info: Dict[str, Any],
    verdict: Dict[str, Any],
    duration: float,
    current_idx: int,
    total_tasks: int,
    passed_count: int,
) -> str:
    """Formats an HTML notification for Telegram."""
    status = verdict.get("status", "UNKNOWN")
    passed = verdict.get("passed", False)
    ptype = task_info.get("type", "frontiermath50")

    if passed:
        icon = "🎉 <b>[PROVEN / RESOLVED]</b>"
    elif status == "SORRY_DETECTED":
        icon = "⚠️ <b>[SORRY SHORTCUT DETECTED]</b>"
    elif status == "TIMEOUT":
        icon = "⏳ <b>[TIMEOUT LIMIT REACHED]</b>"
    elif status == "UNRESOLVED":
        icon = "🔬 <b>[EXPLORED - UNRESOLVED]</b>"
    else:
        icon = "❌ <b>[VERIFICATION FAILED]</b>"

    pid = html.escape(str(task_info.get("id", "Unknown")))
    title = html.escape(str(task_info.get("title", pid)))
    exec_t = round(duration, 2)
    iters = verdict.get("iterations", 1)
    pass_pct = round((passed_count / max(1, current_idx)) * 100, 1)

    diag = ""
    errors = verdict.get("errors", [])
    if errors:
        snippet = html.escape("\n".join(errors[:3]))
        diag = f"\n<pre>{snippet}</pre>"

    msg = (
        f"{icon}\n"
        f"<b>Problem:</b> {title} (<code>{pid}</code>)\n"
        f"<b>Type:</b> {ptype} | <b>Turns:</b> {iters} | <b>Time:</b> {exec_t}s\n"
        f"<b>Status:</b> {status}\n"
        f"<b>Progress:</b> {current_idx}/{total_tasks} ({pass_pct}% Pass Rate: {passed_count} Resolved)\n"
        f"{diag}"
    )
    return msg


def solve_and_judge_one(
    problem: Dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Autonomous prove + judge cycle for one problem (Lean or FrontierMath)."""
    pid = problem["id"]
    ptype = problem.get("type", "frontiermath50")
    out_file = run_dir / "tasks" / f"{pid}.json"

    log(f"[bold cyan]▶ Starting Problem {problem['index']}: {problem['title']} ({pid})[/bold cyan]")
    start_time = time.perf_counter()

    prover = MathProver(
        default_max_iterations=args.max_iterations,
        enable_literature_search=args.literature_search,
        proof_mode=getattr(args, "proof_mode", "auto"),
    )

    if ptype == "lean":
        spec = problem["spec"]
        prompt = f"""You are an elite research mathematician and expert Lean 4 theorem prover.
You are tasked with resolving the following open mathematical conjecture formalized in Lean 4.

### Lean 4 Formal Specification:
```lean
{spec}
```

### Prover Instructions:
1. You may prove either the main conjecture theorem OR its disproof theorem (`{problem.get('disproof', '')}`).
2. Conduct any necessary background mathematical analysis and literature reasoning first.
3. Output the complete, self-contained, working Lean 4 formal proof strictly inside a ```lean ... ``` code block.
4. DO NOT use `sorry` or `admit`. All goals must be closed completely.
5. Ensure all necessary imports and namespaces from the specification are preserved.
"""
        client = TunnelThinkingClient()
        proof_code = ""
        error_msg = None
        try:
            if args.literature_search:
                log(f"  [magenta]🔍 Harvesting literature for {problem['title']}...[/magenta]")
                harvest = prover._harvest_literature(f"{problem['title']} {pid}")
                if harvest:
                    prompt += f"\n\n### Prior Literature & Context Harvest:\n{harvest[:4000]}\n"

            response = client.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=8192,
            )
            proof_code = response.content
        except Exception as exc:
            error_msg = str(exc)
            log(f"  [red]Prover generation error for {pid}: {exc}[/red]")

        if proof_code:
            verdict = judge_lean_proof(proof_code, timeout=args.timeout)
        else:
            verdict = {
                "verdict": "0",
                "passed": False,
                "status": "ERROR",
                "errors": [error_msg or "Empty proof generated"],
                "execution_time": 0.0,
            }
        is_pass = verdict.get("passed", False)
        status = verdict.get("status", "UNKNOWN")
        candidate = proof_code
        iterations_used = 1

    else:
        # FrontierMath 50 problem: Full 8-iteration closed-loop solver
        task_prompt = problem["prompt"]
        ckpt_path = run_dir / "checkpoints" / f"{pid}.ckpt.json"
        solve_res = prover.solve(
            task_id=pid,
            task_prompt=task_prompt,
            max_iterations=args.max_iterations,
            max_time_seconds=args.max_time,
            enable_literature_search=args.literature_search,
            checkpoint_path=ckpt_path,
        )
        is_pass = solve_res.verified
        candidate = solve_res.candidate_proof or ""
        duration = solve_res.duration_s
        iterations_used = solve_res.iterations

        status = "SOLVED" if is_pass else ("TIMEOUT" if (args.max_time and duration >= args.max_time) else "UNRESOLVED")
        verdict = {
            "verdict": "1" if is_pass else "0",
            "passed": is_pass,
            "status": status,
            "iterations": iterations_used,
            "error": solve_res.error,
            "execution_time": duration,
        }

        # If candidate contains Lean 4 code, judge it with Lean 4 compiler
        extracted_lean = extract_lean_code(candidate)
        if extracted_lean:
            lean_verdict = judge_lean_proof(extracted_lean, timeout=args.timeout)
            verdict["lean_judge"] = lean_verdict
            if lean_verdict.get("passed"):
                is_pass = True
                status = "LEAN_PROVEN"
                verdict["status"] = status
                verdict["passed"] = True

    duration = time.perf_counter() - start_time
    status_color = "green" if is_pass else ("yellow" if status in ("UNRESOLVED", "SORRY_DETECTED") else "red")
    log(f"[{status_color}]■ Finished {pid}: Status={status} ({iterations_used} iters, Elapsed: {round(duration, 1)}s)[/{status_color}]")

    result_record = {
        "problem_id": pid,
        "title": problem["title"],
        "index": problem["index"],
        "type": ptype,
        "duration_s": duration,
        "iterations": iterations_used,
        "passed": is_pass,
        "status": status,
        "verdict": verdict,
        "candidate_preview": candidate[:1000] if candidate else "",
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result_record, indent=2), encoding="utf-8")

    with stats_lock:
        state["completed"] += 1
        if is_pass:
            state["passed"] += 1
        current_completed = state["completed"]
        current_passed = state["passed"]

    if state.get("tg_token") and state.get("tg_chat_id"):
        try:
            tg_text = format_telegram_alert(
                problem,
                verdict,
                duration,
                current_completed,
                state["total"],
                current_passed,
            )
            send_telegram_message(state["tg_token"], state["tg_chat_id"], tg_text)
        except Exception as tg_err:
            log(f"  [yellow]Telegram alert failed: {tg_err}[/yellow]")

    return result_record


def run_smoke_test(args: argparse.Namespace) -> None:
    """Executes a fast end-to-end smoke test on a simple Lean 4 theorem."""
    log("[bold blue]=== Starting Lean 4 Prover & Judge Smoke Test ===[/bold blue]")
    client = TunnelThinkingClient()
    prompt = """State and prove a theorem named `my_and_swap` in Lean 4 that for any propositions P and Q, P ∧ Q → Q ∧ P.
Write only the working Lean 4 code inside a ```lean code block."""

    log("1. Generating candidate Lean proof via remote thinking model...")
    t0 = time.perf_counter()
    resp = client.complete(messages=[{"role": "user", "content": prompt}], temperature=0.1)
    gen_time = round(time.perf_counter() - t0, 2)
    log(f"   Generated in {gen_time}s:\n{resp.content.strip()}")

    log("2. Compiling and verifying via Lean 4 Compiler Judge...")
    verdict = judge_lean_proof(resp.content, timeout=30.0)
    log(f"   Verdict: {verdict['verdict']} | Passed: {verdict['passed']} | Status: {verdict['status']}")
    log(f"   Lean Compiler Output: {verdict['compiler_output'] or '[Clean 0-error exit]'}")

    if verdict["passed"]:
        log("[bold green]✔ Smoke Test Succeeded! Pipeline is 100% operational.[/bold green]")
        if args.telegram:
            try:
                token, chat_id = load_telegram_credentials()
                msg = (
                    "🚀 <b>[PRIME MATH - LEAN 4 SMOKE TEST PASSED]</b>\n"
                    "End-to-end Lean prover and compiler judge verified successfully.\n"
                    f"<b>Proof:</b> <code>P ∧ Q → Q ∧ P</code>\n"
                    f"<b>Status:</b> {verdict['status']} ({verdict['execution_time']}s)"
                )
                send_telegram_message(token, chat_id, msg)
                log("[green]Telegram notification sent.[/green]")
            except Exception as e:
                log(f"[yellow]Telegram notification skipped: {e}[/yellow]")
    else:
        log("[bold red]✘ Smoke Test Failed verification.[/bold red]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous Prover for the 50 Open Mathematical Problems")
    parser.add_argument("--domain", choices=["frontiermath50", "lean"], default="frontiermath50",
                        help="Problem domain: 'frontiermath50' (default) or 'lean'")
    parser.add_argument("--problems-dir", type=Path, default=None,
                        help="Custom problems directory")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of problems")
    parser.add_argument("--problem-id", type=str, default=None,
                        help="Specific problem ID or index (e.g. 'baillie-psw' or 'Erdos1.erdos_1')")
    parser.add_argument("--problem-range", type=str, default=None,
                        help="Range of problem indices (e.g. 1-10)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent solver workers (default: 2)")
    parser.add_argument("--literature-search", action="store_true", default=True,
                        help="Enable prior literature search (default: True)")
    parser.add_argument("--no-literature-search", action="store_false", dest="literature_search")
    parser.add_argument("--exclude-solved", action="store_true", default=True,
                        help="Exclude already-solved problems from the run (default: True)")
    parser.add_argument("--include-solved", action="store_false", dest="exclude_solved",
                        help="Include already-solved problems")
    parser.add_argument("--max-iterations", type=int, default=8,
                        help="Max solver turns per problem (default: 8)")
    parser.add_argument("--max-time", type=float, default=None,
                        help="Optional maximum execution time per problem in seconds (default: None, full iterations)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Lean compiler verification timeout in seconds (default: 60s)")
    parser.add_argument("--proof-mode", choices=["auto", "tcs", "computational"], default="auto",
                        help="Proof auditor mode: 'auto' (default), 'tcs', or 'computational'")
    parser.add_argument("--telegram", action="store_true", default=True,
                        help="Send live Telegram alerts per problem (default: True)")
    parser.add_argument("--no-telegram", action="store_false", dest="telegram")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a quick smoke test on an easy theorem")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Output evaluation directory")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test(args)
        return 0

    if args.domain == "frontiermath50":
        pdir = args.problems_dir or (DEFAULT_OPEN_PROBLEMS_DIR / "frontiermath50")
        problems = load_frontiermath50_problems(
            pdir,
            limit=args.limit,
            problem_id=args.problem_id,
            problem_range=args.problem_range,
            exclude_solved=args.exclude_solved,
        )
    else:
        pdir = args.problems_dir or (DEFAULT_OPEN_PROBLEMS_DIR / "lean")
        problems = load_lean_problems(
            pdir,
            limit=args.limit,
            problem_id=args.problem_id,
            problem_range=args.problem_range,
        )

    if not problems:
        log("[red]No problems selected to prove.[/red]")
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (DEFAULT_EVAL_DIR / f"open_problems_{args.domain}_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Telegram configuration
    tg_token, tg_chat_id = None, None
    if args.telegram:
        try:
            tg_token, tg_chat_id = load_telegram_credentials()
            log(f"[green]Telegram reporting enabled for chat {tg_chat_id[:4]}****[/green]")
        except Exception as e:
            log(f"[yellow]Telegram disabled: {e}[/yellow]")

    state = {
        "completed": 0,
        "passed": 0,
        "total": len(problems),
        "tg_token": tg_token,
        "tg_chat_id": tg_chat_id,
    }

    time_str = f"{args.max_time}s" if args.max_time else "Uncapped (full iterations)"
    log(f"[bold blue]=== Starting Open Problems Pipeline: {len(problems)} problem(s), {min(args.workers, len(problems))} worker(s) ===[/bold blue]")
    log(f"Domain: [cyan]{args.domain}[/cyan] | Max Turns: [cyan]{args.max_iterations}[/cyan] | Time Budget: [cyan]{time_str}[/cyan]")
    log(f"Run directory: [dim]{run_dir}[/dim]")
    log(f"Literature Search: [{'green' if args.literature_search else 'yellow'}]{args.literature_search}[/]")
    log(f"Exclude Solved: [{'green' if args.exclude_solved else 'yellow'}]{args.exclude_solved}[/]")

    results = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(problems))) as executor:
        futures = {executor.submit(solve_and_judge_one, p, args, run_dir, state): p for p in problems}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)

    passed_count = sum(1 for r in results if r.get("passed"))
    pass_rate = (passed_count / len(results)) * 100 if results else 0.0

    summary = {
        "domain": args.domain,
        "total": len(results),
        "passed": passed_count,
        "pass_rate_pct": round(pass_rate, 2),
        "literature_search": args.literature_search,
        "exclude_solved": args.exclude_solved,
        "max_iterations": args.max_iterations,
        "max_time_s": args.max_time,
        "run_dir": str(run_dir),
        "timestamp": ts,
        "results": [
            {
                "id": r["problem_id"],
                "passed": r["passed"],
                "status": r["status"],
                "iterations": r.get("iterations", 1),
                "duration_s": round(r["duration_s"], 2),
            }
            for r in results
        ],
    }

    summary_file = run_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log(f"\n[bold green]=== Run Finished: {passed_count}/{len(results)} Passed ({round(pass_rate, 1)}%) ===[/bold green]")
    log(f"Summary written to: {summary_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
