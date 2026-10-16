#!/usr/bin/env python3
"""TCS-Bench leakage / contamination audit -- produces EVIDENCE that the solver never
saw the ground-truth proof. Requires no changes to the solver script.

Architectural guarantee (enforced by run_tcs_bench_prove.py):
  * the prover process receives ONLY the task `prompt`; the `ground_truth_solution`
    field is stripped before the prover is touched;
  * Stage 0 (web research) is disabled -- no arXiv/Wikipedia lookups.
This audit verifies that guarantee empirically:

pre  -- BEFORE the run: query TurboVec procedural memory with each task's target
       statement. The vector store legitimately returns *related* heuristics (learned
       from earlier runs) for most math queries, so "any hit" is NOT the criterion.
       The criterion is verbatim-content overlap: a returned entry is flagged when its
       12-grams overlap the GT proof (>=0.05) or its 8-grams overlap the target
       statement (>=0.20), or when it mentions the source arXiv id. Note the write-side
       guard in the harness itself: add_procedural_heuristic() REJECTS non-universal
       content, so task-specific proof material cannot normally enter the collection.
post -- AFTER Stage 1: for every candidate proof, measure 12-gram containment
       against the withheld GT proof, and scan the task log for evidence of external
       access (arXiv id, arXiv-URL host, long GT substrings). Also counts REPL web-tool
       call attempts in the log (search_math( / read_paper( / oeis_lookup() -- with the
       TCSCompliantKernel these hit protocol stubs, so attempts alone are NOT a leak;
       actual arXiv content in the log is). Also reports same-source-paper task groups:
       solve() always consolidates into the SHARED TurboVec collection (no disable flag
       in the harness), so tasks from one paper share memory exposure within a run.

Interpretation: containment above the threshold on a task whose candidate is a
faithful proof of the same lemma is EXPECTED (shared lemma statement / context
definitions) -- read the distribution, and investigate the top of the table.
Zero log hits on arXiv ids is the clean signal that Stage 0 never fired.

Usage:
  .venv/bin/python scripts/tcs_bench_leakage_audit.py pre
  .venv/bin/python scripts/tcs_bench_leakage_audit.py post --solve-dir data/eval_results/tcs_solve_<ts>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

_LOCAL_TCS_TASKS = PROJECT_ROOT / "data" / "tcs" / "tasks"
_EXTERNAL_TCS_TASKS = PROJECT_ROOT.parent / "TCSBench" / "tasks"
DEFAULT_TASKS_DIR = _LOCAL_TCS_TASKS if _LOCAL_TCS_TASKS.is_dir() else _EXTERNAL_TCS_TASKS
console = Console()

LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")
NON_ALNUM_RE = re.compile(r"[^0-9a-zA-Z]+")
TARGET_RE = re.compile(r"Target Problem \(your task\):\s*(.+)$", re.S)


def load_tasks(tasks_dir: Path) -> List[Dict[str, Any]]:
    tasks = []
    for path in sorted(tasks_dir.glob("task_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        tasks.append({"task_id": path.stem, "arxiv_id": data.get("arxiv_id"),
                      "prompt": data.get("prompt", ""),
                      "gt": data.get("ground_truth_solution", "")})
    return tasks


def extract_target(prompt: str) -> str:
    m = TARGET_RE.search(prompt)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return prompt[-1200:].strip()


def norm_tokens(text: str) -> List[str]:
    text = LATEX_CMD_RE.sub(" ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    return [t for t in text.lower().split() if len(t) > 2]


def shingles(tokens: List[str], k: int) -> set:
    return {tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def containment(cand: str, gt: str, k: int) -> float:
    """Fraction of candidate k-grams that also appear in the GT."""
    S = shingles(norm_tokens(cand), k)
    if not S:
        return 0.0
    T = shingles(norm_tokens(gt), k)
    return len(S & T) / len(S)


def pre_audit(tasks: List[Dict[str, Any]], sample: int | None) -> int:
    from core_engine.memory import CoALAMemoryManager
    mem = CoALAMemoryManager()
    mem.check_health()
    if sample:
        tasks = tasks[:sample]
    console.print(f"[bold]Pre-run audit: querying TurboVec procedural memory with "
                  f"{len(tasks)} target statements (min_score=0.50)...[/bold]")
    console.print("[dim]Criterion: content overlap of returned entries with the GT proof "
                  "or target statement (not mere recall of related heuristics).[/dim]")
    related: int = 0
    flagged: List[Dict[str, Any]] = []
    for i, task in enumerate(tasks, 1):
        query = extract_target(task["prompt"])
        rules: List[str] = mem.query_procedural_heuristics(query, limit=3, min_score=0.50) or []
        if rules:
            related += 1
        reasons: List[str] = []
        for rule in rules:
            gt_cont = containment(rule, task["gt"], 12)
            tgt_cont = containment(rule, query, 8)
            arxiv = str(task.get("arxiv_id") or "")
            if gt_cont >= 0.05:
                reasons.append(f"GT 12-gram overlap {gt_cont:.2f}")
            if tgt_cont >= 0.20:
                reasons.append(f"target 8-gram overlap {tgt_cont:.2f}")
            if arxiv and arxiv in rule:
                reasons.append(f"arxiv id {arxiv} in entry")
        if reasons:
            flagged.append({"task_id": task["task_id"], "arxiv_id": task["arxiv_id"],
                            "reasons": reasons,
                            "entries": [r[:300] for r in rules]})
            console.print(f"[red]FLAGGED {task['task_id']} ({task['arxiv_id']}): "
                          f"{'; '.join(reasons)}[/red]")
        if i % 50 == 0:
            console.print(f"[dim]...{i}/{len(tasks)} queried[/dim]")
    table = Table(title=f"Pre-run memory audit: {len(flagged)} contaminated task(s), "
                        f"{related} task(s) with merely-related heuristics (expected)")
    table.add_column("task")
    table.add_column("arxiv_id")
    table.add_column("reasons")
    for f in flagged:
        table.add_row(f["task_id"], str(f["arxiv_id"]), "; ".join(f["reasons"]))
    if not flagged:
        console.print("[green]No verbatim TCS task/GT material found in shared memory -- clean.[/green]")
    else:
        console.print(table)
    return 2 if flagged else 0


def post_audit(solve_dir: Path, tasks_dir: Path, shingle: int, threshold: float) -> int:
    tasks = {t["task_id"]: t for t in load_tasks(tasks_dir)}
    cand_dir = solve_dir / "tasks"
    if not cand_dir.is_dir():
        sys.exit(f"no tasks/ dir under --solve-dir: {solve_dir}")
    rows: List[Dict[str, Any]] = []
    flagged = 0
    for path in sorted(cand_dir.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        tid = rec.get("task_id", path.stem)
        task = tasks.get(tid)
        row: Dict[str, Any] = {"task_id": tid, "arxiv_id": rec.get("arxiv_id")}
        if rec.get("status") != "completed" or not task:
            row["note"] = "not completed / task json missing -- skipped"
            rows.append(row)
            continue
        cand = rec.get("candidate_proof") or "\n\n".join(rec.get("reasoning_traces") or [])
        row["containment"] = round(containment(cand, task["gt"], shingle), 4)
        row["flagged"] = row["containment"] >= threshold
        flagged += row["flagged"]

        # Log scan: source-paper id + arXiv-URL host + long raw GT substrings inside the
        # harness log; plus a count of REPL web-tool call attempts (informational --
        # with TCSCompliantKernel these hit protocol stubs, so attempts alone are safe).
        log_hits: List[str] = []
        web_tool_attempts = 0
        log_file = Path(rec.get("log_file", "")) if rec.get("log_file") else None
        if log_file and log_file.is_file():
            log_text = log_file.read_text(encoding="utf-8", errors="replace")
            arxiv = str(task.get("arxiv_id") or "")
            if arxiv and arxiv in log_text:
                log_hits.append(f"arxiv_id {arxiv} present in log")
            if re.search(r"arxiv\.org", log_text):
                log_hits.append("arxiv.org URL present in log")
            web_tool_attempts = len(re.findall(
                r"\b(search_math|read_paper|oeis_lookup)\s*\(", log_text))
            gt_raw = task["gt"]
            for i in range(0, max(1, len(gt_raw) - 79), 200):
                chunk = gt_raw[i:i + 80]
                if len(chunk) >= 40 and chunk in log_text:
                    log_hits.append(f"GT substring [{i}:{i + 80}] present in log")
                    break
        row["log_hits"] = log_hits
        row["web_tool_call_attempts"] = web_tool_attempts
        rows.append(row)

    # Same-source-paper grouping (shared TurboVec exposure within the run)
    paper_groups = Counter(str(r.get("arxiv_id")) for r in rows if r.get("arxiv_id"))
    multi = {k: v for k, v in paper_groups.items() if v > 1}

    table = Table(title=f"Post-run leakage audit: {flagged} task(s) at/above containment {threshold}")
    table.add_column("task", max_width=52)
    table.add_column("arxiv_id")
    table.add_column("containment", justify="right")
    table.add_column("flag")
    table.add_column("log hits")
    for r in sorted(rows, key=lambda x: -x.get("containment", 0)):
        if "containment" not in r:
            continue
        table.add_row(
            r["task_id"], str(r.get("arxiv_id")), f"{r['containment']:.4f}",
            "[red]YES[/red]" if r["flagged"] else "no",
            "; ".join(r["log_hits"]) if r["log_hits"] else "-",
        )
    console.print(table)
    if multi:
        groups_str = ", ".join(f"{k} x{v}" for k, v in sorted(multi.items(), key=lambda x: -x[1]))
        console.print(f"[yellow]Same-source-paper groups (shared TurboVec exposure in-run): {groups_str}[/yellow]")
    any_log_hits = any(r.get("log_hits") for r in rows)
    if any_log_hits:
        console.print("[red]LOG HIT(S): GT/arXiv content appeared in solver logs -- investigate.[/red]")

    report = {
        "solve_dir": str(solve_dir),
        "shingle_k": shingle,
        "threshold": threshold,
        "flagged": flagged,
        "any_log_hits": any_log_hits,
        "same_paper_groups": multi,
        "rows": rows,
    }
    out = solve_dir / "leakage_audit.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    console.print(f"[green]Audit report saved to {out}[/green]")
    return 2 if (flagged or any_log_hits) else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="TCS-Bench leakage / contamination audit")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_pre = sub.add_parser("pre", help="pre-run TurboVec memory contamination check")
    p_pre.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    p_pre.add_argument("--sample", type=int, default=None,
                       help="query only the first N tasks")
    p_post = sub.add_parser("post", help="post-run candidate-vs-GT + log audit")
    p_post.add_argument("--prove-dir", "--solve-dir", dest="prove_dir", type=Path, required=True,
                        help="Directory containing proof/solve run outputs")
    p_post.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    p_post.add_argument("--shingle", type=int, default=12,
                        help="n-gram size for containment (default 12)")
    p_post.add_argument("--threshold", type=float, default=0.20,
                        help="containment flag threshold (default 0.20)")
    args = parser.parse_args()

    if not args.tasks_dir.is_dir():
        sys.exit(f"tasks dir not found: {args.tasks_dir}")
    if args.cmd == "pre":
        sys.exit(pre_audit(load_tasks(args.tasks_dir), args.sample))
    sys.exit(post_audit(args.prove_dir, args.tasks_dir, args.shingle, args.threshold))


if __name__ == "__main__":
    main()
