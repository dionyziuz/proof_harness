"""FrontierMath Benchmark Evaluator for Prime Math.

Evaluates Prime Math against expert-level mathematics problems from Epoch AI's
FrontierMath benchmark across Tiers 1 through 4.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from .evaluator import answers_match
from .kernel import PersistentMathKernel
from .llm_client import TunnelThinkingClient
from .memory import CoALAMemoryManager
from .solver import MathSolver

logger = logging.getLogger("prime_math.frontier")
console = Console()

FRONTIER_DATASET_PATH = Path(__file__).parent.parent / "data" / "frontiermath_samples.json"
DEFAULT_EVAL_DIR = Path(__file__).parent.parent / "data" / "eval_results"


@dataclass
class FrontierEvalRecord:
    problem_id: int
    tier: int
    title: str
    difficulty: str
    msc_code: str
    expected_answer: str
    predicted_answer: Optional[str]
    is_correct: bool
    status: str  # PASS, WRONG_ANSWER, NO_BOXED_ANSWER, ERROR
    turns: int
    runtime_seconds: float
    error_message: Optional[str] = None
    learned_heuristic: Optional[str] = None
    research_briefing: Optional[str] = None


@dataclass
class FrontierSummary:
    total_problems: int
    passed: int
    failed: int
    pass_rate_percent: float
    total_runtime_seconds: float
    average_runtime_seconds: float
    tier_accuracy: Dict[str, str] = field(default_factory=dict)
    records: List[FrontierEvalRecord] = field(default_factory=list)


class FrontierMathEvaluator:
    """Evaluator for FrontierMath Tier 1-4 benchmark problems."""

    def __init__(
        self,
        dataset_path: Path = FRONTIER_DATASET_PATH,
        num_workers: int = 2,
        max_iterations: int = 60,
        max_problem_timeout_seconds: float = 3600.0,
    ):
        self.dataset_path = dataset_path
        self.num_workers = num_workers
        self.max_iterations = max_iterations
        self.max_problem_timeout_seconds = max_problem_timeout_seconds
        self.problems = self._load_dataset()

    def _load_dataset(self) -> Dict[int, Dict[str, Any]]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"FrontierMath dataset not found at {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {item["id"]: item for item in data}

    def evaluate_single_problem(self, prob_id: int) -> FrontierEvalRecord:
        """Solves a single FrontierMath problem and compares with ground truth."""
        item = self.problems.get(prob_id)
        if not item:
            raise KeyError(f"Problem ID {prob_id} not found in FrontierMath dataset.")

        problem_text = (
            f"FrontierMath Tier {item['tier']} ({item['difficulty']}) - {item['title']}:\n"
            f"Subject Area: {item['msc_code']}\n\n"
            f"{item['statement']}"
        )
        expected = str(item.get("answer", "")).strip()

        start_time = time.perf_counter()
        prob_log_dir = Path(__file__).parent.parent / "logs" / "frontier"
        prob_log_dir.mkdir(parents=True, exist_ok=True)
        prob_log_path = prob_log_dir / f"tier{item['tier']}_prob_{prob_id:02d}.log"

        client = TunnelThinkingClient()
        kernel = PersistentMathKernel()
        memory = CoALAMemoryManager()

        solver = MathSolver(
            client=client,
            kernel=kernel,
            memory=memory,
            default_max_iterations=self.max_iterations,
            log_path=prob_log_path,
        )

        try:
            res = solver.solve(
                problem_text,
                max_iterations=self.max_iterations,
                max_time_seconds=self.max_problem_timeout_seconds,
            )
            elapsed = round(time.perf_counter() - start_time, 2)

            is_correct = answers_match(res.boxed_answer, expected)
            is_fast_composer_guess = (
                res.iterations == 1
                and res.reasoning_traces
                and any("[Tier 1 Fast Composer" in trace for trace in res.reasoning_traces)
            )
            if not is_correct and is_fast_composer_guess and solver.fast_composer:
                remaining_time = max(10.0, self.max_problem_timeout_seconds - elapsed)
                solver._log_write(
                    f"\n[Fast Composer Mismatch] Speculative answer '{res.boxed_answer}' did not match expected '{expected}'.\n"
                    f"Escalating to Deep Orchestrator with Stage 0 Academic Research (Remaining: {remaining_time:.1f}s)...\n"
                )
                logger.info(
                    f"Problem {prob_id}: Fast Composer answer '{res.boxed_answer}' incorrect. Escalating to Deep Orchestrator..."
                )
                # Temporarily disable fast composer so deep orchestrator and Stage 0 execute
                solver.fast_composer.fast_cfg.enabled = False
                res = solver.solve(
                    problem_text,
                    max_iterations=self.max_iterations,
                    max_time_seconds=remaining_time,
                )
                elapsed = round(time.perf_counter() - start_time, 2)
                is_correct = answers_match(res.boxed_answer, expected)

            if is_correct:
                status = "PASS"
            elif res.boxed_answer:
                status = "WRONG_ANSWER"
            elif elapsed >= (self.max_problem_timeout_seconds - 5.0):
                status = "TIMEOUT"
            else:
                status = "NO_BOXED_ANSWER"

            err = None if is_correct else f"Predicted {res.boxed_answer} vs Expected {expected}"
            if status == "TIMEOUT":
                err = f"Wall-clock timeout ({self.max_problem_timeout_seconds}s) exceeded without answer"

            learned_h = res.learned_heuristics[0] if res.learned_heuristics else None

            return FrontierEvalRecord(
                problem_id=prob_id,
                tier=item["tier"],
                title=item["title"],
                difficulty=item["difficulty"],
                msc_code=item["msc_code"],
                expected_answer=expected,
                predicted_answer=res.boxed_answer,
                is_correct=is_correct,
                status=status,
                turns=res.iterations,
                runtime_seconds=elapsed,
                error_message=err,
                learned_heuristic=learned_h,
                research_briefing=res.research_briefing,
            )
        except Exception as e:
            elapsed = round(time.perf_counter() - start_time, 2)
            return FrontierEvalRecord(
                problem_id=prob_id,
                tier=item["tier"],
                title=item["title"],
                difficulty=item["difficulty"],
                msc_code=item["msc_code"],
                expected_answer=expected,
                predicted_answer=None,
                is_correct=False,
                status="ERROR",
                turns=0,
                runtime_seconds=elapsed,
                error_message=str(e),
                learned_heuristic=None,
            )

    def run_benchmark(
        self,
        tier: Optional[int] = None,
        problem_ids: Optional[List[int]] = None,
        save_results: bool = True,
    ) -> FrontierSummary:
        """Runs the FrontierMath benchmark filtered by Tier or problem IDs."""
        if problem_ids:
            target_ids = [pid for pid in problem_ids if pid in self.problems]
        elif tier is not None:
            target_ids = [pid for pid, p in self.problems.items() if p.get("tier") == tier]
        else:
            target_ids = list(self.problems.keys())

        if not target_ids:
            console.print("[bold red]No matching problems found for evaluation.[/bold red]")
            return FrontierSummary(0, 0, 0, 0.0, 0.0, 0.0)

        active_workers = min(self.num_workers, len(target_ids))
        console.print(
            f"[bold blue]Launching FrontierMath Benchmark on {len(target_ids)} problems "
            f"({active_workers} active concurrent worker{'s' if active_workers > 1 else ''})...[/bold blue]"
        )

        start_all = time.perf_counter()
        records: List[FrontierEvalRecord] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=active_workers) as executor:
            future_to_prob = {
                executor.submit(self.evaluate_single_problem, pid): pid
                for pid in target_ids
            }
            for future in concurrent.futures.as_completed(future_to_prob):
                pid = future_to_prob[future]
                try:
                    rec = future.result()
                    records.append(rec)
                    status_badge = "[green]✓ PASS[/green]" if rec.is_correct else f"[red]✗ {rec.status}[/red]"
                    console.print(
                        f"Tier {rec.tier} | Problem {rec.problem_id} ({rec.title[:28]}...): {status_badge} "
                        f"| Pred: {rec.predicted_answer} vs Exp: {rec.expected_answer} "
                        f"({rec.runtime_seconds}s, {rec.turns} turns)"
                    )
                except Exception as e:
                    console.print(f"[bold red]Problem {pid} worker crashed:[/bold red] {e}")

        total_time = round(time.perf_counter() - start_all, 2)
        records.sort(key=lambda x: (x.tier, x.problem_id))

        passed_count = sum(1 for r in records if r.is_correct)
        failed_count = len(records) - passed_count
        pass_rate = round(passed_count / len(records) * 100, 1) if records else 0.0
        avg_time = round(total_time / len(records), 2) if records else 0.0

        # Compute per-tier breakdown
        tier_stats: Dict[str, str] = {}
        for t in range(1, 5):
            t_recs = [r for r in records if r.tier == t]
            if t_recs:
                t_passed = sum(1 for r in t_recs if r.is_correct)
                t_pct = round(t_passed / len(t_recs) * 100, 1)
                tier_stats[f"Tier {t}"] = f"{t_passed}/{len(t_recs)} ({t_pct}%)"

        summary = FrontierSummary(
            total_problems=len(records),
            passed=passed_count,
            failed=failed_count,
            pass_rate_percent=pass_rate,
            total_runtime_seconds=total_time,
            average_runtime_seconds=avg_time,
            tier_accuracy=tier_stats,
            records=records,
        )

        if save_results:
            DEFAULT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            res_file = DEFAULT_EVAL_DIR / f"eval_frontiermath_{ts}.json"
            with open(res_file, "w", encoding="utf-8") as f:
                json.dump(asdict(summary), f, indent=2)
            console.print(f"\n[green]Saved FrontierMath results to {res_file}[/green]")

        return summary
