"""Project Euler 200 Automated Evaluation Harness and Failure Miner.

Runs multi-worker evaluation batches against data/euler/euler_200.json,
compares predicted boxed answers against verified ground truth,
mines failure patterns into an actionable taxonomy, and commits
successful heuristics into TurboVec procedural memory.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console

from .kernel import PersistentMathKernel
from .llm_client import TunnelThinkingClient
from .memory import CoALAMemoryManager
from .solver import MathSolver, SolveResult

logger = logging.getLogger("prime_math.evaluator")
console = Console()

EULER_DATASET_PATH = Path(__file__).parent.parent / "data" / "euler" / "euler_200.json"
DEFAULT_EVAL_DIR = Path(__file__).parent.parent / "data" / "eval_results"


def normalize_answer(ans: Optional[str]) -> str:
    """Normalizes an answer string for robust equality comparison."""
    if ans is None:
        return ""
    s = str(ans).strip()

    # Remove LaTeX wrappers
    s = re.sub(r"\\boxed\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
    s = s.replace("$", "").replace(",", "").replace(" ", "").strip()

    # Check integer equality
    try:
        if "." not in s:
            return str(int(s))
    except (ValueError, OverflowError):
        pass

    # Check float equality
    try:
        val = float(s)
        return f"{val:.6f}"
    except (ValueError, OverflowError):
        pass

    return s.lower()


def answers_match(pred: Optional[str], expected: Optional[str]) -> bool:
    """Returns True if predicted answer matches expected ground truth."""
    if pred is None or expected is None:
        return False
    norm_p = normalize_answer(pred)
    norm_e = normalize_answer(expected)
    if norm_p == norm_e:
        return True

    # Check float with tolerance
    try:
        p_val = float(norm_p)
        e_val = float(norm_e)
        if abs(p_val - e_val) < 1e-4:
            return True
    except (ValueError, OverflowError):
        pass

    return False


HARNESS_VERSION = "2.0"


@dataclass
class ProblemEvalRecord:
    problem_id: int
    title: str
    expected_answer: str
    predicted_answer: Optional[str]
    is_correct: bool
    status: str  # PASS, WRONG_ANSWER, NO_BOXED_ANSWER, ERROR, TIMEOUT
    turns: int
    runtime_seconds: float
    failure_category: Optional[str] = None
    error_message: Optional[str] = None
    learned_heuristic: Optional[str] = None


@dataclass
class EvaluationSummary:
    total_problems: int
    passed: int
    failed: int
    pass_rate_percent: float
    total_runtime_seconds: float
    average_runtime_seconds: float
    failure_taxonomy: Dict[str, int] = field(default_factory=dict)
    records: List[ProblemEvalRecord] = field(default_factory=list)
    harness_version: str = HARNESS_VERSION


class EulerEvaluator:
    """Automated multi-worker evaluation scaffold for Project Euler benchmarks."""

    def __init__(
        self,
        dataset_path: Path = EULER_DATASET_PATH,
        num_workers: int = 2,
        max_iterations: int = 12,
        enable_web_research: bool = False,
        harness_version: str = HARNESS_VERSION,
    ):
        self.dataset_path = dataset_path
        self.num_workers = num_workers
        self.max_iterations = max_iterations
        self.enable_web_research = enable_web_research
        self.harness_version = harness_version
        self.problems = self._load_dataset()

    def _load_dataset(self) -> Dict[int, Dict[str, Any]]:
        problems: Dict[int, Dict[str, Any]] = {}
        if self.dataset_path.exists():
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    problems[item["id"]] = item

        # Automatically merge euler_800_1000.json if present
        data_dir = self.dataset_path.parent
        extra_files = [
            data_dir / "euler_800_1000.json",
            data_dir / "euler_200.json",
            data_dir / "euler" / "euler_800_1000.json",
            data_dir.parent / "euler" / "euler_800_1000.json",
        ]
        for extra in extra_files:
            if extra.exists() and extra != self.dataset_path:
                try:
                    with open(extra, "r", encoding="utf-8") as f:
                        for item in json.load(f):
                            if item["id"] not in problems:
                                problems[item["id"]] = item
                except Exception as e:
                    logger.debug(f"Failed to load extra dataset {extra}: {e}")

        if not problems:
            raise FileNotFoundError(f"No problem records found at {self.dataset_path}")
        return problems

    def _classify_failure(self, result: SolveResult, expected: str) -> Tuple[str, Optional[str]]:
        """Mines failure signature to classify into an actionable taxonomy."""
        # 1. Did kernel execution hit an error?
        for exec_res in result.execution_history:
            if not exec_res.success and exec_res.error:
                err_text = exec_res.error
                if "ImportError" in err_text or "ModuleNotFoundError" in err_text:
                    return "SYNTAX_OR_IMPORT_ERROR", err_text[:200]
                if "Timeout" in err_text or "Timed out" in err_text:
                    return "TIMEOUT_OR_BLOWUP", err_text[:200]
                if "IndexError" in err_text or "UnboundLocalError" in err_text:
                    return "OFF_BY_ONE_OR_BOUNDARY", err_text[:200]
                return "KERNEL_RUNTIME_ERROR", err_text[:200]

        # 2. Was an answer produced but wrong?
        if result.boxed_answer:
            # Check if reasoning contained anchor bias
            all_reasoning = " ".join(result.reasoning_traces)
            if "known answer" in all_reasoning.lower() or "recall" in all_reasoning.lower():
                return "MEMORY_ANCHOR_BIAS", f"Predicted {result.boxed_answer} vs Expected {expected}"
            return "WRONG_ANSWER", f"Predicted {result.boxed_answer} vs Expected {expected}"

        # 3. Exhausted turns without boxed answer
        if result.iterations >= self.max_iterations:
            return "TURN_LIMIT_EXHAUSTED", f"Exhausted {result.iterations} turns without final answer"

        return "NO_BOXED_ANSWER", "No LaTeX boxed answer found in response"

    def evaluate_single_problem(self, prob_id: int) -> ProblemEvalRecord:
        """Solves a single problem and grades against ground truth."""
        item = self.problems.get(prob_id)
        if not item:
            raise KeyError(f"Problem {prob_id} not found in dataset.")

        problem_text = f"Project Euler Problem {prob_id} ({item['title']}):\n\n{item['statement']}"
        expected = str(item.get("answer", "")).strip()

        start_time = time.perf_counter()
        client = TunnelThinkingClient()
        kernel = PersistentMathKernel()
        memory = CoALAMemoryManager()

        prob_log_dir = Path(__file__).parent.parent / "logs" / "problems"
        prob_log_dir.mkdir(parents=True, exist_ok=True)
        prob_log_path = prob_log_dir / f"problem_{prob_id:03d}.log"

        solver = MathSolver(
            client=client,
            kernel=kernel,
            memory=memory,
            default_max_iterations=self.max_iterations,
            enable_pre_solve_research=self.enable_web_research,
            log_path=prob_log_path,
        )

        try:
            res = solver.solve(problem_text, max_iterations=self.max_iterations)
            elapsed = round(time.perf_counter() - start_time, 2)

            is_correct = answers_match(res.boxed_answer, expected)
            status = "PASS" if is_correct else ("WRONG_ANSWER" if res.boxed_answer else "NO_BOXED_ANSWER")
            res.is_correct = is_correct

            # Strictly gate procedural heuristic consolidation to verified correct results
            solver.consolidate_learning(problem_text, res, is_correct=is_correct)

            failure_category = None
            error_message = None
            if not is_correct:
                failure_category, error_message = self._classify_failure(res, expected)

            learned_h = res.learned_heuristics[0] if res.learned_heuristics else None

            return ProblemEvalRecord(
                problem_id=prob_id,
                title=item["title"],
                expected_answer=expected,
                predicted_answer=res.boxed_answer,
                is_correct=is_correct,
                status=status,
                turns=res.iterations,
                runtime_seconds=elapsed,
                failure_category=failure_category,
                error_message=error_message,
                learned_heuristic=learned_h,
            )
        except Exception as e:
            elapsed = round(time.perf_counter() - start_time, 2)
            return ProblemEvalRecord(
                problem_id=prob_id,
                title=item["title"],
                expected_answer=expected,
                predicted_answer=None,
                is_correct=False,
                status="ERROR",
                turns=0,
                runtime_seconds=elapsed,
                failure_category="HARNESS_EXCEPTION",
                error_message=str(e),
                learned_heuristic=None,
            )

    def get_passed_problem_records(self) -> Dict[int, ProblemEvalRecord]:
        """Loads all previously passed problem records from eval_results directory and verified problem logs."""
        passed: Dict[int, ProblemEvalRecord] = {}
        if DEFAULT_EVAL_DIR.exists():
            for f in sorted(DEFAULT_EVAL_DIR.glob("eval_euler_*.json")):
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    # Smart cache: only reuse cache with matching harness version
                    if data.get("harness_version") != self.harness_version:
                        continue
                    for r_dict in data.get("records", []):
                        if r_dict.get("is_correct"):
                            pid = r_dict.get("problem_id")
                            rec = ProblemEvalRecord(
                                problem_id=pid,
                                title=r_dict.get("title", f"Problem {pid}"),
                                expected_answer=r_dict.get("expected_answer", ""),
                                predicted_answer=r_dict.get("predicted_answer"),
                                is_correct=True,
                                status="PASS",
                                turns=r_dict.get("turns", 1),
                                runtime_seconds=r_dict.get("runtime_seconds", 0.0),
                                learned_heuristic=r_dict.get("learned_heuristic"),
                            )
                            passed[pid] = rec
                except Exception:
                    continue

        # Also inspect problem logs to recover verified solutions from any interrupted runs matching this harness version
        prob_log_dir = Path(__file__).parent.parent / "logs" / "problems"
        if prob_log_dir.exists():
            for log_file in prob_log_dir.glob("problem_*.log"):
                try:
                    pid = int(log_file.stem.split("_")[1])
                    if pid in passed:
                        continue
                    item = self.problems.get(pid)
                    if not item:
                        continue
                    expected = str(item.get("answer", "")).strip()
                    txt = log_file.read_text(encoding="utf-8", errors="ignore")
                    if f"Harness v{self.harness_version}" not in txt and f"v{self.harness_version}" not in txt:
                        continue
                    m = re.search(r"\[(?:SESSION|TIER 1) COMPLETE\] (?:Final |Verified )?Boxed Answer: (\S+)", txt)
                    if m:
                        cand = m.group(1).replace("{", "").replace("}", "").strip()
                        if answers_match(cand, expected):
                            passed[pid] = ProblemEvalRecord(
                                problem_id=pid,
                                title=item["title"],
                                expected_answer=expected,
                                predicted_answer=cand,
                                is_correct=True,
                                status="PASS",
                                turns=1,
                                runtime_seconds=0.0,
                                learned_heuristic=None,
                            )
                except Exception:
                    continue

        return passed

    def run_benchmark(
        self,
        problem_ids: List[int],
        skip_passed: bool = True,
        save_results: bool = True,
    ) -> EvaluationSummary:
        """Runs parallel evaluation across designated problem IDs using concurrent workers."""
        passed_cache = self.get_passed_problem_records() if skip_passed else {}
        to_run = [pid for pid in problem_ids if pid not in passed_cache]
        cached_records = [passed_cache[pid] for pid in problem_ids if pid in passed_cache]

        if cached_records:
            console.print(
                f"[bold cyan]⚡ Smart Cache: Skipping {len(cached_records)} previously solved & verified problems "
                f"({len(to_run)} problems to solve)...[/bold cyan]"
            )

        start_all = time.perf_counter()
        records: List[ProblemEvalRecord] = list(cached_records)

        if to_run:
            console.print(
                f"[bold blue]Launching Euler Benchmark on {len(to_run)} problems "
                f"({self.num_workers} concurrent workers)...[/bold blue]"
            )

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_prob = {
                    executor.submit(self.evaluate_single_problem, pid): pid
                    for pid in to_run
                }
                for future in concurrent.futures.as_completed(future_to_prob):
                    pid = future_to_prob[future]
                    try:
                        rec = future.result()
                        records.append(rec)
                        status_badge = "[green]✓ PASS[/green]" if rec.is_correct else f"[red]✗ {rec.status}[/red]"
                        console.print(
                            f"Problem {rec.problem_id:03d} ({rec.title[:25]}...): {status_badge} "
                            f"| Pred: {rec.predicted_answer} vs Exp: {rec.expected_answer} "
                            f"({rec.runtime_seconds}s, {rec.turns} turns)"
                        )
                    except Exception as e:
                        console.print(f"[bold red]Problem {pid} worker crashed:[/bold red] {e}")

        total_time = round(time.perf_counter() - start_all, 2)
        records.sort(key=lambda x: x.problem_id)

        passed_count = sum(1 for r in records if r.is_correct)
        failed_count = len(records) - passed_count
        pass_rate = round(passed_count / len(records) * 100, 1) if records else 0.0
        avg_time = round(total_time / len(records), 2) if records else 0.0

        taxonomy: Dict[str, int] = {}
        for r in records:
            if not r.is_correct and r.failure_category:
                taxonomy[r.failure_category] = taxonomy.get(r.failure_category, 0) + 1

        summary = EvaluationSummary(
            total_problems=len(records),
            passed=passed_count,
            failed=failed_count,
            pass_rate_percent=pass_rate,
            total_runtime_seconds=total_time,
            average_runtime_seconds=avg_time,
            failure_taxonomy=taxonomy,
            records=records,
            harness_version=self.harness_version,
        )

        if save_results:
            DEFAULT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            res_file = DEFAULT_EVAL_DIR / f"eval_euler_{ts}.json"
            with open(res_file, "w", encoding="utf-8") as f:
                json.dump(asdict(summary), f, indent=2)
            console.print(f"\n[green]Saved evaluation results to {res_file}[/green]")

        return summary
