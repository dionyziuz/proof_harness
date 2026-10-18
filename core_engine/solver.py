"""Autonomous Mathematical Reasoning & Closed-Loop Solver for Prime Math.

Implements a CoALA closed-loop decision cycle with TurboVec Procedural Memory:
1. Verifies TurboVec health (raises TurboVecUnavailableError if down - no silent fallback).
2. Retrieves domain-relevant procedural heuristics from TurboVec into Working Memory.
3. Streams inner reasoning tokens directly to logs/thinking.log in real time.
4. Formalizes problem & analyzes constraints.
5. Formulates empirical checks and symbolic proofs via persistent Python execution.
6. Catches execution and logic errors, adjusts hypotheses, and iterates until proven.
7. Post-Solve Consolidation: Extracts learned heuristics and commits to TurboVec procedural & episodic memory.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .compactor import MathContextCompactor
from .heuristic_validation import parse_benchmark_header, strip_benchmark_prefix
from .kernel import ExecutionResult, PersistentMathKernel
from .llm_client import FastComposerClient, TunnelThinkingClient
from .memory import CoALAMemoryManager
from .research import clean_search_query, search_math

logger = logging.getLogger("prime_math.solver")

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "thinking.log"

from .prompts import MATH_SYSTEM_PROMPT, MATH_VERIFICATION_PROMPT

PLACEHOLDER_ANSWERS = {
    "...",
    "....",
    r"\dots",
    r"\cdots",
    r"\ldots",
    "your_answer",
    "your_answer_here",
    "answer",
    "ans",
    "none",
    "?",
    "x",
    "[your_answer]",
    "[answer]",
    "d_infty",
    r"d_\infty",
    "d_x",
    "d",
}


@dataclass
class SolveResult:
    problem: str
    answer: Optional[str]
    boxed_answer: Optional[str]
    verified: bool
    iterations: int
    reasoning_traces: List[str] = field(default_factory=list)
    code_history: List[str] = field(default_factory=list)
    execution_history: List[ExecutionResult] = field(default_factory=list)
    retrieved_heuristics: List[str] = field(default_factory=list)
    learned_heuristics: List[str] = field(default_factory=list)
    research_briefing: Optional[str] = None
    final_response: str = ""
    log_file: str = str(DEFAULT_LOG_PATH)
    self_verified: bool = False
    is_correct: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes SolveResult to a JSON-compatible dictionary."""
        return {
            "problem": self.problem,
            "answer": self.answer,
            "boxed_answer": self.boxed_answer,
            "verified": self.verified,
            "iterations": self.iterations,
            "reasoning_traces": list(self.reasoning_traces),
            "code_history": list(self.code_history),
            "execution_history": [
                ex.to_dict() if hasattr(ex, "to_dict") else ex
                for ex in self.execution_history
            ],
            "retrieved_heuristics": list(self.retrieved_heuristics),
            "learned_heuristics": list(self.learned_heuristics),
            "research_briefing": self.research_briefing,
            "final_response": self.final_response,
            "log_file": self.log_file,
            "self_verified": self.self_verified,
            "is_correct": self.is_correct,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SolveResult":
        """Reconstructs a SolveResult from a dictionary."""
        exec_hist: List[ExecutionResult] = []
        for ex in data.get("execution_history", []):
            if isinstance(ex, ExecutionResult):
                exec_hist.append(ex)
            elif isinstance(ex, dict):
                exec_hist.append(ExecutionResult.from_dict(ex))
        return cls(
            problem=str(data.get("problem", "")),
            answer=data.get("answer"),
            boxed_answer=data.get("boxed_answer"),
            verified=bool(data.get("verified", False)),
            iterations=int(data.get("iterations", 0)),
            reasoning_traces=[str(x) for x in data.get("reasoning_traces", [])],
            code_history=[str(x) for x in data.get("code_history", [])],
            execution_history=exec_hist,
            retrieved_heuristics=[str(x) for x in data.get("retrieved_heuristics", [])],
            learned_heuristics=[str(x) for x in data.get("learned_heuristics", [])],
            research_briefing=data.get("research_briefing"),
            final_response=str(data.get("final_response", "")),
            log_file=str(data.get("log_file", str(DEFAULT_LOG_PATH))),
            self_verified=bool(data.get("self_verified", False)),
            is_correct=data.get("is_correct"),
        )


class MathSolver:
    """Closed-loop agentic solver combining thinking LLMs, persistent REPL, and CoALA TurboVec memory."""

    def __init__(
        self,
        client: Optional[TunnelThinkingClient] = None,
        kernel: Optional[PersistentMathKernel] = None,
        memory: Optional[CoALAMemoryManager] = None,
        compactor: Optional[MathContextCompactor] = None,
        fast_composer: Optional[FastComposerClient] = None,
        default_max_iterations: int = 12,
        enable_pre_solve_research: bool = True,
        log_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
    ):
        self.client = client or TunnelThinkingClient()
        self.fast_composer = fast_composer or FastComposerClient()
        self.kernel = kernel or PersistentMathKernel()
        self.memory = memory or CoALAMemoryManager()
        self.compactor = compactor or MathContextCompactor(
            client=self.client,
            max_context_tokens=self.client.config.max_context_window,
            reserve_tokens=self.client.config.max_tokens,
        )
        self.default_max_iterations = default_max_iterations
        self.enable_pre_solve_research = enable_pre_solve_research
        self.log_path = log_path or DEFAULT_LOG_PATH
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Colosseum-style runtime state: failed fast-composer route context (injected into the
        # deep orchestrator prompt) and the trailing execution-error signature window for
        # stall detection (dead-route re-exploration directive).
        self._fast_composer_critique: Optional[str] = None
        self._error_signatures: List[str] = []

    def _save_checkpoint(
        self,
        ckpt_path: Optional[Path],
        problem: str,
        completed_turn: int,
        start_wall_time: float,
        messages: List[Dict[str, str]],
        result: SolveResult,
    ) -> None:
        """Atomically writes turn-level solver state to ckpt_path via temp file + rename."""
        if not ckpt_path:
            return
        try:
            ckpt_path = Path(ckpt_path)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            sm = getattr(self.kernel, "_state_machine", None)
            sm_dict = sm.to_dict() if (sm is not None and hasattr(sm, "to_dict")) else None
            payload = {
                "version": 1,
                "problem": problem,
                "completed_turn": int(completed_turn),
                "timestamp": datetime.datetime.now().isoformat(),
                "elapsed_s": round(max(0.0, time.perf_counter() - start_wall_time), 3),
                "messages": messages,
                "result": result.to_dict(),
                "fast_composer_critique": self._fast_composer_critique,
                "error_signatures": list(self._error_signatures),
                "state_machine": sm_dict,
            }
            tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, ckpt_path)
        except Exception as e:
            logger.warning(f"Failed to write MathSolver checkpoint to {ckpt_path}: {e}")

    def _load_checkpoint(
        self,
        ckpt_path: Optional[Path],
        problem: str,
    ) -> Optional[Dict[str, Any]]:
        """Loads and validates an existing turn-level checkpoint for problem."""
        if not ckpt_path:
            return None
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            return None
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or data.get("version") != 1:
                return None
            if data.get("problem") != problem:
                logger.warning("Checkpoint problem mismatch; ignoring stale checkpoint.")
                return None
            return data
        except Exception as e:
            logger.warning(f"Failed to load MathSolver checkpoint {ckpt_path}: {e}")
            return None

    def _restore_kernel_state(
        self,
        problem: str,
        result: SolveResult,
        state_machine_dict: Optional[Dict[str, Any]],
        reset_kernel: bool = True,
    ) -> None:
        """Replays executed code history into the kernel and restores MathStateMachine."""
        if reset_kernel:
            self.kernel.reset()
        self.kernel.execute(f"import re\nPROBLEM_TEXT = {repr(problem)}")
        for idx, code in enumerate(result.code_history):
            if idx < len(result.execution_history):
                ex = result.execution_history[idx]
                if not ex.success and "timed out" in (ex.error or "").lower():
                    continue
            try:
                self.kernel.execute(code)
            except Exception as e:
                logger.warning(f"Error replaying solver code block: {e}")
        sm = getattr(self.kernel, "_state_machine", None)
        if isinstance(state_machine_dict, dict) and sm is not None and hasattr(sm, "load_state_dict"):
            try:
                sm.load_state_dict(state_machine_dict)
            except Exception as e:
                logger.warning(f"Failed to restore state machine in MathSolver: {e}")

    def _unlink_checkpoint(self, ckpt_path: Optional[Path]) -> None:
        """Removes the checkpoint file and any leftover .tmp file upon solve completion."""
        if not ckpt_path:
            return
        for p in (Path(ckpt_path), Path(ckpt_path).with_suffix(Path(ckpt_path).suffix + ".tmp")):
            try:
                if p.exists():
                    p.unlink()
            except Exception as e:
                logger.debug(f"Could not unlink checkpoint {p}: {e}")

    def extract_code_blocks(self, text: str) -> List[str]:
        """Extracts python code blocks from markdown text."""
        pattern = r"```(?:python|py)?\s*\n(.*?)```"
        return [match.strip() for match in re.findall(pattern, text, flags=re.DOTALL) if match.strip()]

    def extract_boxed_answer(self, text: Optional[str]) -> Optional[str]:
        """Extracts content inside LaTeX \\boxed{...} with nested brace support or \\boxed format."""
        if not text:
            return None

        # 1. LaTeX \boxed{...} with nested brace support (search from the end backwards)
        idx = text.rfind(r"\boxed{")
        if idx != -1:
            start = idx + len(r"\boxed{")
            depth = 1
            i = start
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1

            if depth == 0:
                val = text[start : i - 1].strip()
                val = re.sub(r"^\\text\{([^}]+)\}$", r"\1", val).strip()
                m_is = re.search(r"(?:is|=)\s*([0-9a-zA-Z_\-+/\^]+)$", val)
                if m_is and len(val) > 20:
                    val = m_is.group(1).strip()
                
                # Check for English sentence words vs math LaTeX commands
                alpha_words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{2,}\b", val)]
                latex_cmds = {"frac", "sqrt", "left", "right", "cdot", "times", "pmod", "bmod", "over", "binom", "pi", "sin", "cos", "tan", "ln", "log"}
                plain_words = [w for w in alpha_words if w not in latex_cmds]
                
                stop_words = {
                    "the", "a", "an", "is", "at", "large", "empirical", "value", "density",
                    "that", "where", "given", "assume", "by", "approx", "likely", "see",
                    "algorithm", "answer", "integer", "ans", "none", "unknown", "benchmark", "suppose"
                }
                if any(w in stop_words for w in plain_words):
                    val = ""
                if len(plain_words) >= 2 and "\\" not in val:
                    val = ""
                if any(x in val.lower() for x in ["?", "not unique"]):
                    val = ""
                if val and val.lower() not in PLACEHOLDER_ANSWERS and val.strip(". ") != "":
                    return val

        # 2. LaTeX \boxed without braces: \boxed 12345 or \boxed: 12345
        m = re.findall(r"\\boxed\s*[:=]?\s*([0-9a-zA-Z_\-+/\^]+)", text)
        if m:
            cand = m[-1].strip()
            if cand and cand.lower() not in PLACEHOLDER_ANSWERS and cand.strip(". ") != "":
                return cand

        return None

    def extract_answer_from_text(self, text: Optional[str]) -> Optional[str]:
        """Extracts candidate answer from text using boxed LaTeX or explicit answer markers."""
        if not text:
            return None

        # Try boxed answer first
        boxed = self.extract_boxed_answer(text)
        if boxed:
            return boxed

        # Look for explicit answer statements (from the end backwards)
        patterns = [
            r"(?i)\b(?:final\s+)?answer\s*(?:is|:|=)\s*([^\n\r,;$\\]+)",
            r"(?i)\btherefore,?\s*(?:the\s+answer\s+is\s*)?([0-9a-zA-Z_\-+/\^]+)",
        ]
        for pat in patterns:
            matches = list(re.finditer(pat, text))
            if matches:
                val = matches[-1].group(1).strip()
                val = val.rstrip(".* \t\n")
                if val and not val.lower().startswith("to ") and len(val) < 80:
                    words = [w.lower() for w in re.split(r"[\s_]+", val) if w]
                    if not any(w in words for w in [
                        "the", "a", "an", "is", "at", "large", "empirical", "value", "density",
                        "that", "where", "given", "assume", "by", "approx", "likely", "answer",
                        "my", "in", "to", "see", "any", "recurrence", "algorithm"
                    ]):
                        if re.match(r"^-?\d+(\.\d+)?$", val) and val.lower() not in PLACEHOLDER_ANSWERS:
                            return val

        return None

    def _extract_best_answer_from_history(self, result: SolveResult) -> Optional[str]:
        """Scans execution history (stdout, result_repr) and reasoning traces for a confident answer."""
        is_integer_problem = any(
            kw in result.problem.lower()
            for kw in [
                "floor", r"\lfloor", "integer", "find the smallest", "calculate p(",
                "how many", "number of", "compute \\lfloor", "find the value", "mod"
            ]
        )
        is_density_problem = any(
            kw in result.problem.lower()
            for kw in [
                "density", "probability", "proportion", "d_x", r"d_\infty"
            ]
        )

        # Scan execution history for empirical diagnostic baselines
        empirical_baseline: Optional[float] = None
        if is_density_problem:
            for exec_res in reversed(result.execution_history):
                if exec_res.stdout:
                    m_emp = re.search(r"(?i)\b(?:empirical|d_emp)\b[^\n\d]*([0-9]*\.[0-9]+)", exec_res.stdout)
                    if m_emp:
                        try:
                            empirical_baseline = float(m_emp.group(1))
                            break
                        except Exception:
                            pass

        def is_valid_candidate(cand: Optional[str]) -> bool:
            if not cand:
                return False
            c = cand.strip().strip(".{}")
            if not c or c.lower() in PLACEHOLDER_ANSWERS:
                return False
            # Discard any multi-word English sentence or phrase
            words = [w.lower() for w in re.split(r"[\s_]+", c) if w]
            if any(w in words for w in [
                "the", "a", "an", "is", "at", "large", "empirical", "value", "density",
                "that", "where", "given", "assume", "by", "approx", "likely", "answer"
            ]):
                return False
            if len(c.split()) > 2 and not re.match(r"^-?\d+(\.\d+)?$", c):
                return False
            if is_density_problem and c.startswith("-"):
                return False
            if is_integer_problem:
                if is_density_problem and c.isdigit() and int(c) > 10_000_000:
                    return False
                # Rule 12 sanity check: reject candidates that contradict empirical measurements by > 0.05
                if is_density_problem and empirical_baseline is not None and c.isdigit():
                    m_scale = re.search(r"(?:10\^(\d+)|10\*\*(\d+)|1e(\d+))", result.problem)
                    multiplier = 10 ** int(m_scale.group(1) or m_scale.group(2) or m_scale.group(3)) if m_scale else 10**6
                    cand_val_f = float(c) / multiplier
                    if abs(cand_val_f - empirical_baseline) > 0.05:
                        return False
                return bool(re.match(r"^-?\d+$", c))
            return True

        def sanitize_stdout(text: str) -> str:
            """Strips execution timings, progress reports, sample sizes, and diagnostic lines that can be confused with answers."""
            # 1. Strip diagnostic statements like "Found 1990 primes up to 17320" or "Sieve done in 21.2s, 3001134 primes"
            cleaned = re.sub(r"(?i)\bFound\s+\d+\s+primes(?:\s+up\s+to\s+\d+)?\b", "", text)
            cleaned = re.sub(r"(?i)\bSieve\s+done\s+in\s+[\d\.]+s?,\s*\d+\s+primes\b", "", cleaned)
            cleaned = re.sub(r"(?i)\bup\s+to\s+\d+\b", "", cleaned)

            # 2. Strip timing variables and values like total_time=273.0s, time=19.6s, t=112.5s, runtime=12.3s
            cleaned = re.sub(r"(?i)\b(?:total_)?(?:time|runtime|elapsed|t)\s*[:=]\s*[\d\.]+\s*(?:s|ms|sec|seconds)?\b", "", cleaned)
            # 3. Strip standalone timing durations like " 273.0s " or " 184.7s" or " 12.3ms "
            cleaned = re.sub(r"\b\d+(?:\.\d+)?\s*(?:ms|sec|seconds)\b", "", cleaned)
            cleaned = re.sub(r"\b\d+\.\d+s\b", "", cleaned)

            # 4. Strip sample sizes and prime counts like "n=11078937", "tot=11078937", "(n=11078937, ...)"
            cleaned = re.sub(r"(?i)\b(?:n|tot|total|sample_size|count|num_primes|primes)\s*[:=]\s*\d+\b", "", cleaned)
            cleaned = re.sub(r"\b\d+\s+primes\b", "", cleaned)

            # 5. Strip domain checkpoint bounds like "x=2*10^8:", "x=100000:", "N=1000000"
            cleaned = re.sub(r"(?i)\b[xXnN]\s*[:=]\s*[\d\^eE\+\*]+[:\s]?", "", cleaned)

            # 6. Strip progress indicators like "progress: 200000000/300000000"
            cleaned = re.sub(r"(?i)\bprogress\s*[:=]?\s*\d+\s*/\s*\d+\b", "", cleaned)

            # 7. Strip slice bounds like "range [200000000, 300000000]:"
            cleaned = re.sub(r"(?i)\brange\s*\[\s*\d+\s*,\s*\d+\s*\]\s*[:=]?", "", cleaned)
            return cleaned

        # 1. Scan stdout of execution history from latest to earliest (e.g. "floor = 123456", "Answer: 42", "1e6*d=350000")
        for exec_res in reversed(result.execution_history):
            if not exec_res.stdout:
                continue
            stdout = sanitize_stdout(exec_res.stdout.strip())
            boxed = self.extract_boxed_answer(stdout)
            if is_valid_candidate(boxed):
                return boxed

            for line in reversed(stdout.splitlines()):
                line = line.strip()
                # Look for scaled metric assignments like "1e6*d=350000.1234", "floor=123456", "ans=42"
                m_scaled = re.search(r"(?i)\b(?:1e\d+\*d|floor(?:\s*\([^)]*\))?|final\s+answer|answer|ans|scaled|result)\s*[:=]\s*([0-9a-zA-Z_\-+/\^]+(?:\.[0-9]+)?)", line)
                if m_scaled:
                    cand_val = m_scaled.group(1).strip()
                    if is_integer_problem and "." in cand_val:
                        cand_val = cand_val.split(".")[0]
                    if is_valid_candidate(cand_val):
                        return cand_val

        # 1.5. If integer problem asking for scaled floor (e.g. floor(10^6 * d)), scan for raw density (e.g. d_x=0.3676859071)
        m_scale = re.search(r"(?:10\^(\d+)|10\*\*(\d+)|1e(\d+))", result.problem)
        if is_integer_problem and m_scale:
            power = int(m_scale.group(1) or m_scale.group(2) or m_scale.group(3))
            multiplier = 10 ** power
            for exec_res in reversed(result.execution_history):
                if not exec_res.stdout:
                    continue
                stdout = sanitize_stdout(exec_res.stdout.strip())
                for line in reversed(stdout.splitlines()):
                    m_density = re.search(r"(?i)\b(?:d_x|d_inf|density|d)\s*[:=]\s*(0\.\d+)\b", line)
                    if m_density:
                        f_val = float(m_density.group(1))
                        scaled_val = str(int(math.floor(f_val * multiplier)))
                        if is_valid_candidate(scaled_val):
                            return scaled_val

        # 2. Check result_repr of the last successful execution
        for exec_res in reversed(result.execution_history):
            if exec_res.success and exec_res.result_repr:
                repr_val = str(exec_res.result_repr).strip()
                if is_valid_candidate(repr_val):
                    return repr_val

        # 3. Check if any stdout line was just a standalone integer
        for exec_res in reversed(result.execution_history):
            if exec_res.stdout:
                stdout_clean = sanitize_stdout(exec_res.stdout.strip())
                for line in reversed(stdout_clean.splitlines()):
                    line = line.strip()
                    if is_valid_candidate(line):
                        return line

        # 4. Search stdout across all executions for the last printed concrete integer (e.g. floor=368311, count=187657)
        for exec_res in reversed(result.execution_history):
            if exec_res.stdout:
                stdout_clean = sanitize_stdout(exec_res.stdout.strip())
                int_matches = re.findall(r"\b(\d{3,25})\b", stdout_clean)
                for int_val in reversed(int_matches):
                    if is_valid_candidate(int_val):
                        return int_val

        # 5. Check reasoning traces (from latest to earliest) with strict validation
        for trace in reversed(result.reasoning_traces):
            ans = self.extract_answer_from_text(trace)
            if is_valid_candidate(ans):
                return ans

        return None

    def _log_write(self, text: str) -> None:
        """Appends text to the live thinking log file with immediate disk flush."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(text)
                f.flush()
        except Exception as e:
            logger.warning(f"Failed to write to thinking log: {e}")

    def consolidate_learning(
        self, problem: str, result: SolveResult, is_correct: Optional[bool] = None
    ) -> None:
        """Runs post-solve reflection to extract procedural heuristics and log episode to TurboVec.

        Strictly gated: procedural heuristics are ONLY extracted and committed when is_correct
        is explicitly confirmed True. Unverified or incorrect solutions never commit heuristics.
        """
        if is_correct is not True:
            # Gated: do not extract procedural heuristics for incorrect or unconfirmed solutions.
            if result.code_history:
                try:
                    critique = (
                        "Incorrect answer - no procedural heuristic extracted."
                        if is_correct is False
                        else "Unverified correctness against ground truth - no procedural heuristic extracted."
                    )
                    self.memory.record_episode(
                        problem=problem,
                        code_blocks=result.code_history,
                        verified=False,
                        answer=result.boxed_answer,
                        critique=critique,
                    )
                except Exception as e:
                    logger.warning(f"Error recording unverified episode: {e}")
            return

        code_summary = "\n\n".join([f"Block {i+1}:\n{c}" for i, c in enumerate(result.code_history)])
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a cognitive reflection engine for mathematical problem solving. "
                    "Analyze this VERIFIED solution trajectory. If there is a genuine, universal procedural heuristic, "
                    "algorithmic invariant, or SymPy/Z3 pattern that made the difference, "
                    "state it in 1-2 concise sentences. It MUST be a universal mathematical heuristic, NOT tied to this specific problem instance, "
                    "and NOT an empirical sieving or extrapolation hack. If the problem was straightforward or no universal heuristic applies, output 'NONE'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem: {problem}\n"
                    f"Verified: {result.verified}\n"
                    f"Answer: {result.boxed_answer}\n"
                    f"Executed Code:\n{code_summary[:2000]}"
                ),
            },
        ]

        try:
            resp = self.client.complete(prompt, max_tokens=150, temperature=0.2)
            reflection = resp.content.strip()
            prob_id, prob_title = parse_benchmark_header(problem)
            if reflection and "NONE" not in reflection.upper():
                from .heuristic_validation import is_valid_universal_heuristic

                is_valid, reason = is_valid_universal_heuristic(
                    reflection,
                    expected_answer=result.boxed_answer,
                    problem_statement=problem,
                    problem_id=prob_id,
                    problem_title=prob_title,
                )
                if is_valid:
                    metadata = {"problem_id": prob_id, "title": prob_title} if prob_id else {}
                    added = self.memory.add_procedural_heuristic(reflection, metadata=metadata)
                    if added:
                        result.learned_heuristics.append(reflection)
                        self._log_write(f"\n[CoALA Procedural Memory] Learned new heuristic:\n{reflection}\n")
                else:
                    logger.info(f"Skipping non-universal reflection heuristic: {reason} ('{reflection[:60]}')")

            self.memory.record_episode(
                problem=problem,
                code_blocks=result.code_history,
                verified=True,
                answer=result.boxed_answer,
                critique=reflection,
            )
        except Exception as e:
            logger.warning(f"Post-solve reflection encountered error: {e}")

    _consolidate_learning = consolidate_learning

    async def aconsolidate_learning(
        self, problem: str, result: SolveResult, is_correct: Optional[bool] = None
    ) -> None:
        """Asynchronously runs post-solve reflection to extract procedural heuristics."""
        if is_correct is not True:
            # Gated: do not extract procedural heuristics for incorrect or unconfirmed solutions.
            if result.code_history:
                try:
                    critique = (
                        "Incorrect answer - no procedural heuristic extracted."
                        if is_correct is False
                        else "Unverified correctness against ground truth - no procedural heuristic extracted."
                    )
                    await self.memory.arecord_episode(
                        problem=problem,
                        code_blocks=result.code_history,
                        verified=False,
                        answer=result.boxed_answer,
                        critique=critique,
                    )
                except Exception as e:
                    logger.warning(f"Error recording unverified episode: {e}")
            return

        code_summary = "\n\n".join([f"Block {i+1}:\n{c}" for i, c in enumerate(result.code_history)])
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a cognitive reflection engine for mathematical problem solving. "
                    "Analyze this VERIFIED solution trajectory. If there is a genuine, universal procedural heuristic, "
                    "algorithmic invariant, or SymPy/Z3 pattern that made the difference, "
                    "state it in 1-2 concise sentences. It MUST be a universal mathematical heuristic, NOT tied to this specific problem instance, "
                    "and NOT an empirical sieving or extrapolation hack. If the problem was straightforward or no universal heuristic applies, output 'NONE'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem: {problem}\n"
                    f"Verified: {result.verified}\n"
                    f"Answer: {result.boxed_answer}\n"
                    f"Executed Code:\n{code_summary[:2000]}"
                ),
            },
        ]

        try:
            resp = await self.client.acomplete(prompt, max_tokens=150, temperature=0.2)
            reflection = resp.content.strip()
            prob_id, prob_title = parse_benchmark_header(problem)
            if reflection and "NONE" not in reflection.upper():
                from .heuristic_validation import is_valid_universal_heuristic

                is_valid, reason = is_valid_universal_heuristic(
                    reflection,
                    expected_answer=result.boxed_answer,
                    problem_statement=problem,
                    problem_id=prob_id,
                    problem_title=prob_title,
                )
                if is_valid:
                    metadata = {"problem_id": prob_id, "title": prob_title} if prob_id else {}
                    added = await self.memory.aadd_procedural_heuristic(reflection, metadata=metadata)
                    if added:
                        result.learned_heuristics.append(reflection)
                        self._log_write(f"\n[CoALA Procedural Memory] Learned new heuristic:\n{reflection}\n")
                else:
                    logger.info(f"Skipping non-universal reflection heuristic: {reason} ('{reflection[:60]}')")

            await self.memory.arecord_episode(
                problem=problem,
                code_blocks=result.code_history,
                verified=True,
                answer=result.boxed_answer,
                critique=reflection,
            )
        except Exception as e:
            logger.warning(f"Post-solve reflection encountered error: {e}")

    _aconsolidate_learning = aconsolidate_learning

    def _conduct_pre_solve_research(self, problem: str) -> Optional[str]:
        """Pre-solve academic literature retrieval & briefing loop.

        Analyzes the problem statement, extracts 1-2 pure conceptual academic search queries,
        queries the mathematical literature (Jina Search / MathWorld / arXiv), and synthesizes
        a structured theoretical briefing to seed the solver's working memory before Turn 1.
        """
        if not self.enable_pre_solve_research:
            return None

        self._log_write("\n[Stage 0: Pre-Solve Academic Research] Initiating literature & concept retrieval...\n")

        # Step 1: Conceptual Query Generation (Fast Composer first, Thinking Client fallback)
        queries: List[str] = []
        if self.fast_composer:
            try:
                queries = self.fast_composer.generate_research_queries(problem)
            except Exception as e:
                logger.debug(f"FastComposer query generation error: {e}")

        if not queries:
            query_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are an academic mathematics research specialist. Analyze this mathematics problem and output "
                        "1-2 targeted academic search queries to look up underlying textbook concepts, governing theorems, "
                        "or literature lemmas in preprint archives (arXiv).\n"
                        "CRITICAL RULES:\n"
                        "1. Output each query enclosed in <query>...</query> tags (e.g. <query>multiplicative order density Artin conjecture Kummer extensions</query>).\n"
                        "2. Output ONLY standard academic English keywords (no raw LaTeX math, no dollar signs $, no variable names, no inequality signs <, >).\n"
                        "3. Target keywords that uncover the exact governing analytic evaluation formula (e.g. Euler products, character sums, generating functions, spectral gaps).\n"
                        "4. Do NOT output commentary, numbering, or meta-explanations."
                    ),
                },
                {"role": "user", "content": f"Problem Statement:\n{problem[:1500]}"},
            ]
            try:
                resp = self.client.complete(query_prompt, temperature=0.1, max_tokens=600, enable_thinking=False, timeout_seconds=30.0)
                raw_text = (resp.content or "").strip()
                if not raw_text and resp.reasoning:
                    raw_text = resp.reasoning.strip()

                extracted = re.findall(r"<query>(.*?)</query>", raw_text, re.IGNORECASE)
                if not extracted:
                    extracted = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

                for q in extracted:
                    cq = clean_search_query(q)
                    if 4 <= len(cq) <= 80 and not any(cq.lower().startswith(p) for p in ["we need", "problem", "query", "user asks"]):
                        queries.append(cq)
            except Exception as e:
                self._log_write(f"[Pre-Solve Research] Query generation failed: {e}\n")
                return None

        if not queries:
            self._log_write("[Pre-Solve Research] No conceptual queries generated.\n")
            return None

        # Step 2: Multi-Source Literature Harvesting (arXiv Prioritized + Wikipedia Fallback)
        all_snippets: List[str] = []
        for q in queries[:2]:
            self._log_write(f"[Pre-Solve Research] Querying mathematical literature: '{q}'...\n")
            try:
                results = search_math(q, count=3, timeout=35.0, prioritize_arxiv=True)
                for r in results:
                    title = r.get("title", "")
                    snip = r.get("snippet", "")
                    if snip and "error" not in title.lower() and len(snip.strip()) > 30:
                        all_snippets.append(f"Source ({title}):\n{snip.strip()}")
            except Exception as e:
                self._log_write(f"[Pre-Solve Research] Search for '{q}' encountered error: {e}\n")

        if not all_snippets:
            self._log_write("[Pre-Solve Research] No relevant literature snippets retrieved.\n")
            return None

        # Step 3: Theoretical Briefing Synthesis
        synth_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a pure mathematics research specialist. Synthesize the following research literature snippets "
                    "into a dense, structured theoretical briefing (under 300 words) to equip an elite mathematical solver.\n"
                    "CRITICAL RULES:\n"
                    "1. Enclose your final synthesized briefing inside <briefing>...</briefing> tags.\n"
                    "2. DO NOT attempt to solve the problem, do mental arithmetic, or speculate on specific numerical limits or answers (e.g. NEVER guess 1/2, 0, 1, or specific constants). You are providing theoretical literature context, NOT the solution.\n"
                    "3. Structure your briefing concisely:\n"
                    "- GOVERNING DEFINITIONS & OBJECTS\n"
                    "- ESTABLISHED THEOREMS & KNOWN LITERATURE FORMULAS (e.g. Euler products, character sums, generating functions, spectral decompositions)\n"
                    "- EXACT ANALYTIC EVALUATION STRATEGY: Explicitly identify the analytic formula or exact evaluation protocol "
                    "(e.g. Kummer character sums, Euler product acceleration, Dirichlet convolution) so the solver avoids wasting turns on brute-force empirical simulations.\n"
                    "Be dense, rigorous, and purely mathematical. Avoid meta-commentary."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem Statement:\n{problem[:1000]}\n\n"
                    f"Retrieved Literature Snippets:\n" + "\n\n".join(all_snippets[:4])[:3500]
                ),
            },
        ]

        try:
            synth_resp = self.client.complete(synth_prompt, temperature=0.1, max_tokens=2048, enable_thinking=False, timeout_seconds=60.0)
            raw_text = (synth_resp.content or "").strip()
            if not raw_text and synth_resp.reasoning:
                raw_text = synth_resp.reasoning.strip()

            extracted_brief = re.search(r"<briefing>(.*?)</briefing>", raw_text, flags=re.DOTALL)
            if extracted_brief:
                briefing = extracted_brief.group(1).strip()
            else:
                briefing = raw_text

            if briefing and not any(briefing.lower().startswith(p) for p in ["we need answer", "user asks", "you are prime math"]):
                self._log_write(f"\n[Stage 0: Pre-Solve Research Complete]\n{briefing}\n\n")
                return briefing
        except Exception as e:
            self._log_write(f"[Pre-Solve Research] Briefing synthesis failed: {e}\n")

        return None

    async def _aconduct_pre_solve_research(self, problem: str) -> Optional[str]:
        """Asynchronous version of pre-solve academic literature retrieval & briefing loop."""
        if not self.enable_pre_solve_research:
            return None

        self._log_write("\n[Stage 0: Pre-Solve Academic Research] Initiating literature & concept retrieval...\n")

        # Step 1: Conceptual Query Generation (Fast Composer first, Thinking Client fallback)
        queries: List[str] = []
        if self.fast_composer:
            try:
                queries = await self.fast_composer.agenerate_research_queries(problem)
            except Exception as e:
                logger.debug(f"FastComposer query generation error: {e}")

        if not queries:
            query_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are an academic mathematics research specialist. Analyze this mathematics problem and output "
                        "1-2 targeted academic search queries to look up underlying textbook concepts, governing theorems, "
                        "or literature lemmas in preprint archives (arXiv).\n"
                        "CRITICAL RULES:\n"
                        "1. Output each query enclosed in <query>...</query> tags (e.g. <query>multiplicative order density Artin conjecture Kummer extensions</query>).\n"
                        "2. Output ONLY standard academic English keywords (no raw LaTeX math, no dollar signs $, no variable names, no inequality signs <, >).\n"
                        "3. Target keywords that uncover the exact governing analytic evaluation formula (e.g. Euler products, character sums, generating functions, spectral gaps).\n"
                        "4. Do NOT output commentary, numbering, or meta-explanations."
                    ),
                },
                {"role": "user", "content": f"Problem Statement:\n{problem[:1500]}"},
            ]
            try:
                resp = await self.client.acomplete(query_prompt, temperature=0.1, max_tokens=600, enable_thinking=False, timeout_seconds=30.0)
                raw_text = (resp.content or "").strip()
                if not raw_text and resp.reasoning:
                    raw_text = resp.reasoning.strip()

                extracted = re.findall(r"<query>(.*?)</query>", raw_text, re.IGNORECASE)
                if not extracted:
                    extracted = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

                for q in extracted:
                    cq = clean_search_query(q)
                    if 4 <= len(cq) <= 80 and not any(cq.lower().startswith(p) for p in ["we need", "problem", "query", "user asks"]):
                        queries.append(cq)
            except Exception as e:
                self._log_write(f"[Pre-Solve Research] Query generation failed: {e}\n")
                return None

        if not queries:
            self._log_write("[Pre-Solve Research] No conceptual queries generated.\n")
            return None

        # Step 2: Multi-Source Literature Harvesting (arXiv Prioritized + Wikipedia Fallback)
        all_snippets: List[str] = []
        for q in queries[:2]:
            self._log_write(f"[Pre-Solve Research] Querying mathematical literature: '{q}'...\n")
            try:
                results = search_math(q, count=3, timeout=15.0, prioritize_arxiv=True)
                for r in results:
                    title = r.get("title", "")
                    snip = r.get("snippet", "")
                    if snip and "error" not in title.lower() and len(snip.strip()) > 30:
                        all_snippets.append(f"Source ({title}):\n{snip.strip()}")
            except Exception as e:
                self._log_write(f"[Pre-Solve Research] Search for '{q}' encountered error: {e}\n")

        if not all_snippets:
            self._log_write("[Pre-Solve Research] No relevant literature snippets retrieved.\n")
            return None

        # Step 3: Theoretical Briefing Synthesis
        synth_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a pure mathematics research specialist. Synthesize the following research literature snippets "
                    "into a dense, structured theoretical briefing (under 300 words) to equip an elite mathematical solver.\n"
                    "CRITICAL: Enclose your final synthesized briefing inside <briefing>...</briefing> tags.\n"
                    "Structure your briefing concisely:\n"
                    "- GOVERNING DEFINITIONS & OBJECTS\n"
                    "- ESTABLISHED THEOREMS & KNOWN FORMULAS (e.g. Euler products, character sums, generating functions, spectral decompositions)\n"
                    "- EXACT ANALYTIC EVALUATION STRATEGY: Explicitly identify the analytic formula or exact evaluation protocol "
                    "(e.g. Kummer character sums, Euler product acceleration, Dirichlet convolution) so the solver avoids wasting turns on brute-force empirical simulations.\n"
                    "Be dense, rigorous, and purely mathematical. Avoid meta-commentary."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem Statement:\n{problem[:1000]}\n\n"
                    f"Retrieved Literature Snippets:\n" + "\n\n".join(all_snippets[:4])[:3500]
                ),
            },
        ]

        try:
            synth_resp = await self.client.acomplete(synth_prompt, temperature=0.1, max_tokens=2048, enable_thinking=False, timeout_seconds=60.0)
            raw_text = (synth_resp.content or "").strip()
            if not raw_text and synth_resp.reasoning:
                raw_text = synth_resp.reasoning.strip()

            extracted_brief = re.search(r"<briefing>(.*?)</briefing>", raw_text, flags=re.DOTALL)
            if extracted_brief:
                briefing = extracted_brief.group(1).strip()
            else:
                briefing = raw_text

            if briefing and not any(briefing.lower().startswith(p) for p in ["we need answer", "user asks", "you are prime math"]):
                self._log_write(f"\n[Stage 0: Pre-Solve Research Complete]\n{briefing}\n\n")
                return briefing
        except Exception as e:
            self._log_write(f"[Pre-Solve Research] Briefing synthesis failed: {e}\n")

        return None

    def _try_fast_composer(self, problem: str, on_step: Optional[Any] = None) -> Optional[SolveResult]:
        """Attempts a 1-turn speculative solve using the local Gemma 4 composer model."""
        try:
            self._log_write("[Two-Tier Routing] Launching Tier 1 Fast Composer (Local Gemma 4 on LAN)...\n")
            code = self.fast_composer.synthesize_code(problem)
            if not code:
                return None

            self._log_write(f"--- [TIER 1 FAST COMPOSER CODE] ---\n```python\n{code}\n```\n")
            exec_res = self.kernel.execute(code, timeout_seconds=15.0)
            if not exec_res.success:
                self._log_write(f"[Tier 1 Error] Execution failed: {exec_res.error}\n")
                return None

            stdout_clean = exec_res.stdout.strip()
            self._log_write(f"[Tier 1 Output] Stdout: {stdout_clean}\n")
            if not stdout_clean:
                return None

            lines = [ln.strip() for ln in stdout_clean.splitlines() if ln.strip()]
            if not lines:
                return None
            last_line = lines[-1]

            cand = last_line
            if ":" in cand:
                cand = cand.split(":")[-1].strip()
            match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", cand)
            if not match:
                return None

            ans = match.group(0).strip()
            if not ans or len(ans) > 60:
                return None

            boxed_ans = ans
            res = SolveResult(
                problem=problem,
                answer=boxed_ans,
                boxed_answer=boxed_ans,
                verified=True,
                iterations=1,
                reasoning_traces=["[Tier 1 Fast Composer (Gemma 4)] Direct synthesis"],
                code_history=[code],
                execution_history=[exec_res],
                final_response=f"Answer: \\boxed{{{boxed_ans}}}",
                log_file=str(self.log_path or DEFAULT_LOG_PATH),
            )
            # Colosseum cross-model critique: the stronger tunnel model adversarially
            # falsifies the speculative Tier 1 answer before it is accepted.
            verdict, reason = self._verify_final_answer(problem, res)
            if verdict == "VERIFIED":
                res.self_verified = True
                self._fast_composer_critique = None
                self._log_write(
                    f"\n{'=' * 80}\n"
                    f"[TIER 1 COMPLETE] Verified Boxed Answer: {boxed_ans} in 1 turn (Execution: {exec_res.execution_time}s)\n"
                    f"{'=' * 80}\n"
                )
                if on_step:
                    on_step("answer_boxed", boxed_ans)
                return res
            elif verdict == "WRONG" and reason:
                self._log_write(
                    f"[Two-Tier Routing] Tier 1 answer '{boxed_ans}' FALSIFIED by adversarial verification: {reason}\n"
                    "Escalating to Deep Orchestrator with failed-route context...\n"
                )
                self._fast_composer_critique = (
                    f"A prior speculative (Fast Composer) attempt claimed the answer {boxed_ans}. "
                    f"An adversarial verifier falsified it: {reason}. Do NOT reuse that approach or its formulation; "
                    "derive and verify your own result independently from PROBLEM_TEXT."
                )
                return None
            else:
                self._log_write(
                    f"[Two-Tier Routing] Tier 1 answer '{boxed_ans}' verification was INCONCLUSIVE ({verdict}). "
                    "Escalating to Deep Orchestrator for full reasoning...\n"
                )
                self._fast_composer_critique = None
                return None
        except Exception as e:
            self._log_write(f"[Tier 1 Exception] {e}\n")
            return None

    async def _atry_fast_composer(self, problem: str, on_step: Optional[Any] = None) -> Optional[SolveResult]:
        """Asynchronously attempts a 1-turn speculative solve using the local Gemma 4 composer model."""
        try:
            self._log_write("[Two-Tier Routing] Launching Tier 1 Fast Composer (Local Gemma 4 on LAN)...\n")
            code = await self.fast_composer.asynthesize_code(problem)
            if not code:
                return None

            self._log_write(f"--- [TIER 1 FAST COMPOSER CODE] ---\n```python\n{code}\n```\n")
            exec_res = self.kernel.execute(code, timeout_seconds=15.0)
            if not exec_res.success:
                self._log_write(f"[Tier 1 Error] Execution failed: {exec_res.error}\n")
                return None

            stdout_clean = exec_res.stdout.strip()
            self._log_write(f"[Tier 1 Output] Stdout: {stdout_clean}\n")
            if not stdout_clean:
                return None

            lines = [ln.strip() for ln in stdout_clean.splitlines() if ln.strip()]
            if not lines:
                return None
            last_line = lines[-1]

            cand = last_line
            if ":" in cand:
                cand = cand.split(":")[-1].strip()
            match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", cand)
            if not match:
                return None

            ans = match.group(0).strip()
            if not ans or len(ans) > 60:
                return None

            boxed_ans = ans
            res = SolveResult(
                problem=problem,
                answer=boxed_ans,
                boxed_answer=boxed_ans,
                verified=True,
                iterations=1,
                reasoning_traces=["[Tier 1 Fast Composer (Gemma 4)] Direct synthesis"],
                code_history=[code],
                execution_history=[exec_res],
                final_response=f"Answer: \\boxed{{{boxed_ans}}}",
                log_file=str(self.log_path or DEFAULT_LOG_PATH),
            )
            # Colosseum cross-model critique: the stronger tunnel model adversarially
            # falsifies the speculative Tier 1 answer before it is accepted.
            verdict, reason = await self._averify_final_answer(problem, res)
            if verdict == "VERIFIED":
                res.self_verified = True
                self._fast_composer_critique = None
                self._log_write(
                    f"\n{'=' * 80}\n"
                    f"[TIER 1 COMPLETE] Verified Boxed Answer: {boxed_ans} in 1 turn (Execution: {exec_res.execution_time}s)\n"
                    f"{'=' * 80}\n"
                )
                if on_step:
                    on_step("answer_boxed", boxed_ans)
                return res
            elif verdict == "WRONG" and reason:
                self._log_write(
                    f"[Two-Tier Routing] Tier 1 answer '{boxed_ans}' FALSIFIED by adversarial verification: {reason}\n"
                    "Escalating to Deep Orchestrator with failed-route context...\n"
                )
                self._fast_composer_critique = (
                    f"A prior speculative (Fast Composer) attempt claimed the answer {boxed_ans}. "
                    f"An adversarial verifier falsified it: {reason}. Do NOT reuse that approach or its formulation; "
                    "derive and verify your own result independently from PROBLEM_TEXT."
                )
                return None
            else:
                self._log_write(
                    f"[Two-Tier Routing] Tier 1 answer '{boxed_ans}' verification was INCONCLUSIVE ({verdict}). "
                    "Escalating to Deep Orchestrator for full reasoning...\n"
                )
                self._fast_composer_critique = None
                return None
        except Exception as e:
            self._log_write(f"[Tier 1 Exception] {e}\n")
            return None

    # ========================================================================
    # Adversarial Verification & Stall Detection (Colosseum-inspired)
    # ========================================================================

    def _execution_error_feedback(self, exec_res: ExecutionResult) -> str:
        """Builds execution-error feedback with Colosseum-style dead-route stall detection.

        Tracks the signature (first line) of consecutive execution errors. When the identical
        error repeats 3 times in a row, the current route is declared a dead route and the
        model receives an explicit strategy-reset (re-exploration) directive instead of the
        generic 'analyze and continue' feedback.
        """
        error_text = (exec_res.error or "").strip()
        first_line = error_text.splitlines()[0][:120] if error_text else "unknown error"

        self._error_signatures.append(first_line)
        if len(self._error_signatures) > 3:
            self._error_signatures.pop(0)

        feedback = (
            f"[Execution Error]:\n{exec_res.error}\n"
            "Please analyze why this error occurred, adjust your code or approach, and continue."
        )
        if len(self._error_signatures) == 3 and all(s == first_line for s in self._error_signatures):
            feedback = (
                f"[Execution Error]:\n{exec_res.error}\n"
                f"[STALLED ROUTE] This exact error has now occurred {len(self._error_signatures)} consecutive times. "
                "The current approach is a dead route: minor tweaks to the same code will keep failing. "
                "State in one sentence why this route cannot work, then switch to a fundamentally different strategy "
                "(e.g. reformulate the problem in Z3, use a different combinatorial decomposition or generating function, "
                "look up OEIS / literature, derive symbolically with SymPy, or probe smaller cases) and execute new Python code."
            )
        if "ExecutionBlockedByRule11" in (exec_res.error or ""):
            feedback += (
                "\n[ANALYTICAL PERSISTENCE DIRECTIVE]: Brute-force simulation over large ranges was blocked by Rule 11. "
                "Do NOT escalate empirical simulations. Persist with your theoretical derivation from first principles: "
                "formulate the governing analytic formula, local factors, or Euler product, and evaluate it in Python."
            )
        return feedback

    def _last_code_and_stdout(self, result: SolveResult) -> Tuple[str, str]:
        """Returns the last executed code block and the last non-empty stdout tail."""
        last_code = result.code_history[-1] if result.code_history else ""
        last_stdout = ""
        for er in reversed(result.execution_history):
            if er.stdout and er.stdout.strip():
                last_stdout = er.stdout.strip()
                break
        return last_code, last_stdout

    def _build_verification_messages(self, problem: str, result: SolveResult) -> List[Dict[str, str]]:
        """Builds the adversarial verification prompt for the final boxed answer."""
        last_code, last_stdout = self._last_code_and_stdout(result)
        return [
            {"role": "system", "content": MATH_VERIFICATION_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Problem:\n{problem}\n\n"
                    f"Claimed final answer: \\boxed{{{result.boxed_answer}}}\n\n"
                    f"Solver's last executed code:\n```python\n{last_code[-3000:]}\n```\n\n"
                    f"Execution output (tail):\n{last_stdout[-2000:]}\n\n"
                    "Now verify the claimed answer. Output exactly one of VERIFIED, INCONCLUSIVE, "
                    "or WRONG: <concrete defect> as the first line of your reply."
                ),
            },
        ]

    @staticmethod
    def _parse_verification_verdict(text: Optional[str]) -> Tuple[str, str]:
        """Parses the critic's first-line verdict. Conservative: anything ambiguous is INCONCLUSIVE."""
        if not text or not text.strip():
            return "INCONCLUSIVE", ""
        first_line = text.strip().splitlines()[0].strip().upper()
        if first_line.startswith("WRONG") and (len(first_line) == 5 or first_line[5] in ": \t"):
            reason = text.strip()
            if len(reason) > 5:
                reason = reason[5:].lstrip(": \t")
            return "WRONG", reason[:500].strip()
        if first_line.startswith("VERIFIED"):
            return "VERIFIED", ""
        if first_line.startswith("INCONCLUSIVE"):
            return "INCONCLUSIVE", ""
        return "INCONCLUSIVE", ""

    def _verify_final_answer(self, problem: str, result: SolveResult) -> Tuple[str, str]:
        """One-shot adversarial verification of the final boxed answer (Colosseum global
        verification + targeted falsification). Returns (verdict, reason). Never raises: any
        failure yields ("INCONCLUSIVE", "") so the original answer is preserved.
        """
        if not result.boxed_answer:
            return "INCONCLUSIVE", ""
        self._log_write(
            f"\n--- [ADVERSARIAL VERIFICATION] Falsifying final answer \\boxed{{{result.boxed_answer}}} ---\n"
        )
        try:
            resp = self.client.complete(
                self._build_verification_messages(problem, result),
                temperature=0.1, max_tokens=1500, enable_thinking=False, timeout_seconds=60.0,
            )
            text = str(getattr(resp, "content", "") or "")
        except Exception as e:
            self._log_write(f"[Verification] Verification call failed: {e}\n")
            return "INCONCLUSIVE", ""
        verdict, reason = self._parse_verification_verdict(text)
        self._log_write(f"[Verification] Critic verdict: {verdict}" + (f" | {reason}\n" if reason else "\n"))
        return verdict, reason

    async def _averify_final_answer(self, problem: str, result: SolveResult) -> Tuple[str, str]:
        """Asynchronous one-shot adversarial verification of the final boxed answer."""
        if not result.boxed_answer:
            return "INCONCLUSIVE", ""
        self._log_write(
            f"\n--- [ADVERSARIAL VERIFICATION] Falsifying final answer \\boxed{{{result.boxed_answer}}} ---\n"
        )
        try:
            resp = await self.client.acomplete(
                self._build_verification_messages(problem, result),
                temperature=0.1, max_tokens=1500, enable_thinking=False, timeout_seconds=60.0,
            )
            text = str(getattr(resp, "content", "") or "")
        except Exception as e:
            self._log_write(f"[Verification] Verification call failed: {e}\n")
            return "INCONCLUSIVE", ""
        verdict, reason = self._parse_verification_verdict(text)
        self._log_write(f"[Verification] Critic verdict: {verdict}" + (f" | {reason}\n" if reason else "\n"))
        return verdict, reason

    def _build_revision_prompt(self, old_answer: str, reason: str) -> str:
        """Builds the revision-turn prompt carrying the verifier's finding (findings stay
        attached to the candidate, Colosseum revision path)."""
        return (
            f"[ADVERSARIAL VERIFICATION DEFECT] An independent adversarial verifier rejected your final "
            f"answer \\boxed{{{old_answer}}} with a concrete defect:\n{reason}\n\n"
            "The previous route is DISCARDED. Do NOT reuse its formulation, code, or intermediate results. "
            "Re-solve the problem by a fundamentally different route, execute the computation in Python "
            "(PROBLEM_TEXT is preloaded), and output the new final answer strictly as \\boxed{your_answer}."
        )

    def _run_stream_turn(
        self,
        messages: List[Dict[str, str]],
        result: SolveResult,
        turn: int,
        max_turns: int,
        on_step: Optional[Any] = None,
    ) -> Optional[str]:
        """Runs a single CoALA reasoning turn: streams from the model, executes code blocks in the
        persistent kernel, and appends feedback/reprompt messages. Mutates messages/result in place.
        Returns the accepted boxed answer (caller breaks the turn loop); None if the turn continues.
        """
        self._log_write(f"\n--- [TURN {turn}] INNER THINKING (STREAMING FROM QWEN ENGINE) ---\n")
        if on_step:
            on_step("turn_start", turn)

        reasoning_chunks: List[str] = []
        content_chunks: List[str] = []
        switched_to_content = False

        for typ, delta in self.client.stream_complete(messages):
            if typ == "reasoning":
                reasoning_chunks.append(delta)
                self._log_write(delta)
                if on_step:
                    on_step("reasoning_delta", delta)
            elif typ == "content":
                if not switched_to_content:
                    self._log_write(f"\n\n--- [TURN {turn}] MODEL GENERATED PROPOSAL & CODE ---\n")
                    switched_to_content = True
                    if on_step:
                        on_step("content_start", "")
                content_chunks.append(delta)
                self._log_write(delta)
                if on_step:
                    on_step("content_delta", delta)

        reasoning_text = "".join(reasoning_chunks).strip()
        model_text = "".join(content_chunks).strip()

        if reasoning_text:
            result.reasoning_traces.append(reasoning_text)
        result.final_response = model_text
        if on_step:
            on_step("turn_done", {"turn": turn, "reasoning": reasoning_text, "content": model_text})

        code_blocks = self.extract_code_blocks(model_text)
        if not code_blocks and reasoning_text:
            code_blocks = self.extract_code_blocks(reasoning_text)

        if code_blocks:
            msg_content = model_text if model_text.strip() else f"```python\n{code_blocks[0]}\n```"
            messages.append({"role": "assistant", "content": msg_content})
            exec_feedbacks = []

            for code in code_blocks:
                result.code_history.append(code)
                self._log_write(f"\n\n--- [TURN {turn}] EXECUTING CODE IN PERSISTENT REPL ---\n```python\n{code}\n```\n")
                if on_step:
                    on_step("code_start", code)

                exec_res = self.kernel.execute(code)
                result.execution_history.append(exec_res)
                if on_step:
                    on_step("code_done", exec_res)

                if exec_res.success:
                    self._error_signatures.clear()
                    self._log_write(f"--- [TURN {turn}] EXECUTION SUCCESS ({exec_res.execution_time}s) ---\n")
                    if exec_res.stdout:
                        self._log_write(f"Stdout:\n{exec_res.stdout}\n")
                    if exec_res.result_repr:
                        self._log_write(f"Evaluated Value: {exec_res.result_repr}\n")

                    feedback = f"[Execution Output]:\n{exec_res.stdout}"
                    if exec_res.result_repr:
                        feedback += f"\n[Result Value]: {exec_res.result_repr}"
                    if not exec_res.stdout and not exec_res.result_repr:
                        feedback += "[Executed successfully with no output]"
                else:
                    self._log_write(f"--- [TURN {turn}] EXECUTION ERROR ({exec_res.execution_time}s) ---\n{exec_res.error}\n")
                    feedback = self._execution_error_feedback(exec_res)
                exec_feedbacks.append(feedback)

            if turn >= max_turns - 3:
                exec_feedbacks.append(
                    f"[TURN ALERT: Turn {turn} of {max_turns}. You are approaching the turn limit. "
                    f"Per Rule 17, terminate open-ended research. Execute your final numerical calculation in Python "
                    f"and output your final answer strictly as \\boxed{{integer}}.]"
                )

            messages.append({
                "role": "user",
                "content": "\n\n".join(exec_feedbacks),
            })

        else:
            boxed = self.extract_boxed_answer(model_text)
            if boxed and (turn >= 2 or len(result.code_history) > 0):
                result.boxed_answer = boxed
                result.answer = boxed
                result.verified = len(result.code_history) > 0
                if on_step:
                    on_step("answer_boxed", boxed)
                return boxed

            content_to_add = model_text.strip()
            if reasoning_text and len(content_to_add) < 400:
                tail = reasoning_text[-3000:].strip()
                if content_to_add:
                    combined = f"{content_to_add}\n\nTheoretical formulation:\n{tail}\n\nI will now implement this formulation in Python."
                else:
                    combined = f"Theoretical formulation:\n{tail}\n\nI will now implement this formulation in Python."
                messages.append({"role": "assistant", "content": combined})
            elif content_to_add:
                messages.append({"role": "assistant", "content": content_to_add})
            else:
                messages.append({"role": "assistant", "content": "I will write Python code using PROBLEM_TEXT to compute the answer."})

            if reasoning_text:
                reprompt = (
                    "CRITICAL: Continue directly from your theoretical formulation above. "
                    "You MUST write executable Python code enclosed in ```python ... ``` blocks to evaluate this formula / Euler product. "
                    "Do NOT abandon this analytical path or retreat to brute force sieving. "
                    "Implement and execute the Python computation now."
                )
            else:
                reprompt = (
                    "CRITICAL: You MUST write executable Python code enclosed in ```python ... ``` blocks to compute the answer. "
                    "Do NOT guess, recall from memory, or do mental arithmetic. "
                    "The variable PROBLEM_TEXT is already preloaded in Python. Write and run code now."
                )
            if boxed and len(result.code_history) == 0:
                reprompt = (
                    f"CRITICAL: You declared \\boxed{{{boxed}}} without executing any Python code to verify it! "
                    "Mental calculation or unverified guessing is STRICTLY PROHIBITED. "
                    "You MUST write and execute Python code in ```python ... ``` blocks to compute and verify this answer now."
                )
            messages.append({
                "role": "user",
                "content": reprompt,
            })

        boxed = self.extract_boxed_answer(model_text)
        if boxed and (turn >= 2 or len(result.code_history) > 0):
            result.boxed_answer = boxed
            result.answer = boxed
            result.verified = True
            if on_step:
                on_step("answer_boxed", boxed)
            return boxed
        return None

    async def _arun_stream_turn(
        self,
        messages: List[Dict[str, str]],
        result: SolveResult,
        turn: int,
        max_turns: int,
    ) -> Optional[str]:
        """Asynchronous version of _run_stream_turn."""
        self._log_write(f"\n--- [TURN {turn}] INNER THINKING (STREAMING FROM QWEN ENGINE) ---\n")

        reasoning_chunks: List[str] = []
        content_chunks: List[str] = []
        switched_to_content = False

        async for typ, delta in self.client.astream_complete(messages):
            if typ == "reasoning":
                reasoning_chunks.append(delta)
                self._log_write(delta)
            elif typ == "content":
                if not switched_to_content:
                    self._log_write(f"\n\n--- [TURN {turn}] MODEL GENERATED PROPOSAL & CODE ---\n")
                    switched_to_content = True
                content_chunks.append(delta)
                self._log_write(delta)

        reasoning_text = "".join(reasoning_chunks).strip()
        model_text = "".join(content_chunks).strip()

        if reasoning_text:
            result.reasoning_traces.append(reasoning_text)
        result.final_response = model_text

        code_blocks = self.extract_code_blocks(model_text)
        if not code_blocks and reasoning_text:
            code_blocks = self.extract_code_blocks(reasoning_text)

        if code_blocks:
            msg_content = model_text if model_text.strip() else f"```python\n{code_blocks[0]}\n```"
            messages.append({"role": "assistant", "content": msg_content})
            exec_feedbacks = []

            for code in code_blocks:
                result.code_history.append(code)
                self._log_write(f"\n\n--- [TURN {turn}] EXECUTING CODE IN PERSISTENT REPL ---\n```python\n{code}\n```\n")

                exec_res = self.kernel.execute(code)
                result.execution_history.append(exec_res)

                if exec_res.success:
                    self._error_signatures.clear()
                    self._log_write(f"--- [TURN {turn}] EXECUTION SUCCESS ({exec_res.execution_time}s) ---\n")
                    if exec_res.stdout:
                        self._log_write(f"Stdout:\n{exec_res.stdout}\n")
                    if exec_res.result_repr:
                        self._log_write(f"Evaluated Value: {exec_res.result_repr}\n")

                    feedback = f"[Execution Output]:\n{exec_res.stdout}"
                    if exec_res.result_repr:
                        feedback += f"\n[Result Value]: {exec_res.result_repr}"
                    if not exec_res.stdout and not exec_res.result_repr:
                        feedback += "[Executed successfully with no output]"
                else:
                    self._log_write(f"--- [TURN {turn}] EXECUTION ERROR ({exec_res.execution_time}s) ---\n{exec_res.error}\n")
                    feedback = self._execution_error_feedback(exec_res)
                exec_feedbacks.append(feedback)

            if turn >= max_turns - 3:
                exec_feedbacks.append(
                    f"[TURN ALERT: Turn {turn} of {max_turns}. You are approaching the turn limit. "
                    f"Per Rule 17, terminate open-ended research. Execute your final numerical calculation in Python "
                    f"and output your final answer strictly as \\boxed{{integer}}.]"
                )

            messages.append({
                "role": "user",
                "content": "\n\n".join(exec_feedbacks),
            })

        else:
            boxed = self.extract_boxed_answer(model_text)
            if boxed and (turn >= 2 or len(result.code_history) > 0):
                result.boxed_answer = boxed
                result.answer = boxed
                result.verified = len(result.code_history) > 0
                return boxed

            content_to_add = model_text.strip()
            if reasoning_text and len(content_to_add) < 400:
                tail = reasoning_text[-3000:].strip()
                if content_to_add:
                    combined = f"{content_to_add}\n\nTheoretical formulation:\n{tail}\n\nI will now implement this formulation in Python."
                else:
                    combined = f"Theoretical formulation:\n{tail}\n\nI will now implement this formulation in Python."
                messages.append({"role": "assistant", "content": combined})
            elif content_to_add:
                messages.append({"role": "assistant", "content": content_to_add})
            else:
                messages.append({"role": "assistant", "content": "I will write Python code using PROBLEM_TEXT to compute the answer."})

            if reasoning_text:
                reprompt = (
                    "CRITICAL: Continue directly from your theoretical formulation above. "
                    "You MUST write executable Python code enclosed in ```python ... ``` blocks to evaluate this formula / Euler product. "
                    "Do NOT abandon this analytical path or retreat to brute force sieving. "
                    "Implement and execute the Python computation now."
                )
            else:
                reprompt = (
                    "CRITICAL: You MUST write executable Python code enclosed in ```python ... ``` blocks to compute the answer. "
                    "Do NOT guess, recall from memory, or do mental arithmetic. "
                    "The variable PROBLEM_TEXT is already preloaded in Python. Write and run code now."
                )
            if boxed and len(result.code_history) == 0:
                reprompt = (
                    f"CRITICAL: You declared \\boxed{{{boxed}}} without executing any Python code to verify it! "
                    "Mental calculation or unverified guessing is STRICTLY PROHIBITED. "
                    "You MUST write and execute Python code in ```python ... ``` blocks to compute and verify this answer now."
                )
            messages.append({
                "role": "user",
                "content": reprompt,
            })

        boxed = self.extract_boxed_answer(model_text)
        if boxed and (turn >= 2 or len(result.code_history) > 0):
            result.boxed_answer = boxed
            result.answer = boxed
            result.verified = True
            return boxed
        return None

    def solve(
        self,
        problem: str,
        max_iterations: Optional[int] = None,
        max_time_seconds: Optional[float] = None,
        reset_kernel: bool = True,
        on_step: Optional[Any] = None,
        checkpoint_path: Optional[Path] = None,
    ) -> SolveResult:
        """Solves a math problem synchronously through the CoALA reasoning cycle with TurboVec memory."""
        # 1. Check TurboVec health strictly (raises TurboVecUnavailableError if down)
        self.memory.check_health()

        max_turns = max_iterations or self.default_max_iterations
        ckpt_path = Path(checkpoint_path) if checkpoint_path else self.checkpoint_path
        ckpt = self._load_checkpoint(ckpt_path, problem) if ckpt_path else None
        if ckpt is not None and ckpt.get("problem") == problem:
            prior_elapsed = float(ckpt.get("elapsed_s", 0.0))
            start_wall_time = time.perf_counter() - prior_elapsed
            completed_turn = int(ckpt.get("completed_turn", 0))
            start_turn = completed_turn + 1
            messages: List[Dict[str, str]] = ckpt["messages"]
            result = SolveResult.from_dict(ckpt["result"])
            self._error_signatures = list(ckpt.get("error_signatures", []))
            self._fast_composer_critique = ckpt.get("fast_composer_critique")
            self._restore_kernel_state(
                problem=problem,
                result=result,
                state_machine_dict=ckpt.get("state_machine"),
                reset_kernel=reset_kernel,
            )
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._log_write(
                f"\n\n{'=' * 80}\n"
                f"[{now_str}] PRIME MATH RESUMING SOLVE SESSION FROM CHECKPOINT\n"
                f"Completed Turn: {completed_turn} | Resuming at Turn: {start_turn}/{max_turns} | Prior Elapsed: {prior_elapsed:.1f}s\n"
                f"{'=' * 80}\n"
            )
            if on_step:
                on_step("checkpoint_resume", {"completed_turn": completed_turn, "start_turn": start_turn})
        else:
            start_wall_time = time.perf_counter()
            start_turn = 1
            if reset_kernel:
                self.kernel.reset()
            # Preload raw problem text into persistent kernel
            self.kernel.execute(f"import re\nPROBLEM_TEXT = {repr(problem)}")

            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._log_write(
                f"\n\n{'=' * 80}\n"
                f"[{now_str}] PRIME MATH NEW SOLVE SESSION\n"
                f"Problem: {problem}\n"
                f"{'=' * 80}\n"
            )
            # Reset per-problem Colosseum runtime state (critique injection + stall window)
            self._fast_composer_critique = None
            self._error_signatures = []

            # 0. TIER 1: Speculative Fast Composer Execution (Local Gemma 4 on LAN)
            if self.fast_composer and self.fast_composer.fast_cfg.enabled:
                fast_res = self._try_fast_composer(problem, on_step=on_step)
                if fast_res:
                    self._unlink_checkpoint(ckpt_path)
                    return fast_res
                self._log_write("[Two-Tier Routing] Fast Composer did not resolve. Escalating to Deep Orchestrator (Qwen 3.8)...\n")

            # 1. Stage 0: Pre-Solve Academic Research & Literature Briefing
            research_briefing = self._conduct_pre_solve_research(problem)

            # 2. Retrieve domain-relevant procedural heuristics from TurboVec
            # Formulate clean conceptual query from title/keywords to prevent 500-word embedding dilution
            prob_id, prob_title = parse_benchmark_header(problem)
            stripped_problem = strip_benchmark_prefix(problem)
            first_line = stripped_problem.splitlines()[0] if stripped_problem.splitlines() else ""
            clean_title = prob_title or (first_line.split(" - ")[-1].split(":")[0].strip() if " - " in first_line else first_line[:120])
            concept_query = clean_title or stripped_problem[:200]
            if self.fast_composer:
                try:
                    queries = self.fast_composer.generate_research_queries(stripped_problem)
                    if queries:
                        concept_query = f"{clean_title} {' '.join(queries)}".strip()
                except Exception:
                    pass

            retrieved = self.memory.query_procedural_heuristics(
                concept_query, limit=5, min_score=0.50, exclude_problem_id=prob_id
            )
            if retrieved:
                self._log_write(f"[CoALA Procedural Memory] Retrieved {len(retrieved)} relevant rules from TurboVec for '{concept_query}':\n")
                for h in retrieved:
                    self._log_write(f"  • {h}\n")

            system_content = MATH_SYSTEM_PROMPT
            if retrieved:
                heuristics_block = "\n\nRELEVANT PROCEDURAL RULES & PROVEN HEURISTICS (FROM COALA MEMORY):\n"
                for h in retrieved:
                    heuristics_block += f"- {h}\n"
                system_content += heuristics_block

            if research_briefing:
                briefing_block = (
                    "\n\nACADEMIC LITERATURE & THEORETICAL BRIEFING (PRE-SOLVE RESEARCH):\n"
                    f"{research_briefing}\n"
                )
                system_content += briefing_block

            user_content = f"Solve the following mathematical problem:\n\n{problem}"
            if self._fast_composer_critique:
                user_content += (
                    "\n\n[FAST COMPOSER CRITIQUE - FALSIFIED SPECULATIVE ROUTE] "
                    f"{self._fast_composer_critique}"
                )
                self._log_write(
                    f"[Two-Tier Routing] Injecting falsified-route critique into Deep Orchestrator prompt:\n"
                    f"{self._fast_composer_critique}\n"
                )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            result = SolveResult(
                problem=problem,
                answer=None,
                boxed_answer=None,
                verified=False,
                iterations=0,
                retrieved_heuristics=retrieved,
                research_briefing=research_briefing,
                log_file=str(self.log_path),
            )
            self._save_checkpoint(
                ckpt_path,
                problem=problem,
                completed_turn=0,
                start_wall_time=start_wall_time,
                messages=messages,
                result=result,
            )

        for turn in range(start_turn, max_turns + 1):
            result.iterations = turn

            if max_time_seconds and (time.perf_counter() - start_wall_time) > max_time_seconds:
                self._log_write(
                    f"\n[Solver Timeout] Wall-clock time limit ({max_time_seconds:.1f}s) exceeded; "
                    f"terminating reasoning turns and extracting best answer.\n"
                )
                break

            if self.compactor.should_compact(messages):
                comp_res = self.compactor.compact(messages)
                if comp_res.compacted:
                    messages = comp_res.compacted_messages
                    self._log_write(
                        f"\n[Context Compaction] Context window compacted: "
                        f"{comp_res.original_tokens} -> {comp_res.new_tokens} tokens\n"
                    )

            early_stop = self._run_stream_turn(messages, result, turn, max_turns, on_step)
            self._save_checkpoint(
                ckpt_path,
                problem=problem,
                completed_turn=turn,
                start_wall_time=start_wall_time,
                messages=messages,
                result=result,
            )
            if early_stop:
                break

        # 2.5. Post-solve adversarial verification (Colosseum): falsify the final boxed answer.
        # Runs before answer extraction so the wrap-up below applies if the revision ends boxed-less.
        if result.boxed_answer:
            verdict, reason = self._verify_final_answer(problem, result)
            if verdict == "WRONG" and reason:
                rejected = result.boxed_answer
                self._log_write(
                    f"\n--- [ADVERSARIAL VERIFICATION] Final answer \\boxed{{{rejected}}} REJECTED by independent verifier:\n"
                    f"{reason}\nRunning one revision turn via a fundamentally different route...\n"
                )
                # The revision discards the falsified route: reset the stall window
                self._error_signatures = []
                result.boxed_answer = None
                result.answer = None
                result.verified = False
                messages.append({"role": "user", "content": self._build_revision_prompt(rejected, reason)})
                revision_turn = result.iterations + 1
                self._run_stream_turn(messages, result, revision_turn, max_turns, on_step)
                result.iterations = revision_turn
            elif verdict == "VERIFIED":
                result.self_verified = True

        if not result.boxed_answer:
            result.boxed_answer = self.extract_boxed_answer(result.final_response)
            result.answer = result.boxed_answer
            if result.boxed_answer:
                result.verified = len(result.code_history) > 0

        # If still no boxed answer, but code was executed, prompt a fast wrap-up to extract final answer
        if not result.boxed_answer and result.code_history:
            last_stdout = ""
            for er in reversed(result.execution_history):
                if er.stdout and er.stdout.strip():
                    last_stdout = er.stdout.strip()
                    break
            last_code = result.code_history[-1] if result.code_history else ""

            compact_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a mathematical answer extraction assistant. "
                        "State the exact final numerical or algebraic answer strictly inside LaTeX boxed format: \\boxed{your_answer}.\n"
                        "CRITICAL SANITY CHECKS:\n"
                        "- For density, probability, or proportion problems (e.g. d_x, d_\\infty), the answer MUST be strictly non-negative (0 <= ans). Never output a negative answer for a density problem; reject negative regression or fit artifacts in favor of empirical sample medians or positive limits.\n"
                        "Provide a brief direct answer line, then output \\boxed{answer} immediately without lengthy re-derivation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Problem:\n{problem}\n\n"
                        f"Last executed Python code:\n```python\n{last_code}\n```\n\n"
                        f"Execution output:\n{last_stdout}\n\n"
                        "Based on the above problem and Python execution output, state the final answer "
                        "strictly inside LaTeX boxed format: \\boxed{your_answer}."
                    ),
                },
            ]
            try:
                self._log_write("\n--- [FINAL EXTRACTION WRAP-UP TURN] ---\n")
                resp = self.client.complete(compact_messages, temperature=0.1, max_tokens=1024, enable_thinking=False, timeout_seconds=30.0)
                wrap_text = resp.content.strip()
                self._log_write(wrap_text + "\n")
                boxed = self.extract_boxed_answer(wrap_text) or self.extract_boxed_answer(resp.reasoning or "")
                if not boxed:
                    boxed = self.extract_answer_from_text(wrap_text)

                # Reject unphysical negative values for density problems
                is_density = any(kw in problem.lower() for kw in ["density", "probability", "proportion", "d_x", r"d_\infty"])
                if boxed and is_density and boxed.strip().startswith("-"):
                    logger.warning(f"Rejecting unphysical negative density in wrap-up: {boxed}")
                    boxed = None

                if boxed:
                    result.boxed_answer = boxed
                    result.answer = boxed
                    result.verified = True
                    result.final_response = wrap_text
            except Exception as e:
                self._log_write(f"Wrap-up extraction failed: {e}\n")

        # Final fallback from history if still None
        if not result.boxed_answer:
            fallback = self._extract_best_answer_from_history(result)
            if fallback:
                result.boxed_answer = fallback
                result.answer = fallback
                result.verified = len(result.code_history) > 0
                self._log_write(f"[FINAL FALLBACK] Extracted answer from history: {fallback}\n")

        self._log_write(
            f"\n{'=' * 80}\n"
            f"[SESSION COMPLETE] Final Boxed Answer: {result.boxed_answer} (Verified: {result.verified}, Turns: {result.iterations})\n"
            f"{'=' * 80}\n"
        )
        self._unlink_checkpoint(ckpt_path)

        return result

    async def asolve(
        self,
        problem: str,
        max_iterations: Optional[int] = None,
        max_time_seconds: Optional[float] = None,
        reset_kernel: bool = True,
        checkpoint_path: Optional[Path] = None,
    ) -> SolveResult:
        """Asynchronous version of solve with CoALA TurboVec integration."""
        # 1. Check TurboVec health strictly
        await self.memory.acheck_health()

        max_turns = max_iterations or self.default_max_iterations
        ckpt_path = Path(checkpoint_path) if checkpoint_path else self.checkpoint_path
        ckpt = self._load_checkpoint(ckpt_path, problem) if ckpt_path else None
        if ckpt is not None and ckpt.get("problem") == problem:
            prior_elapsed = float(ckpt.get("elapsed_s", 0.0))
            start_wall_time = time.perf_counter() - prior_elapsed
            completed_turn = int(ckpt.get("completed_turn", 0))
            start_turn = completed_turn + 1
            messages: List[Dict[str, str]] = ckpt["messages"]
            result = SolveResult.from_dict(ckpt["result"])
            self._error_signatures = list(ckpt.get("error_signatures", []))
            self._fast_composer_critique = ckpt.get("fast_composer_critique")
            self._restore_kernel_state(
                problem=problem,
                result=result,
                state_machine_dict=ckpt.get("state_machine"),
                reset_kernel=reset_kernel,
            )
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._log_write(
                f"\n\n{'=' * 80}\n"
                f"[{now_str}] PRIME MATH RESUMING SOLVE SESSION FROM CHECKPOINT\n"
                f"Completed Turn: {completed_turn} | Resuming at Turn: {start_turn}/{max_turns} | Prior Elapsed: {prior_elapsed:.1f}s\n"
                f"{'=' * 80}\n"
            )
        else:
            start_wall_time = time.perf_counter()
            start_turn = 1
            if reset_kernel:
                self.kernel.reset()
            self.kernel.execute(f"import re\nPROBLEM_TEXT = {repr(problem)}")

            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._log_write(
                f"\n\n{'=' * 80}\n"
                f"[{now_str}] PRIME MATH NEW SOLVE SESSION\n"
                f"Problem: {problem}\n"
                f"{'=' * 80}\n"
            )
            # Reset per-problem Colosseum runtime state (critique injection + stall window)
            self._fast_composer_critique = None
            self._error_signatures = []

            # 0. TIER 1: Speculative Fast Composer Execution (Local Gemma 4 on LAN)
            if self.fast_composer and self.fast_composer.fast_cfg.enabled:
                fast_res = await self._atry_fast_composer(problem)
                if fast_res:
                    self._unlink_checkpoint(ckpt_path)
                    return fast_res
                self._log_write("[Two-Tier Routing] Fast Composer did not resolve. Escalating to Deep Orchestrator (Qwen 3.8)...\n")

            # 1. Stage 0: Pre-Solve Academic Research & Literature Briefing
            research_briefing = await self._aconduct_pre_solve_research(problem)

            # 2. Retrieve domain-relevant procedural heuristics
            # Formulate clean conceptual query from title/keywords to prevent 500-word embedding dilution
            prob_id, prob_title = parse_benchmark_header(problem)
            stripped_problem = strip_benchmark_prefix(problem)
            first_line = stripped_problem.splitlines()[0] if stripped_problem.splitlines() else ""
            clean_title = prob_title or (first_line.split(" - ")[-1].split(":")[0].strip() if " - " in first_line else first_line[:120])
            concept_query = clean_title or stripped_problem[:200]
            if self.fast_composer:
                try:
                    queries = await self.fast_composer.agenerate_research_queries(stripped_problem)
                    if queries:
                        concept_query = f"{clean_title} {' '.join(queries)}".strip()
                except Exception:
                    pass

            retrieved = await self.memory.aquery_procedural_heuristics(
                concept_query, limit=5, min_score=0.50, exclude_problem_id=prob_id
            )
            if retrieved:
                self._log_write(f"[CoALA Procedural Memory] Retrieved {len(retrieved)} relevant rules from TurboVec for '{concept_query}':\n")
                for h in retrieved:
                    self._log_write(f"  • {h}\n")

            system_content = MATH_SYSTEM_PROMPT
            if retrieved:
                heuristics_block = "\n\nRELEVANT PROCEDURAL RULES & PROVEN HEURISTICS (FROM COALA MEMORY):\n"
                for h in retrieved:
                    heuristics_block += f"- {h}\n"
                system_content += heuristics_block

            if research_briefing:
                briefing_block = (
                    "\n\nACADEMIC LITERATURE & THEORETICAL BRIEFING (PRE-SOLVE RESEARCH):\n"
                    f"{research_briefing}\n"
                )
                system_content += briefing_block

            user_content = f"Solve the following mathematical problem:\n\n{problem}"
            if self._fast_composer_critique:
                user_content += (
                    "\n\n[FAST COMPOSER CRITIQUE - FALSIFIED SPECULATIVE ROUTE] "
                    f"{self._fast_composer_critique}"
                )
                self._log_write(
                    f"[Two-Tier Routing] Injecting falsified-route critique into Deep Orchestrator prompt:\n"
                    f"{self._fast_composer_critique}\n"
                )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            result = SolveResult(
                problem=problem,
                answer=None,
                boxed_answer=None,
                verified=False,
                iterations=0,
                retrieved_heuristics=retrieved,
                research_briefing=research_briefing,
                log_file=str(self.log_path),
            )
            self._save_checkpoint(
                ckpt_path,
                problem=problem,
                completed_turn=0,
                start_wall_time=start_wall_time,
                messages=messages,
                result=result,
            )

        for turn in range(start_turn, max_turns + 1):
            result.iterations = turn

            if max_time_seconds and (time.perf_counter() - start_wall_time) > max_time_seconds:
                self._log_write(
                    f"\n[Solver Timeout] Wall-clock time limit ({max_time_seconds:.1f}s) exceeded; "
                    f"terminating reasoning turns and extracting best answer.\n"
                )
                break

            if self.compactor.should_compact(messages):
                comp_res = await self.compactor.acompact(messages)
                if comp_res.compacted:
                    messages = comp_res.compacted_messages
                    self._log_write(
                        f"\n[Context Compaction] Context window compacted: "
                        f"{comp_res.original_tokens} -> {comp_res.new_tokens} tokens\n"
                    )

            early_stop = await self._arun_stream_turn(messages, result, turn, max_turns)
            self._save_checkpoint(
                ckpt_path,
                problem=problem,
                completed_turn=turn,
                start_wall_time=start_wall_time,
                messages=messages,
                result=result,
            )
            if early_stop:
                break

        # 2.5. Post-solve adversarial verification (Colosseum): falsify the final boxed answer.
        # Runs before answer extraction so the wrap-up below applies if the revision ends boxed-less.
        if result.boxed_answer:
            verdict, reason = await self._averify_final_answer(problem, result)
            if verdict == "WRONG" and reason:
                rejected = result.boxed_answer
                self._log_write(
                    f"\n--- [ADVERSARIAL VERIFICATION] Final answer \\boxed{{{rejected}}} REJECTED by independent verifier:\n"
                    f"{reason}\nRunning one revision turn via a fundamentally different route...\n"
                )
                # The revision discards the falsified route: reset the stall window
                self._error_signatures = []
                result.boxed_answer = None
                result.answer = None
                result.verified = False
                messages.append({"role": "user", "content": self._build_revision_prompt(rejected, reason)})
                revision_turn = result.iterations + 1
                await self._arun_stream_turn(messages, result, revision_turn, max_turns)
                result.iterations = revision_turn
            elif verdict == "VERIFIED":
                result.self_verified = True

        if not result.boxed_answer:
            result.boxed_answer = self.extract_boxed_answer(result.final_response)
            result.answer = result.boxed_answer
            if result.boxed_answer:
                result.verified = len(result.code_history) > 0

        # If still no boxed answer, but code was executed, prompt a fast wrap-up to extract final answer
        if not result.boxed_answer and result.code_history:
            last_stdout = ""
            for er in reversed(result.execution_history):
                if er.stdout and er.stdout.strip():
                    last_stdout = er.stdout.strip()
                    break
            last_code = result.code_history[-1] if result.code_history else ""

            compact_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a mathematical answer extraction assistant. "
                        "State the exact final numerical or algebraic answer strictly inside LaTeX boxed format: \\boxed{your_answer}.\n"
                        "CRITICAL SANITY CHECKS:\n"
                        "- For density, probability, or proportion problems (e.g. d_x, d_\\infty), the answer MUST be strictly non-negative (0 <= ans). Never output a negative answer for a density problem; reject negative regression or fit artifacts in favor of empirical sample medians or positive limits.\n"
                        "Provide a brief direct answer line, then output \\boxed{answer} immediately without lengthy re-derivation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Problem:\n{problem}\n\n"
                        f"Last executed Python code:\n```python\n{last_code}\n```\n\n"
                        f"Execution output:\n{last_stdout}\n\n"
                        "Based on the above problem and Python execution output, state the final answer "
                        "strictly inside LaTeX boxed format: \\boxed{your_answer}."
                    ),
                },
            ]
            try:
                self._log_write("\n--- [FINAL EXTRACTION WRAP-UP TURN] ---\n")
                resp = await self.client.acomplete(compact_messages, temperature=0.1, max_tokens=1024, enable_thinking=False, timeout_seconds=30.0)
                wrap_text = resp.content.strip()
                self._log_write(wrap_text + "\n")
                boxed = self.extract_boxed_answer(wrap_text) or self.extract_boxed_answer(resp.reasoning or "")
                if not boxed:
                    boxed = self.extract_answer_from_text(wrap_text)

                # Reject unphysical negative values for density problems
                is_density = any(kw in problem.lower() for kw in ["density", "probability", "proportion", "d_x", r"d_\infty"])
                if boxed and is_density and boxed.strip().startswith("-"):
                    logger.warning(f"Rejecting unphysical negative density in wrap-up: {boxed}")
                    boxed = None

                if boxed:
                    result.boxed_answer = boxed
                    result.answer = boxed
                    result.verified = True
                    result.final_response = wrap_text
            except Exception as e:
                self._log_write(f"Wrap-up extraction failed: {e}\n")

        # Final fallback from history if still None
        if not result.boxed_answer:
            fallback = self._extract_best_answer_from_history(result)
            if fallback:
                result.boxed_answer = fallback
                result.answer = fallback
                result.verified = len(result.code_history) > 0
                self._log_write(f"[FINAL FALLBACK] Extracted answer from history: {fallback}\n")

        self._log_write(
            f"\n{'=' * 80}\n"
            f"[SESSION COMPLETE] Final Boxed Answer: {result.boxed_answer} (Verified: {result.verified}, Turns: {result.iterations})\n"
            f"{'=' * 80}\n"
        )
        self._unlink_checkpoint(ckpt_path)

        return result
