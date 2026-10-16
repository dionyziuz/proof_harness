"""Inductive Mathematical Discoverer & Law Miner.

Empowers Prime Math to uncover hidden mathematical laws, closed-form formulas,
recurrence relations, and dynamical invariants from raw sequence data, empirical
observations, or uncharacterized operators.
"""

from __future__ import annotations

import ast
import datetime
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .kernel import ExecutionResult, PersistentMathKernel
from .llm_client import TunnelThinkingClient
from .memory import CoALAMemoryManager
from .prompts import MATH_DISCOVERER_SYSTEM_PROMPT

logger = logging.getLogger("prime_math.discoverer")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DISCOVERER_LOG = REPO_ROOT / "logs" / "discoverer.log"


@dataclass
class DiscoveryResult:
    input_data: str
    status: str  # "DISCOVERED", "INCONCLUSIVE"
    law: Optional[str] = None
    conjecture: Optional[str] = None
    evidence: Optional[str] = None
    implementation: Optional[str] = None
    code_history: List[str] = field(default_factory=list)
    execution_history: List[ExecutionResult] = field(default_factory=list)
    iterations: int = 0
    final_response: str = ""
    log_file: str = str(DEFAULT_DISCOVERER_LOG)


class MathDiscoverer:
    """Inductive law miner, sequence analyst, and invariant discoverer."""

    def __init__(
        self,
        client: Optional[TunnelThinkingClient] = None,
        kernel: Optional[PersistentMathKernel] = None,
        memory: Optional[CoALAMemoryManager] = None,
        default_max_iterations: int = 8,
        log_path: Optional[Path] = None,
    ):
        self.client = client or TunnelThinkingClient()
        self.kernel = kernel or PersistentMathKernel()
        self.memory = memory or CoALAMemoryManager()
        self.default_max_iterations = default_max_iterations
        self.log_path = log_path or DEFAULT_DISCOVERER_LOG
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def extract_code_blocks(self, text: str) -> List[str]:
        """Extracts executable python code blocks from markdown."""
        pattern = r"```(?:python)?\s*\n(.*?)```"
        return [b.strip() for b in re.findall(pattern, text, flags=re.DOTALL) if b.strip()]

    def _log_write(self, text: str) -> None:
        """Appends text to the discoverer log file."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(text)
                f.flush()
        except Exception as e:
            logger.warning(f"Failed to write to discoverer log: {e}")

    def _try_parse_sequence(self, input_str: str) -> Optional[List[Any]]:
        """Attempts to parse a list of numbers or tuples from the input string."""
        s = input_str.strip()
        # Case 1: Python list literal [1, 2, 3]
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                    return list(parsed)
            except Exception:
                pass

        # Case 2: Comma or space-separated numbers: "1, 2, 5, 14, 42"
        tokens = [t.strip().rstrip(",") for t in re.split(r"[\s,]+", s) if t.strip()]
        if len(tokens) >= 3:
            parsed_nums = []
            for t in tokens:
                try:
                    if "." in t or "e" in t.lower():
                        parsed_nums.append(float(t))
                    else:
                        parsed_nums.append(int(t))
                except ValueError:
                    return None
            return parsed_nums
        return None

    def discover(
        self,
        data_or_description: str,
        max_iterations: Optional[int] = None,
        reset_kernel: bool = True,
        on_step: Optional[Any] = None,
    ) -> DiscoveryResult:
        """Inductively discovers governing mathematical laws or closed forms."""
        max_turns = max_iterations or self.default_max_iterations
        if reset_kernel:
            self.kernel.reset()

        # Check if the input is a sequence of numbers to preload
        seq = self._try_parse_sequence(data_or_description)
        preloaded_msg = ""
        if seq is not None:
            preload_code = f"DATA_SEQUENCE = {repr(seq)}\nimport numpy as np\nseq = np.array(DATA_SEQUENCE)"
            self.kernel.execute(preload_code)
            preloaded_msg = (
                "\n[Environment Setup]: The input sequence has been preloaded into your Python environment as "
                "`DATA_SEQUENCE` (and `seq = np.array(DATA_SEQUENCE)`). "
                "You can directly access, slice, and compute differences/ratios on it immediately in Turn 1."
            )

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_write(
            f"\n\n{'=' * 80}\n"
            f"[{now_str}] PRIME MATH NEW DISCOVERY SESSION\n"
            f"Input Data / System:\n{data_or_description}\n"
            f"{'=' * 80}\n"
        )

        user_content = (
            f"Analyze and inductively discover the mathematical law, closed form, recurrence relation, "
            f"or invariant governing the following input:\n\n"
            f"{data_or_description}\n{preloaded_msg}\n\n"
            f"Use difference tables (`np.diff`), ratios, prime factorization (`factorint`), `oeis_lookup`, "
            f"`sympy.rsolve`, `mpmath.pslq`, or symbolic regression in Python to discover the governing formula. "
            f"Structure your final response with <law>, <conjecture>, <evidence>, and <implementation> tags."
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": MATH_DISCOVERER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        result = DiscoveryResult(
            input_data=data_or_description,
            status="INCONCLUSIVE",
            log_file=str(self.log_path),
        )

        for turn in range(1, max_turns + 1):
            result.iterations = turn
            self._log_write(f"\n--- [TURN {turn}] INNER THINKING ---\n")
            if on_step:
                on_step("turn_start", turn)

            reasoning_chunks = []
            content_chunks = []
            switched_to_content = False

            for typ, delta in self.client.stream_complete(messages):
                if typ == "reasoning":
                    reasoning_chunks.append(delta)
                    self._log_write(delta)
                    if on_step:
                        on_step("reasoning_delta", delta)
                elif typ == "content":
                    if not switched_to_content:
                        self._log_write(f"\n\n--- [TURN {turn}] GENERATED DISCOVERY CODE & ANALYSIS ---\n")
                        switched_to_content = True
                    content_chunks.append(delta)
                    self._log_write(delta)
                    if on_step:
                        on_step("content_delta", delta)

            reasoning_text = "".join(reasoning_chunks).strip()
            model_text = "".join(content_chunks).strip()
            result.final_response = model_text

            # Extract code blocks, sanitizing scratchpad code
            code_blocks = self.extract_code_blocks(model_text)
            if not code_blocks and reasoning_text:
                raw_blocks = self.extract_code_blocks(reasoning_text)
                valid_blocks = []
                for b in raw_blocks:
                    try:
                        compile(b, "<string>", "exec")
                        valid_blocks.append(b)
                    except SyntaxError:
                        continue
                code_blocks = valid_blocks

            if code_blocks:
                messages.append({"role": "assistant", "content": model_text})
                exec_feedbacks = []

                for code in code_blocks:
                    result.code_history.append(code)
                    self._log_write(f"\n[Executing Discovery Code]\n```python\n{code}\n```\n")
                    if on_step:
                        on_step("code_start", code)

                    exec_res = self.kernel.execute(code)
                    result.execution_history.append(exec_res)
                    if on_step:
                        on_step("code_done", exec_res)

                    if exec_res.success:
                        feedback = f"[Execution Output]:\n{exec_res.stdout}"
                        if exec_res.result_repr:
                            feedback += f"\n[Result Value]: {exec_res.result_repr}"
                    else:
                        feedback = f"[Execution Error]:\n{exec_res.error}\nPlease adapt the discovery script."
                    exec_feedbacks.append(feedback)

                messages.append({"role": "user", "content": "\n\n".join(exec_feedbacks)})

            # Check for discovery tags
            m_law = re.search(r"<law>(.*?)</law>", model_text, flags=re.DOTALL)
            m_conj = re.search(r"<conjecture>(.*?)</conjecture>", model_text, flags=re.DOTALL)
            m_evi = re.search(r"<evidence>(.*?)</evidence>", model_text, flags=re.DOTALL)
            m_impl = re.search(r"<implementation>(.*?)</implementation>", model_text, flags=re.DOTALL)

            if (m_law or m_conj) and len(result.code_history) > 0:
                result.status = "DISCOVERED"
                if m_law:
                    result.law = m_law.group(1).strip()
                if m_conj:
                    result.conjecture = m_conj.group(1).strip()
                if m_evi:
                    result.evidence = m_evi.group(1).strip()
                if m_impl:
                    result.implementation = m_impl.group(1).strip()
                break

            if not code_blocks and turn < max_turns:
                messages.append({
                    "role": "user",
                    "content": (
                        "Please write and execute Python code to compute differences, ratios, factorizations, "
                        "or recurrence fits on the data. Then present your discovered formula enclosed in "
                        "<law>, <conjecture>, <evidence>, and <implementation> tags."
                    ),
                })

        return result
