"""Adversarial Mathematical Falsifier & Counterexample Hunter.

Empowers Prime Math to stress-test mathematical claims, model invariants, and conjectures
by actively discovering minimal counterexamples via Z3 SMT solving, boundary probing,
and randomized adversarial search.
"""

from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .kernel import ExecutionResult, PersistentMathKernel
from .llm_client import TunnelThinkingClient
from .memory import CoALAMemoryManager
from .prompts import MATH_FALSIFIER_SYSTEM_PROMPT

logger = logging.getLogger("prime_math.falsifier")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FALSIFIER_LOG = REPO_ROOT / "logs" / "falsifier.log"


@dataclass
class FalsificationResult:
    claim: str
    status: str  # "FALSIFIED", "UNFALSIFIED", "INCONCLUSIVE"
    counterexample: Optional[str] = None
    explanation: Optional[str] = None
    proof: Optional[str] = None
    implementation: Optional[str] = None
    code_history: List[str] = field(default_factory=list)
    execution_history: List[ExecutionResult] = field(default_factory=list)
    iterations: int = 0
    final_response: str = ""
    log_file: str = str(DEFAULT_FALSIFIER_LOG)


class MathFalsifier:
    """Adversarial counterexample hunter and mathematical falsifier."""

    def __init__(
        self,
        client: Optional[TunnelThinkingClient] = None,
        kernel: Optional[PersistentMathKernel] = None,
        memory: Optional[CoALAMemoryManager] = None,
        default_max_iterations: int = 6,
        log_path: Optional[Path] = None,
    ):
        self.client = client or TunnelThinkingClient()
        self.kernel = kernel or PersistentMathKernel()
        self.memory = memory or CoALAMemoryManager()
        self.default_max_iterations = default_max_iterations
        self.log_path = log_path or DEFAULT_FALSIFIER_LOG
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def extract_code_blocks(self, text: str) -> List[str]:
        """Extracts executable python code blocks from markdown."""
        pattern = r"```(?:python)?\s*\n(.*?)```"
        return [b.strip() for b in re.findall(pattern, text, flags=re.DOTALL) if b.strip()]

    def _log_write(self, text: str) -> None:
        """Appends text to the falsifier log file."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(text)
                f.flush()
        except Exception as e:
            logger.warning(f"Failed to write to falsifier log: {e}")

    def falsify(
        self,
        claim: str,
        max_iterations: Optional[int] = None,
        reset_kernel: bool = True,
        on_step: Optional[Any] = None,
    ) -> FalsificationResult:
        """Actively attempts to discover a counterexample disproving the given claim."""
        max_turns = max_iterations or self.default_max_iterations
        if reset_kernel:
            self.kernel.reset()

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_write(
            f"\n\n{'=' * 80}\n"
            f"[{now_str}] PRIME MATH NEW FALSIFICATION SESSION\n"
            f"Claim to Falsify:\n{claim}\n"
            f"{'=' * 80}\n"
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": MATH_FALSIFIER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Actively test and attempt to falsify the following mathematical claim:\n\n"
                    f"{claim}\n\n"
                    f"Use Z3, SymPy, or numerical boundary search in Python to search for a counterexample. "
                    f"Structure your final answer with <verdict> (FALSIFIED or UNFALSIFIED), "
                    f"<counterexample>, <explanation>, and <implementation> tags."
                ),
            },
        ]

        result = FalsificationResult(
            claim=claim,
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
                        self._log_write(f"\n\n--- [TURN {turn}] GENERATED FALSIFICATION SEARCH & CODE ---\n")
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
                    self._log_write(f"\n[Executing Adversarial Search Code]\n```python\n{code}\n```\n")
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
                        feedback = f"[Execution Error]:\n{exec_res.error}\nPlease adapt the adversarial search script."
                    exec_feedbacks.append(feedback)

                messages.append({"role": "user", "content": "\n\n".join(exec_feedbacks)})

            # Check verdict tags
            m_verdict = re.search(r"<verdict>(.*?)</verdict>", model_text, flags=re.DOTALL | re.IGNORECASE)
            m_cex = re.search(r"<counterexample>(.*?)</counterexample>", model_text, flags=re.DOTALL)
            m_exp = re.search(r"<explanation>(.*?)</explanation>", model_text, flags=re.DOTALL)
            m_impl = re.search(r"<implementation>(.*?)</implementation>", model_text, flags=re.DOTALL)
            m_proof = re.search(r"<proof>(.*?)</proof>", model_text, flags=re.DOTALL)

            if m_verdict and len(result.code_history) > 0:
                v_str = m_verdict.group(1).strip().upper()
                if "FALSIFIED" in v_str and "UNFALSIFIED" not in v_str:
                    result.status = "FALSIFIED"
                elif "UNFALSIFIED" in v_str:
                    result.status = "UNFALSIFIED"

                if m_cex:
                    result.counterexample = m_cex.group(1).strip()
                if m_exp:
                    result.explanation = m_exp.group(1).strip()
                if m_impl:
                    result.implementation = m_impl.group(1).strip()
                if m_proof:
                    result.proof = m_proof.group(1).strip()
                break

            if not code_blocks and turn < max_turns:
                messages.append({
                    "role": "user",
                    "content": (
                        "Please write and execute Python code using Z3 or numerical search to find an edge-case counterexample. "
                        "Then format your final response with <verdict> (FALSIFIED or UNFALSIFIED), "
                        "<counterexample>, <explanation>, and <implementation> tags."
                    ),
                })

        return result
