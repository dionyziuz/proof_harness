"""Mathematical Context Compactor for Prime Math.

Implements hybrid deterministic and neural context compaction:
1. Detects when accumulated conversation turns approach the context threshold.
2. Preserves the system prompt, original problem statement, and recent K turns.
3. Condenses older intermediate turns into a structured mathematical summary:
   - Accomplished Goals & Verified Lemmata
   - Active Persistent REPL Variables & Functions
   - Disproven Conjectures & Dead Ends
   - Current Calculation Frontier
4. Keeps persistent Python REPL variables intact so execution continues seamlessly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .llm_client import TunnelThinkingClient

logger = logging.getLogger("prime_math.compactor")

MATH_COMPACTION_SYSTEM_PROMPT = """You are an expert mathematical context compactor.
Your job is to condense a multi-turn mathematical exploration and proof session into a structured, highly informative summary.
Preserve all crucial mathematical invariants, variable names in the persistent Python kernel, verified intermediate values, and known dead ends.

Respond strictly in the following Markdown format:

### 1. Core Goal & Formal Constraints
(Summary of what is being solved, key equations, and target bounds)

### 2. Verified Mathematical Milestones & Lemmata
(List all exact numerical values and mathematical statements verified via Python code execution)

### 3. Active Persistent REPL Variables & Functions
(List the exact names and roles of Python functions and variables that were defined and remain available in the kernel)

### 4. Disproven Hypotheses & Dead Ends
(Algorithms or assumptions that failed, timed out, or produced wrong results to avoid repeating them)

### 5. Current Mathematical Frontier & Next Immediate Actions
(Where the proof or computation currently stands and the next step to reach the final boxed answer)
"""


def estimate_tokens(text: str) -> int:
    """Fast, conservative approximation of token count (~3.5 characters per token)."""
    if not text:
        return 0
    return max(1, int(len(text) / 3.5))


def estimate_messages_tokens(messages: List[Dict[str, str]]) -> int:
    """Estimates the total token count of a chat message sequence."""
    total = 0
    for msg in messages:
        total += 4  # Per-message formatting overhead
        total += estimate_tokens(msg.get("content", ""))
    return total


@dataclass
class CompactionResult:
    compacted: bool
    original_tokens: int
    new_tokens: int
    compacted_messages: List[Dict[str, str]]
    summary: Optional[str] = None


class MathContextCompactor:
    """Automated context compaction manager for long-running mathematical proofs."""

    def __init__(
        self,
        client: Optional[TunnelThinkingClient] = None,
        max_context_tokens: int = 32768,  # Safe 32k context envelope for vLLM under concurrent load
        reserve_tokens: int = 8192,       # 8k reserve for generation / thinking tokens
        keep_recent_turns: int = 1,
    ):
        self.client = client or TunnelThinkingClient()
        self.max_context_tokens = max_context_tokens
        self.reserve_tokens = reserve_tokens
        self.keep_recent_turns = keep_recent_turns
        self.compaction_threshold = max_context_tokens - reserve_tokens

    def should_compact(self, messages: List[Dict[str, str]]) -> bool:
        """Returns True if message tokens exceed threshold or any single intermediate message is giant."""
        tokens = estimate_messages_tokens(messages)
        # Check only intermediate scratchpad messages (index 2+), never system (0) or problem prompt (1)
        has_giant_msg = any(estimate_tokens(m.get("content", "")) > 8000 for m in messages[2:])
        return (tokens >= self.compaction_threshold or has_giant_msg) and len(messages) >= 3

    @staticmethod
    def _clamp_oversized(messages: List[Dict[str, str]], protected_ids: set) -> None:
        """Cap oversized intermediate scratchpads (index 2+) in place, never touching
        the system prompt (0), the problem statement (1) or protected anchor messages."""
        for msg in messages[2:]:
            if id(msg) in protected_ids:
                continue
            cont = msg.get("content", "")
            if len(cont) > 20000:
                head = cont[:4000]
                tail = cont[-12000:]
                msg["content"] = f"{head}\n\n% [... intermediate scratchpad omitted to preserve context envelope ...]\n\n{tail}"

    def _plan(
        self,
        messages: List[Dict[str, str]],
        protected: Optional[Sequence[Dict[str, str]]],
    ) -> Optional[Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]]:
        """Split messages into (anchors, older-to-summarize, recent).  Returns None if nothing to summarize."""
        protected_ids = {id(m) for m in (protected or [])}
        self._clamp_oversized(messages, protected_ids)
        recent_msg_count = self.keep_recent_turns * 2
        split_idx = len(messages) - recent_msg_count
        if split_idx <= 2:
            return None
        middle = messages[2:split_idx]
        anchors = [m for m in middle if id(m) in protected_ids]
        older = [m for m in middle if id(m) not in protected_ids]
        if not older:
            return None
        return anchors, older, messages[split_idx:]

    @staticmethod
    def _extract_deterministic_ledger(older_messages: List[Dict[str, str]]) -> str:
        """Deterministically extracts structured [CLAIM] statements, REPL verification outputs,
        and referee defect critiques from older messages so zero structured state is lost.
        """
        claims: List[str] = []
        auditor_critiques: List[str] = []
        repl_verified: List[str] = []

        for m in older_messages:
            c = m.get("content", "")
            if m.get("role") == "assistant":
                for cm in re.finditer(r"\[CLAIM\]:?\s*(.+?)(?=\n\[|\n```|\n\n|\Z)", c, re.DOTALL):
                    cl_text = " ".join(cm.group(1).split())[:400]
                    if cl_text and cl_text not in claims:
                        claims.append(cl_text)
            if "ADVERSARIAL PROOF AUDIT" in c or "DEFECTS DETECTED" in c:
                m_iss = re.search(r"identified the following defects:\s*(.+?)(?:\n\n|\Z)", c, re.DOTALL)
                if m_iss:
                    crit = m_iss.group(1).strip()[:1200]
                    if crit and crit not in auditor_critiques:
                        auditor_critiques.append(crit)
            if "[Execution Output]:" in c or "REPL Verified:" in c:
                for vm in re.finditer(r"(?:\[Execution Output\]:|REPL Verified:)\s*(.+?)(?=\n\[|\n\n|\Z)", c, re.DOTALL):
                    v_text = " ".join(vm.group(1).split())[:300]
                    if v_text and v_text not in repl_verified:
                        repl_verified.append(v_text)

        parts: List[str] = []
        if claims:
            parts.append("### Deterministic Claim Ledger (Extracted Verbatim)\n" + "\n".join(f"- {cl}" for cl in claims[-6:]))
        if repl_verified:
            parts.append("### Deterministic REPL Outputs\n" + "\n".join(f"- {rv}" for rv in repl_verified[-4:]))
        if auditor_critiques:
            parts.append("### Prior Referee Defects (Must Remain Resolved)\n" + "\n".join(f"- {cr}" for cr in auditor_critiques[-2:]))
        return "\n\n".join(parts)

    @staticmethod
    def _summarization_prompt(older_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        transcript_parts = []
        for msg in older_messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            transcript_parts.append(f"[{role} TURN]:\n{content}\n")
        return [
            {"role": "system", "content": MATH_COMPACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Please compact the following older turns of this mathematical proof session into "
                    f"a structured summary:\n\n{''.join(transcript_parts)}"
                ),
            },
        ]

    @staticmethod
    def _assemble(
        messages: List[Dict[str, str]],
        anchors: List[Dict[str, str]],
        recent: List[Dict[str, str]],
        summary_text: str,
        orig_tokens: int,
    ) -> CompactionResult:
        compact_summary_msg = {
            "role": "assistant",
            "content": f"[MATHEMATICAL CONTEXT COMPACTION SUMMARY]\n\n{summary_text}",
        }
        continuity_user_msg = {
            "role": "user",
            "content": (
                "Context compaction complete. All Python variables and functions listed above remain "
                "available in the persistent kernel. Continue your proof and calculation from the current frontier."
            ),
        }
        # Protected anchors (e.g. the candidate proof under revision + its revision prompt) are
        # kept verbatim, in their original order, right after the problem statement.
        new_messages = [messages[0], messages[1]] + anchors + [compact_summary_msg, continuity_user_msg] + recent
        new_tokens = estimate_messages_tokens(new_messages)
        logger.info(
            f"Context successfully compacted: {orig_tokens} -> {new_tokens} tokens "
            f"(reduced by {orig_tokens - new_tokens} tokens)."
        )
        return CompactionResult(
            compacted=True,
            original_tokens=orig_tokens,
            new_tokens=new_tokens,
            compacted_messages=new_messages,
            summary=summary_text,
        )

    def compact(
        self,
        messages: List[Dict[str, str]],
        force: bool = False,
        protected: Optional[Sequence[Dict[str, str]]] = None,
    ) -> CompactionResult:
        """Compact older messages into a structured mathematical summary.

        ``protected`` messages (matched by identity) are never summarized or clamped.
        """
        orig_tokens = estimate_messages_tokens(messages)
        if not force and not self.should_compact(messages):
            return CompactionResult(False, orig_tokens, orig_tokens, messages)

        plan = self._plan(messages, protected)
        if plan is None:
            new_toks = estimate_messages_tokens(messages)
            return CompactionResult(new_toks < orig_tokens, orig_tokens, new_toks, messages)
        anchors, older, recent = plan

        det_ledger = self._extract_deterministic_ledger(older)
        summary_text = ""
        try:
            resp = self.client.complete(self._summarization_prompt(older), max_tokens=2048, temperature=0.1)
            summary_text = (resp.content or "").strip()
        except Exception as e:
            logger.error(f"Context compaction LLM call failed with exception: {e}")

        if det_ledger and summary_text:
            summary_text = f"{det_ledger}\n\n{summary_text}"
        elif det_ledger and not summary_text:
            summary_text = det_ledger

        if not summary_text:
            logger.error("Context compaction returned an empty summary; keeping original messages.")
            return CompactionResult(False, orig_tokens, orig_tokens, messages)
        return self._assemble(messages, anchors, recent, summary_text, orig_tokens)

    async def acompact(
        self,
        messages: List[Dict[str, str]],
        force: bool = False,
        protected: Optional[Sequence[Dict[str, str]]] = None,
    ) -> CompactionResult:
        """Asynchronously compact older messages (same semantics as :meth:`compact`)."""
        orig_tokens = estimate_messages_tokens(messages)
        if not force and not self.should_compact(messages):
            return CompactionResult(False, orig_tokens, orig_tokens, messages)

        plan = self._plan(messages, protected)
        if plan is None:
            new_toks = estimate_messages_tokens(messages)
            return CompactionResult(new_toks < orig_tokens, orig_tokens, new_toks, messages)
        anchors, older, recent = plan

        det_ledger = self._extract_deterministic_ledger(older)
        summary_text = ""
        try:
            resp = await self.client.acomplete(self._summarization_prompt(older), max_tokens=2048, temperature=0.1)
            summary_text = (resp.content or "").strip()
        except Exception as e:
            logger.error(f"Context compaction LLM call failed with exception: {e}")

        if det_ledger and summary_text:
            summary_text = f"{det_ledger}\n\n{summary_text}"
        elif det_ledger and not summary_text:
            summary_text = det_ledger

        if not summary_text:
            logger.error("Context compaction returned an empty summary; keeping original messages.")
            return CompactionResult(False, orig_tokens, orig_tokens, messages)
        return self._assemble(messages, anchors, recent, summary_text, orig_tokens)
