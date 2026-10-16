"""Flagship Mathematical Theorem Prover & Formal Verification Engine for Prime Math.

Unifies deep reasoning, multi-subagent falsification, persistent Python REPL execution,
and formal theorem verification (Lean 4 and publication-grade LaTeX).

Supports:
- Zero-internet strict protocol for benchmarks (TCS-Bench, etc.).
- Optional pre-proof literature harvesting (arXiv / MathWorld) for open research conjectures.
- Lean 4 compiler verification via LeanJudge.
- Dynamic context compaction (MathContextCompactor) and TurboVec memory integration.
"""

from __future__ import annotations

import ast
import datetime
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .compactor import MathContextCompactor
from .kernel import PersistentMathKernel
from .llm_client import TunnelThinkingClient
from .memory import CoALAMemoryManager
from .prompt_loader import load_tcs_prompt
from .prompts import MATH_PROVER_SYSTEM_PROMPT as MATH_PROVER_SYSTEM_PROMPT
from .tcs_config import TCSConfig, load_tcs_config


logger = logging.getLogger("prime_math.prover")

from .prover_types import (
    DEFAULT_LOG_PATH,
    CandidatePool,
    CandidateRecord,
    ProofResult,
    THRESHOLDS,
    TCSSolveResult,
)
from .prover_kernel import TCSCompliantKernel
from .prover_auditor import ProverAuditorMixin
from .prover_stages import (
    SKEPTIC_REFUTED,
    SKEPTIC_VERIFIED,
    ProverStagesMixin,
)

TCS_BENCH_SYSTEM_PROMPT = load_tcs_prompt("system")

from .proof_extractor import (
    extract_latex_proof,
    is_valid_latex_proof,
    sanitize_informal_proof,
)


class MathProver(ProverAuditorMixin, ProverStagesMixin):
    """Flagship agentic theorem prover and formal verification engine for Prime Math.

    Integrates:
    - Multi-subagent closed-loop reasoning (Architect, Falsifier, Skeptic, Scribe, Auditor).
    - Persistent Python/SymPy/Z3 REPL with safe APU memory limits and combinatorial ceilings.
    - Formal verification with Lean 4 and LeanJudge.
    - Dynamic context compaction (MathContextCompactor).
    - TurboVec procedural and episodic memory.
    - Optional pre-proof academic literature retrieval (arXiv / Jina) for open conjectures.
    """

    def __init__(
        self,
        client: Optional[TunnelThinkingClient] = None,
        kernel: Optional[PersistentMathKernel] = None,
        memory: Optional[CoALAMemoryManager] = None,
        compactor: Optional[MathContextCompactor] = None,
        default_max_iterations: Optional[int] = None,
        log_path: Optional[Path] = None,
        enable_thinking: Optional[bool] = None,
        enable_pre_proof_research: Optional[bool] = None,
        enable_literature_search: Optional[bool] = None,
        config: Optional[TCSConfig] = None,
        proof_mode: Optional[str] = None,
    ):
        self.config = config or load_tcs_config()
        self.client = client or TunnelThinkingClient()
        self.proof_mode = proof_mode or getattr(self.config, "proof_mode", "auto")
        self.auditor_kernel_timeout = getattr(self.config, "auditor_kernel_timeout_seconds", 15.0)
        self.enable_literature_search = (
            enable_literature_search
            if enable_literature_search is not None
            else (enable_pre_proof_research if enable_pre_proof_research is not None
                  else (os.environ.get("PROVER_ENABLE_LITERATURE_SEARCH", "").lower() in ("true", "1")
                        or os.environ.get("TCS_ENABLE_LITERATURE_SEARCH", "").lower() in ("true", "1")))
        )
        self.enable_pre_proof_research = self.enable_literature_search
        self.kernel = kernel or TCSCompliantKernel(allow_internet=self.enable_literature_search)
        self.memory = memory or CoALAMemoryManager()
        self.compactor = compactor or MathContextCompactor(
            client=self.client,
            max_context_tokens=self.config.compactor_max_context_tokens,
            reserve_tokens=self.config.compactor_reserve_tokens,
            keep_recent_turns=self.config.compactor_keep_recent_turns,
        )
        self.default_max_iterations = default_max_iterations or self.config.solver_max_iterations
        self.log_path = log_path or DEFAULT_LOG_PATH
        self.enable_thinking = self.config.solver_enable_thinking if enable_thinking is None else enable_thinking
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def extract_code_blocks(self, text: str) -> List[str]:
        """Extracts python code blocks from markdown text, ignoring LaTeX/prose fences."""
        explicit_pattern = r"```(?:python|py)\b[ \t]*\n(.*?)```"
        blocks = [m.strip() for m in re.findall(explicit_pattern, text, flags=re.DOTALL) if m.strip()]
        if blocks:
            return blocks
        # Fallback: check bare ``` ... ``` blocks only if they contain valid Python AST and no LaTeX proof environment
        bare_pattern = r"```[ \t]*\n(.*?)```"
        valid_bare: List[str] = []
        for m in re.findall(bare_pattern, text, flags=re.DOTALL):
            cand = m.strip()
            if not cand or r"\begin{proof}" in cand or r"\end{proof}" in cand:
                continue
            try:
                ast.parse(cand)
                valid_bare.append(cand)
            except SyntaxError:
                continue
        return valid_bare

    def _log_write(self, text: str) -> None:
        """Appends text to the live thinking log file with immediate disk flush."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(text)
                f.flush()
        except Exception as e:
            logger.warning(f"Failed to write to thinking log: {e}")

    def _safe_record_lemma(self, claim_text: str) -> None:
        """Safely records a certified lemma into the state machine ledger without string injection."""
        clean_claim = " ".join(claim_text.split())[:1500]
        if hasattr(self.kernel, "_state_machine") and self.kernel._state_machine:
            try:
                self.kernel._state_machine.record_lemma(clean_claim)
                return
            except Exception:
                pass
        if "state" in getattr(self.kernel, "namespace", {}) and hasattr(self.kernel.namespace["state"], "record_lemma"):
            try:
                self.kernel.namespace["state"].record_lemma(clean_claim)
                return
            except Exception:
                pass
        self.kernel.execute(f"state.record_lemma({repr(clean_claim)})")


    def _pre_proof_research(self, conjecture: str) -> Optional[str]:
        """Pre-proof academic literature retrieval & briefing loop via arXiv and academic web search.

        Extracts 1-2 targeted queries, searches literature, and synthesizes a structured briefing
        to equip the reasoning engine before Turn 1.
        """
        if not self.enable_literature_search:
            return None

        self._log_write("\n[Stage 0: Pre-Proof Academic Research] Initiating literature & lemma retrieval...\n")

        from .research import clean_search_query, search_math

        query_prompt = [
            {
                "role": "system",
                "content": (
                    "You are an academic mathematics research specialist. Analyze this conjecture or theorem specification "
                    "and output 1-2 targeted academic search queries to look up governing theorems, lemma properties, "
                    "or preprint literature in arXiv / MathWorld.\n"
                    "CRITICAL RULES:\n"
                    "1. Output each query enclosed in <query>...</query> tags (e.g. <query>Jones polynomial trivial unknot conjecture</query>).\n"
                    "2. Output ONLY standard academic English keywords (no raw LaTeX math, no dollar signs $, no variable names).\n"
                    "3. Target keywords that uncover governing structural properties or related literature.\n"
                    "4. Do NOT output commentary or explanations."
                ),
            },
            {"role": "user", "content": f"Conjecture / Specification:\n{conjecture[:1500]}"},
        ]

        queries: List[str] = []
        try:
            resp = self.client.complete(query_prompt, temperature=0.1, max_tokens=400, enable_thinking=False, timeout_seconds=25.0)
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
            self._log_write(f"[Pre-Proof Research] Query generation error: {e}\n")
            return None

        if not queries:
            return None

        all_snippets: List[str] = []
        for q in queries[:2]:
            self._log_write(f"[Pre-Proof Research] Searching literature for: '{q}'...\n")
            try:
                results = search_math(q, count=3, timeout=30.0, prioritize_arxiv=True)
                for r in results:
                    title = r.get("title", "")
                    snip = r.get("snippet", "")
                    if snip and "error" not in title.lower() and len(snip.strip()) > 30:
                        all_snippets.append(f"Source ({title}):\n{snip.strip()}")
            except Exception as e:
                self._log_write(f"[Pre-Proof Research] Search for '{q}' failed: {e}\n")

        if not all_snippets:
            return None

        synth_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a pure mathematics research specialist. Synthesize the following research literature snippets "
                    "into a concise, structured theoretical briefing (under 250 words) to equip an elite theorem prover.\n"
                    "Structure concisely:\n"
                    "- GOVERNING OPERATORS & RELEVANT DEFINITIONS\n"
                    "- ESTABLISHED THEOREMS & LITERATURE LEMMAS\n"
                    "- PROOF & COUNTEREXAMPLE/COMPUTATIONAL PATHWAY\n"
                    "Enclose your briefing inside <briefing>...</briefing> tags."
                ),
            },
            {
                "role": "user",
                "content": f"Conjecture / Problem:\n{conjecture[:1000]}\n\nLiterature Snippets:\n" + "\n\n".join(all_snippets[:4]),
            },
        ]

        try:
            resp = self.client.complete(synth_prompt, temperature=0.1, max_tokens=600, enable_thinking=False, timeout_seconds=30.0)
            raw = (resp.content or "").strip()
            if not raw and resp.reasoning:
                raw = resp.reasoning.strip()
            m = re.search(r"<briefing>(.*?)</briefing>", raw, re.DOTALL | re.IGNORECASE)
            briefing = m.group(1).strip() if m else raw
            self._log_write(f"\n[Stage 0: Pre-Proof Research Briefing Synthesized]\n{briefing}\n")
            return briefing
        except Exception as e:
            self._log_write(f"[Pre-Proof Research] Synthesis failed: {e}\n")
            return None

    def prove(
        self,
        conjecture_or_spec: str,
        max_iterations: Optional[int] = None,
        reset_kernel: bool = True,
        on_step: Optional[Any] = None,
        enable_literature_search: Optional[bool] = None,
    ) -> ProofResult:
        """Derives, proves, and computationally verifies a mathematical conjecture or theorem.

        Provides a unified API for both CLI theorem proving and TCS benchmark evaluation.
        """
        lit_search = self.enable_literature_search if enable_literature_search is None else enable_literature_search
        res = self.solve(
            task_id="conjecture",
            task_prompt=conjecture_or_spec,
            max_iterations=max_iterations,
            reset_kernel=reset_kernel,
            on_step=on_step,
            enable_literature_search=lit_search,
        )
        if res.final_response:
            m_thm = re.search(r"<theorem>(.*?)</theorem>", res.final_response, flags=re.DOTALL)
            m_prf = re.search(r"<proof>(.*?)</proof>", res.final_response, flags=re.DOTALL)
            m_impl = re.search(r"<implementation>(.*?)</implementation>", res.final_response, flags=re.DOTALL)
            m_inv = re.search(r"<invariants>(.*?)</invariants>", res.final_response, flags=re.DOTALL)
            if m_thm:
                res.theorem = m_thm.group(1).strip()
            if m_prf:
                res.proof = m_prf.group(1).strip()
            if m_impl:
                res.implementation = m_impl.group(1).strip()
            if m_inv:
                res.verified_invariants = [ln.strip("- *") for ln in m_inv.group(1).strip().splitlines() if ln.strip()]

        if res.verified or (res.candidate_proof and is_valid_latex_proof(res.candidate_proof)):
            res.status = "PROVEN"
        elif res.status == "completed":
            res.status = "PROVEN" if res.candidate_proof else "INCONCLUSIVE"
        return res

    def _consolidate_learning(self, task_id: str, problem: str, result: TCSSolveResult) -> None:
        """Records the solve episode to TurboVec memory and extracts any procedural heuristics."""
        if not result.candidate_proof:
            return

        reflection_note = "TCS-Bench candidate proof generated."
        if result.verified:
            try:
                from .heuristic_validation import is_valid_universal_heuristic

                target_match = re.search(r"Target Problem \(your task\):\s*(.+)$", problem, re.DOTALL)
                target_text = target_match.group(1).strip() if target_match else problem[-1500:]
                prompt = [
                    {
                        "role": "system",
                        "content": (
                            "You are a cognitive reflection engine for theoretical computer science and mathematical proof construction. "
                            "Analyze this VERIFIED theorem and proof trajectory. Distill 1-2 concise sentences stating a universal "
                            "proof strategy, reduction invariant, potential-function technique, or quantifier/boundary discipline that "
                            "made the proof rigorous. "
                            "STRICT RULES: "
                            "(1) Must be 100% universal and domain-reusable across problems. "
                            "(2) NEVER mention paper titles, author names, Theorem/Lemma/Claim numbers, arXiv IDs, or task-specific variable values. "
                            "(3) Start with an actionable directive (e.g., 'When bounding...', 'To prove...', 'Always verify...'). "
                            "If no universal proof heuristic applies, output 'NONE'."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Target Statement:\n{target_text[:1200]}\n\n"
                            f"Verified Proof Excerpt:\n{result.candidate_proof[:2500]}"
                        ),
                    },
                ]
                resp = self.client.complete(
                    prompt,
                    max_tokens=180,
                    temperature=0.2,
                    enable_thinking=False,
                )
                reflection = (resp.content or "").strip()
                if reflection and "NONE" not in reflection.upper():
                    is_valid, reason = is_valid_universal_heuristic(
                        reflection,
                        problem_statement=target_text,
                        problem_id=task_id,
                    )
                    if is_valid:
                        metadata = {"task_id": task_id, "problem_id": task_id}
                        added = self.memory.add_procedural_heuristic(reflection, metadata=metadata)
                        if added:
                            result.learned_heuristics.append(reflection)
                            reflection_note = reflection
                            self._log_write(f"\n[CoALA Procedural Memory] Learned new heuristic:\n{reflection}\n")
                    else:
                        logger.info(f"Skipping non-universal TCS reflection heuristic: {reason} ('{reflection[:60]}')")
            except Exception as e:
                logger.warning(f"Post-proof procedural heuristic extraction error: {e}")

        try:
            self.memory.record_episode(
                problem=f"[{task_id}] {problem[:500]}",
                code_blocks=result.code_history,
                verified=result.verified,
                answer=result.candidate_proof[:200],
                critique=reflection_note,
            )
        except Exception as e:
            logger.warning(f"Error recording TCS episode: {e}")

        if result.verified and getattr(result, "blueprint", None):
            try:
                architect_memory = CoALAMemoryManager(agent_id="tcs_architect")
                architect_memory.record_episode(
                    problem=f"[{task_id}] Blueprint for {problem[:400]}",
                    code_blocks=[],
                    verified=True,
                    answer=result.blueprint[:300],
                    critique="Valid reduction blueprint.",
                )
            except Exception as e:
                logger.debug(f"Architect episodic memory record error: {e}")

    def _compact_conversation_history(
        self,
        messages: List[Dict[str, str]],
        current_turn: int,
    ) -> List[Dict[str, str]]:
        """Compacts older conversational turns to maintain a clean, high-attention context
        window while preserving problem prompt and recent feedback.
        """
        if len(messages) <= 4:
            return messages

        system_msg = messages[0]
        task_msg = messages[1]
        recent_messages = messages[-2:]
        anchor_ids = {id(m) for m in getattr(self, "_anchor_msgs", [])}
        middle = messages[2:-2]
        # The candidate proof under revision and its revision prompt are pinned: they are kept
        # verbatim so later turns never lose the draft they are supposed to be repairing.
        anchors = [m for m in middle if id(m) in anchor_ids]
        intermediate = [m for m in middle if id(m) not in anchor_ids]
        if not intermediate:
            return messages

        summary_lines = [
            f"[CONCISE SUMMARY OF PRIOR EXPLORATION TURNS (TURNS 1 TO {max(1, current_turn - 1)})]",
            f"• {len(intermediate)} earlier exploration messages were condensed to save context; "
            "Python variables defined in them remain available in the persistent kernel.",
        ]

        claims: List[str] = []
        auditor_critiques: List[str] = []
        repl_outputs: List[str] = []
        for m in intermediate:
            c = m.get("content", "")
            if m.get("role") == "assistant":
                for cm in re.finditer(r"\[CLAIM\]:?\s*(.+?)(?=\n\[|\n```|\n\n|\Z)", c, re.DOTALL):
                    claims.append(" ".join(cm.group(1).split())[:400])
            if "ADVERSARIAL PROOF AUDIT" in c or "DEFECTS DETECTED" in c:
                m_iss = re.search(r"identified the following defects:\s*(.+?)(?:\n\n|\Z)", c, re.DOTALL)
                if m_iss:
                    auditor_critiques.append(m_iss.group(1).strip()[:1500])
            if "[Execution Output]:" in c:
                for rm in re.finditer(r"\[Execution Output\]:\s*(.+?)(?=\n\[|\n\n|\Z)", c, re.DOTALL):
                    out_s = " ".join(rm.group(1).split())[:300]
                    if out_s and out_s not in repl_outputs:
                        repl_outputs.append(out_s)

        if claims:
            summary_lines.append("• Intermediate claims proposed earlier (not necessarily proved):")
            for cl in claims[-4:]:
                summary_lines.append(f"  - {cl}")
        if repl_outputs:
            summary_lines.append("• Earlier REPL verification outputs:")
            for ro in repl_outputs[-4:]:
                summary_lines.append(f"  - {ro}")
        if auditor_critiques:
            summary_lines.append("• Prior referee defects (must stay resolved):")
            for crit in auditor_critiques[-2:]:
                summary_lines.append(f"  - {crit}")

        summary_lines.append("• Directive: Build directly upon the latest messages below without repeating earlier flawed conjectures.")

        compacted_assistant = {
            "role": "assistant",
            "content": "\n".join(summary_lines),
        }
        compacted_user_ack = {
            "role": "user",
            "content": "[Acknowledged. Proceed with addressing the active turn's requirements and candidate proof construction.]",
        }

        return [system_msg, task_msg] + anchors + [compacted_assistant, compacted_user_ack] + recent_messages

    def _init_solve_session(
        self,
        task_id: str,
        task_prompt: str,
        max_turns: int,
        reset_kernel: bool = True,
        on_step: Optional[Any] = None,
        enable_literature_search: Optional[bool] = None,
    ) -> Tuple[List[Dict[str, str]], TCSSolveResult, List[str]]:
        """Pre-solve initialization: health checks, TurboVec heuristics retrieval,
        context architect blueprint, and falsifier subagents.
        """
        self.memory.check_health()
        self._prior_audit_defects = []

        if reset_kernel:
            self.kernel.reset()

        # Preload task prompt into kernel
        self.kernel.execute(f"import re\nPROBLEM_TEXT = {repr(task_prompt)}")

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        safe_task_id = re.sub(r"\d{4}\.\d{4,5}", "[ID]", str(task_id))
        self._log_write(
            f"\n\n{'=' * 80}\n"
            f"[{now_str}] TCS-BENCH FORMAL PROVE SESSION: {safe_task_id}\n"
            f"{'=' * 80}\n"
        )

        # Query TurboVec for domain-relevant procedural heuristics
        target_match = re.search(r"Target Problem \(your task\):\s*(.+)$", task_prompt, re.DOTALL)
        target_text = target_match.group(1).strip() if target_match else task_prompt[-2000:]
        concept_query = target_text[:300].strip()

        retrieved: List[str] = []
        try:
            retrieved = self.memory.query_procedural_heuristics(
                concept_query, limit=5, min_score=0.68, exclude_task_id=task_id
            )
            if retrieved:
                self._log_write(f"[CoALA Procedural Memory] Retrieved {len(retrieved)} relevant rules for '{concept_query[:80]}':\n")
                for h in retrieved:
                    self._log_write(f"  • {h}\n")
        except Exception as e:
            logger.debug(f"TurboVec retrieval error: {e}")

        research_briefing = None
        lit_search = self.enable_literature_search if enable_literature_search is None else enable_literature_search
        if isinstance(self.kernel, TCSCompliantKernel):
            self.kernel.allow_internet = lit_search
        if lit_search:
            try:
                research_briefing = self._pre_proof_research(task_prompt)
            except Exception as e:
                logger.debug(f"Pre-proof research error: {e}")

        system_content = TCS_BENCH_SYSTEM_PROMPT
        if research_briefing:
            system_content += f"\n\nACADEMIC LITERATURE & THEORETICAL FOUNDATIONS (STAGE 0 RESEARCH):\n{research_briefing}\n"
        if retrieved:
            heuristics_block = "\n\nRELEVANT THEORETICAL & ALGORITHMIC PRINCIPLES:\n"
            for h in retrieved:
                heuristics_block += f"- {h}\n"
            system_content += heuristics_block

        # Subagent Pre-Computation: Context Architect & Computational Falsifier
        blueprint = self._run_context_architect(task_id, task_prompt, on_step=on_step)
        verified_facts, falsified_alerts = self._run_computational_falsifier(task_id, blueprint, on_step=on_step)

        subagent_context_block = ""
        if blueprint or verified_facts or falsified_alerts:
            subagent_sections = ["--- Pre-Computed Subagent Architecture & REPL Verification ---"]
            if blueprint:
                subagent_sections.append(
                    "### Reduction Blueprint (UNVERIFIED CONJECTURAL SKETCH — INTERMEDIATE BOUNDS MAY BE LOOSE OR FALSE):\n"
                    f"{blueprint}\n\n"
                    "CRITICAL LOGICAL RULE ON BLUEPRINT BOUNDS & UNIVERSAL DOMAINS:\n"
                    "- Any intermediate inequality $A(x) \\le B(x) \\le C(x)$ proposed in the Blueprint above is an unverified heuristic.\n"
                    "- If a Python REPL test falsifies $A(x) \\le B(x)$ or $B(x) \\le C(x)$ for some $x \\in D$, **discard or replace the Blueprint's intermediate expression $B(x)$** — NEVER conclude that the Target Theorem $A(x) \\le C(x)$ only holds on a restricted regime.\n"
                    "- If your proof strategy partitions the universally quantified domain $D = D_1 \\cup D_2$, you MUST explicitly prove both $x \\in D_1$ and the complementary regime $x \\in D \\setminus D_1$. NEVER introduce unstated domain restrictions to salvage a loose intermediate step.\n"
                )
            if verified_facts:
                subagent_sections.append("### Verified Mathematical Facts (REPL Kernel):\n" + "\n".join(f"✓ {f}" for f in verified_facts) + "\n")
            if falsified_alerts:
                subagent_sections.append(
                    "### ⚠️ Critical Falsification Warnings (Blueprint Claim Failed in REPL — Discard the Broken Blueprint Step, NOT the Theorem Domain):\n"
                    + "\n".join(f"⚠️ {a}" for a in falsified_alerts)
                    + "\n"
                )
            subagent_sections.append("-------------------------------------------------------------------\n\n")
            subagent_context_block = "\n".join(subagent_sections)

        initial_user_prompt = task_prompt
        prefix_blocks = []
        if retrieved:
            heuristics_reminder = (
                "--- Relevant Theoretical & Algorithmic Principles ---\n"
                + "\n".join(f"- {h}" for h in retrieved)
                + "\n--------------------------------------------------------------------------\n\n"
            )
            prefix_blocks.append(heuristics_reminder)
        if subagent_context_block:
            prefix_blocks.append(subagent_context_block)
        if prefix_blocks:
            initial_user_prompt = "".join(prefix_blocks) + task_prompt
        task_only_prompt = initial_user_prompt

        if max_turns > 1:
            turn1_discipline = (
                f"\n\n[COGNITIVE DIVISION OF LABOR: DEDUCTIVE LOGICIAN TURN 1 OF {max_turns} (EXPLORATION GATE)]\n"
                f"You are the Deductive Logician on Turn 1 of a multi-turn proving process (Turn 1 of {max_turns}).\n\n"
                "CRITICAL CONSTRAINT (TURN-1 PROOF BAN):\n"
                "You are strictly FORBIDDEN from writing a complete formal proof (\\begin{proof} ... \\end{proof}) on this turn.\n"
                "Attempting a monolithic proof or exhaustive brute-force case split on Turn 1 before verifying structural invariants is strictly prohibited.\n\n"
                "REQUIRED OPERATING DISCIPLINE:\n"
                "1. Context Lemma Inventory: Carefully inspect the 'Context' section of the prompt. Catalog all lemmas, theorems, and definitions provided. Plan your proof around composing these context tools.\n"
                "2. Invariant-First Formulation: Do NOT attempt an exhaustive combinatorial case split across all objects. Instead, search for a unifying topological, boundary, or partition invariant (such as star-neighborhood coverage, boundary cuts, or minimal path obstructions) that groups or reduces the cases.\n"
                "3. Structured Intermediate Claims: Formulate your core reduction steps as structured claims:\n"
                "   [CLAIM]: <concise statement of the intermediate lemma or topological invariant>\n"
                "   [REDUCTION]: <how this claim discharges a hypothesis or reduces the target problem>\n"
                "   The Adversarial Skeptic Subagent will independently audit each claim in the REPL and register certified lemmas into the Verified Lemma Ledger (state.record_lemma).\n"
                "4. State Machine & Python REPL Verification: Write executable Python code in ```python ... ``` blocks using `state.register(...)`, `state.derive(...)`, and `state.checkpoint(...)` (or NetworkX/SymPy simulations) to test boundary configurations and verify whether your candidate invariant holds.\n"
                "5. Complete formal LaTeX proof synthesis (\\begin{proof} ... \\end{proof}) will be unlocked starting on Turn 2 once intermediate claims and invariants have been audited in the REPL."
            )
            initial_user_prompt += turn1_discipline

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": initial_user_prompt},
        ]
        # Task statement + context WITHOUT the Turn-1 "explore, do not rush a proof" instructions.
        # Revision, synthesis and repair contexts use this so they never inherit stale Turn-1 directives.
        self._task_only_msg = {"role": "user", "content": task_only_prompt}
        self._anchor_msgs = []
        self._last_messages = messages

        result = TCSSolveResult(
            problem=task_prompt,
            task_id=task_id,
            candidate_proof=None,
            verified=False,
            iterations=0,
            retrieved_heuristics=retrieved,
            blueprint=blueprint,
            verified_facts=verified_facts,
            falsified_alerts=falsified_alerts,
            log_file=str(self.log_path),
            conjecture=task_prompt,
            research_briefing=research_briefing,
        )

        return messages, result, retrieved

    def _run_exploratory_turns(
        self,
        task_id: str,
        task_prompt: str,
        messages: List[Dict[str, str]],
        result: TCSSolveResult,
        max_turns: int,
        start_wall_time: float,
        max_time_seconds: Optional[float] = None,
        on_step: Optional[Any] = None,
    ) -> Tuple[Optional[CandidateRecord], str]:
        """Executes multi-turn exploratory reasoning, Python REPL verification,
        adversarial skeptic audits, and candidate proof validation.
        """
        last_auditor_issues = ""
        pool = CandidatePool()
        best_candidate: Optional[CandidateRecord] = None
        rejection_count = 0
        prev_rejected_validity = 0.0
        prev_structural_validity = 0.0
        current_max_turns = max_turns
        dead_routes: List[str] = []
        repl_consecutive_turns = 0
        pivot_repl_gate_active = False
        pivot_gate_msg_ref: Optional[Dict[str, str]] = None
        pivot_unlocked_revision_prompt: str = ""
        turn = 0
        self.kernel.execute("state.checkpoint('init')")

        while turn < current_max_turns:
            turn += 1
            result.iterations = turn

            if max_time_seconds and (time.perf_counter() - start_wall_time) > max_time_seconds:
                self._log_write(
                    f"\n[Solver Timeout] Wall-clock time limit ({max_time_seconds:.1f}s) exceeded; "
                    f"terminating turn loop and extracting best proof.\n"
                )
                break

            # Context compaction check
            if self.compactor.should_compact(messages):
                comp_res = self.compactor.compact(messages, protected=getattr(self, "_anchor_msgs", []))
                if comp_res.compacted:
                    messages = comp_res.compacted_messages
                    self._log_write(
                        f"\n[Context Compaction] Context window compacted: "
                        f"{comp_res.original_tokens} -> {comp_res.new_tokens} tokens\n"
                    )

            self._log_write(f"\n--- [TURN {turn}/{current_max_turns}] INNER THINKING (STREAMING FROM QWEN ENGINE) ---\n")
            if on_step:
                on_step("turn_start", turn)

            def _on_reasoning(delta: str) -> None:
                if on_step:
                    on_step("reasoning_delta", delta)

            def _on_content(delta: str) -> None:
                if on_step:
                    on_step("content_delta", delta)

            def _on_turn_switch() -> None:
                self._log_write(f"\n\n--- [TURN {turn}/{max_turns}] MODEL GENERATED PROPOSAL & PROOF ---\n")
                if on_step:
                    on_step("content_start", "")

            reasoning_text, model_text = self._stream_completion_with_retry(
                messages=messages,
                max_tokens=self.client.config.max_tokens,
                enable_thinking=self.enable_thinking,
                thinking_budget_tokens=self.client.config.thinking_token_budget or 16384,
                temperature=0.2,
                timeout_seconds=600.0,
                max_retries=5,
                on_reasoning=_on_reasoning,
                on_content=_on_content,
                on_turn_switch=_on_turn_switch,
            )

            # Rescue empty content turns if thinking budget exhausted max_tokens
            if not model_text.strip() and reasoning_text:
                self._log_write("\n[Warning] Reasoning engine exhausted token budget inside <think>; requesting non-thinking continuation...\n")
                cont_messages = list(messages) + [
                    {"role": "assistant", "content": f"<think>\n{reasoning_text}\n</think>\n"},
                    {"role": "user", "content": "Based on your thinking above, state your concrete mathematical reduction, any verification Python code, and current proof draft now without repeating internal thoughts:"}
                ]
                try:
                    cont_resp = self.client.complete(
                        cont_messages,
                        max_tokens=4096,
                        enable_thinking=False,
                        temperature=0.2,
                        timeout_seconds=120.0,
                    )
                    if cont_resp and cont_resp.content:
                        model_text = cont_resp.content.strip()
                        self._log_write(f"[Continuation Extracted {len(model_text)} chars of content]\n")
                except Exception as cont_err:
                    self._log_write(f"[Continuation Failed]: {cont_err}\n")

            if reasoning_text:
                result.reasoning_traces.append(reasoning_text)
            result.final_response = model_text

            if on_step:
                on_step("turn_done", {"turn": turn, "reasoning": reasoning_text, "content": model_text})

            # Check if a formal LaTeX proof was provided in model_text
            extracted_proof = extract_latex_proof(model_text)
            if not extracted_proof and reasoning_text:
                candidate_proof = extract_latex_proof(reasoning_text)
                if candidate_proof and is_valid_latex_proof(candidate_proof):
                    extracted_proof = candidate_proof
                elif (
                    turn > 1
                    and not pivot_repl_gate_active
                    and r"\begin{proof}" in reasoning_text
                ):
                    last_bp = reasoning_text.rfind(r"\begin{proof}")
                    partial_proof = reasoning_text[last_bp:].strip()
                    if r"\end{proof}" not in partial_proof and len(partial_proof) > 400:
                        self._log_write(
                            "\n[Notice] Found unclosed \\begin{proof} inside <think> (token budget reached); "
                            "requesting non-thinking continuation to complete \\end{proof}...\n"
                        )
                        cont_messages = list(messages) + [
                            {"role": "assistant", "content": partial_proof},
                            {
                                "role": "user",
                                "content": (
                                    "Continue and complete the formal LaTeX proof above from the exact point where it was cut off, "
                                    "and conclude with \\end{proof}. Output ONLY the remaining LaTeX lines closing with \\end{proof}:"
                                ),
                            },
                        ]
                        try:
                            cont_resp = self.client.complete(
                                cont_messages,
                                max_tokens=4096,
                                enable_thinking=False,
                                temperature=0.2,
                                timeout_seconds=120.0,
                            )
                            if cont_resp and cont_resp.content:
                                stitched = partial_proof + "\n" + cont_resp.content.strip()
                                cand_stitched = extract_latex_proof(stitched)
                                if cand_stitched and is_valid_latex_proof(cand_stitched):
                                    extracted_proof = cand_stitched
                                    self._log_write(
                                        f"[Unclosed <think> Proof Rescued]: {len(extracted_proof)} chars.\n"
                                    )
                        except Exception as cont_err:
                            self._log_write(f"[Unclosed <think> Proof Continuation Failed]: {cont_err}\n")

            # Only execute verification code explicitly provided in model output
            code_blocks = self.extract_code_blocks(model_text)

            # Programmatic Turn-1 Gate: on Turn 1 of a multi-turn run, if REPL code blocks were emitted,
            # always execute the REPL verification and inspect its output before auditing any premature proof draft.
            if turn == 1 and max_turns > 1 and code_blocks:
                extracted_proof = None

            # If this turn was gated by the 1-Turn REPL-First Pivot Gate after a Clean-Slate Hard Pivot:
            # 1. Unlock the pivot prompt in messages so the next turn can synthesize \begin{proof} ... \end{proof}.
            # 2. Clear pinned pivot anchor messages so "FRESH REPL VERIFICATION" does not stay pinned forever.
            # 3. Suppress premature proof auditing on the REPL exploration gate turn.
            if pivot_repl_gate_active:
                pivot_repl_gate_active = False
                if pivot_gate_msg_ref is not None and pivot_unlocked_revision_prompt:
                    pivot_gate_msg_ref["content"] = pivot_unlocked_revision_prompt
                self._anchor_msgs = []
                if code_blocks:
                    extracted_proof = None

            # 1. Execute any code blocks the model provided in REPL first
            exec_feedbacks: List[str] = []
            if code_blocks:
                for code in code_blocks:
                    result.code_history.append(code)
                    self._log_write(f"\n\n--- [TURN {turn}] EXECUTING VERIFICATION CODE IN REPL ---\n```python\n{code}\n```\n")
                    if on_step:
                        on_step("code_start", code)

                    exec_res = self.kernel.execute(code)
                    result.execution_history.append(exec_res)
                    if on_step:
                        on_step("code_done", exec_res)

                    if exec_res.success:
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
                        if "AssertionError" in (exec_res.error or ""):
                            feedback = (
                                f"[REPL Assertion Failure]:\n{exec_res.error}\n"
                                "CRITICAL WARNING: The Python verification script encountered an AssertionError.\n"
                                "If your intermediate claim, algebraic bound, or identity fails on concrete test instances, "
                                "the formal proof relying on it cannot be mathematically valid.\n"
                                "You must immediately determine whether this failure is due to:\n"
                                "  1. An unhandled boundary or complementary regime $x \\in D \\setminus D_1$ where your intermediate bound is loose (prove $D \\setminus D_1$ via a direct complementary argument so the full domain $D$ is covered),\n"
                                "  2. A false intermediate claim from the Blueprint that must be replaced by a sound deduction, or\n"
                                "  3. An implicit standard scale/parameter normalization inherent to the statement's literal formula (if direct numerical testing of the target formula itself shows it holds under standard unit/scale normalization, state that canonical normalization explicitly and NEVER fabricate false inequalities or invalid periodicity claims for unnormalized parameters).\n"
                                "NEVER invent mathematically false inequalities to force an unnormalized boundary case."
                            )
                        else:
                            feedback = f"[Execution Error]:\n{exec_res.error}\nPlease correct the Python implementation."
                    exec_feedbacks.append(feedback)

                # If Mandatory Proof Synthesis Directive was already active (repl_consecutive_turns >= 2),
                # yet <think> consumed nearly its full budget (>= 12,000 chars) and only emitted REPL code
                # without a \begin{proof}, immediately run a non-thinking continuation seeded with the
                # <think> derivation + REPL execution output to synthesize the formal proof on this turn!
                if (
                    repl_consecutive_turns >= 2
                    and not extracted_proof
                    and reasoning_text
                    and len(reasoning_text) >= 12000
                    and turn < current_max_turns
                ):
                    self._log_write(
                        f"\n[Mandatory Synthesis Continuation] Turn {turn} exhausted <think> ({len(reasoning_text)} chars) "
                        "during consecutive REPL turns without emitting \\begin{proof}; invoking non-thinking proof synthesis continuation...\n"
                    )
                    synth_cont_msgs = list(messages) + [
                        {
                            "role": "assistant",
                            "content": f"<think>\n{reasoning_text[-24000:]}\n</think>\n\n{model_text}",
                        },
                        {
                            "role": "user",
                            "content": (
                                "\n\n".join(exec_feedbacks)
                                + "\n\n[IMMEDIATE NON-THINKING PROOF EMISSION]\n"
                                "Using your mathematical derivation in <think> above and the REPL verification output, "
                                "immediately write the complete, self-contained formal LaTeX proof of the Target Problem "
                                "enclosed strictly in \\begin{proof} ... \\end{proof}. Start directly with \\begin{proof}:"
                            ),
                        },
                    ]
                    try:
                        sc_resp = self.client.complete(
                            synth_cont_msgs,
                            max_tokens=8192,
                            enable_thinking=False,
                            temperature=0.2,
                            timeout_seconds=180.0,
                        )
                        if sc_resp and sc_resp.content:
                            sc_proof = extract_latex_proof(sc_resp.content)
                            if sc_proof and not is_valid_latex_proof(sc_proof):
                                sc_proof = sanitize_informal_proof(sc_proof)
                            if sc_proof and is_valid_latex_proof(sc_proof):
                                extracted_proof = sc_proof
                                model_text = model_text + "\n\n" + sc_proof
                                result.final_response = model_text
                                self._log_write(
                                    f"[Mandatory Synthesis Continuation Succeeded]: {len(extracted_proof)} chars.\n"
                                )
                    except Exception as sc_err:
                        self._log_write(f"[Mandatory Synthesis Continuation Failed]: {sc_err}\n")

            # Check for proposed intermediate claim from Deductive Logician
            claim_match = re.search(r"\[CLAIM\]:?\s*(.+?)(?=\n\[REDUCTION\]|\n\[|\n```|\Z)", model_text, re.DOTALL)
            if not claim_match:
                claim_match = re.search(r"(?:^|\n)(?:Claim|Lemma)\s*\d*:\s*(.+?)(?=\n\n|\n```|\Z)", model_text, re.DOTALL)

            claim_text = claim_match.group(1).strip() if claim_match else ""
            if claim_text and len(claim_text) > 20 and len(claim_text) < 1500 and not extracted_proof:
                self.kernel.execute("state.checkpoint('pre_claim')")
                st_res = self.kernel.execute("print(state.status())")
                state_summary = (st_res.stdout or "").strip()

                skeptic_status, verdict_exp, skeptic_code = self._run_adversarial_skeptic(
                    task_id, task_prompt, claim_text, state_summary, on_step=on_step
                )
                if skeptic_status == SKEPTIC_REFUTED:
                    self.kernel.execute("state.rollback('pre_claim')")
                    self._log_write("[State Machine] Claim refuted by Adversarial Skeptic. Rolled back to 'pre_claim'.\n")
                    skeptic_feedback = (
                        f"\n[ADVERSARIAL SKEPTIC REFUTATION]\n"
                        f"The Adversarial Skeptic Subagent attacked your proposed claim and found a counterexample or failed check:\n"
                        f"{verdict_exp}\n"
                        f"The state machine has executed state.rollback('pre_claim') to restore clean invariants.\n"
                        f"First double-check that the attack really tests your claim as stated. If it does, this intermediate claim "
                        f"is false: abandon it and find an alternative step, or prove the complementary regime separately. (The Target Problem itself remains true over its full domain.)\n"
                    )
                    exec_feedbacks.append(skeptic_feedback)
                elif skeptic_status == SKEPTIC_VERIFIED:
                    self._safe_record_lemma(claim_text)
                    self._log_write("[State Machine] Claim survived the Adversarial Skeptic and was recorded to the claim ledger.\n")
                    skeptic_feedback = (
                        f"\n[ADVERSARIAL SKEPTIC: NO COUNTEREXAMPLE FOUND]\n"
                        f"The Adversarial Skeptic stress-tested your intermediate claim and found no counterexample: {verdict_exp}\n"
                        f"It has been recorded in the claim ledger (state.record_lemma). This is evidence, not a proof: "
                        f"you must still prove the claim rigorously in the final proof.\n"
                    )
                    exec_feedbacks.append(skeptic_feedback)
                else:
                    self._log_write("[State Machine] Skeptic inconclusive; claim NOT recorded to the ledger.\n")
                    skeptic_feedback = (
                        f"\n[ADVERSARIAL SKEPTIC: INCONCLUSIVE]\n"
                        f"The Adversarial Skeptic could not check your claim ({verdict_exp}). The claim is NOT certified; "
                        f"justify it carefully (or test it yourself in Python) before relying on it.\n"
                    )
                    exec_feedbacks.append(skeptic_feedback)

            # Check if ready for Publication Scribe (with Epistemic State Machine Gate)
            explicit_ready = bool(re.search(r"\[\s*ALL CLAIMS ESTABLISHED\s*:\s*READY FOR PUBLICATION SCRIBE\s*\]", model_text, re.IGNORECASE))
            if explicit_ready and turn < current_max_turns - 1 and hasattr(self.kernel, "_state_machine"):
                sm = self.kernel._state_machine
                revoked_objs = [
                    f"{obj.name} (revoked: { sorted(obj.revoked_properties) })"
                    for obj in getattr(sm, "_objects", {}).values()
                    if getattr(obj, "revoked_properties", None)
                ]
                if revoked_objs:
                    explicit_ready = False
                    alert_msg = (
                        "[EPISTEMIC STATE MACHINE GATE]: Early Scribe handoff paused because the following derived "
                        f"objects have unresolved revoked fragile properties: {', '.join(revoked_objs)}. "
                        "Re-establish these properties via `state.assert_property(name, prop, witness=...)` or prove why they are not needed."
                    )
                    self._log_write(f"\n{alert_msg}\n")
                    exec_feedbacks.append(alert_msg)

            # Do NOT hijack penultimate REPL verification turns (e.g. Turn 7/8 with code_blocks);
            # let REPL stdout feed into Turn 8 so the solver can write the final proof with verified facts.
            ready_for_scribe = explicit_ready or (
                turn >= current_max_turns and not extracted_proof and not code_blocks
            )

            if ready_for_scribe and not result.verified and not extracted_proof:
                self._log_write(f"\n[Cognitive Partitioning] Invoking Publication Scribe Subagent on Turn {turn}...\n")
                st_res = self.kernel.execute("print(state.status())")
                state_summary = (st_res.stdout or "").strip()
                v_lemmas = []
                if hasattr(self.kernel, "_state_machine"):
                    v_lemmas = list(self.kernel._state_machine._verified_lemmas)

                scribe_proof = self._run_publication_scribe(
                    task_id=task_id,
                    task_prompt=task_prompt,
                    verified_ledger=v_lemmas,
                    state_summary=state_summary,
                    candidate_draft=model_text,
                    on_step=on_step,
                )
                if scribe_proof and is_valid_latex_proof(scribe_proof):
                    extracted_proof = scribe_proof

            # 2. Check if a formal LaTeX proof was provided
            if extracted_proof and not is_valid_latex_proof(extracted_proof):
                san_p = sanitize_informal_proof(extracted_proof)
                if san_p and is_valid_latex_proof(san_p):
                    extracted_proof = san_p

            if extracted_proof and is_valid_latex_proof(extracted_proof):
                repl_consecutive_turns = 0
                self._log_write(
                    f"\n[Proof Draft Generated] Turn {turn}: {len(extracted_proof)} chars. "
                    f"Running closed-loop adversarial audit...\n"
                )
                if on_step:
                    on_step("audit_start", extracted_proof)
                audit_res = self._audit_proof_draft(task_prompt, extracted_proof)
                verdict = audit_res.verdict
                issues = audit_res.issues
                validity = audit_res.validity
                if on_step:
                    on_step("audit_result", {"verdict": verdict, "issues": issues, "validity": validity})

                # Track best candidate across all turns via CandidatePool
                if is_valid_latex_proof(extracted_proof):
                    if pool.offer(extracted_proof, validity, verdict, issues, turn, audit_result=audit_res):
                        best_candidate = pool.best
                        self._log_write(f"[Best Candidate Updated] Turn {turn}: validity={validity:.2f} ({len(extracted_proof)} chars)\n")
                    else:
                        best_candidate = pool.best

                # High-Confidence Early Exit (requires validity >= 0.90)
                # Borderline drafts (e.g. validity 0.80 - 0.89) are saved in best_candidate,
                # but the loop continues across the full turn budget to resolve subtle defects.
                no_referee_issues = not issues or issues.strip().lower() == "none"
                is_high_confidence_exit = (
                    verdict == "VALID"
                    and (
                        validity >= THRESHOLDS.early_exit_validity
                        or (turn >= current_max_turns and no_referee_issues)
                    )
                )
                if is_high_confidence_exit:
                    self._log_write(f"[Audit Passed] Proof verified on Turn {turn} (validity: {validity:.2f}). Early-terminating to prevent multi-turn degradation.\n")
                    if on_step:
                        on_step("proof_accepted", extracted_proof)
                    result.candidate_proof = extracted_proof
                    result.final_response = extracted_proof
                    result.verified = True
                    break
                else:
                    audit_inconclusive = verdict == "INCONCLUSIVE"
                    if audit_inconclusive:
                        # The referee output could not be parsed: that is not evidence the proof is bad.
                        # Do not count it as a rejection and do not let its placeholder 0.0 trigger a pivot.
                        validity = pool.best_validity if pool.best else THRESHOLDS.judge_pass_threshold
                        issues = (
                            "The referee audit for this draft was inconclusive (its output could not be parsed). "
                            "Re-check every step of the proof for gaps, unproved claims, missing base cases and "
                            "unverified lemma preconditions, and tighten the argument."
                        )
                    elif verdict == "VALID" and no_referee_issues:
                        issues = (
                            f"The referee scored this draft as VALID (validity: {validity:.2f}) with no fatal flaws, "
                            f"but below the {THRESHOLDS.early_exit_validity:.2f} early-termination threshold. "
                            "Tighten every intermediate justification, explicitly verify all boundary and base cases, "
                            "and ensure every inequality direction and theorem precondition is proved step-by-step."
                        )
                    else:
                        rejection_count += 1
                    self._log_write(f"[Audit Flagged Issues] (verdict: {verdict}, validity: {validity:.2f}, rejection #{rejection_count})\n{issues}\n")
                    last_auditor_issues = issues
                    if is_valid_latex_proof(extracted_proof):
                        result.candidate_proof = extracted_proof

                    # High-Potential Overtime Extension Gate:
                    # When reaching nominal turn ceiling, if the candidate proof has demonstrated
                    # strong internal validity (>= threshold, default 0.40) but is not yet fully verified,
                    # dynamically grant a 4-turn overtime extension for surgical repair.
                    enable_overtime = getattr(self.config, "enable_overtime", True)
                    overtime_iters = getattr(self.config, "overtime_iterations", 4)
                    overtime_thresh = getattr(self.config, "overtime_validity_threshold", 0.35)
                    overtime_triggered_now = False
                    if (
                        enable_overtime
                        and turn == current_max_turns
                        and current_max_turns == max_turns
                        and not result.verified
                    ):
                        best_val = pool.best_validity if pool.best else validity
                        if best_val >= overtime_thresh:
                            current_max_turns += overtime_iters
                            overtime_triggered_now = True
                            overtime_seconds = float(getattr(self.config, "overtime_seconds_per_iter", 450.0) * overtime_iters)
                            if max_time_seconds is not None:
                                max_time_seconds += overtime_seconds
                            if getattr(self, "_solve_deadline", None) is not None:
                                self._solve_deadline += overtime_seconds
                            clock_info = f", +{overtime_seconds:.0f}s wall-clock to {max_time_seconds:.0f}s" if max_time_seconds is not None else ""
                            self._log_write(
                                f"\n================================================================================\n"
                                f"[HIGH-POTENTIAL OVERTIME GATE TRIGGERED]\n"
                                f"Candidate proof achieved strong directional validity ({best_val:.2f} >= {overtime_thresh:.2f}).\n"
                                f"Granting {overtime_iters}-turn overtime extension (Turn {turn} -> Max turns: {current_max_turns}{clock_info}) for surgical repair.\n"
                                f"================================================================================\n\n"
                            )
                            if on_step:
                                on_step("overtime_triggered", {"new_max_turns": current_max_turns, "validity": best_val, "extended_seconds": overtime_seconds})

                    if turn < current_max_turns:
                        # Golden Balance Anti-Perseveration Gate:
                        # 1. Single-strike hard pivot ONLY for complete gibberish / empty / hallucinatory output (<= hard_pivot_floor, 0.05)
                        #    without substantive structure.
                        # 2. Repeated rejections: require rejection_count >= 2 AND validity < THRESHOLDS.pivot_threshold (0.25)
                        #    AND not has_substantive_structure (completeness < 0.35 and rigor < 0.35)
                        #    AND no upward progress (neither validity nor structural_validity improved by > 0.05).
                        # 3. NEVER trigger a Clean-Slate Hard Pivot when overtime_triggered_now is True OR when pool.best
                        #    already holds a high-validity / substantive candidate foundation — restore pool.best for Surgical Revision instead!
                        has_substantive_structure = getattr(audit_res, "has_substantive_structure", False)
                        structural_val = getattr(audit_res, "structural_validity", validity)
                        pool_has_viable_foundation = bool(
                            pool.best
                            and (
                                pool.best_validity >= THRESHOLDS.judge_pass_threshold
                                or pool.best.has_substantive_structure
                                or pool.best.structural_validity >= 0.40
                            )
                        )
                        is_improving = rejection_count >= 2 and (
                            validity > prev_rejected_validity + 0.05
                            or structural_val > prev_structural_validity + 0.05
                        )
                        would_hard_pivot_raw = (not audit_inconclusive) and (
                            (validity <= THRESHOLDS.hard_pivot_floor)
                            or (rejection_count >= 2 and validity < THRESHOLDS.pivot_threshold and not is_improving)
                        )
                        if would_hard_pivot_raw and (has_substantive_structure or overtime_triggered_now or pool_has_viable_foundation):
                            self._log_write(
                                f"[Hard Pivot Prevented — Substantive/Best Candidate Protected] "
                                f"Turn {turn} validity was {validity:.2f}, but a viable foundation exists "
                                f"(current structural={structural_val:.2f}, best validity={pool.best_validity:.2f}). "
                                f"Proceeding with Surgical Revision instead of wiping candidate.\n"
                            )
                        is_stuck_low_validity = (
                            would_hard_pivot_raw
                            and not has_substantive_structure
                            and not overtime_triggered_now
                            and not pool_has_viable_foundation
                        )
                        prev_rejected_validity = validity
                        prev_structural_validity = structural_val

                        if is_stuck_low_validity:
                            clean_issue = issues.strip().replace("\n", " ")
                            if len(clean_issue) > 300:
                                clean_issue = clean_issue[:300] + "..."
                            dead_route_summary = f"Refuted Reduction (Turn {turn}, Validity {validity:.2f}): {clean_issue}"
                            if dead_route_summary not in dead_routes:
                                dead_routes.append(dead_route_summary)

                            dead_routes_directive = "\n".join(f"  ❌ {dr}" for dr in dead_routes)

                            # 1. Prune the low-validity refuted candidate from the pool so it cannot anchor search or become the final output
                            pruned = pool.prune_abandoned(THRESHOLDS.pivot_threshold)
                            if pruned:
                                self._log_write(
                                    f"[Candidate Pool Pruned] Purged refuted candidate from Turn {pruned.turn} "
                                    f"(validity: {pruned.validity:.2f}) to prevent anchoring on a dead reduction path.\n"
                                )
                                best_candidate = pool.best
                                if pool.best:
                                    result.candidate_proof = pool.best.proof
                                else:
                                    result.candidate_proof = ""

                            # 2. Reset the Python execution kernel and state machine completely (clean slate)
                            self.kernel.reset()
                            self.kernel.execute("state.checkpoint('init')")
                            self._log_write(
                                "[Kernel State Purged] Python namespace and mathematical state machine reset to clean init. "
                                "All dead-end variables, invariants, and lemmas cleared.\n"
                            )

                            # 3. Reset consecutive rejection counter and validity tracking so the new architecture has a clean window to develop
                            rejection_count = 0
                            prev_rejected_validity = 0.0
                            prev_structural_validity = 0.0

                            self._log_write(
                                f"[Clean-Slate Hard Pivot Triggered] Abandoning refuted path on Turn {turn}. "
                                f"Initiating clean-slate search with fresh proof architecture.\n"
                            )
                            if on_step:
                                on_step("hard_pivot", {"turn": turn, "dead_route": dead_route_summary})

                            candidate_block = (
                                "% [PREVIOUS REDUCTION PATH ABANDONED DUE TO REFUTATION]\n"
                                "% BANNED PREMISES / REFUTED ROUTES (DO NOT RE-ATTEMPT OR RESCUE):\n"
                                f"{dead_routes_directive}\n"
                                "%\n"
                                "% All scratchpad state has been completely purged.\n"
                                "% Formulate a fundamentally fresh proof architecture from first principles."
                            )

                            unlocked_pivot_prompt = (
                                f"[MANDATORY REDUCTIVE PIVOT - PREVIOUS APPROACH REFUTED & PURGED (Turn {turn})]\n"
                                f"Your previous reduction attempt reached a fatal mathematical obstacle or false premise (validity: {validity:.2f}).\n"
                                f"All Python kernel variables, invariants, and intermediate claims from that attempt have been purged from memory.\n\n"
                                f"[BANNED PREMISES & REFUTED DEAD ENDS - DO NOT RETRY OR RESCUE]:\n"
                                f"{dead_routes_directive}\n\n"
                                "CRITICAL PIVOT DIRECTIVE:\n"
                                "1. DO NOT DEFEND THE REFUTED PATH: Do NOT attempt cosmetic re-patching, helper lemmas, or circular deductions to rescue the refuted premise.\n"
                                "2. CONSTRUCT A FUNDAMENTALLY DISTINCT PROOF ARCHITECTURE:\n"
                                "   - Step back and discover an alternative structural invariant, global symmetry/conservation law, extremal property, or dual representation from first principles.\n"
                                "   - If previous approach was local/inductive, consider global invariants (e.g. partition cores, abacus, generating functions, or potential arguments).\n"
                                "3. THEOREM FIDELITY: Construct the complete formal proof establishing the Target Problem in full.\n"
                                "4. CASE & BOUNDARY INTEGRITY: Explicitly cover all base cases, boundary parameters, and edge conditions required by the statement from first principles.\n"
                                "5. PROBABILITY & ALGORITHMIC RIGOR: Check probability conditioning, ensure recursive calls exclude degenerate sets, and verify normalization.\n"
                                "6. NO HAND-WAVING: State and prove all intermediate claims explicitly with rigorous justifications.\n"
                                "7. FRESH REPL VERIFICATION: Use ```python ... ``` blocks to verify your new invariants from first principles before committing to the final proof in \\begin{proof} ... \\end{proof}."
                            )

                            # Enforce 1-Turn REPL-First Pivot Gate when at least 2 turns remain (Turn t+1 explores in REPL, Turn t+2 writes proof)
                            if turn <= current_max_turns - 2:
                                pivot_repl_gate_active = True
                                repl_consecutive_turns = 1
                                pivot_unlocked_revision_prompt = unlocked_pivot_prompt
                                self._log_write(
                                    f"[REPL-First Pivot Gate Activated] Enforcing 1-turn mandatory Python REPL exploration on Turn {turn + 1} "
                                    f"before unlocking formal proof synthesis on Turn {turn + 2}.\n"
                                )
                                revision_prompt = (
                                    f"[MANDATORY REDUCTIVE PIVOT — 1-TURN REPL EXPLORATION GATE (Turn {turn + 1} of {current_max_turns})]\n"
                                    f"Your previous reduction attempt reached a fatal mathematical obstacle or false premise (validity: {validity:.2f}).\n"
                                    f"All Python kernel variables, invariants, and intermediate claims from that attempt have been purged from memory.\n\n"
                                    f"[BANNED PREMISES & REFUTED DEAD ENDS - DO NOT RETRY OR RESCUE]:\n"
                                    f"{dead_routes_directive}\n\n"
                                    "CRITICAL CONSTRAINT (1-TURN REPL-FIRST PIVOT GATE):\n"
                                    "You are strictly FORBIDDEN from writing a complete formal proof (\\begin{proof} ... \\end{proof}) on this turn.\n"
                                    "Do NOT jump directly into a monolithic proof of an untested alternative idea.\n\n"
                                    "REQUIRED PIVOT OPERATING DISCIPLINE (THIS TURN ONLY):\n"
                                    "1. DO NOT DEFEND THE REFUTED PATH: Abandon the banned premises above completely.\n"
                                    "2. FORMULATE 2-3 FRESH CANDIDATE REDUCTIONS / INVARIANTS: Step back and propose alternative structural invariants, couplings, conservation laws, or dual representations from first principles and Context lemmas.\n"
                                    "3. MANDATORY PYTHON REPL VERIFICATION: Write an executable ```python ... ``` script to empirically test your new candidate invariants/inequalities on concrete instances and register verified claims via `state.register(...)` / `state.derive(...)`.\n"
                                    "4. Formulate your strongest new intermediate step as:\n"
                                    "   [CLAIM]: <concise statement of the fresh intermediate lemma or invariant>\n"
                                    "   [REDUCTION]: <how this claim discharges the target problem without the refuted premise>\n"
                                    f"5. Complete formal LaTeX proof synthesis (\\begin{{proof}} ... \\end{{proof}}) will automatically unlock on Turn {turn + 2} once your fresh reduction has been tested in the REPL."
                                )
                            else:
                                pivot_repl_gate_active = False
                                revision_prompt = unlocked_pivot_prompt

                        else:
                            # Best-Candidate Restoration for Overtime & Revision:
                            # If an earlier iteration produced a higher-validity (or higher structural_validity) draft than
                            # the current turn (e.g. Turn 6 was 0.78 but Turn 8 truncated to 0.15, or Turn 10 had rigor=0.75
                            # while Turn 11 dropped to rigor=0.35), restore the highest-scoring candidate draft and its
                            # referee issues so revision directly repairs the best mathematical foundation!
                            if pool.best and (
                                pool.best_validity >= THRESHOLDS.judge_pass_threshold
                                or pool.best.structural_validity >= 0.45
                            ) and (
                                overtime_triggered_now
                                or (turn >= max_turns and (validity < pool.best_validity or structural_val < pool.best.structural_validity))
                                or (pool.best_validity > validity + THRESHOLDS.restore_margin)
                                or (pool.best.structural_validity > structural_val + THRESHOLDS.restore_margin and pool.best_validity >= validity)
                            ):
                                self._log_write(
                                    f"[Best Candidate Restored for Revision] Restoring highest-scoring candidate from Turn {pool.best.turn} "
                                    f"(validity: {pool.best.validity:.2f}, structural: {pool.best.structural_validity:.2f}) over Turn {turn} "
                                    f"(validity: {validity:.2f}, structural: {structural_val:.2f}) to guarantee progress on the best mathematical foundation.\n"
                                )
                                extracted_proof = pool.best.proof
                                issues = pool.best.issues
                                validity = pool.best.validity
                                # DO NOT decrement rejection_count: preserve accurate rejection tracking

                            self._log_write(f"[Self-Improvement] Initiating surgical revision turn {turn + 1} with auditor feedback.\n")
                            if on_step:
                                on_step("revision_start", issues)

                            # Dynamic Critique Falsification: test referee objections in REPL
                            critique_verified, critique_alerts = self._run_critique_falsifier(
                                task_id, issues, extracted_proof, on_step=on_step
                            )

                            evidence_parts = []
                            if exec_feedbacks:
                                evidence_parts.append("--- Verified REPL Computational Facts (This Turn) ---\n" + "\n".join(exec_feedbacks))
                            if critique_verified:
                                evidence_parts.append(
                                    "--- ✓ Dynamic REPL Check: Candidate Invariant Held on Tested Instances (Retain Verified Math; Clarify Exposition) ---\n"
                                    + "\n".join(f"✓ {f}" for f in critique_verified)
                                )
                            if critique_alerts:
                                evidence_parts.append("--- ⚠️ Critical REPL Falsification of Claimed Step ---\n" + "\n".join(f"⚠️ {a}" for a in critique_alerts))
                            repl_evidence = ""
                            if evidence_parts:
                                repl_evidence = "\n".join(evidence_parts) + "\n----------------------------------------------------\n\n"

                            strategic_pivot_directive = (
                                "\n[STRATEGIC SURGICAL REVISION DIRECTIVE]\n"
                                f"Your previous candidate proof draft achieved strong validity ({validity:.2f}) and is provided in full above.\n"
                                "An independent mathematical referee audited it and flagged only specific, isolated defects.\n"
                                "- SURGICAL REPAIR OVER REWRITES: Make targeted, surgical repairs directly to the specific defect(s) flagged by the referee (e.g., supply missing base cases, prove complementary domain regimes, correct parameter bounds/inequalities, or articulate the missing deductive link in the flagged step).\n"
                                "- PRESERVE VALID SUB-CLAIMS: Retain all established lemmas, verified equations, and valid inductive machinery from your previous draft.\n"
                                "- EXPLICIT GAP RESOLUTION: Ensure the specific step, claim, or boundary condition flagged by the referee is now stated and verified with complete, rigorous deductions from first principles.\n\n"
                            )

                            st_res = self.kernel.execute("print(state.status())")
                            state_summary = (st_res.stdout or "").strip()
                            state_ledger_block = ""
                            if state_summary and "Total Invariants: 0" not in state_summary and "Error" not in state_summary:
                                state_ledger_block = f"--- Formally Registered Mathematical State & Invariants ---\n{state_summary}\n------------------------------------------------------------\n\n"

                            clean_candidate = extracted_proof if (extracted_proof and is_valid_latex_proof(extracted_proof)) else (result.candidate_proof or model_text[:15000])
                            candidate_block = (
                                "Here is the candidate proof under revision:\n\n"
                                f"{clean_candidate}\n\n"
                                "% [End of candidate proof under revision]"
                            )
                            rule2_text = "2. Modular Lemma Immutability: Keep all sound lemmas, definitions, and unflagged derivations from your previous draft verbatim. Do NOT rephrase, alter, or weaken sound parts. Focus 100% of your mathematical repair on the specific defective step or claim identified by the referee."

                            revision_prompt = (
                                f"{state_ledger_block}"
                                f"{repl_evidence}"
                                f"[ADVERSARIAL PROOF AUDIT - DEFECTS DETECTED IN YOUR DRAFT (Validity: {validity:.2f})]\n"
                                f"An independent mathematical referee audited your draft and identified the following defects:\n"
                                f"{issues}\n\n"
                                f"{strategic_pivot_directive}"
                                f"REVISION INSTRUCTIONS:\n"
                                f"1. Address and resolve every defect identified above via surgical, targeted corrections to your proof.\n"
                                f"{rule2_text}\n"
                                f"3. Theorem & Universal Domain Fidelity: Construct the complete formal proof establishing the Target Problem over its ENTIRE universally quantified domain $D$. If an intermediate step only holds on $D_1 \\subsetneq D$, explicitly prove the complementary regime $D \\setminus D_1$ via a sound complementary argument. However, if direct numerical verification confirms that the statement's literal formula carries a canonical scale/unit normalization, state that standard normalization cleanly and NEVER fabricate false algebraic inequalities or invalid periodicity claims.\n"
                                f"4. Case & Boundary Integrity: Explicitly cover all base cases, boundary parameters, and edge conditions required by the statement from first principles.\n"
                                f"5. Probability & Algorithmic Rigor: Check probability conditioning (all-fail vs. at-least-one-fail), ensure recursive calls exclude degenerate/empty sets, and verify probability densities and normalization.\n"
                                f"6. No Hand-Waving: State and prove all intermediate claims explicitly with rigorous justifications. Never dismiss cases as 'analogous' or 'trivial'.\n"
                                f"7. Context Adherence: Use ONLY definitions and lemmas provided in the problem prompt. Never cite unstated external theorems.\n"
                                f"8. Statement Property Labels: When the Target Problem defines numbered, lettered, or labeled properties, explicitly reference each property using its exact statement label when proving that it is satisfied.\n"
                                f"9. Objective Verification of Referee Notes: Reviewers may occasionally challenge standard mathematical conventions or suggest irrelevant external theories. Always verify referee objections against the Target Problem's explicit hypotheses and first principles. If an objection is resolved by standard definitions or preconditions, articulate the deduction rigorously from first principles; under NO circumstances invent non-existent structural axioms, adopt suggested external theorem names, or hallucinate domain theories to appease a referee.\n"
                                f"10. You may execute further Python code in ```python ... ``` blocks to verify claims on the kernel, or output your revised publication LaTeX proof enclosed strictly in \\begin{{proof}} ... \\end{{proof}}."
                            )

                        # Stateless Blackboard Architecture:
                        # Give revision turn a clean, high-attention context containing:
                        # 1. System prompt (rules & identity)
                        # 2. Problem prompt & Context lemmas
                        # 3. Clean candidate proof under repair (or fresh slate if hard pivot)
                        # 4. Auditor defects & surgical repair instructions
                        candidate_msg = {"role": "assistant", "content": candidate_block}
                        revision_msg = {"role": "user", "content": revision_prompt}
                        if pivot_repl_gate_active:
                            pivot_gate_msg_ref = revision_msg
                        messages = [
                            messages[0],  # System prompt
                            getattr(self, "_task_only_msg", None) or messages[1],  # Task prompt & context (no Turn-1 directives)
                            candidate_msg,
                            revision_msg,
                        ]
                        # Pin the draft under repair so neither compaction path can erase it.
                        self._anchor_msgs = [candidate_msg, revision_msg]
                        self._last_messages = messages
                        continue

                    else:
                        self._log_write(f"[Max Turns Reached] Storing current candidate proof from Turn {turn}.\n")
                        break

            elif code_blocks:
                repl_consecutive_turns += 1
                if len(messages) > 1 and getattr(self, "_task_only_msg", None) is not None:
                    messages[1] = self._task_only_msg
                # Prevent conversational prefix-copying ("Looking at this problem... Let me first verify...") across turns
                # and preserve the tail of <think> so mathematical derivations are not lost between REPL turns.
                code_str = "\n\n".join(f"```python\n{cb}\n```" for cb in code_blocks)
                claim_and_red = ""
                if claim_text:
                    claim_and_red = f"[CLAIM]: {claim_text}\n\n"
                think_summary = ""
                if reasoning_text and len(reasoning_text.strip()) > 100:
                    think_tail = reasoning_text.strip()[-2000:]
                    think_summary = f"[Mathematical Derivation Summary (Turn {turn})]:\n{think_tail}\n\n"
                msg_content = f"{think_summary}{claim_and_red}{code_str}".strip()
                messages.append({"role": "assistant", "content": msg_content})
                next_prompt = "\n\n".join(exec_feedbacks)

                dead_block = ""
                if dead_routes:
                    dead_block = "\n[REMINDER OF REFUTED / BANNED PREMISES]:\n" + "\n".join(f"  ❌ {dr}" for dr in dead_routes) + "\n\n"

                if repl_consecutive_turns >= 2 and turn < current_max_turns - 1:
                    next_prompt += (
                        f"\n\n[MANDATORY PROOF SYNTHESIS DIRECTIVE - TURN {turn + 1} of {current_max_turns}]\n"
                        f"{dead_block}"
                        "You have completed your exploratory REPL verification. You must now synthesize your reduction chain "
                        "into a complete formal LaTeX proof enclosed strictly in \\begin{proof} ... \\end{proof}.\n"
                        "Ensure your proof covers the ENTIRE universally quantified domain of the Target Problem (including any complementary regime where an intermediate bound does not apply) without introducing unauthorized restrictions.\n"
                        "Do not linger in scratchpad exploration; commit to the reduction chain and write the complete proof draft now."
                    )
                elif turn < current_max_turns - 1:
                    next_prompt += (
                        f"\n\n[DEDUCTIVE LOGICIAN TURN {turn + 1} of {current_max_turns}]\n"
                        f"{dead_block}"
                        "Computational verification results and state invariant feedback are provided above.\n"
                        "Continue building and stress-testing your modular reduction chain:\n"
                        "1. Formulate intermediate lemmas with [CLAIM]: ... and [REDUCTION]: ...\n"
                        "2. Execute REPL verification code in ```python ... ``` using `state.register`, `state.derive`, `state.checkpoint`, and graph/probability testers.\n"
                        "3. The Adversarial Skeptic Subagent will audit any proposed [CLAIM] in an isolated context.\n"
                        "4. If your reduction is established across the full domain, output your complete formal LaTeX proof in \\begin{proof} ... \\end{proof} (or state [ALL CLAIMS ESTABLISHED: READY FOR PUBLICATION SCRIBE])."
                    )
                else:
                    next_prompt += (
                        f"\n\n[MANDATORY CONVERGENCE - PENULTIMATE/FINAL TURN {turn + 1} of {current_max_turns}]\n"
                        f"{dead_block}"
                        "Finalize your modular reduction now.\n"
                        "Output your complete formal LaTeX proof in \\begin{proof} ... \\end{proof}, or state your final verified claims with [CLAIM]: ... or declare [ALL CLAIMS ESTABLISHED: READY FOR PUBLICATION SCRIBE].\n"
                        "The Publication Scribe Subagent will typeset the verified lemmas into the final publication LaTeX proof if needed."
                    )
                messages.append({"role": "user", "content": next_prompt})
                if len(messages) >= 8 or self.compactor.should_compact(messages):
                    messages = self._compact_conversation_history(messages, turn + 1)
                self._last_messages = messages

            else:
                repl_consecutive_turns = 0
                if len(messages) > 1 and getattr(self, "_task_only_msg", None) is not None:
                    messages[1] = self._task_only_msg
                if turn < current_max_turns:
                    had_rejected_proof_block = (r"\begin{proof}" in model_text and r"\end{proof}" in model_text)
                    if had_rejected_proof_block:
                        self._log_write(
                            f"[Turn {turn}] Candidate \\begin{{proof}} block failed formal validation (conversational markers or placeholders); requesting clean publication proof on Turn {turn + 1}.\n"
                        )
                    else:
                        self._log_write(
                            f"[Turn {turn}] Theoretical reasoning in progress; prompting for modular claim or REPL verification in Turn {turn + 1}.\n"
                        )
                    content_to_add = model_text.strip()
                    if len(content_to_add) >= 150:
                        if len(content_to_add) > 20000:
                            content_to_add = (
                                content_to_add[:6000]
                                + "\n\n% [... intermediate scratchpad truncated to preserve context budget ...]\n\n"
                                + content_to_add[-12000:]
                            )
                        messages.append({"role": "assistant", "content": content_to_add})
                    elif reasoning_text:
                        tail = reasoning_text[-3000:].strip()
                        messages.append({
                            "role": "assistant",
                            "content": (
                                f"Theoretical analysis and current formulation:\n{tail}\n\n"
                                "I will now formulate the core intermediate claims and reductions."
                            ),
                        })
                    else:
                        messages.append({
                            "role": "assistant",
                            "content": "I have completed the structural exploration and invariant analysis. I will now formulate the core intermediate claims."
                        })

                    if had_rejected_proof_block:
                        prompt_next = (
                            f"[FORMAL PROOF VALIDATION GATE - TURN {turn + 1} of {current_max_turns}]\n"
                            "Your previous \\begin{proof} ... \\end{proof} block was rejected by the publication syntax & rigor validator because it contained conversational self-correction markers (e.g., 'Wait', 'Let me reconsider', 'Let me think', 'I realize') or omission placeholders (e.g., 'proof omitted', '(as above)').\n"
                            "Re-emit your complete, self-contained formal LaTeX proof enclosed strictly in \\begin{proof} ... \\end{proof} with all conversational scratchpad sentences removed and all steps written in rigorous mathematical prose.\n"
                        )
                    else:
                        prompt_next = (
                            f"[DEDUCTIVE LOGICIAN - TURN {turn + 1} of {current_max_turns}]\n"
                            "In your previous response you did not output an executable Python code block or a structured [CLAIM].\n"
                            "DO NOT repeat introductory analysis, problem restatements, or conversational preambles.\n"
                            "You must immediately execute ONE of the following concrete actions:\n"
                            "1. DIRECT PROOF SYNTHESIS: If the reduction is clear from Context lemmas, output your complete formal LaTeX proof enclosed strictly in \\begin{proof} ... \\end{proof}.\n"
                            "2. REPL EXPERIMENTATION & INVARIANT GATES: Output executable Python verification code in ```python ... ``` using `state.register`, `state.derive`, `state.checkpoint`, and testers.\n"
                            "3. MODULAR CLAIM FORMULATION: Formulate the core intermediate lemma as [CLAIM]: ...\n"
                            "4. DECLARE READY: If all intermediate reductions are verified, output [ALL CLAIMS ESTABLISHED: READY FOR PUBLICATION SCRIBE].\n"
                        )
                    messages.append({"role": "user", "content": prompt_next})
                    if len(messages) >= 8 or self.compactor.should_compact(messages):
                        messages = self._compact_conversation_history(messages, turn + 1)
                    self._last_messages = messages
                    continue
                else:
                    break

        self._last_messages = messages
        return best_candidate, last_auditor_issues

    def _run_synthesis_pipeline(
        self,
        task_id: str,
        task_prompt: str,
        messages: List[Dict[str, str]],
        result: TCSSolveResult,
        best_candidate: Optional[CandidateRecord],
        last_issues: str,
        retrieved: List[str],
        on_step: Optional[Any] = None,
    ) -> Optional[CandidateRecord]:
        """Runs the dedicated Publication Scribe synthesis turn, referee audit loop,
        and surgical repair passes.
        """
        self._log_write("\n--- [DEDICATED PUBLICATION SCRIBE SYNTHESIS TURN] ---\n")
        if on_step:
            on_step("synthesis_start", "")

        pool = CandidatePool()
        if best_candidate and is_valid_latex_proof(best_candidate.proof):
            pool.offer(
                best_candidate.proof,
                best_candidate.validity,
                best_candidate.verdict,
                best_candidate.issues,
                best_candidate.turn,
                rigor=best_candidate.rigor,
                completeness=best_candidate.completeness,
                boundary_cases=best_candidate.boundary_cases,
            )

        # Attempt 1: Isolated Publication Scribe Subagent using Verified Lemma Ledger
        # ONLY invoke the from-scratch Scribe when exploratory turns ended without a valid LaTeX candidate proof!
        # When best_candidate already has a valid LaTeX proof, skip Attempt 1 to save 300-600s and proceed directly
        # to Attempt 2 / Revision seeded with best_candidate.
        if not (best_candidate and is_valid_latex_proof(best_candidate.proof)):
            st_res = self.kernel.execute("print(state.status())")
            state_summary = (st_res.stdout or "").strip()
            v_lemmas = []
            if hasattr(self.kernel, "_state_machine"):
                v_lemmas = list(self.kernel._state_machine._verified_lemmas)

            scribe_proof = self._run_publication_scribe(
                task_id=task_id,
                task_prompt=task_prompt,
                verified_ledger=v_lemmas,
                state_summary=state_summary,
                candidate_draft=result.final_response or "",
                on_step=on_step,
            )
            if scribe_proof and is_valid_latex_proof(scribe_proof):
                self._log_write("\n[Auditing Synthesized Proof Draft from Publication Scribe]...\n")
                if on_step:
                    on_step("audit_start", scribe_proof)
                s_audit = self._audit_proof_draft(task_prompt, scribe_proof)
                s_verdict = s_audit.verdict
                s_issues = s_audit.issues
                s_val = s_audit.validity
                if on_step:
                    on_step("audit_result", {"verdict": s_verdict, "issues": s_issues, "validity": s_val})
                if pool.offer(scribe_proof, s_val, s_verdict, s_issues, turn=990, audit_result=s_audit):
                    best_candidate = pool.best
                else:
                    best_candidate = pool.best
                if s_verdict == "VALID":
                    self._log_write(f"[Publication Scribe Audit Passed!] Proof verified (validity: {s_val:.2f}).\n")
                    if on_step:
                        on_step("proof_accepted", scribe_proof)
                    result.candidate_proof = scribe_proof
                    result.final_response = scribe_proof
                    result.verified = True
                    return best_candidate
        else:
            self._log_write(
                f"[Synthesis Optimization] Skipping redundant from-scratch Scribe pass; "
                f"seeding directly from Best Candidate (Turn {best_candidate.turn}, validity: {best_candidate.validity:.2f}).\n"
            )

        # Attempt 2: Direct model synthesis turn with contract directive
        heuristics_reminder = ""
        if retrieved:
            heuristics_reminder = (
                "--- Relevant Theoretical & Algorithmic Principles ---\n"
                + "\n".join(f"- {h}" for h in retrieved)
                + "\n--------------------------------------------------------------------------\n\n"
            )

        repl_summary = ""
        if result.execution_history:
            succ = [ex for ex in result.execution_history if ex.success and (ex.stdout or ex.result_repr)]
            if succ:
                succ_lines = []
                for ex in succ[-3:]:
                    out_text = (ex.stdout or ex.result_repr or "").strip()
                    if out_text:
                        succ_lines.append(out_text[:1000])
                if succ_lines:
                    repl_summary = (
                        "--- Computational Verification & Kernel Observations ---\n"
                        + "\n".join(succ_lines)
                        + "\n-------------------------------------------------------\n\n"
                    )

        critique_block = ""
        if best_candidate and best_candidate.issues:
            critique_block = (
                f"--- Critical Defects Flagged by Mathematical Referee (Must Resolve) ---\n"
                f"{best_candidate.issues}\n"
                f"NOTE: The Target Problem is an established theorem and is true as stated, so ignore any referee speculation "
                f"that the theorem itself is false. However, a counterexample to one of YOUR intermediate claims or steps is a real "
                f"defect: replace or repair that step rather than defending it.\n"
                f"-------------------------------------------------------------------------\n\n"
            )
        elif last_issues:
            critique_block = (
                f"--- Critical Defects Flagged by Mathematical Referee (Must Resolve) ---\n"
                f"{last_issues}\n"
                f"NOTE: The Target Problem is an established theorem and is true as stated, so ignore any referee speculation "
                f"that the theorem itself is false. However, a counterexample to one of YOUR intermediate claims or steps is a real "
                f"defect: replace or repair that step rather than defending it.\n"
                f"-------------------------------------------------------------------------\n\n"
            )

        best_proof_block = ""
        if best_candidate and is_valid_latex_proof(best_candidate.proof):
            best_proof_block = (
                f"--- Highest-Scoring Candidate Proof from Exploration (Turn {best_candidate.turn}, validity: {best_candidate.validity:.2f}) ---\n"
                f"{best_candidate.proof}\n"
                f"Refine and complete this candidate proof to resolve all remaining defects and produce the final publication proof.\n"
                f"---------------------------------------------------------------------------------------------------\n\n"
            )

        contract_directive = (
            f"{heuristics_reminder}"
            f"{repl_summary}"
            f"{best_proof_block}"
            f"{critique_block}"
            "[FINAL PUBLICATION LATEX PROOF CONTRACT]\n"
            "Now, based on all your deductions, invariants, and computational verifications above, assemble and write the complete, rigorous publication-grade LaTeX proof of the Target Problem.\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. IMMEDIATE PROOF EMISSION & JOURNAL STRUCTURE: Start your response directly with \\begin{proof} on the very first line. Do NOT write any preliminary outline, preamble, meta-commentary, or introductory remarks. Multi-part theorems (such as proving structural existence, algorithmic constructions, polynomial-time computability, or complexity bounds) must be organized directly under standard declarative paragraph headings (e.g. \\medskip\\noindent\\textbf{Part Name.}) matching the components of the target theorem. You may use short helper claims (\\begin{claim}...\\end{claim}\\begin{proof}...\\end{proof}) for modular lemmas, but do NOT wrap major theorem parts in outer \\begin{claim} blocks that duplicate the theorem statement. The entire output must conclude with a single \\end{proof} on the final line without duplicate trailing \\end{proof}.\n"
            "2. PURE DECLARATIVE JOURNAL LATEX: Write ONLY declarative mathematical statements establishing the theorem in standard journal style. Under NO circumstances output conversational self-talk, exploratory questions, stream-of-consciousness, or scratchpad thinking ('Wait', 'Let me check', 'Could this be?', 'Contradiction?').\n"
            "3. REPL IS CLOSED: Do NOT output Python code blocks (```python) or computational tests.\n"
            "4. STRICT THEOREM FIDELITY: The Target Problem is an established theorem from published theoretical computer science literature. Construct the rigorous mathematical proof that ESTABLISHES the theorem in full. Never attempt to disprove the theorem itself. Your own intermediate lemmas can be wrong, though: if a step of yours fails on an example, replace it with a correct argument.\n"
            "5. COMPLETE INVARIANT, BASE CASE & BOUNDARY INTEGRITY: Directly reference and verify each property required by the theorem statement throughout the argument. Rigorously establish all inductive steps, base cases, bounds, and complexity/algorithmic guarantees required by the theorem from first principles.\n"
            "6. STATEMENT PROPERTY LABELS: When the Target Problem defines numbered, lettered, or labeled properties/conditions, explicitly reference each property using its exact statement label when proving that it is satisfied. This ensures exact alignment with the theorem's specification.\n"
            "7. STRICT CONTEXT ADHERENCE: Rely ONLY on the definitions, lemmas, and facts stated in the problem context. Explicitly verify all preconditions before applying any context lemma. Never cite unstated external papers or theorems.\n"
            "8. NO HAND-WAVING: Do NOT refer to earlier chat turns or write phrases like 'as shown in the main text' or 'as shown in earlier turns'. Write out every deduction in full; the proof must be 100% self-contained.\n"
            "9. CONCISENESS & JOURNAL ELEGANCE: Top journal proofs (STOC, FOCS, SODA) are focused, direct, and elegant (typically 500 to 1,200 words, 3,000 to 7,500 characters). Avoid bloated multi-page lemmas, redundant definitions, or endless case splitting. Keep the entire proof strictly under 15,000 characters so that it is self-contained without truncation. Organize multi-part proofs with \\medskip\\noindent\\textbf{Section name.} rather than wrapping the entire theorem in sub-claims.\n"
            "10. STRICT DEFINITION FIDELITY: Strictly adhere to all explicit definitions, conventions, and notation in the problem statement. Never introduce alternative 'degenerate conventions' or redefine stated variables. Branch directly from the problem definitions.\n"
            "11. BOUNDARY & PARAMETER CONSISTENCY: In recursive algorithms or inductive proofs, verify that degenerate base instances (such as empty sets or boundary parameters) are handled rigorously from first principles. Ensure all parameter inequalities, exponents, and metric bounds align in the exact direction needed for the terminal statement.\n"
            "12. Conclude your response immediately with \\end{proof}."
        )

        synthesis_history = []
        if len(messages) >= 2:
            synthesis_history.append(messages[0])
            synthesis_history.append(getattr(self, "_task_only_msg", None) or messages[1])
            last_assistant = [m for m in messages if m.get("role") == "assistant"]
            if last_assistant:
                last_cont = last_assistant[-1].get("content", "")
                # Do NOT seed synthesis_history with a Python-code-only assistant message (which induces code emission in synthesis)
                has_code_without_proof = ("```python" in last_cont or "```py" in last_cont) and r"\begin{proof}" not in last_cont
                if not has_code_without_proof:
                    if len(last_cont) > 20000:
                        last_cont = (
                            last_cont[:6000]
                            + "\n\n% [... intermediate scratchpad condensed for synthesis ...]\n\n"
                            + last_cont[-12000:]
                        )
                    synthesis_history.append({"role": "assistant", "content": last_cont})
                elif result.reasoning_traces:
                    tail_r = result.reasoning_traces[-1][-6000:].strip()
                    if tail_r:
                        synthesis_history.append({
                            "role": "assistant",
                            "content": f"% [Summary of Mathematical Derivations from Exploration]:\n{tail_r}",
                        })

        synthesis_messages = synthesis_history + [
            {"role": "user", "content": contract_directive},
        ]
        if self.compactor.should_compact(synthesis_messages):
            comp_res = self.compactor.compact(synthesis_messages)
            if comp_res.compacted:
                synthesis_messages = comp_res.compacted_messages

        try:
            if best_candidate and is_valid_latex_proof(best_candidate.proof):
                result.candidate_proof = best_candidate.proof
                result.final_response = best_candidate.proof
                if best_candidate.verdict == "VALID" and (
                    not best_candidate.issues or best_candidate.issues.strip().lower() == "none"
                ):
                    self._log_write(
                        f"\n[Synthesis Optimization] Best Candidate (Turn {best_candidate.turn}, "
                        f"validity: {best_candidate.validity:.2f}) already passed two-pass audit with no issues; "
                        f"skipping redundant revision passes.\n"
                    )
                    result.verified = True
                    return best_candidate
                self._log_write(
                    f"\n[Synthesis Optimization] Skipping redundant Attempt 2 (Synthesis Turn); "
                    f"entering Synthesis Revision 1/2 directly on Best Candidate (Turn {best_candidate.turn}, "
                    f"validity: {best_candidate.validity:.2f}, structural: {best_candidate.structural_validity:.2f}).\n"
                )
                current_proof = best_candidate.proof
                current_issues = (
                    best_candidate.issues
                    if (best_candidate.issues and best_candidate.issues.strip().lower() != "none")
                    else last_issues
                )
                current_synth_text = best_candidate.proof
                synth_text = best_candidate.proof
            else:
                self._log_write("\n[Synthesis Turn: Emitting Publication Proof]\n")
                synth_reasoning_text, synth_text = self._stream_completion_with_retry(
                    messages=synthesis_messages,
                    temperature=0.2,
                    max_tokens=self.client.config.max_tokens,
                    enable_thinking=self.enable_thinking,
                    thinking_budget_tokens=self.client.config.thinking_token_budget or 16384,
                    timeout_seconds=600.0,
                    max_retries=5,
                )
                if not synth_text and synth_reasoning_text:
                    p_fall = extract_latex_proof(synth_reasoning_text)
                    if p_fall:
                        synth_text = p_fall
                self._log_write("\n" + synth_text + "\n")
                proof = extract_latex_proof(synth_text)
                if not proof and synth_text and synth_text.strip().endswith(r"\end{proof}") and not synth_text.strip().startswith(r"\begin{proof}"):
                    proof = extract_latex_proof("\\begin{proof}\n" + synth_text.strip())
                if proof and not is_valid_latex_proof(proof):
                    san = sanitize_informal_proof(proof)
                    if san and is_valid_latex_proof(san):
                        proof = san
                if not proof or not is_valid_latex_proof(proof):
                    self._log_write("[Synthesis Notice] Direct extraction did not yield a valid publication proof. Invoking Journal Typesetter pass...\n")
                    combined_synth_source = (
                        f"{synth_reasoning_text[-25000:]}\n\n{synth_text}".strip()
                        if synth_reasoning_text
                        else synth_text
                    )
                    typeset_cand = self._typeset_publication_proof(task_prompt, combined_synth_source)
                    if not typeset_cand:
                        all_assistant_text = "\n\n--- Next Exploration Trace ---\n\n".join(
                            m["content"] for m in messages if m.get("role") == "assistant" and m.get("content")
                        )
                        if result.reasoning_traces:
                            all_assistant_text = result.reasoning_traces[-1][-15000:] + "\n\n" + all_assistant_text
                        if len(all_assistant_text.strip()) > 200:
                            typeset_cand = self._typeset_publication_proof(task_prompt, all_assistant_text)
                    if typeset_cand:
                        self._log_write(f"[Journal Typesetter Success] Produced valid LaTeX proof ({len(typeset_cand)} chars).\n")
                        proof = typeset_cand

                if not proof or not is_valid_latex_proof(proof):
                    s_verdict = "REJECT"
                    s_issues = (
                        "- Format and Closure Defect: The previous generation failed publication proof validation. "
                        "Either \\begin{proof} ... \\end{proof} was incomplete, truncated, or missing proper closure, "
                        "or the body contained informal stream-of-consciousness/conversational remarks, unproven assertions, "
                        "or internal self-doubt. You must construct a complete, self-contained, formal mathematical LaTeX proof "
                        "enclosed strictly in \\begin{proof} ... \\end{proof}."
                    )
                    self._log_write(f"[Synthesis Format Notice] {s_issues}\n")
                    current_issues = s_issues
                    current_proof = None
                    current_synth_text = synth_text
                else:
                    self._log_write("\n[Auditing Synthesized Proof Draft]...\n")
                    if on_step:
                        on_step("audit_start", proof)
                    s_audit = self._audit_proof_draft(task_prompt, proof)
                    s_verdict = s_audit.verdict
                    s_issues = s_audit.issues
                    s_val = s_audit.validity
                    if on_step:
                        on_step("audit_result", {"verdict": s_verdict, "issues": s_issues, "validity": s_val})
                    if is_valid_latex_proof(proof):
                        if pool.offer(proof, s_val, s_verdict, s_issues, turn=990, audit_result=s_audit):
                            self._log_write(f"[Best Candidate Updated in Synthesis] validity={s_val:.2f} ({len(proof)} chars)\n")
                        best_candidate = pool.best
                    if s_verdict == "VALID":
                        self._log_write(f"[Synthesis Audit Passed] Final publication proof verified by referee (validity: {s_val:.2f}).\n")
                        if on_step:
                            on_step("proof_accepted", proof)
                        result.candidate_proof = proof
                        result.final_response = proof
                        result.verified = True
                        return best_candidate
                    else:
                        self._log_write(f"[Synthesis Audit Notes] (validity: {s_val:.2f}) {s_issues}\n")
                        current_issues = s_issues
                        current_proof = proof
                        current_synth_text = synth_text
                        if is_valid_latex_proof(proof) and best_candidate is not None and best_candidate.proof == proof:
                            result.candidate_proof = proof
                            result.final_response = proof

                if on_step:
                    on_step("synthesis_done", {"proof": proof, "valid": bool(proof and is_valid_latex_proof(proof))})

            # Attempt 3: Revision passes (up to 2 iterations)
            if not result.verified:
                # Always ensure Synthesis Revision starts from the highest-scoring candidate in pool.best
                # rather than a degraded or lower-scoring [Synthesis Turn] draft!
                if best_candidate and is_valid_latex_proof(best_candidate.proof) and best_candidate.proof != current_proof:
                    self._log_write(
                        f"[Synthesis Revision Seeding] Seeding revision passes with highest-scoring candidate from Turn {best_candidate.turn} "
                        f"(validity: {best_candidate.validity:.2f}, structural: {best_candidate.structural_validity:.2f}).\n"
                    )
                    current_proof = best_candidate.proof
                    if best_candidate.issues:
                        current_issues = best_candidate.issues

                max_revision_iters = 2
                deadline = getattr(self, "_solve_deadline", None)
                if (
                    deadline is not None
                    and (deadline - time.perf_counter()) < 600.0
                    and best_candidate
                    and (best_candidate.validity >= THRESHOLDS.pivot_threshold or best_candidate.has_substantive_structure)
                ):
                    self._solve_deadline = time.perf_counter() + 600.0
                    self._log_write(
                        f"[Synthesis Runway Extension] Extended synthesis deadline by +600s to allow complete referee revision "
                        f"passes for viable candidate (validity: {best_candidate.validity:.2f}).\n"
                    )

                for rev_iter in range(1, max_revision_iters + 1):
                    deadline = getattr(self, "_solve_deadline", None)
                    if deadline is not None and time.perf_counter() > deadline:
                        self._log_write("[Synthesis Revision] Wall-clock limit exceeded; skipping remaining revision passes.\n")
                        break
                    self._log_write(f"\n[Synthesis Revision {rev_iter}/{max_revision_iters}] Initiating referee revision pass to resolve flagged issues...\n")
                    if on_step:
                        on_step("revision_start", current_issues)
                    critique_verified, critique_alerts = self._run_critique_falsifier(
                        task_id, current_issues, current_proof or synth_text, on_step=on_step
                    )
                    critique_falsifier_feedback = ""
                    if critique_verified or critique_alerts:
                        c_lines = []
                        if critique_verified:
                            c_lines.append(
                                "--- ✓ Dynamic REPL Check: Candidate Invariant Held on Tested Instances (Retain Verified Math; Clarify Exposition) ---\n"
                                + "\n".join(f"✓ {f}" for f in critique_verified)
                            )
                        if critique_alerts:
                            c_lines.append("--- ⚠️ Critical REPL Falsification of Flagged Step ---\n" + "\n".join(f"⚠️ {a}" for a in critique_alerts))
                        critique_falsifier_feedback = "\n".join(c_lines) + "\n----------------------------------------------------\n\n"

                    repair_prompt = (
                        f"{heuristics_reminder}"
                        f"{repl_summary}"
                        f"{critique_falsifier_feedback}"
                        f"[CRITICAL REFEREE REVISION DIRECTIVE - ATTEMPT {rev_iter}/{max_revision_iters}]\n"
                        f"An independent mathematical referee audited the candidate proof approach and identified these defects:\n"
                        f"{current_issues}\n"
                        f"NOTE: The Target Problem is an established theorem and is true as stated, so ignore any referee speculation "
                        f"that the theorem itself is false. However, a counterexample to one of YOUR intermediate claims or steps is a real "
                        f"defect: replace or repair that step rather than defending it.\n\n"
                        f"CRITICAL REFEREE REVISION GUIDANCE:\n"
                        f"Address and resolve every defect flagged by the referee:\n"
                        f"- MODULAR LEMMA PRESERVATION: Identify which parts of the candidate proof were sound and verified (e.g., core invariants, base reductions, metric bounds). PRESERVE sound lemmas and established definitions VERBATIM. Do NOT alter, weaken, or rewrite unflagged premises, graph walk simplicity assumptions, or notation in the sound sections.\n"
                        f"- SURGICAL REPAIR OVER REWRITES: Focus your mathematical derivation strictly on resolving the specific defect, missing data structure, or unclosed gap flagged by the referee.\n"
                        f"- Missing Proof / Unjustified Claim: If the referee flagged an unproved step or assertion, provide the complete, rigorous deductive proof from first principles without hand-waving.\n"
                        f"- Preconditions & Definitions: Verify that every definition and lemma used has its preconditions strictly satisfied as specified in the context.\n"
                        f"- Logical Flaw: If an intermediate claim is mathematically false, pivot to a sound structural argument.\n\n"
                        f"STRICT RE-DERIVATION REQUIREMENTS:\n"
                        f"1. IMMEDIATE PROOF EMISSION & JOURNAL STRUCTURE: Start your response directly with \\begin{{proof}} on the very first line. Do NOT write any preliminary outline, preamble, meta-commentary, or introductory remarks. Multi-part theorems (such as proving structural existence, algorithmic constructions, polynomial-time computability, or complexity bounds) must be organized directly under standard declarative paragraph headings (e.g. \\medskip\\noindent\\textbf{{Part Name.}}) matching the components of the target theorem. Use \\begin{{claim}}...\\end{{claim}}\\begin{{proof}}...\\end{{proof}} only for short modular helper lemmas, and do NOT wrap major theorem parts in outer \\begin{{claim}} blocks. The output must conclude with a single \\end{{proof}} on the final line without duplicate trailing \\end{{proof}}.\n"
                        f"2. PURE DECLARATIVE JOURNAL LATEX: Write ONLY declarative mathematical statements establishing the theorem in standard journal style. Strictly NO conversational self-talk, exploratory questions, or scratchpad thinking ('Wait', 'Let me check', 'Could this be?').\n"
                        f"3. RESOLVE ALL REFEREE DEFECTS: Completely address and resolve every issue, defect, or missing case identified above. Do NOT patch flawed reasoning with hand-waving; reconstruct the argument rigorously from first principles.\n"
                        f"4. TARGETED REPL TESTING: You may include an optional short ```python ... ``` block to test your corrected parameter bound, recurrence inequality, or graph property. Conclude your response with the complete formal LaTeX proof enclosed strictly in \\begin{{proof}} ... \\end{{proof}}.\n"
                        f"5. COMPLETE INVARIANT, BASE CASE & BOUNDARY INTEGRITY: Directly reference and verify each property required by the theorem statement throughout the argument. Rigorously establish all inductive steps, base cases, bounds, and complexity/algorithmic guarantees required by the theorem from first principles.\n"
                        f"6. STATEMENT PROPERTY LABELS: When the Target Problem defines numbered, lettered, or labeled properties/conditions, explicitly reference each property using its exact statement label when proving that it is satisfied. This ensures exact alignment with the theorem's specification.\n"
                        f"7. STRICT CONTEXT ADHERENCE: Use only definitions and lemmas provided in the context; verify all preconditions.\n"
                        f"8. NO HAND-WAVING: Write out every deduction in full without referring to earlier turns or using phrases like 'as shown in the main text' or 'as shown in earlier turns'.\n"
                        f"9. CONCISENESS & JOURNAL ELEGANCE: Structure the proof into direct, elegant mathematical paragraphs (typically 500 to 1,200 words, 3,000 to 7,500 characters). Keep the proof strictly under 15,000 characters so that it is self-contained without truncation. Organize multi-part proofs with \\medskip\\noindent\\textbf{{Section name.}} rather than wrapping the entire theorem in sub-claims.\n"
                        f"10. STRICT DEFINITION FIDELITY: Strictly adhere to all explicit definitions, conventions, and notation in the problem statement. Never introduce alternative 'degenerate conventions' or redefine stated variables. Branch directly from the problem definitions, evaluating boundary cases on the exact objects designated by the statement.\n"
                        f"11. Conclude your response immediately with \\end{{proof}}."
                    )
                    candidate_under_repair = current_proof or current_synth_text or synth_text
                    candidate_block = (
                        "--- Complete Candidate Proof Under Referee Revision ---\n"
                        f"{candidate_under_repair}\n"
                        "--- [End of Candidate Proof Under Revision] ---\n\n"
                    )
                    # Clean Stateless Blackboard Architecture for Synthesis Revision
                    repair_messages = [
                        synthesis_messages[0],  # System prompt
                        getattr(self, "_task_only_msg", None) or synthesis_messages[1],  # Task prompt & context lemmas
                        {"role": "assistant", "content": candidate_block},
                        {"role": "user", "content": repair_prompt},
                    ]
                    if self.compactor.should_compact(repair_messages):
                        comp_res = self.compactor.compact(repair_messages, protected=repair_messages[2:])
                        if comp_res.compacted:
                            repair_messages = comp_res.compacted_messages
                    try:
                        repair_reasoning, repair_text = self._stream_completion_with_retry(
                            messages=repair_messages,
                            temperature=0.2,
                            max_tokens=self.client.config.max_tokens,
                            enable_thinking=self.enable_thinking,
                            thinking_budget_tokens=self.client.config.thinking_token_budget or 16384,
                            timeout_seconds=600.0,
                            max_retries=5,
                        )
                        rev_code_blocks = self.extract_code_blocks(repair_text)
                        if rev_code_blocks:
                            for c in rev_code_blocks:
                                result.code_history.append(c)
                                self._log_write(f"[Synthesis Revision REPL Execution]:\n```python\n{c}\n```\n")
                                ex_res = self.kernel.execute(c)
                                result.execution_history.append(ex_res)
                                if ex_res.stdout:
                                    self._log_write(f"Stdout:\n{ex_res.stdout}\n")

                        r_proof = extract_latex_proof(repair_text)
                        if not r_proof and repair_text and repair_text.strip().endswith(r"\end{proof}") and not repair_text.strip().startswith(r"\begin{proof}"):
                            r_proof = extract_latex_proof("\\begin{proof}\n" + repair_text.strip())
                        if r_proof and not is_valid_latex_proof(r_proof):
                            san_r = sanitize_informal_proof(r_proof)
                            if san_r and is_valid_latex_proof(san_r):
                                r_proof = san_r
                        if not r_proof or not is_valid_latex_proof(r_proof):
                            combined_rev_source = (
                                f"{repair_reasoning[-25000:]}\n\n{repair_text}".strip()
                                if repair_reasoning
                                else repair_text
                            )
                            typeset_r = self._typeset_publication_proof(task_prompt, combined_rev_source)
                            if not typeset_r:
                                all_assistant_text = "\n\n--- Next Exploration Trace ---\n\n".join(
                                    m["content"] for m in messages if m.get("role") == "assistant" and m.get("content")
                                )
                                if len(all_assistant_text.strip()) > 200:
                                    typeset_r = self._typeset_publication_proof(task_prompt, all_assistant_text)
                            if typeset_r:
                                r_proof = typeset_r
                        if r_proof and is_valid_latex_proof(r_proof):
                            if on_step:
                                on_step("audit_start", r_proof)
                            r_audit = self._audit_proof_draft(task_prompt, r_proof)
                            r_verdict = r_audit.verdict
                            r_issues = r_audit.issues
                            r_val = r_audit.validity
                            if on_step:
                                on_step("audit_result", {"verdict": r_verdict, "issues": r_issues, "validity": r_val})
                            if is_valid_latex_proof(r_proof):
                                if pool.offer(r_proof, r_val, r_verdict, r_issues, turn=990 + rev_iter, audit_result=r_audit):
                                    best_candidate = pool.best
                                    self._log_write(f"[Best Candidate Updated in Revision {rev_iter}] validity={r_val:.2f} ({len(r_proof)} chars)\n")
                                else:
                                    best_candidate = pool.best
                            if r_verdict == "VALID":
                                self._log_write(f"[Revised Proof Passed Audit on Attempt {rev_iter}!] Proof verified by referee (validity: {r_val:.2f}).\n")
                                if on_step:
                                    on_step("proof_accepted", r_proof)
                                result.candidate_proof = r_proof
                                result.final_response = r_proof
                                result.verified = True
                                break
                            else:
                                self._log_write(f"[Revised Proof Audit Notes Attempt {rev_iter}] (validity: {r_val:.2f}) {r_issues}\n")
                                if best_candidate and is_valid_latex_proof(best_candidate.proof):
                                    result.candidate_proof = best_candidate.proof
                                    result.final_response = best_candidate.proof
                                    current_proof = best_candidate.proof
                                    current_issues = best_candidate.issues or r_issues
                                else:
                                    current_issues = r_issues
                                    if is_valid_latex_proof(r_proof):
                                        current_proof = r_proof
                                current_synth_text = repair_text
                                if on_step:
                                    on_step("synthesis_done", {"proof": r_proof, "valid": False})
                        else:
                            self._log_write(f"[Synthesis Revision Attempt {rev_iter}] Revision did not yield a valid LaTeX proof; retaining previous draft.\n")
                            if current_proof:
                                result.candidate_proof = current_proof
                                result.final_response = current_proof
                            if on_step:
                                on_step("synthesis_done", {"proof": r_proof, "valid": bool(r_proof and is_valid_latex_proof(r_proof))})
                    except Exception as rep_err:
                        self._log_write(f"[Synthesis Revision Error Attempt {rev_iter}] {rep_err}\n")
                        if current_proof:
                            result.candidate_proof = current_proof
                            result.final_response = current_proof
                        break
            elif synth_text:
                result.final_response = synth_text
        except Exception as e:
            self._log_write(f"Final proof synthesis failed: {e}\n")

        return best_candidate

    def _finalize_result(
        self,
        task_id: str,
        task_prompt: str,
        result: TCSSolveResult,
        best_candidate: Optional[CandidateRecord],
        start_wall_time: float,
    ) -> TCSSolveResult:
        """Best candidate recovery, final formatting, duration accounting,
        and TurboVec consolidation.
        """
        if best_candidate and is_valid_latex_proof(best_candidate.proof):
            if result.candidate_proof != best_candidate.proof:
                self._log_write(
                    f"[Best Candidate Recovery] Restoring highest-scoring candidate from Turn {best_candidate.turn} "
                    f"(validity: {best_candidate.validity:.2f}, {len(best_candidate.proof)} chars).\n"
                )
                result.candidate_proof = best_candidate.proof
                result.final_response = best_candidate.proof
            if best_candidate.verdict == "VALID":
                result.verified = True

        if not result.candidate_proof:
            proof = extract_latex_proof(result.final_response)
            if proof and is_valid_latex_proof(proof):
                result.candidate_proof = proof
            else:
                for tr in reversed(result.reasoning_traces):
                    p = extract_latex_proof(tr)
                    if p and is_valid_latex_proof(p):
                        result.candidate_proof = p
                        break
            if not result.candidate_proof:
                draft_src = result.final_response or (result.reasoning_traces[-1] if result.reasoning_traces else "")
                if draft_src:
                    self._log_write("[Fallback Notice] Attempting Journal Typesetter on best available draft...\n")
                    p_fallback = self._typeset_publication_proof(task_prompt, draft_src)
                    if p_fallback:
                        result.candidate_proof = p_fallback
            if result.candidate_proof:
                result.final_response = result.candidate_proof

        result.duration_s = round(time.perf_counter() - start_wall_time, 2)
        n_chars = len(result.candidate_proof or "")
        safe_task_id = re.sub(r"\d{4}\.\d{4,5}", "[ID]", str(task_id))
        self._log_write(
            f"\n{'=' * 80}\n"
            f"[SESSION COMPLETE] Task: {safe_task_id} | Proof Length: {n_chars} chars | "
            f"Turns: {result.iterations} | Duration: {result.duration_s}s\n"
            f"{'=' * 80}\n"
        )

        self._consolidate_learning(task_id, task_prompt, result)
        return result

    def solve(
        self,
        task_id: str,
        task_prompt: str,
        max_iterations: Optional[int] = None,
        max_time_seconds: Optional[float] = None,
        reset_kernel: bool = True,
        on_step: Optional[Any] = None,
        enable_literature_search: Optional[bool] = None,
    ) -> TCSSolveResult:
        """Solves a TCS-Bench theorem proving task through closed-loop reasoning."""
        if not task_prompt or not task_prompt.strip():
            self._log_write(f"[Solver] Empty task prompt for {task_id}; skipping.\n")
            return TCSSolveResult(
                problem=task_prompt or "",
                task_id=task_id,
                candidate_proof=None,
                verified=False,
                iterations=0,
                error="Empty task prompt in dataset",
                duration_s=0.0,
            )

        start_wall_time = time.perf_counter()
        max_turns = max_iterations or self.default_max_iterations
        if max_time_seconds is None:
            max_time_seconds = float(getattr(getattr(self, "config", None), "solver_timeout_seconds", 3600.0) or 3600.0)
        self._solve_deadline = (start_wall_time + max_time_seconds) if max_time_seconds else None

        # 1. Setup session & query TurboVec / subagents
        messages, result, retrieved = self._init_solve_session(
            task_id=task_id,
            task_prompt=task_prompt,
            max_turns=max_turns,
            reset_kernel=reset_kernel,
            on_step=on_step,
            enable_literature_search=enable_literature_search,
        )

        # 2. Multi-turn reasoning & REPL invariant verification loop
        best_candidate, last_issues = self._run_exploratory_turns(
            task_id=task_id,
            task_prompt=task_prompt,
            messages=messages,
            result=result,
            max_turns=max_turns,
            start_wall_time=start_wall_time,
            max_time_seconds=max_time_seconds,
            on_step=on_step,
        )

        # 3. Final publication proof synthesis & referee audit loop
        if not result.verified or not is_valid_latex_proof(result.candidate_proof):
            best_candidate = self._run_synthesis_pipeline(
                task_id=task_id,
                task_prompt=task_prompt,
                messages=getattr(self, "_last_messages", None) or messages,
                result=result,
                best_candidate=best_candidate,
                last_issues=last_issues,
                retrieved=retrieved,
                on_step=on_step,
            )

        # 4. Finalize result, recover best candidate, and consolidate learning to TurboVec
        return self._finalize_result(
            task_id=task_id,
            task_prompt=task_prompt,
            result=result,
            best_candidate=best_candidate,
            start_wall_time=start_wall_time,
        )


__all__ = [
    "MathProver",
    "MATH_PROVER_SYSTEM_PROMPT",
    "TCS_BENCH_SYSTEM_PROMPT",
]
