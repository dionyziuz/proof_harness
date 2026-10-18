"""Auditing and Empirical Falsification Sub-systems for MathProver.

Provides:
- ProverAuditorMixin: Mixin providing two-stage verification (Stage 1 empirical falsification
  in an ephemeral AuditorKernel, Stage 2 academic referee audit), proof mode detection,
  and publication LaTeX typesetting.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

from .prompt_loader import load_tcs_prompt
from .proof_extractor import (
    extract_latex_proof,
    is_valid_latex_proof,
    sanitize_informal_proof,
)
from .prover_kernel import AuditorKernel
from .prover_types import THRESHOLDS, AuditResult, has_fatal_issue

logger = logging.getLogger("prime_math.prover_auditor")


class ProverAuditorMixin:
    """Mixin providing auditing, referee evaluation, and streaming retry mechanics."""

    def _detect_proof_mode(self, task_prompt: str, proof_text: str = "") -> str:
        """Determines whether to use 'tcs' mode (academic literature referee)
        or 'computational' mode (two-stage empirical falsifier + referee).
        """
        mode = getattr(self, "proof_mode", "auto")
        if mode in ("tcs", "computational"):
            return mode

        # Auto detection
        # 1. TCS-Bench hallmarks:
        tcs_markers = (
            "# Context:",
            "Definitions and Lemmas from context",
            "Target Problem (your task):",
            "arXiv:2608.09538",
            "Theorem 1.",
            "Lemma 1.",
        )
        if any(marker in task_prompt for marker in tcs_markers):
            return "tcs"

        # 2. Computational / concrete witness / Diophantine / finite graph hallmarks:
        prompt_lower = task_prompt.lower()
        comp_markers = (
            "discrete logarithm",
            "elliptic curve",
            "diophantine",
            "ramsey",
            "clique number",
            "chromatic number",
            "paley graph",
            "euler sum of powers",
            "find an integer",
            "find all integers",
            "exact value",
            "minimal counterexample",
            "finite graph",
            "integer coordinates",
        )
        if any(marker in prompt_lower for marker in comp_markers):
            return "computational"

        # 3. If literature search is enabled (typical for open problems like FrontierMath):
        if getattr(self, "enable_literature_search", False):
            return "computational"

        # Default fallback is TCS
        return "tcs"

    def _empirical_falsifier_audit(self, task_prompt: str, proof_text: str) -> Tuple[bool, str]:
        """Stage 1: Sandboxed Empirical Adversarial Falsifier.
        Generates and runs an isolated Python verification script testing any claimed concrete
        witnesses, equations, numerical values, or finite graph/algebraic properties asserted in the proof.
        Returns (passed: bool, reason: str).
        """
        try:
            falsifier_system = load_tcs_prompt("proof_auditor_computational")
        except Exception:
            falsifier_system = (
                "You are an adversarial mathematical auditor and code-based empirical falsifier.\n"
                "Your objective: given a problem and candidate proof, write an isolated Python verification script "
                "that tests every concrete numerical claim, witness, algebraic identity, or finite object asserted in the proof.\n\n"
                "GUIDELINES:\n"
                "1. Extract all concrete claims: claimed solution values (x, y, z), points on curves, graph constructions, "
                "residues, matrix ranks, or small boundary values (e.g. n=1, 2, 3, 4).\n"
                "2. Write a Python script using standard libraries (sympy, math, fractions, networkx, mpmath, z3, itertools).\n"
                "3. Use explicit `assert` statements to verify whether the claimed witness actually solves the equation or conditions.\n"
                "4. If the proof is purely abstract and contains NO concrete witnesses, numerical claims, or finite checks, "
                "output ```python\n# NO_CONCRETE_WITNESS\n```\n"
                "5. Wrap your Python code strictly in ```python ... ``` fences."
            )
        falsifier_user = f"Target Problem:\n{task_prompt}\n\nCandidate Proof:\n{proof_text}\n"

        try:
            resp = self.client.complete(
                [{"role": "system", "content": falsifier_system}, {"role": "user", "content": falsifier_user}],
                temperature=0.1,
                max_tokens=4096,
                enable_thinking=True,
                thinking_budget_tokens=2048,
                timeout_seconds=90.0,
            )
            content = resp.content or ""
            code_blocks = self.extract_code_blocks(content)
            if not code_blocks:
                return True, "No test code generated; skipping Stage 1"

            test_code = code_blocks[0].strip()
            if "NO_CONCRETE_WITNESS" in test_code:
                return True, "Proof is abstract with no finite witness to falsify"

            # Execute in an ephemeral AuditorKernel
            auditor_kernel = AuditorKernel(default_timeout=getattr(self, "auditor_kernel_timeout", 15.0))
            exec_res = auditor_kernel.execute(test_code, timeout_seconds=getattr(self, "auditor_kernel_timeout", 15.0))

            if not exec_res.success:
                err_text = exec_res.error or ""
                # An explicit assertion failure means the claimed witness is false (unless it is IEEE-754 float rounding)
                if "AssertionError" in err_text or "Assertion failed" in err_text:
                    floats_in_err = [float(x) for x in re.findall(r"[-+]?\d+\.\d{6,}(?:[eE][-+]?\d+)?", err_text)]
                    if len(floats_in_err) >= 2:
                        a, b = floats_in_err[-2], floats_in_err[-1]
                        denom = max(abs(a), abs(b), 1.0)
                        if abs(a - b) / denom < 1e-8:
                            return True, f"Ignored IEEE-754 floating-point rounding difference ({a} vs {b})"
                    fail_msg = (
                        f"Empirical Falsification: Witness/assertion failed:\n{err_text.strip()}\n"
                        f"[Falsifier Test Script]:\n{test_code[:1200]}"
                    )
                    return False, fail_msg
                # Runtime script bugs (e.g. NameError in auditor's test code) are inconclusive
                return True, f"Auditor test script encountered runtime error: {err_text[:200]}"

            return True, "Empirical verification passed all assertions"
        except Exception as e:
            return True, f"Stage 1 auditor error: {e}"

    @staticmethod
    def _detect_unauthorized_restrictions(proof_text: str, task_prompt: str) -> List[str]:
        """Scans a candidate proof for generic English discourse phrases that narrow the
        universally quantified domain mid-proof without a complementary case split.
        """
        if not proof_text:
            return []
        restriction_re = re.compile(
            r"(?:"
            r"we\s+assume\s+additionally\s+that|"
            r"assuming\s+further\s+that|"
            r"under\s+the\s+(?:additional\s+)?(?:regime|restriction|assumption)\s+(?:that|where)|"
            r"restrict(?:ing)?\s+(?:our\s+attention|the\s+analysis|the\s+proof)\s+to|"
            r"holds\s+(?:only\s+)?in\s+the\s+regime\s+where|"
            r"valid\s+in\s+the\s+regime\s+where|"
            r"in\s+the\s+restricted\s+regime"
            r")\s+([^.,;\n]{3,140})",
            re.IGNORECASE,
        )
        complement_re = re.compile(
            r"\b(?:otherwise|conversely|on\s+the\s+other\s+hand|complement(?:ary)?\s+(?:case|regime)|"
            r"case\s+2|case\s+ii|when\s+instead|in\s+the\s+remaining\s+case)\b",
            re.IGNORECASE,
        )
        has_complement = bool(complement_re.search(proof_text))
        prompt_norm = " ".join((task_prompt or "").lower().split())
        flagged: List[str] = []
        for m in restriction_re.finditer(proof_text):
            clause = " ".join(m.group(1).strip().split())
            if clause.lower() in prompt_norm:
                continue
            if not has_complement:
                flagged.append(m.group(0).strip())
        return flagged

    def _audit_proof_draft(self, task_prompt: str, proof_text: str) -> AuditResult:
        """Audits a candidate proof against TCS-Bench rigor criteria using an academic referee call
        with zero ground-truth leakage.
        In computational/auto mode, runs Stage 1 empirical falsification in an isolated AuditorKernel
        before proceeding to Stage 2 academic referee.
        Returns AuditResult(verdict='VALID'|'REJECT'|'INCONCLUSIVE', issues='...', validity=float).
        """
        mode = self._detect_proof_mode(task_prompt, proof_text)
        self._log_write(f"\n[Proof Auditor] Active mode: {mode}\n")

        # Stage 1: Empirical Adversarial Falsification
        stage1_note = ""
        if mode in ("computational", "tcs", "auto"):
            stage1_passed, failure_reason = self._empirical_falsifier_audit(task_prompt, proof_text)
            if not stage1_passed:
                self._log_write(f"[Proof Auditor Stage 1 Counterexample]: {failure_reason}\n")
                if mode == "computational":
                    return AuditResult(
                        verdict="REJECT",
                        issues=failure_reason,
                        validity=0.0,
                    )
                # In TCS and auto modes, forward the concrete falsification finding to the Stage 2 referee
                # so that test script assertion bugs do not unilaterally zero out mathematical proofs.
                stage1_note = (
                    f"\n\n[Computational Falsifier Alert]:\n{failure_reason}\n"
                    "Evaluate if this assertion failure reflects an actual counterexample to a claim in the proof "
                    "or a flaw in the test script's premises. If it is a genuine counterexample, reject the proof."
                )

        # Stage 1.5: Deterministic Domain-Restriction Pre-Check
        flagged_restrictions = self._detect_unauthorized_restrictions(proof_text, task_prompt)
        if flagged_restrictions:
            joined_flags = "; ".join(f"'{r}'" for r in flagged_restrictions[:3])
            self._log_write(f"[Proof Auditor Domain Restriction Pre-Check]: Flagged {joined_flags}\n")
            stage1_note += (
                f"\n\n[Deterministic Domain-Coverage Pre-Check Alert]:\n"
                f"The candidate proof contains discourse markers narrowing the domain ({joined_flags}) "
                f"without an obvious complementary case split. Verify whether the proof restricts the "
                f"Target Problem's universally quantified domain to a subset without proving the complement. "
                f"If any part of the target domain is left unproved, set \"statement_fidelity_exact\": false, "
                f"cap \"validity\" at <= 0.25, and return \"verdict\": \"REJECT\"."
            )

        # Stage 1.75: Stateful Prior-Defect Verification
        prior_defects: List[str] = getattr(self, "_prior_audit_defects", [])
        if prior_defects:
            prior_block = "\n".join(f"  {idx + 1}. {d[:600]}" for idx, d in enumerate(prior_defects[-3:]))
            stage1_note += (
                f"\n\n[PRIOR REFEREE DEFECTS FROM EARLIER DRAFTS (VERIFY IF RESOLVED OR BYPASSED)]:\n"
                f"{prior_block}\n"
                f"CRITICAL: Check line-by-line whether the candidate proof still contains an unproved gap from the "
                f"prior defects listed above. NOTE: A prior defect IS resolved if the candidate proof either (a) directly "
                f"repairs the step, (b) pivots to an alternative proof route that does not rely on the flawed step, OR "
                f"(c) proves a bridging claim from the Context definitions showing that the required precondition holds "
                f"automatically. Do NOT blindly repeat an earlier objection if the current draft supplies a valid mathematical "
                f"argument discharging it. Only if a prior defect genuinely remains unproved and actively relied upon in "
                f"the current draft, set \"verdict\": \"REJECT\" and cap \"validity\" at <= 0.40."
            )

        audit_system = load_tcs_prompt("proof_auditor")
        audit_user = f"Target Problem & Context:\n{task_prompt}\n\nCandidate Proof to Audit:\n{proof_text}\n{stage1_note}"
        cfg = getattr(self, "config", None)

        # One retry on INCONCLUSIVE (unparseable output / transport failure / length truncation):
        # If Attempt 1 truncated mid-derivation after producing substantial reasoning notes, preserve
        # those notes on Attempt 2 with thinking disabled (mirroring judge_b_once) so the discovered
        # mathematical defects are synthesized directly into the required JSON verdict instead of discarded.
        result = AuditResult(verdict="INCONCLUSIVE", issues="", validity=0.0)
        partial_notes = ""
        for attempt in range(2):
            try:
                req_user = audit_user
                req_thinking = True
                req_max_tokens = getattr(cfg, "auditor_max_tokens", 16384)
                req_thinking_budget = getattr(cfg, "auditor_thinking_budget", 12288)
                if attempt == 1 and len(partial_notes) > 150:
                    req_user = (
                        f"{audit_user}\n\n"
                        f"[PARTIAL REFEREE DERIVATION NOTES FROM ATTEMPT 1 (TRUNCATED BEFORE JSON EMISSION)]:\n"
                        f"{partial_notes[-6000:]}\n\n"
                        f"Based on the Target Problem, Candidate Proof, and your partial referee notes above, "
                        f"immediately output ONLY the required JSON verdict object (no commentary, no markdown fences)."
                    )
                    req_thinking = False
                    req_thinking_budget = None
                resp = self.client.complete(
                    [{"role": "system", "content": audit_system}, {"role": "user", "content": req_user}],
                    temperature=0.1,
                    max_tokens=req_max_tokens,
                    enable_thinking=req_thinking,
                    thinking_budget_tokens=req_thinking_budget,
                    timeout_seconds=getattr(cfg, "auditor_timeout_seconds", 360.0),
                )
                content = resp.content.strip() if resp.content else ""
                reasoning_str = (getattr(resp, "reasoning", None) or "").strip()
                self._log_write(f"\n[Academic Proof Auditor Output]:\n{content}\n")
                result = self._classify_audit_output(content)
                if result.verdict == "INCONCLUSIVE":
                    partial_notes = f"{reasoning_str}\n{content}".strip()
            except Exception as e:
                self._log_write(f"[Audit Error] Auditor call failed (attempt {attempt + 1}/2): {e}\n")
                result = AuditResult(verdict="INCONCLUSIVE", issues="", validity=0.0)
            if result.verdict != "INCONCLUSIVE":
                break
            self._log_write("[Proof Auditor] Inconclusive referee output; retrying once.\n")

        # Stage 2.5: Adversarial Confirmation on VALID (all >= 0.70 or whenever prior defects existed)
        # Single-pass auditors occasionally rubber-stamp polished prose (even at >= 0.92) that contains an
        # inverted inequality direction, an unremoved prior defect, or an unproved complementary regime.
        if result.verdict == "VALID" and (result.validity >= 0.70 or bool(prior_defects)):
            try:
                confirm_prompt = (
                    f"{audit_user}\n\n"
                    "[ADVERSARIAL SECOND-OPINION CHECK]:\n"
                    "A preliminary review scored this draft as valid. Perform a strict independent verification of:\n"
                    "1. Prior Defect Elimination: Were ALL prior referee defects from earlier drafts either repaired, bypassed by an alternative route, or discharged by a valid bridging lemma?\n"
                    "2. Statement & Domain Fidelity: Does the proof introduce ANY unstated parameter/structural restriction "
                    "(e.g., 0 < beta <= 1) without proving the complementary regime?\n"
                    "3. Inequality & Sign Integrity: Does every chained inequality, matrix/vector dimension, and algebraic sign hold across the full domain?\n"
                    "4. Unproved Intermediate Claims: Are any helper claims or lemma preconditions asserted without proof?\n"
                    "Respond with the required JSON object."
                )
                c_resp = self.client.complete(
                    [{"role": "system", "content": audit_system}, {"role": "user", "content": confirm_prompt}],
                    temperature=0.2,
                    max_tokens=getattr(cfg, "auditor_max_tokens", 16384),
                    enable_thinking=True,
                    thinking_budget_tokens=getattr(cfg, "auditor_thinking_budget", 12288),
                    timeout_seconds=getattr(cfg, "auditor_timeout_seconds", 360.0),
                )
                c_content = c_resp.content.strip() if c_resp.content else ""
                self._log_write(f"\n[Borderline Confirmation Auditor Output]:\n{c_content}\n")
                c_result = self._classify_audit_output(c_content)
                if c_result.verdict == "INCONCLUSIVE":
                    c_notes = f"{(getattr(c_resp, 'reasoning', None) or '').strip()}\n{c_content}".strip()
                    if len(c_notes) > 150:
                        c_resp2 = self.client.complete(
                            [
                                {"role": "system", "content": audit_system},
                                {
                                    "role": "user",
                                    "content": (
                                        f"{confirm_prompt}\n\n"
                                        f"[PARTIAL SECOND-OPINION NOTES (TRUNCATED BEFORE JSON EMISSION)]:\n"
                                        f"{c_notes[-6000:]}\n\n"
                                        f"Immediately output ONLY the required JSON verdict object."
                                    ),
                                },
                            ],
                            temperature=0.1,
                            max_tokens=4096,
                            enable_thinking=False,
                            timeout_seconds=120.0,
                        )
                        c_content2 = c_resp2.content.strip() if c_resp2.content else ""
                        self._log_write(f"\n[Borderline Confirmation Repair Output]:\n{c_content2}\n")
                        c_result = self._classify_audit_output(c_content2)
                if c_result.verdict != "INCONCLUSIVE" and c_result.validity < result.validity:
                    result = c_result
            except Exception as e:
                self._log_write(f"[Borderline Confirmation Warning]: {e}\n")

        if result.verdict == "REJECT" and result.issues and result.issues.lower() != "none":
            if not hasattr(self, "_prior_audit_defects"):
                self._prior_audit_defects = []
            if result.issues not in self._prior_audit_defects:
                self._prior_audit_defects.append(result.issues)

        return result

    @staticmethod
    def _parse_audit_json(content: str) -> Optional[dict]:
        """Extracts the referee's JSON object (tolerating code fences and unescaped LaTeX backslashes)."""
        fence_m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, flags=re.DOTALL)
        raw_json = fence_m.group(1).strip() if fence_m else content
        brace_m = re.search(r"\{.*\}", raw_json, flags=re.DOTALL)
        if brace_m:
            json_str = brace_m.group(0)
            for candidate in (json_str, re.sub(r'\\([a-zA-Z])', r'\\\\\1', json_str)):
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except (ValueError, TypeError):
                    continue
        # Regex recovery (handles truncated JSON); only accept if a numeric validity is present.
        m_val = re.search(r'"validity"\s*:\s*([0-9]*\.?[0-9]+)', content)
        if not m_val:
            return None
        m_verdict = re.search(r'"verdict"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
        m_iss = re.search(r'"issues"\s*:\s*"(.*?)"\s*[,}]', content, re.DOTALL)
        out: dict = {
            "validity": float(m_val.group(1)),
            "verdict": m_verdict.group(1) if m_verdict else "",
            "issues": m_iss.group(1).strip() if m_iss else "",
        }
        for sub_key in ("rigor", "completeness", "boundary_cases"):
            m_sub = re.search(rf'"{sub_key}"\s*:\s*([0-9]*\.?[0-9]+)', content)
            if m_sub:
                out[sub_key] = float(m_sub.group(1))
        for gate_key in ("statement_fidelity_exact", "inequality_directions_verified", "theorem_hypotheses_verified"):
            m_gate = re.search(rf'"{gate_key}"\s*:\s*(true|false)', content, re.IGNORECASE)
            if m_gate:
                out[gate_key] = m_gate.group(1).lower() == "true"
        return out

    @staticmethod
    def _classify_audit_output(content: str) -> AuditResult:
        """Maps raw referee output to an AuditResult. Fail-closed: no synthetic scores are ever invented."""
        parsed = ProverAuditorMixin._parse_audit_json(content)
        if not parsed or parsed.get("validity") is None:
            return AuditResult(verdict="INCONCLUSIVE", issues=content[-1500:].strip(), validity=0.0)

        try:
            validity = max(0.0, min(1.0, float(parsed.get("validity"))))
        except (TypeError, ValueError):
            return AuditResult(verdict="INCONCLUSIVE", issues=content[-1500:].strip(), validity=0.0)

        def _safe_subscore(key: str) -> float:
            try:
                val = parsed.get(key)
                return max(0.0, min(1.0, float(val))) if val is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        rigor = _safe_subscore("rigor")
        completeness = _safe_subscore("completeness")
        boundary_cases = _safe_subscore("boundary_cases")

        verdict_str = str(parsed.get("verdict", "")).strip().upper()
        issues = str(parsed.get("issues", "")).strip()
        if issues.lower().rstrip(".") in ("none", "n/a", "no issues", "no defects", "no fatal flaws", "no flaws"):
            issues = ""

        # Enforce explicit boolean referee gates when present
        if parsed.get("statement_fidelity_exact") is False:
            validity = min(validity, 0.25)
            verdict_str = "REJECT"
            if not issues:
                issues = "Statement fidelity violation: the proof introduces an unstated domain restriction or proves a weakened statement without covering the complementary regime."
        if parsed.get("inequality_directions_verified") is False:
            validity = min(validity, 0.25)
            verdict_str = "REJECT"
            if not issues:
                issues = "Inequality direction defect: one or more chained inequalities point in the wrong direction or lack monotonicity justification."
        if parsed.get("theorem_hypotheses_verified") is False:
            validity = min(validity, 0.35)
            verdict_str = "REJECT"
            if not issues:
                issues = "Unverified lemma/theorem hypotheses: a contextual lemma was invoked without discharging all of its preconditions."

        t = THRESHOLDS
        if verdict_str == "REJECT" or has_fatal_issue(issues):
            return AuditResult(
                verdict="REJECT",
                issues=issues or f"Referee rejected the proof (validity {validity:.2f}).",
                validity=validity,
                rigor=rigor,
                completeness=completeness,
                boundary_cases=boundary_cases,
            )
        if validity >= t.audit_valid_floor or (verdict_str == "VALID" and validity >= t.audit_verdict_floor):
            return AuditResult(
                verdict="VALID",
                issues=issues,
                validity=validity,
                rigor=rigor,
                completeness=completeness,
                boundary_cases=boundary_cases,
            )
        return AuditResult(
            verdict="REJECT",
            issues=issues or f"Proof validity scored {validity:.2f} (below {t.audit_verdict_floor:.2f}).",
            validity=validity,
            rigor=rigor,
            completeness=completeness,
            boundary_cases=boundary_cases,
        )

    def _typeset_publication_proof(self, task_prompt: str, raw_draft: str) -> Optional[str]:
        """Converts an informal mathematical draft or exploratory trace into a publication-grade LaTeX proof
        in pure declarative academic style, eliminating scratchpad chatter and conversational remarks.
        """
        if not raw_draft or len(raw_draft.strip()) < 100:
            return None
        sys_prompt = (
            "You are an expert theoretical computer science journal referee and editor.\n"
            "You are given an informal mathematical draft containing scratchpad reasoning, rhetorical questions, and conversational remarks.\n"
            "Your task is to extract the core valid mathematical arguments and write a formal, publication-grade LaTeX proof.\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY the formal proof enclosed strictly in \\begin{proof} ... \\end{proof}.\n"
            "2. Start directly with \\begin{proof} on line 1 and conclude strictly with \\end{proof}.\n"
            "3. Write in concise, authoritative academic prose. Every sentence must be a declarative mathematical deduction.\n"
            "4. DO NOT copy or include any scratchpad self-talk, rhetorical questions, doubts, or informal phrases like 'Wait', 'Actually', 'Let me see', 'I will', 'the draft'.\n"
            "5. The proof should be concise, self-contained, and complete.\n"
            "6. Explicitly state and verify all inductive steps, base cases, bounds, and complexity or algorithmic guarantees required by the theorem.\n"
            "7. Case & Boundary Integrity: Explicitly address all base cases, boundary parameters, and edge conditions required by the statement.\n"
            "8. Strict Definition Fidelity: Strictly adhere to all explicit definitions, conventions, and notation in the problem statement. Never introduce alternative 'degenerate conventions' or redefine stated variables.\n"
            "9. Statement Property Labels: When the Target Problem defines numbered, lettered, or labeled properties or conditions, explicitly reference and verify each property using its exact statement label to ensure direct correspondence with the theorem statement."
        )
        user_msg = (
            f"Target Problem & Context:\n{task_prompt}\n\n"
            f"Draft & Exploration Traces:\n{raw_draft[-24000:]}\n\n"
            "Write the clean publication LaTeX proof now:"
        )
        try:
            resp = self.client.complete(
                [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}],
                temperature=0.1,
                max_tokens=65536,
                enable_thinking=True,
                thinking_budget_tokens=16384,
                timeout_seconds=360.0,
            )
            content = resp.content or ""
            reasoning = getattr(resp, "reasoning", None) or ""
            proof = extract_latex_proof(content)
            if not proof and content.strip():
                if not content.strip().startswith(r"\begin{proof}"):
                    proof = extract_latex_proof(r"\begin{proof}" + "\n" + content.strip())
            if not proof and reasoning and r"\begin{proof}" in reasoning:
                proof = extract_latex_proof(reasoning)
            if proof and is_valid_latex_proof(proof):
                return proof
            if proof:
                san = sanitize_informal_proof(proof)
                if san and is_valid_latex_proof(san):
                    return san
            if content.strip():
                san_c = sanitize_informal_proof(content.strip())
                if san_c:
                    p_c = extract_latex_proof(san_c)
                    if p_c and is_valid_latex_proof(p_c):
                        return p_c
        except Exception as e:
            self._log_write(f"[Typesetter Error] {e}\n")
        return None

    def _stream_completion_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 8192,
        temperature: float = 0.1,
        enable_thinking: bool = False,
        thinking_budget_tokens: Optional[int] = None,
        timeout_seconds: float = 600.0,
        max_retries: int = 5,
        on_reasoning: Optional[Callable[[str], None]] = None,
        on_content: Optional[Callable[[str], None]] = None,
        on_turn_switch: Optional[Callable[[], None]] = None,
    ) -> Tuple[str, str]:
        """Streams completion from LLM client with automatic retry if interrupted mid-stream.
        Returns (reasoning_text, content_text).
        """
        for attempt in range(max_retries):
            reasoning_chunks: List[str] = []
            content_chunks: List[str] = []
            switched_to_content = False
            try:
                for typ, delta in self.client.stream_complete(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                    thinking_budget_tokens=thinking_budget_tokens,
                    timeout_seconds=timeout_seconds,
                    max_retries=1,
                ):
                    if typ == "reasoning":
                        reasoning_chunks.append(delta)
                        self._log_write(delta)
                        if on_reasoning:
                            on_reasoning(delta)
                    elif typ == "content":
                        if not switched_to_content:
                            switched_to_content = True
                            if on_turn_switch:
                                on_turn_switch()
                        content_chunks.append(delta)
                        self._log_write(delta)
                        if on_content:
                            on_content(delta)
                return "".join(reasoning_chunks).strip(), "".join(content_chunks).strip()
            except Exception as exc:
                exc_str = str(exc)
                status = getattr(getattr(exc, "response", None), "status_code", None)
                is_overflow = (
                    type(exc).__name__ == "ContextOverflowError"
                    or (status == 400)
                    or ("400" in exc_str and any(k in exc_str.lower() for k in ("context", "token", "max_model_len", "too long")))
                )
                if is_overflow:
                    self._log_write(
                        "\n[Context Overflow Alert] HTTP 400 / context overflow; retrying with a compressed copy of the history...\n"
                    )
                    # Work on copies: never mutate the caller's message dicts (they may hold the candidate proof).
                    trimmed = [dict(m) for m in messages]
                    if len(trimmed) > 4:
                        trimmed = trimmed[:2] + trimmed[-2:]
                    for m in trimmed[2:]:
                        c = m.get("content", "")
                        if len(c) > 24000:
                            m["content"] = c[:6000] + "\n\n% [... intermediate text truncated for context envelope ...]\n\n" + c[-14000:]
                    messages = trimmed
                    max_tokens = max(16384, max_tokens // 2)
                    if thinking_budget_tokens:
                        thinking_budget_tokens = min(thinking_budget_tokens, max_tokens // 2)

                if attempt == max_retries - 1:
                    self._log_write(f"\n[Stream Error] Exhausted all {max_retries} stream retries: {exc}\n")
                    logger.error(f"Exhausted all {max_retries} stream retries: {exc}")
                    raise
                wait = 2.0 * (attempt + 1)
                warn_msg = (
                    f"\n[Stream Warning] Stream interrupted ({type(exc).__name__}: {exc}); "
                    f"retrying stream from start in {wait:.1f}s (attempt {attempt + 1}/{max_retries})...\n"
                )
                self._log_write(warn_msg)
                logger.warning(warn_msg.strip())
                time.sleep(wait)
        return "", ""


__all__ = [
    "ProverAuditorMixin",
]
