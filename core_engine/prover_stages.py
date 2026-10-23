"""Multi-Agent Stage Pipelines for MathProver.

Houses specialized mathematical subagents:
- Context Architect: Catalogs contextual lemmas and generates a reduction blueprint.
- Computational Falsifier: Executes Python Monte Carlo / SymPy sanity checks on blueprint claims.
- Dynamic Critique Falsifier: Empirically stress-tests referee rejection critiques in isolated kernel.
- Adversarial Skeptic: Actively attacks intermediate claims using Z3/NetworkX/SymPy.
- Publication Scribe: Typesets publication-grade LaTeX proofs from certified lemma ledgers.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Tuple

from .memory import CoALAMemoryManager
from .prompt_loader import load_tcs_prompt
from .proof_extractor import (
    extract_latex_proof,
    is_valid_latex_proof,
    sanitize_informal_proof,
)

logger = logging.getLogger("prime_math.prover_stages")

SKEPTIC_VERIFIED = "VERIFIED"
SKEPTIC_REFUTED = "REFUTED"
SKEPTIC_INCONCLUSIVE = "INCONCLUSIVE"

# A verdict line must start with VERIFIED: / REFUTED: (allowing markdown bullets / bold / heading marks).
_SKEPTIC_VERDICT_RE = re.compile(
    r"^[ \t>*#\-]*\**\b(VERIFIED|REFUTED)\b\**\s*:\s*\**(.*)$",
    re.MULTILINE,
)

# Max characters of the Deductive Logician draft handed to the Scribe (was a 4000-char tail).
SCRIBE_DRAFT_MAX_CHARS = 20000


class ProverStagesMixin:
    """Mixin implementing specialized subagent execution routines for MathProver."""

    def _active_extension_hint(self) -> str:
        """Returns a concise summary of preloaded domain kernel helpers when one of the
        scoped extensions (Z3/SMT, Topological/Chord, Quantum/PSD) is active; else empty string.
        """
        hints: List[str] = []
        if getattr(self, "_smt_extension_active", False):
            hints.append(
                "[Z3 / SMT Kernel Helpers Preloaded]: Call `smt_disjoint_paths(...)`, "
                "`smt_slice_terminal_pairing(...)`, or `smt_check_parameter_hierarchy(...)` (and native `z3`) "
                "in Python REPL blocks to verify bipartite terminal parities, parameter hierarchies (`0 < p_1 << p_2 << ...`), and degree budgets."
            )
        if getattr(self, "_topo_extension_active", False):
            hints.append(
                "[Topological & Chord Kernel Helpers Preloaded]: Call `planar_dual_and_faces(G)` "
                "or `check_cycle_chords_and_holes(G=None, cycle_vertices=None, chords=None)` in Python REPL blocks "
                "(keep test instances small, N <= 20; never brute-force large itertools.combinations)."
            )
        if getattr(self, "_quantum_extension_active", False):
            hints.append(
                "[Quantum & PSD Kernel Helpers Preloaded]: Call `check_psd_inequality(matrix_diff_fn, dim=4)`, "
                "`check_quantum_channel_claim(predicate_fn, dim=2)`, `partial_trace(rho, dims, keep)`, "
                "or `uhlmann_fidelity(rho, sigma)` in Python REPL blocks."
            )
        return ("\n" + "\n".join(hints) + "\n") if hints else ""

    def _run_context_architect(self, task_id: str, task_prompt: str, on_step: Optional[Any] = None) -> str:
        """Context Architect Subagent:
        Operates with its own isolated CoALA memory collection 'tcs_architect'.
        Deconstructs the Target Problem and Context, catalogs contextual lemmas,
        and derives a modular reduction blueprint with testable mathematical claims.
        """
        safe_task_id = re.sub(r"\d{4}\.\d{4,5}", "[ID]", str(task_id))
        self._log_write(f"\n--- [SUBAGENT: CONTEXT ARCHITECT] Analyzing reduction structure for {safe_task_id} ---\n")
        if on_step:
            on_step("architect_start", "")
        try:
            architect_memory = CoALAMemoryManager(agent_id="tcs_architect")
            retrieved = architect_memory.query_procedural_heuristics(f"[{safe_task_id}] {task_prompt[:400]}", limit=2, min_score=0.68)
        except Exception:
            retrieved = []

        heuristics_text = ""
        if retrieved:
            heuristics_text = (
                "\n--- Proven Architectural Patterns ---\n"
                + "\n".join(f"- {h}" for h in retrieved)
                + "\n------------------------------------\n\n"
            )

        arch_system = load_tcs_prompt("context_architect")
        ext_hint = self._active_extension_hint()
        arch_user = f"{heuristics_text}Target Problem & Context:\n{task_prompt}\n{ext_hint}\nProduce the Reduction Blueprint:"

        try:
            resp = self.client.complete(
                [{"role": "system", "content": arch_system}, {"role": "user", "content": arch_user}],
                temperature=0.2,
                **self._stage_budget("architect", 8192, 2048, 180.0),
            )
            blueprint = resp.content.strip() if resp.content else ""
            # Guard against leaked scratchpad fragments when reasoning budget exhausts output tokens
            if blueprint and len(blueprint) < 250 and "###" not in blueprint and "Strategy" not in blueprint:
                self._log_write(f"[Context Architect] Discarding truncated scratchpad fragment ({len(blueprint)} chars).\n")
                blueprint = ""
            self._log_write(f"[Context Architect Output]:\n{blueprint}\n")
            if blueprint and hasattr(self, "kernel"):
                self._seed_state_from_blueprint(blueprint)
            if on_step and blueprint:
                on_step("architect_done", blueprint)
            return blueprint
        except Exception as e:
            self._log_write(f"[Context Architect] Failed or timed out: {e}\n")
            return ""

    def _seed_state_from_blueprint(self, blueprint: str) -> None:
        """Extracts safe `state.register(...)` and `state.add_subgoal(...)` calls from the
        Context Architect's blueprint and executes them to seed the MathStateMachine.
        """
        try:
            code_blocks = self.extract_code_blocks(blueprint)
            seed_lines: List[str] = []
            for block in code_blocks:
                for line in block.splitlines():
                    stripped = line.strip()
                    if stripped.startswith(("state.register(", "state.add_subgoal(")):
                        seed_lines.append(stripped)
            if seed_lines:
                seed_script = "\n".join(seed_lines[:15])
                res = self.kernel.execute(seed_script, timeout_seconds=5.0)
                if res.success:
                    self._log_write(f"[State Machine Seeded from Blueprint]: {len(seed_lines)} statements executed.\n")
        except Exception as e:
            logger.debug("Blueprint state seeding skipped: %s", e)

    @staticmethod
    def _is_assertion_failure(err_msg: str) -> bool:
        """True only when a script failure represents a mathematical AssertionError rather than
        a Python script bug (SyntaxError, NameError, TypeError, IndexError, KeyError, etc.).
        """
        if not err_msg:
            return False
        if "AssertionError" in err_msg or "Assertion failed" in err_msg:
            return True
        script_bug_markers = (
            "SyntaxError",
            "NameError",
            "TypeError",
            "IndexError",
            "KeyError",
            "AttributeError",
            "ValueError",
            "ZeroDivisionError",
            "ImportError",
            "ModuleNotFoundError",
            "UnboundLocalError",
            "IndentationError",
            "RecursionError",
            "MemoryError",
        )
        if any(marker in err_msg for marker in script_bug_markers):
            return False
        return False

    @staticmethod
    def _is_stdout_counterexample(out: str) -> bool:
        """True when a successful script's stdout explicitly reports a mathematical counterexample
        or falsified invariant (excluding 'counterexample: None', 'counterexamples: []', etc.).
        """
        if not out:
            return False
        for raw_line in out.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            low = line.lower()
            if any(neg in low for neg in ("no counterexample", "0 counterexample", "counterexample: none", "counterexamples: []", "counterexample? false")):
                continue
            if re.search(
                r"(?i)\bcounterexample(?:s|\s+found|\s+to\b[^:\n]*)?\s*[:=]\s*(?!none\b|false\b|0\b|\[\s*\]|\{\s*\}|\(\s*\))\S+",
                line,
            ):
                return True
            if re.search(r"(?i)\b(?:counterexample_found|violation\s+found|defect\s+confirmed)\b", line):
                return True
            if re.search(r"(?i)^violation\b", line):
                return True
            if re.search(r"(?i)\b(?:invariant\s+holds|claim\s+holds|holds)\s*[:=]\s*false\b", line):
                return True
        return False

    def _run_computational_falsifier(self, task_id: str, blueprint: str, on_step: Optional[Any] = None) -> Tuple[List[str], List[str]]:
        """Computational / Monte Carlo Falsifier Subagent:
        Operates with its own isolated CoALA memory collection 'tcs_verifier'.
        Takes the blueprint's intermediate mathematical claims and writes Python scripts
        (SymPy, Monte Carlo) to verify or falsify the formulas on the isolated TCSCompliantKernel.
        Returns (verified_facts, falsified_alerts).
        """
        if not blueprint or not blueprint.strip():
            return [], []

        safe_task_id = re.sub(r"\d{4}\.\d{4,5}", "[ID]", str(task_id))
        self._log_write(f"\n--- [SUBAGENT: COMPUTATIONAL FALSIFIER] Testing mathematical claims for {safe_task_id} ---\n")
        if on_step:
            on_step("falsifier_start", "")
        try:
            verifier_memory = CoALAMemoryManager(agent_id="tcs_verifier")
            retrieved = verifier_memory.query_procedural_heuristics(f"[{safe_task_id}] {blueprint[:400]}", limit=2, min_score=0.68)
        except Exception:
            retrieved = []

        heuristics_text = ""
        if retrieved:
            heuristics_text = (
                "\n--- Proven Falsification Heuristics ---\n"
                + "\n".join(f"- {h}" for h in retrieved)
                + "\n----------------------------------------\n\n"
            )

        verifier_system = load_tcs_prompt("computational_falsifier")
        smt_falsifier_hint = ""
        if getattr(self, "_smt_extension_active", False):
            smt_falsifier_hint += (
                "\n[Z3 / SMT Falsification Tools Available]:\n"
                "Preloaded in REPL: `z3`, `smt_disjoint_paths(...)`, `smt_slice_terminal_pairing(...)`, "
                "and `smt_check_parameter_hierarchy(...)`. Use them to verify parameter hierarchy ordering "
                "(`0 < p_1 << p_2 << ...`), degree budgets (for both even and odd k), exponent arithmetic, "
                "and bipartite terminal parity balance.\n"
            )
        if getattr(self, "_topo_extension_active", False):
            smt_falsifier_hint += (
                "\n[Topological & Structural Graph Tools Available]:\n"
                "Preloaded in REPL: `planar_dual_and_faces(G)` and `check_cycle_chords_and_holes(G=None, cycle_vertices=None, chords=None)`. "
                "Use them to test planar dual cut-vertex/face claims, induced holes, and crossing vs nested cycle chords ($L_1 < L_2 < R_1 < R_2$ crosses!).\n"
            )
        if getattr(self, "_quantum_extension_active", False):
            smt_falsifier_hint += (
                "\n[Quantum & PSD Operator Tools Available]:\n"
                "Preloaded in REPL: `check_psd_inequality(matrix_diff_fn, dim=4)`, `check_quantum_channel_claim(predicate_fn, dim=2)`, "
                "`random_density_matrix(dim)`, `random_unitary(dim)`, `partial_trace(rho, dims, keep)`, and `uhlmann_fidelity(rho, sigma)`. "
                "Use them to falsify invalid channel decompositions (e.g. on dephasing/amplitude-damping channels) or false PSD inequalities.\n"
            )
        verifier_user = (
            f"{heuristics_text}Reduction Blueprint:\n{blueprint}\n{smt_falsifier_hint}\n"
            "Write Python code in ```python ... ``` blocks to verify or falsify these claims:"
        )

        verified_facts: List[str] = []
        falsified_alerts: List[str] = []

        try:
            resp = self.client.complete(
                [{"role": "system", "content": verifier_system}, {"role": "user", "content": verifier_user}],
                temperature=0.1,
                **self._stage_budget("falsifier", 4096, 1536, 120.0),
            )
            code_blocks = self.extract_code_blocks(resp.content or "")
            for idx, code in enumerate(code_blocks[:2]):
                self._log_write(f"[Falsifier Script {idx+1}]:\n{code}\n")
                exec_res = self.kernel.execute(code, timeout_seconds=self._stage_kernel_timeout())
                out = (exec_res.stdout or exec_res.result_repr or "").strip()
                if exec_res.success:
                    if out:
                        if self._is_stdout_counterexample(out):
                            self._log_write(f"[Falsifier Kernel ALERT (Contradiction Found in Output)]: {out[:400]}\n")
                            falsified_alerts.append(f"REPL Falsification Warning: {out[:300]}. Check conditioning, indices, or inequality direction!")
                        else:
                            self._log_write(f"[Falsifier Kernel Output (Verified)]: {out[:500]}\n")
                            verified_facts.append(f"REPL Verified: {out[:300]}")
                else:
                    err_msg = exec_res.error or "Assertion failed"
                    if self._is_assertion_failure(err_msg):
                        self._log_write(f"[Falsifier Kernel ALERT (Contradiction Found)]: {err_msg[:400]}\n")
                        falsified_alerts.append(f"REPL Falsification Warning: Assertion failed with: {err_msg[:300]}` (stdout: {out[:200]}). Check conditioning, indices, or inequality direction!")
                    else:
                        self._log_write(f"[Falsifier Kernel Script Error (Inconclusive — Ignored)]: {err_msg[:250]}\n")
        except Exception as e:
            self._log_write(f"[Computational Falsifier] Exception: {e}\n")

        if on_step and (verified_facts or falsified_alerts):
            on_step("falsifier_done", {"verified": verified_facts, "falsified": falsified_alerts})

        return verified_facts, falsified_alerts

    def _run_critique_falsifier(
        self,
        task_id: str,
        issues: str,
        candidate_proof: str,
        on_step: Optional[Any] = None,
    ) -> Tuple[List[str], List[str]]:
        """Dynamic Critique Falsifier Subagent:
        Takes the referee's rejection critique and candidate proof, and writes a targeted
        Python verification script (NetworkX for graph/subgraph claims, SymPy for algebraic/spectral
        inequalities, or numerical sampling for probabilistic dependencies) to stress-test the referee's
        objection in the isolated TCSCompliantKernel.
        Returns (verified_facts, falsified_alerts).
        """
        if not issues or not issues.strip():
            return [], []

        safe_task_id = re.sub(r"\d{4}\.\d{4,5}", "[ID]", str(task_id))
        self._log_write(f"\n--- [SUBAGENT: DYNAMIC CRITIQUE FALSIFIER] Stress-testing referee objections for {safe_task_id} ---\n")
        if on_step:
            on_step("critique_falsifier_start", "")

        verifier_system = load_tcs_prompt("critique_falsifier")
        proof_context = candidate_proof if len(candidate_proof) <= 16000 else candidate_proof[:6000] + "\n\n% [... intermediate proof steps ...]\n\n" + candidate_proof[-10000:]
        verifier_user = (
            f"Referee Rejection Critique:\n{issues}\n\n"
            f"Candidate Proof Under Review:\n{proof_context}\n\n"
            "CRITICAL: Do NOT test whether the overall Target Theorem holds (the Target Theorem is already known to be true!). "
            "Write a Python script to test ONLY the specific disputed intermediate claim, decomposition, or inequality "
            "in the Candidate Proof that the referee objected to:"
        )

        verified_facts: List[str] = []
        falsified_alerts: List[str] = []

        try:
            resp = self.client.complete(
                [{"role": "system", "content": verifier_system}, {"role": "user", "content": verifier_user}],
                temperature=0.1,
                **self._stage_budget("skeptic", 2048, 1024, 90.0),
            )
            code_blocks = self.extract_code_blocks(resp.content or "")
            for idx, code in enumerate(code_blocks[:1]):
                self._log_write(f"[Critique Falsifier Script]:\n{code}\n")
                exec_res = self.kernel.execute(code, timeout_seconds=self._stage_kernel_timeout())
                out = (exec_res.stdout or exec_res.result_repr or "").strip()
                if exec_res.success:
                    if out:
                        if self._is_stdout_counterexample(out):
                            self._log_write(f"[Critique Kernel ALERT (Counterexample/Defect Confirmed in Output)]: {out[:400]}\n")
                            falsified_alerts.append(
                                f"REPL Counterexample / Defect Confirmed: {out[:300]}. "
                                "The referee's objection is validated by code; you must alter this deduction or pivot."
                            )
                        else:
                            self._log_write(f"[Critique Kernel Output (Verified)]: {out[:500]}\n")
                            verified_facts.append(f"REPL Verified Disputed Step: {out[:300]}")
                else:
                    err_msg = exec_res.error or "Assertion failed"
                    if "timed out" in err_msg.lower() or "timeoutexception" in err_msg.lower():
                        self._log_write(f"[Critique Kernel TIMEOUT]: Script execution timed out (inconclusive, no counterexample found).\n")
                    elif self._is_assertion_failure(err_msg):
                        detail = f"{err_msg[:300]}" + (f" | stdout: {out[:200]}" if out else "")
                        self._log_write(f"[Critique Kernel ALERT (Counterexample/Defect Confirmed)]: {detail[:400]}\n")
                        falsified_alerts.append(
                            f"REPL Counterexample / Defect Confirmed: {detail}. "
                            "The referee's objection is validated by code; you must alter this deduction or pivot."
                        )
                    else:
                        self._log_write(f"[Critique Kernel Script Error (Inconclusive — Ignored)]: {err_msg[:250]}\n")
        except Exception as e:
            self._log_write(f"[Critique Falsifier] Exception: {e}\n")

        if on_step and (verified_facts or falsified_alerts):
            on_step("critique_falsifier_done", {"verified": verified_facts, "falsified": falsified_alerts})

        return verified_facts, falsified_alerts

    def _run_adversarial_skeptic(
        self,
        task_id: str,
        task_prompt: str,
        claim: str,
        state_summary: str,
        on_step: Optional[Any] = None,
    ) -> Tuple[str, str, Optional[str]]:
        """Adversarial Skeptic Subagent:
        Operates with its own isolated CoALA memory collection 'tcs_skeptic' and a completely
        independent context memory (zero context leakage with Deductive Logician or Scribe).

        Takes a proposed intermediate mathematical claim, reduction, or topological assertion
        and actively attacks it using Python / NetworkX / SymPy / Z3 in the isolated kernel
        to detect non-tree chords, dependent conditioning, parameter bounds violations, or boundary collapses.

        Returns (status, verdict_explanation, counterexample_code) where status is one of
        SKEPTIC_VERIFIED, SKEPTIC_REFUTED or SKEPTIC_INCONCLUSIVE.  The skeptic is fail-closed:
        crashes, timeouts, unparseable output or a missing verdict yield INCONCLUSIVE, never VERIFIED.
        VERIFIED only means "no counterexample found on small instances", not a proof.
        """
        if not claim or not claim.strip():
            return SKEPTIC_INCONCLUSIVE, "No claim to attack.", None

        safe_task_id = re.sub(r"\d{4}\.\d{4,5}", "[ID]", str(task_id))
        self._log_write(f"\n--- [SUBAGENT: ADVERSARIAL SKEPTIC] Attacking proposed claim for {safe_task_id} ---\nClaim: {claim[:250]}...\n")
        if on_step:
            on_step("skeptic_start", claim[:200])

        try:
            skeptic_memory = CoALAMemoryManager(agent_id="tcs_skeptic")
            retrieved = skeptic_memory.query_procedural_heuristics(f"[{safe_task_id}] {claim[:400]}", limit=2, min_score=0.68)
        except Exception:
            retrieved = []

        heuristics_text = ""
        if retrieved:
            heuristics_text = "\n".join(f"- {h}" for h in retrieved) + "\n\n"

        skeptic_system = load_tcs_prompt("adversarial_skeptic")
        ext_hint = self._active_extension_hint()

        skeptic_user = (
            f"{heuristics_text}"
            f"Target Problem & Context Definitions:\n{task_prompt}\n\n"
            f"Active Mathematical State & Invariants:\n{state_summary[:4000]}\n\n"
            f"Proposed Mathematical Claim to Attack:\n{claim}\n"
            f"{ext_hint}\n"
            "Aggressively attack this claim with Python in ```python ... ``` now:"
        )

        def _finish(status: str, reason: str, code: Optional[str]) -> Tuple[str, str, Optional[str]]:
            self._log_write(f"[Adversarial Skeptic Result]: {status}: {reason[:300]}\n")
            if on_step:
                on_step(
                    "skeptic_done",
                    {"status": status, "verified": status == SKEPTIC_VERIFIED, "reason": reason[:200]},
                )
            return status, reason, code

        try:
            resp = self.client.complete(
                [{"role": "system", "content": skeptic_system}, {"role": "user", "content": skeptic_user}],
                temperature=0.1,
                **self._stage_budget("skeptic", 2048, 1024, 90.0),
            )
            content = resp.content or ""
            code_blocks = self.extract_code_blocks(content)
            executed_code = None
            script_error: Optional[str] = None
            out = ""

            for code in code_blocks[:1]:
                executed_code = code
                self._log_write(f"[Skeptic Attack Script]:\n{code}\n")
                exec_res = self.kernel.execute(code, timeout_seconds=self._stage_kernel_timeout())
                out = (exec_res.stdout or exec_res.result_repr or "").strip()
                if not exec_res.success:
                    err_msg = (exec_res.error or "").strip()
                    if "AssertionError" in err_msg or "Counterexample" in err_msg or "StateInvariantViolation" in err_msg:
                        self._log_write(f"[Skeptic Kernel ALERT: Assertion Failed / Counterexample Triggered]: {err_msg[:400]}\n")
                        return _finish(
                            SKEPTIC_REFUTED,
                            f"Script assertion failed: {err_msg[:300]}. The proposed claim does not hold on a small instance "
                            "(or the attack script itself encoded the claim incorrectly; double-check before abandoning).",
                            code,
                        )
                    # Script crashed for a non-mathematical reason (NameError, timeout, ImportError...).
                    script_error = err_msg[:300] or "unknown script error"
                    self._log_write(f"[Skeptic Kernel: script error, attack inconclusive]: {err_msg[:400]}\n")
                elif "counterexample found" in out.lower() or self._is_stdout_counterexample(out):
                    self._log_write(f"[Skeptic Kernel ALERT: Counterexample Discovered]: {out[:400]}\n")
                    return _finish(SKEPTIC_REFUTED, f"Counterexample discovered: {out[:300]}", code)
                else:
                    self._log_write(f"[Skeptic Kernel Output]: {out[:300]}\n")

            if script_error is not None:
                return _finish(SKEPTIC_INCONCLUSIVE, f"Attack script crashed ({script_error}); claim not checked.", executed_code)

            verdict, reason = self._parse_skeptic_verdict(content)
            if verdict == SKEPTIC_REFUTED:
                # Fail-closed against single-turn hallucinated text refutations:
                # Because the LLM writes its text verdict in the same completion before code executes,
                # a REFUTED text verdict without an assertion failure or stdout counterexample/refutation
                # is unverified speculation and must be downgraded to INCONCLUSIVE.
                out_low = out.lower()
                if (
                    executed_code is not None
                    and out
                    and any(kw in out_low for kw in ("counterexample", "refuted", "violat", "false", "fail", "mismatch", "invalid"))
                    and "no counterexample" not in out_low
                    and "0 counterexample" not in out_low
                ):
                    return _finish(SKEPTIC_REFUTED, f"{reason or 'Skeptic refuted the claim.'} (stdout: {out[:200]})", executed_code)
                self._log_write(
                    f"[Skeptic Kernel: unverified text REFUTED downgraded to INCONCLUSIVE (no empirical counterexample in stdout)]: {reason[:200]}\n"
                )
                return _finish(
                    SKEPTIC_INCONCLUSIVE,
                    f"Skeptic expressed textual doubt ({reason[:200]}), but attack script produced no concrete counterexample.",
                    executed_code,
                )
            if verdict == SKEPTIC_VERIFIED and script_error is None:
                return _finish(SKEPTIC_VERIFIED, reason or "No counterexample found on small instances.", executed_code)
            return _finish(SKEPTIC_INCONCLUSIVE, "Skeptic produced no explicit VERIFIED/REFUTED verdict.", executed_code)

        except Exception as e:
            self._log_write(f"[Adversarial Skeptic] Exception: {e}\n")
            return _finish(SKEPTIC_INCONCLUSIVE, f"Skeptic skipped due to timeout/exception: {e}", None)

    @staticmethod
    def _parse_skeptic_verdict(content: str) -> Tuple[Optional[str], str]:
        """Parse the skeptic's final verdict line.

        Only lines that *start* with ``VERIFIED:`` or ``REFUTED:`` (optionally after markdown
        emphasis / list markers) count, and the last such line wins.  Word boundaries make
        sure ``UNVERIFIED`` never matches ``VERIFIED``.
        """
        last: Optional[Tuple[str, str]] = None
        for m in _SKEPTIC_VERDICT_RE.finditer(content or ""):
            last = (m.group(1).upper(), m.group(2).strip())
        if last is None:
            return None, ""
        status = SKEPTIC_VERIFIED if last[0] == "VERIFIED" else SKEPTIC_REFUTED
        return status, last[1]

    def _stage_budget(self, stage: str, max_tokens: int, thinking: int, timeout: float) -> dict:
        """Read per-subagent generation budgets from TCSConfig (falls back to the given defaults)."""
        cfg = getattr(self, "config", None)
        return {
            "max_tokens": int(getattr(cfg, f"{stage}_max_tokens", max_tokens) or max_tokens),
            "enable_thinking": True,
            "thinking_budget_tokens": int(getattr(cfg, f"{stage}_thinking_budget", thinking) or thinking),
            "timeout_seconds": float(getattr(cfg, f"{stage}_timeout_seconds", timeout) or timeout),
        }

    def _stage_kernel_timeout(self, default: float = 15.0) -> float:
        cfg = getattr(self, "config", None)
        return float(getattr(cfg, "auditor_kernel_timeout_seconds", default) or default)

    def _run_publication_scribe(
        self,
        task_id: str,
        task_prompt: str,
        verified_ledger: List[str],
        state_summary: str,
        candidate_draft: Optional[str] = None,
        on_step: Optional[Any] = None,
    ) -> Optional[str]:
        """Publication Scribe Subagent:
        Operates with its own isolated CoALA memory collection 'tcs_scribe' and completely
        independent context memory (zero context leakage with Deductive Logician or Skeptic).

        Translates the sequence of verified lemmas, state invariants, and reduction steps
        into a publication-grade LaTeX proof enclosed strictly in \\begin{proof} ... \\end{proof}.
        The Scribe's attention heads are 100% focused on declarative journal prose, notation
        fidelity, and theorem clause traceability.
        """
        safe_task_id = re.sub(r"\d{4}\.\d{4,5}", "[ID]", str(task_id))
        self._log_write(f"\n--- [SUBAGENT: PUBLICATION SCRIBE] Typesetting publication LaTeX proof for {safe_task_id} ---\n")
        if on_step:
            on_step("scribe_start", "")

        try:
            scribe_memory = CoALAMemoryManager(agent_id="tcs_scribe")
            retrieved = scribe_memory.query_procedural_heuristics(f"[{safe_task_id}] {task_prompt[:400]}", limit=2, min_score=0.68)
        except Exception:
            retrieved = []

        heuristics_text = ""
        if retrieved:
            heuristics_text = (
                "--- Proven Publication Patterns ---\n"
                + "\n".join(f"- {h}" for h in retrieved)
                + "\n-----------------------------------\n\n"
            )

        scribe_system = load_tcs_prompt("publication_scribe")

        ledger_text = ""
        if verified_ledger:
            ledger_text = (
                "--- Intermediate Claim Ledger (stress-tested by the Skeptic on small instances; no counterexample found) ---\n"
                "NOTE: these claims are NOT proved. Each one you use must still be rigorously justified in the proof.\n"
                + "\n".join(f"- {lemma}" for lemma in verified_ledger)
                + "\n--------------------------------------------------------------------------\n\n"
            )

        notes_text = ""
        if candidate_draft and len(candidate_draft.strip()) > 100:
            draft = candidate_draft.strip()
            if len(draft) > SCRIBE_DRAFT_MAX_CHARS:
                head = SCRIBE_DRAFT_MAX_CHARS // 4
                draft = draft[:head] + "\n\n% [... middle of draft omitted for length ...]\n\n" + draft[-(SCRIBE_DRAFT_MAX_CHARS - head):]
            notes_text = f"--- Deductive Logician Exploration & Draft Notes ---\n{draft}\n----------------------------------------------------\n\n"

        state_text = ""
        if state_summary and len(state_summary.strip()) > 20:
            state_text = f"--- Active State Invariants ---\n{state_summary[:4000]}\n------------------------------\n\n"

        scribe_user = (
            f"{heuristics_text}"
            f"Target Problem & Context:\n{task_prompt}\n\n"
            f"{ledger_text}"
            f"{state_text}"
            f"{notes_text}"
            "CRITICAL CONTRACT:\n"
            "1. Do NOT abbreviate or omit proofs of any claims you include (e.g. NEVER write '(as above)', 'proof omitted', or leave claims without proof). Every stated claim MUST have its complete formal deduction written out from first principles or context lemmas.\n"
            "2. CRITICAL-PATH PRUNING: Trace backwards from the final theorem conclusion and include ONLY the minimal chain of intermediate lemmas actually used. Omit unused exploratory side-lemmas or superseded bounds.\n"
            "3. FULL DOMAIN COVERAGE: Establish the Target Problem over its entire universally quantified domain D. If an intermediate bound only applies to a subset D_1, explicitly prove the complementary regime D \\ D_1 as well; never restrict the theorem statement to a sub-regime.\n\n"
            "Typeset the complete, publication-grade LaTeX proof now enclosed strictly in \\begin{proof} ... \\end{proof}:"
        )

        try:
            resp = self.client.complete(
                [{"role": "system", "content": scribe_system}, {"role": "user", "content": scribe_user}],
                temperature=0.15,
                **self._stage_budget("scribe", 65536, 16384, 360.0),
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
                self._log_write(f"[Publication Scribe Success] Generated valid proof ({len(proof)} chars).\n")
                if on_step:
                    on_step("scribe_done", {"proof": proof, "valid": True})
                return proof

            if proof:
                san = sanitize_informal_proof(proof)
                if san and is_valid_latex_proof(san):
                    self._log_write(f"[Publication Scribe Sanitized] Produced valid proof ({len(san)} chars).\n")
                    if on_step:
                        on_step("scribe_done", {"proof": san, "valid": True})
                    return san

            typeset_src = f"{reasoning[-25000:]}\n\n{content}".strip() if reasoning else content
            typeset_cand = self._typeset_publication_proof(task_prompt, typeset_src)
            if typeset_cand:
                self._log_write(f"[Publication Scribe Typeset Pass] Produced valid proof ({len(typeset_cand)} chars).\n")
                if on_step:
                    on_step("scribe_done", {"proof": typeset_cand, "valid": True})
                return typeset_cand

            if on_step:
                on_step("scribe_done", {"proof": None, "valid": False})
            return None
        except Exception as e:
            self._log_write(f"[Publication Scribe Exception]: {e}\n")
            if on_step:
                on_step("scribe_done", {"proof": None, "valid": False})
            return None


__all__ = [
    "ProverStagesMixin",
    "SKEPTIC_VERIFIED",
    "SKEPTIC_REFUTED",
    "SKEPTIC_INCONCLUSIVE",
]
