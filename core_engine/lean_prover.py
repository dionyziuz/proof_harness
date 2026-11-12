"""lean_prover.py — Autonomous Lean 4 Formal Theorem Prover & Kernel Verification Engine.

Sits alongside `core_engine/prover.py` (`MathProver`, which generates and audits publication-grade
LaTeX proofs) as Prime Math's dedicated formal theorem prover for machine-certified Lean 4 (`.lean`)
artifacts.

Architecture:
1. Stage-0 Problem-Formulation & Axiom Purity Auditor:
   Verifies zero circular definitions, zero pre-baked conclusions, and ensures the target
   can be proved in Lean 4 with ZERO custom axioms.
2. Preloaded Lean 4 Interactive Kernel (`LeanKernel` / `LeanCompliantKernel`):
   Injects `verify_lean4_kernel`, `run_lean`, `judge_lean_proof`, and `audit_lean_semantic_depth`
   directly into the persistent Python REPL alongside `sympy` and `mpmath`.
3. Deterministic 5-Gate Lean 4 Compiler & Kernel Axiom Judge (`verify_lean4_kernel`):
   Replaces subjective LaTeX refereeing with 5 hard compiler gates:
   - Gate 1: Exact Lean 4 compilation (`returncode == 0`, 0 syntax/type errors, 0 unsolved goals).
   - Gate 2: Zero `sorry` / `admit` shortcuts and zero compiler warnings.
   - Gate 3: Automated `#print axioms <thm>` inspection across every theorem in the file,
     enforcing that ONLY standard Lean 4 core axioms (`propext`, `Classical.choice`, `Quot.sound`)
     appear. Any custom `axiom`, `constant`, `sorryAx`, or `unsafe` triggers immediate rejection.
   - Gate 4: Semantic depth & anti-tautology gate (`audit_lean_semantic_depth`).
   - Gate 5: Target theorem signature completeness check (`required_theorems`).
4. Closed-Loop Compiler Repair & Certified `.lean` Artifact Synthesis:
   Feeds exact Lean 4 compiler line numbers and goal states back into the reasoning loop until
   all 5 gates pass (`100% CERTIFIED HIT`).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .judging.lean_judge import (
    ALLOWED_CORE_AXIOMS,
    LeanKernelValidationResult,
    compute_lean_audit_scores,
    extract_lean_code,
    extract_lean_declarations,
    format_lean_progress_bar,
    judge_lean_proof,
    run_lean_with_optional_lake,
    strip_lean_comments,
    verify_lean4_kernel,
)
from .lean_runner import (
    LeanRunResult,
    audit_lean_semantic_depth,
    find_lean_binary,
    get_lean_version,
    run_lean,
)
from .prover import MathProver
from .prover_kernel import TCSCompliantKernel
from .prover_types import AuditResult, ProofResult, TCSSolveResult

logger = logging.getLogger("prime_math.lean_prover")

LEAN_PROVER_SYSTEM_PROMPT = """
================================================================================
LEAN 4 FORMAL THEOREM PROVER — ZERO-AXIOM MACHINE VERIFICATION ENGINE
================================================================================
You are the Autonomous Lean 4 Formal Theorem Prover for Prime Math.
Instead of producing subjective LaTeX prose, your objective is to produce a 100%
machine-certified Lean 4 formal proof file (`.lean`) that compiles cleanly with
ZERO errors, ZERO warnings, ZERO `sorry` shortcuts, and ZERO custom axioms.

CRITICAL LEAN 4 FORMAL VERIFICATION RULES:
1. Interactive REPL Testing First (`verify_lean4_kernel`):
   - Your Python REPL kernel has `verify_lean4_kernel(lean_code, required_theorems=...)`
     and `run_lean(lean_code)` preloaded!
   - On EVERY turn, test your candidate Lean 4 code inside a ```python ... ``` block using:
     ```python
     lean_src = r'''
     import Std
     ...
     '''
     res = verify_lean4_kernel(lean_src)
     print("PASSED:", res.passed, "STATUS:", res.status)
     print("THEOREMS:", res.theorems_found)
     print("AXIOMS:", res.axiom_report)
     if not res.passed:
         print("ISSUES:", res.issues_summary)
         print("COMPILER OUTPUT:\\n", res.compiler_output)
     ```
   - Fix every compiler error, type mismatch, or unused variable warning inside the REPL
     BEFORE outputting your final candidate proof!

2. Lean 4 Environment & Tactics (`import Std`):
   - Always begin your Lean file with `import Std` (unless a specific library import is instructed).
   - Use standard kernel-verified types (`Nat`, `Int`, `Bool`, `List`, `Fin`, `structure`, `def`)
     and tactics (`decide`, `omega`, `rfl`, `simp`, `simp only`, `dsimp`, `rw`, `exact`,
     `apply`, `induction`, `cases`, `rcases`, `obtain`, `constructor`, `intro`, `have`, `split`).
   - NEVER use `sorry`, `admit`, `axiom`, `constant`, `unsafe`, or `extern`.
   - Every theorem must pass `#print axioms <thm>` using ONLY standard Lean 4 core axioms
     (`propext`, `Classical.choice`, `Quot.sound`, or no axioms).
   - Avoid unused variables in theorem signatures (or prefix intentionally unused hypothesis names
     with an underscore `_h`, e.g., `(_hS : S > 0)`), because Lean compiler warnings are treated
     as failures!

3. How to Output Your Final Candidate Proof:
   - Once `verify_lean4_kernel(lean_src).passed` is `True` in the Python REPL, output your complete,
     self-contained Lean 4 source file inside a ```lean ... ``` code block!
   - Passing the 5-Gate Lean 4 compiler + `#print axioms` audit is a 100% deterministic Certified Hit!
"""

LEAN_STAGE0_AUDITOR_PROMPT = """You are the Stage-0 Formal Problem-Formulation & Axiom Purity Auditor.
Before the Lean 4 Formal Prover executes, audit the PROBLEM STATEMENT and CONTEXT to verify:
1. No Made-Up Circular Definitions (`no_made_up_definitions`):
   - Definitions and structures are mathematically sound and standard.
2. Zero Custom Axioms Required (`zero_custom_axioms_required`):
   - All target theorems are either unconditional mathematical truths OR explicitly formulated
     as pure implication theorems `(h : Hypothesis) -> Conclusion` that require zero global `axiom` declarations.
3. Conclusion Not Pre-Baked (`conclusion_not_pre_baked`):
   - The problem requires genuine formal definitions and non-trivial proofs (not `0 = 0`).
4. Well-Posed Lean 4 Specification (`lean_formalizable_specification`):
   - Target theorem names, types, and mathematical claims are well-specified for Lean 4 formalization.

Output ONLY a single valid JSON object:
{
  "no_made_up_definitions": true/false,
  "zero_custom_axioms_required": true/false,
  "conclusion_not_pre_baked": true/false,
  "lean_formalizable_specification": true/false,
  "verdict": "VALID" or "REJECT",
  "issues": "none" or detailed explanation
}
"""

LEAN_SYNTHESIS_SCRIBE_PROMPT = """You are the Lean 4 Formal Verification Scribe.
Given the Target Problem and the exploration/REPL traces, output ONLY a complete, self-contained,
compilable Lean 4 source file enclosed in a single ```lean ... ``` code block.

STRICT RULES:
1. Start with `import Std` at the top of the Lean 4 file (unless another import is specified).
2. Include all required `def` / `structure` declarations and prove all required `theorem` declarations.
3. NEVER use `sorry`, `admit`, `axiom`, `constant`, or `unsafe`.
4. Ensure zero unused variable warnings (prefix unused hypothesis identifiers with `_`, e.g. `_h`).
5. Output ONLY the ```lean ... ``` code block.
"""


class LeanCompliantKernel(TCSCompliantKernel):
    """Persistent Python REPL Kernel preloaded with `verify_lean4_kernel`, `run_lean`,
    `judge_lean_proof`, `sympy`, and `mpmath` so `LeanProver` can compile and debug Lean 4
    code interactively on every turn.
    """

    def __init__(
        self,
        *args: Any,
        required_theorems: Optional[List[str]] = None,
        lake_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        self.required_theorems: List[str] = list(required_theorems or [])
        self.lake_dir: Optional[Path] = lake_dir
        self.last_verified_lean_result: Optional[LeanKernelValidationResult] = None
        super().__init__(*args, **kwargs)
        self._inject_lean_tools()

    def reset(self) -> None:
        super().reset()
        self._inject_lean_tools()

    def _inject_lean_tools(self) -> None:
        def _repl_verify_lean4_kernel(
            code: str,
            required_theorems: Optional[List[str]] = None,
            timeout: float = 90.0,
        ) -> LeanKernelValidationResult:
            reqs = required_theorems if required_theorems is not None else self.required_theorems
            res = verify_lean4_kernel(
                code,
                required_theorems=reqs,
                timeout=timeout,
                lake_dir=self.lake_dir,
            )
            if res.passed:
                self.last_verified_lean_result = res
            return res

        tools = {
            "run_lean": run_lean,
            "judge_lean_proof": judge_lean_proof,
            "audit_lean_semantic_depth": audit_lean_semantic_depth,
            "verify_lean4_kernel": _repl_verify_lean4_kernel,
            "extract_lean_declarations": extract_lean_declarations,
            "ALLOWED_CORE_AXIOMS": ALLOWED_CORE_AXIOMS,
            "REQUIRED_THEOREMS": list(self.required_theorems),
        }
        for name, obj in tools.items():
            self.namespace[name] = obj


# Concise alias
LeanKernel = LeanCompliantKernel


class LeanProver(MathProver):
    """Autonomous Lean 4 Formal Theorem Prover and Kernel Verification Engine.

    Replaces subjective LaTeX refereeing and LaTeX typesetting with deterministic
    Lean 4 compilation, `#print axioms` kernel verification, and closed-loop compiler repair.
    Passing the 5-Gate Lean 4 engine check (`verify_lean4_kernel`) yields a 100% certified hit.
    """

    def __init__(
        self,
        *args: Any,
        required_theorems: Optional[List[str]] = None,
        lake_dir: Optional[Path] = None,
        max_compiler_repair_attempts: int = 4,
        **kwargs: Any,
    ) -> None:
        self.required_theorems: List[str] = list(required_theorems or [])
        self.lake_dir: Optional[Path] = lake_dir
        self.max_compiler_repair_attempts: int = max_compiler_repair_attempts
        self.last_problem_audit_result: Optional[Dict[str, Any]] = None
        self.last_lean_validation: Optional[LeanKernelValidationResult] = None

        if "kernel" not in kwargs or kwargs["kernel"] is None:
            kwargs["kernel"] = LeanCompliantKernel(
                required_theorems=self.required_theorems,
                lake_dir=self.lake_dir,
            )
        elif isinstance(kwargs["kernel"], LeanCompliantKernel):
            kwargs["kernel"].required_theorems = list(self.required_theorems)
            kwargs["kernel"].lake_dir = self.lake_dir

        super().__init__(*args, **kwargs)

    def _get_base_prover_system_prompt(self) -> str:
        return LEAN_PROVER_SYSTEM_PROMPT

    def audit_problem_statement(self, task_prompt: str) -> Dict[str, Any]:
        """Stage-0 Formal Problem-Formulation & Axiom Purity Auditor."""
        self._log_write(
            "[Stage-0 Lean Problem-Statement Auditor]: Auditing target specification & axiom purity...\n"
        )
        try:
            resp = self.client.complete(
                [
                    {"role": "system", "content": LEAN_STAGE0_AUDITOR_PROMPT},
                    {
                        "role": "user",
                        "content": f"Audit this Lean 4 Task Specification:\n\n{task_prompt}",
                    },
                ],
                temperature=0.1,
                max_tokens=4096,
                enable_thinking=True,
                thinking_budget_tokens=2048,
                timeout_seconds=180.0,
            )
            content = (resp.content or "").strip()
            self._log_write(f"\n[Stage-0 Lean Problem Auditor Output]:\n{content}\n")
            parsed = self._parse_audit_json(content) or {}
            gates = (
                "no_made_up_definitions",
                "zero_custom_axioms_required",
                "conclusion_not_pre_baked",
                "lean_formalizable_specification",
            )
            failed = [g for g in gates if parsed.get(g) is False]
            verdict = (
                "REJECT"
                if (failed or str(parsed.get("verdict", "")).upper() == "REJECT")
                else "VALID"
            )
            parsed["verdict"] = verdict
            if failed and parsed.get("issues") in (None, "", "none"):
                parsed["issues"] = f"Failed Stage-0 formal specification gates: {failed}"
            self.last_problem_audit_result = parsed
            return parsed
        except Exception as exc:
            self._log_write(f"[Stage-0 Lean Problem Auditor Warning]: {exc}\n")
            res = {"verdict": "VALID", "issues": f"Stage-0 LLM audit skipped: {exc}"}
            self.last_problem_audit_result = res
            return res

    def validate_lean_candidate(self, candidate_text: str) -> LeanKernelValidationResult:
        """Runs the deterministic 5-gate Lean 4 compiler & `#print axioms` validator."""
        val_res = verify_lean4_kernel(
            candidate_text,
            required_theorems=self.required_theorems,
            lake_dir=self.lake_dir,
            require_imports_or_structures=True,
        )
        if not val_res.passed and isinstance(self.kernel, LeanCompliantKernel):
            k_res = self.kernel.last_verified_lean_result
            if k_res is not None and k_res.passed:
                self._log_write(
                    "[LeanProver Kernel Recovery]: Recovered 100% certified Lean 4 proof directly from REPL kernel state.\n"
                )
                val_res = k_res

        self.last_lean_validation = val_res
        return val_res

    def _audit_proof_draft(self, task_prompt: str, proof_text: str) -> AuditResult:
        """Overrides subjective LaTeX LLM refereeing with 100% deterministic Lean 4 compiler validation
        and continuous multi-dimensional progress scoring.
        """
        val_res = self.validate_lean_candidate(proof_text)
        bar = format_lean_progress_bar(val_res.validity_score)
        self._log_write(
            f"\n[Deterministic Lean 4 Kernel Judge]: status={val_res.status}, passed={val_res.passed}, "
            f"theorems={val_res.theorems_found}, verified={val_res.verified_theorems}, "
            f"time={val_res.execution_time:.2f}s\n"
            f"[Audit Scores]: Validity={bar} | CompilerHealth={val_res.compiler_health_score:.2f} | "
            f"Rigor={val_res.rigor_score:.2f} | Completeness={val_res.completeness_score:.2f} | "
            f"BoundaryCases={val_res.boundary_cases_score:.2f}\n"
        )
        if val_res.axiom_report:
            self._log_write(f"[Kernel #print axioms Audit]: {json.dumps(val_res.axiom_report)}\n")

        if val_res.passed:
            self._log_write("[Deterministic Lean 4 Kernel Judge]: 100% CERTIFIED HIT!\n")
            return AuditResult(
                verdict="VALID",
                issues="none",
                validity=1.0,
                rigor=1.0,
                completeness=1.0,
                boundary_cases=1.0,
            )

        self._log_write(f"[Deterministic Lean 4 Kernel Judge REJECT]: {val_res.issues_summary}\n")
        detailed_feedback = (
            f"[LEAN 4 COMPILER & AXIOM AUDIT FAILED ({val_res.status})]:\n"
            f"• Validity Progress: {bar} (CompilerHealth={val_res.compiler_health_score:.2f}, "
            f"Rigor={val_res.rigor_score:.2f}, Completeness={val_res.completeness_score:.2f})\n"
            f"• Verified Theorems So Far: {val_res.verified_theorems}\n"
            f"• Issues: {val_res.issues_summary}\n"
            f"• Compiler Output:\n{val_res.compiler_output[:2500]}\n"
            f"Action Required: Fix the exact Lean 4 compiler errors/warnings in `verify_lean4_kernel` "
            f"inside the Python REPL until `res.passed == True`."
        )
        return AuditResult(
            verdict="REJECT",
            issues=detailed_feedback,
            validity=val_res.validity_score,
            rigor=val_res.rigor_score,
            completeness=val_res.completeness_score,
            boundary_cases=val_res.boundary_cases_score,
        )

    def _repair_lean_code_loop(
        self,
        task_prompt: str,
        initial_code: str,
        val_res: LeanKernelValidationResult,
    ) -> LeanKernelValidationResult:
        """Closed-loop Lean 4 compiler repair: sends exact Lean 4 compiler errors and line numbers
        to the thinking model to repair syntax/tactic errors until `verify_lean4_kernel` passes.
        """
        current_code = initial_code
        current_val = val_res
        best_repair_val = val_res

        for attempt in range(1, self.max_compiler_repair_attempts + 1):
            if current_val.passed:
                return current_val

            bar = format_lean_progress_bar(current_val.validity_score)
            self._log_write(
                f"\n[Lean 4 Compiler Repair Loop — Attempt {attempt}/{self.max_compiler_repair_attempts}]: "
                f"status={current_val.status}, errors={len(current_val.compiler_errors)}, "
                f"warnings={len(current_val.compiler_warnings)}, validity={bar}\n"
            )
            numbered_code = "\n".join(
                f"{idx + 1:4d}: {line}" for idx, line in enumerate(current_code.splitlines())
            )
            repair_user = (
                f"--- TARGET SPECIFICATION ---\n{task_prompt}\n\n"
                f"--- REQUIRED THEOREMS ---\n{self.required_theorems}\n\n"
                f"--- CURRENT LEAN 4 CODE (WITH LINE NUMBERS) ---\n{numbered_code}\n\n"
                f"--- LEAN 4 COMPILER DIAGNOSTICS ({current_val.status}) ---\n"
                f"Audit Progress: {bar} (Verified Theorems: {current_val.verified_theorems})\n"
                f"Issues Summary: {current_val.issues_summary}\n"
                f"Raw Compiler Output:\n{current_val.compiler_output}\n\n"
                "Repair every compiler error, warning, unsolved goal, or missing theorem above. "
                "Preserve all already-verified helper lemmas verbatim. "
                "Output the complete, self-contained, corrected Lean 4 file inside a single ```lean ... ``` block."
            )
            try:
                resp = self.client.complete(
                    [
                        {"role": "system", "content": LEAN_SYNTHESIS_SCRIBE_PROMPT},
                        {"role": "user", "content": repair_user},
                    ],
                    temperature=0.1,
                    max_tokens=32768,
                    enable_thinking=True,
                    thinking_budget_tokens=8192,
                    timeout_seconds=360.0,
                )
                repaired_text = (resp.content or "").strip()
                extracted = extract_lean_code(repaired_text)
                if extracted:
                    current_code = extracted
                    current_val = self.validate_lean_candidate(current_code)
                    if current_val.validity_score >= best_repair_val.validity_score:
                        best_repair_val = current_val
                    if current_val.passed:
                        self._log_write(
                            f"[Lean 4 Compiler Repair Loop SUCCESS on Attempt {attempt}]: "
                            f"100% Certified in {current_val.execution_time:.2f}s!\n"
                        )
                        self.last_lean_validation = current_val
                        return current_val
            except Exception as exc:
                self._log_write(f"[Lean 4 Compiler Repair Loop Exception on Attempt {attempt}]: {exc}\n")

        self.last_lean_validation = best_repair_val
        return best_repair_val

    def prove_lean(
        self,
        task_id: str,
        task_prompt: str,
        max_iterations: Optional[int] = None,
        max_time_seconds: float = 3600.0,
        on_step: Optional[Callable[[str, Any], None]] = None,
    ) -> Tuple[TCSSolveResult, LeanKernelValidationResult]:
        """Dedicated Lean 4 Formal Proving Loop.

        On each turn:
        1. Streams reasoning + Python REPL + Lean 4 code from the model.
        2. Executes any ```python ... ``` blocks in `LeanCompliantKernel` (where `verify_lean4_kernel`
           is preloaded and automatically records any passing Lean 4 proof).
        3. Extracts any ```lean ... ``` candidate block (or recovers the verified Lean code from the REPL).
        4. Validates deterministically with `verify_lean4_kernel` (5-Gate Lean 4 compiler + `#print axioms` audit).
        5. If compilation fails, runs the targeted Lean 4 compiler repair loop or feeds exact compiler
           diagnostics into the next turn.
        6. As soon as `verify_lean4_kernel` passes (0 errors, 0 warnings, 0 sorry, 0 custom axioms),
           returns immediately with a 100% deterministic Certified Hit!
        """
        max_turns = max_iterations or self.default_max_iterations
        start_time = time.perf_counter()

        if isinstance(self.kernel, LeanCompliantKernel):
            self.kernel.required_theorems = list(self.required_theorems)
            self.kernel.last_verified_lean_result = None
            self.kernel.reset()

        req_banner = ""
        if self.required_theorems:
            req_banner = (
                "\n\nREQUIRED LEAN 4 THEOREM DECLARATIONS (must all be proved with 0 sorry and 0 custom axioms):\n"
                + "\n".join(f"  - `{t}`" for t in self.required_theorems)
            )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._get_base_prover_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"{task_prompt}{req_banner}\n\n"
                    "Use the Python REPL (`verify_lean4_kernel(lean_src)`) to test and verify your Lean 4 "
                    "formalization interactively, then output the complete, certified ```lean ... ``` file."
                ),
            },
        ]

        reasoning_traces: List[str] = []
        code_history: List[str] = []
        best_val: Optional[LeanKernelValidationResult] = None

        for turn in range(1, max_turns + 1):
            elapsed = time.perf_counter() - start_time
            if elapsed >= max_time_seconds:
                self._log_write(f"\n[LeanProver Timeout] Reached {elapsed:.1f}s >= {max_time_seconds:.1f}s.\n")
                break

            if on_step:
                on_step("turn_start", turn)
            self._log_write(f"\n==================== LEAN PROVER TURN {turn}/{max_turns} ====================\n")

            r_text, c_text = self._stream_completion_with_retry(
                messages,
                max_tokens=32768,
                temperature=0.1,
                enable_thinking=self.enable_thinking,
                thinking_budget_tokens=12288,
                timeout_seconds=min(600.0, max(120.0, max_time_seconds - elapsed)),
            )
            if r_text:
                reasoning_traces.append(r_text)

            messages.append({"role": "assistant", "content": c_text})

            # 1. Execute any ```python ... ``` blocks in LeanKernel
            py_blocks = self.extract_code_blocks(c_text)
            repl_outputs: List[str] = []
            for idx, py_code in enumerate(py_blocks):
                if on_step:
                    on_step("code_start", {"turn": turn, "block": idx + 1})
                code_history.append(py_code)
                exec_res = self.kernel.execute(py_code, timeout_seconds=120.0)
                out_str = (exec_res.stdout or "").strip()
                err_str = (exec_res.error or exec_res.stderr or "").strip()
                combined_out = out_str + (f"\n[Error]: {err_str}" if err_str else "")
                repl_outputs.append(f"[Python REPL Block {idx + 1} Output]:\n{combined_out[:4000]}")
                self._log_write(f"\n[Turn {turn} Python REPL Block {idx + 1}]:\n{combined_out[:4000]}\n")

            # 2. Check if the REPL already certified a Lean 4 proof on this turn!
            if isinstance(self.kernel, LeanCompliantKernel) and self.kernel.last_verified_lean_result is not None:
                k_val = self.kernel.last_verified_lean_result
                if k_val.passed:
                    best_val = k_val
                    self.last_lean_validation = k_val
                    if on_step:
                        on_step("audit_result", {"verdict": "VALID", "validity": 1.0, "status": k_val.status})
                        on_step("proof_accepted", k_val.lean_code)
                    duration = round(time.perf_counter() - start_time, 2)
                    solve_res = TCSSolveResult(
                        task_id=task_id,
                        verified=True,
                        candidate_proof=k_val.lean_code,
                        audit_verdict="VALID",
                        audit_validity=1.0,
                        iterations=turn,
                        duration_s=duration,
                        reasoning_traces=reasoning_traces,
                        code_history=code_history,
                    )
                    return solve_res, k_val

            # 3. Extract any ```lean ... ``` block from assistant output (or reasoning fallback)
            lean_cand = extract_lean_code(c_text) or extract_lean_code(r_text)
            if lean_cand:
                if on_step:
                    on_step("audit_start", lean_cand)
                val_res = self.validate_lean_candidate(lean_cand)
                prev_score = best_val.validity_score if best_val is not None else 0.0
                if not val_res.passed and turn == max_turns:
                    val_res = self._repair_lean_code_loop(task_prompt, lean_cand, val_res)
                    repair_ran = True
                else:
                    repair_ran = False

                delta = round(val_res.validity_score - prev_score, 2)
                delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
                bar = format_lean_progress_bar(val_res.validity_score)
                self._log_write(
                    f"\n[Turn {turn} Audit Progress]: Validity {bar} ({delta_str}) | "
                    f"CompilerHealth={val_res.compiler_health_score:.2f} | "
                    f"Rigor={val_res.rigor_score:.2f} | "
                    f"Completeness={val_res.completeness_score:.2f} | "
                    f"VerifiedTheorems={val_res.verified_theorems}\n"
                )

                if best_val is None or val_res.validity_score >= best_val.validity_score or val_res.passed:
                    best_val = val_res
                self.last_lean_validation = val_res
                if on_step:
                    on_step(
                        "audit_result",
                        {
                            "verdict": val_res.verdict,
                            "validity": val_res.validity_score,
                            "rigor": val_res.rigor_score,
                            "completeness": val_res.completeness_score,
                            "boundary_cases": val_res.boundary_cases_score,
                            "compiler_health": val_res.compiler_health_score,
                            "verified_theorems": val_res.verified_theorems,
                            "progress_bar": bar,
                            "delta": delta,
                            "status": val_res.status,
                            "issues": val_res.issues_summary,
                        },
                    )

                if val_res.passed:
                    if on_step:
                        on_step("proof_accepted", val_res.lean_code)
                    duration = round(time.perf_counter() - start_time, 2)
                    solve_res = TCSSolveResult(
                        task_id=task_id,
                        verified=True,
                        candidate_proof=val_res.lean_code,
                        audit_verdict="VALID",
                        audit_validity=1.0,
                        iterations=turn,
                        duration_s=duration,
                        reasoning_traces=reasoning_traces,
                        code_history=code_history,
                    )
                    return solve_res, val_res

                feedback_parts = []
                if repl_outputs:
                    feedback_parts.append("\n\n".join(repl_outputs))
                feedback_parts.append(
                    f"[LEAN 4 COMPILER & AXIOM AUDIT ({val_res.status})]:\n"
                    f"• Composite Progress: {bar} ({delta_str} from previous best)\n"
                    f"• Scores: CompilerHealth={val_res.compiler_health_score:.2f}, "
                    f"Rigor={val_res.rigor_score:.2f}, Completeness={val_res.completeness_score:.2f}\n"
                    f"• Verified Theorems So Far (keep verbatim): {val_res.verified_theorems}\n"
                    f"• Issues: {val_res.issues_summary}\n"
                    f"• Theorems Found: {val_res.theorems_found}\n"
                    f"• Missing Required Theorems: {val_res.missing_required_theorems}\n"
                    f"• Compiler Output:\n{val_res.compiler_output[:3000]}\n\n"
                    "Please repair the exact Lean 4 errors above, verify with `verify_lean4_kernel(lean_src)` "
                    "in a ```python``` block until `res.passed == True`, and output the complete ```lean``` file."
                )
                messages.append({"role": "user", "content": "\n\n".join(feedback_parts)})
            else:
                repair_ran = False
                feedback_parts = []
                if repl_outputs:
                    feedback_parts.append("\n\n".join(repl_outputs))
                feedback_parts.append(
                    "Now test your complete Lean 4 proof in the Python REPL using `verify_lean4_kernel(lean_src)` "
                    "and output the complete, self-contained ```lean ... ``` file."
                )
                messages.append({"role": "user", "content": "\n\n".join(feedback_parts)})

        if best_val is not None and not best_val.passed and best_val.lean_code and not locals().get("repair_ran", False):
            best_val = self._repair_lean_code_loop(task_prompt, best_val.lean_code, best_val)

        if best_val is None:
            best_val = LeanKernelValidationResult(
                passed=False,
                verdict="REJECT",
                status="NO_LEAN_CODE",
                lean_code="",
                issues_summary="Prover completed turns without producing a certified Lean 4 file.",
            )

        duration = round(time.perf_counter() - start_time, 2)
        solve_res = TCSSolveResult(
            task_id=task_id,
            verified=best_val.passed,
            candidate_proof=best_val.lean_code,
            audit_verdict=best_val.verdict,
            audit_validity=best_val.validity_score,
            iterations=max_turns,
            duration_s=duration,
            reasoning_traces=reasoning_traces,
            code_history=code_history,
        )
        return solve_res, best_val


__all__ = [
    "ALLOWED_CORE_AXIOMS",
    "LeanKernelValidationResult",
    "compute_lean_audit_scores",
    "format_lean_progress_bar",
    "verify_lean4_kernel",
    "extract_lean_declarations",
    "strip_lean_comments",
    "run_lean_with_optional_lake",
    "LeanCompliantKernel",
    "LeanKernel",
    "LeanProver",
]
