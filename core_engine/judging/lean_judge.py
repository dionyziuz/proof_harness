"""Lean 4 Formal Compiler & Kernel Axiom Judge for Prime Math.

Evaluates candidate formal proofs deterministically using the Lean 4 compiler and kernel
axiom inspector (`#print axioms`) rather than subjective LLM judges.

Provides two verification levels:
1. `verify_lean4_kernel`: Full 5-Gate Formal Verification Engine used by `LeanProver`:
   - Gate 1: Exact Lean 4 compilation (`returncode == 0`, 0 errors, 0 unsolved goals).
   - Gate 2: Zero `sorry` / `admit` shortcuts and zero compiler warnings.
   - Gate 3: Automated `#print axioms <thm>` inspection across every theorem in the file,
     enforcing that ONLY standard Lean 4 core axioms (`propext`, `Classical.choice`, `Quot.sound`)
     appear. Any custom `axiom`, `constant`, `sorryAx`, or `unsafe` triggers immediate rejection.
   - Gate 4: Semantic depth & anti-tautology gate (`audit_lean_semantic_depth`).
   - Gate 5: Target theorem signature completeness check (`required_theorems`).
2. `judge_lean_proof`: Benchmark & snippet judge returning `LeanJudgeRecord` with compilation,
   `sorry` detection, and `#print axioms` kernel purity verification.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from core_engine.lean_runner import (
    LeanRunResult,
    audit_lean_semantic_depth,
    find_lean_binary,
    get_lean_version,
    run_lean,
)

logger = logging.getLogger("core_engine.judging.lean_judge")

# Standard core axioms permitted by Lean 4 & Mathlib
ALLOWED_CORE_AXIOMS: Set[str] = {"propext", "Classical.choice", "Quot.sound"}

# Patterns to extract Lean code from markdown responses
_LEAN_BLOCK_RE = re.compile(r"```(?:lean|lean4)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
_LEAN_KEYWORDS = {"theorem", "lemma", "def", "inductive", "structure", "open", "import", "example"}

_THEOREM_DECL_RE = re.compile(
    r"^\s*(?:private\s+|protected\s+)?(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_']*)\b",
    re.MULTILINE,
)
_DEF_DECL_RE = re.compile(
    r"^\s*(?:private\s+|protected\s+)?(?:def|abbrev|structure|inductive|class)\s+([A-Za-z_][A-Za-z0-9_']*)\b",
    re.MULTILINE,
)
_AXIOM_LINE_NO_DEPS_RE = re.compile(r"^'([^']+)'\s+does not depend on any axioms\s*$")
_AXIOM_LINE_DEPS_RE = re.compile(r"^'([^']+)'\s+depends on axioms:\s*\[([^\]]*)\]\s*$")


@dataclass
class LeanKernelValidationResult:
    """Comprehensive deterministic 5-gate verification report from the Lean 4 Kernel Judge."""
    passed: bool
    verdict: str  # "VALID" or "REJECT"
    status: str  # "CERTIFIED_HIT", "COMPILE_ERROR", "AXIOM_VIOLATION", "SORRY_DETECTED", "MISSING_THEOREMS", "SEMANTIC_VIOLATION", "NO_LEAN_CODE"
    lean_code: str
    theorems_found: List[str] = field(default_factory=list)
    definitions_found: List[str] = field(default_factory=list)
    missing_required_theorems: List[str] = field(default_factory=list)
    axiom_report: Dict[str, List[str]] = field(default_factory=dict)
    unauthorized_axioms: Dict[str, List[str]] = field(default_factory=dict)
    compiler_errors: List[str] = field(default_factory=list)
    compiler_warnings: List[str] = field(default_factory=list)
    unsolved_goals: List[str] = field(default_factory=list)
    semantic_violations: List[str] = field(default_factory=list)
    has_sorry: bool = False
    execution_time: float = 0.0
    lean_version: Optional[str] = None
    compiler_output: str = ""
    issues_summary: str = "none"
    verified_theorems: List[str] = field(default_factory=list)
    validity_score: float = 0.0
    rigor_score: float = 0.0
    completeness_score: float = 0.0
    boundary_cases_score: float = 0.0
    compiler_health_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "verdict": self.verdict,
            "status": self.status,
            "theorems_found": self.theorems_found,
            "definitions_found": self.definitions_found,
            "missing_required_theorems": self.missing_required_theorems,
            "verified_theorems": self.verified_theorems,
            "axiom_report": self.axiom_report,
            "unauthorized_axioms": self.unauthorized_axioms,
            "compiler_errors": self.compiler_errors,
            "compiler_warnings": self.compiler_warnings,
            "unsolved_goals": self.unsolved_goals,
            "semantic_violations": self.semantic_violations,
            "has_sorry": self.has_sorry,
            "execution_time": self.execution_time,
            "lean_version": self.lean_version,
            "compiler_output": self.compiler_output,
            "issues_summary": self.issues_summary,
            "validity_score": self.validity_score,
            "rigor_score": self.rigor_score,
            "completeness_score": self.completeness_score,
            "boundary_cases_score": self.boundary_cases_score,
            "compiler_health_score": self.compiler_health_score,
        }


@dataclass
class LeanJudgeRecord:
    """Structured verdict from Lean 4 formal verification."""
    verdict: str  # "1" (Pass) or "0" (Fail)
    passed: bool
    status: str  # "PASS", "FAIL", "SORRY_DETECTED", "AXIOM_VIOLATION", "NO_LEAN_CODE", "TIMEOUT", "ERROR"
    has_sorry: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    unsolved_goals: List[str] = field(default_factory=list)
    theorems_found: List[str] = field(default_factory=list)
    definitions_found: List[str] = field(default_factory=list)
    axiom_report: Dict[str, List[str]] = field(default_factory=dict)
    unauthorized_axioms: Dict[str, List[str]] = field(default_factory=dict)
    compiler_output: str = ""
    execution_time: float = 0.0
    lean_version: Optional[str] = None
    extracted_code_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "passed": self.passed,
            "status": self.status,
            "has_sorry": self.has_sorry,
            "errors": self.errors,
            "warnings": self.warnings,
            "unsolved_goals": self.unsolved_goals,
            "theorems_found": self.theorems_found,
            "definitions_found": self.definitions_found,
            "axiom_report": self.axiom_report,
            "unauthorized_axioms": self.unauthorized_axioms,
            "compiler_output": self.compiler_output,
            "execution_time": self.execution_time,
            "lean_version": self.lean_version,
            "extracted_code_preview": self.extracted_code_preview,
        }


def strip_lean_comments(code: str) -> str:
    """Removes /- ... -/ block comments and -- line comments from Lean 4 source."""
    no_block = re.sub(r"/-.*?-/", "", code, flags=re.DOTALL)
    return "\n".join(line.split("--")[0] for line in no_block.splitlines())


def extract_lean_declarations(code: str) -> Tuple[List[str], List[str]]:
    """Returns (theorem_names, definition_names) declared in the Lean 4 source code."""
    clean = strip_lean_comments(code)
    theorems: List[str] = []
    for m in _THEOREM_DECL_RE.finditer(clean):
        name = m.group(1)
        if name not in theorems:
            theorems.append(name)
    defs: List[str] = []
    for m in _DEF_DECL_RE.finditer(clean):
        name = m.group(1)
        if name not in defs:
            defs.append(name)
    return theorems, defs


def extract_lean_code(text: str) -> str:
    """Extracts Lean 4 source code from candidate markdown response."""
    blocks = _LEAN_BLOCK_RE.findall(text)
    if blocks:
        joined = "\n\n".join(b.strip() for b in blocks if b.strip())
        # Clean stray markdown backtick line artifacts at start/end
        cleaned_lines = []
        for line in joined.splitlines():
            if line.strip() in ("`", "``", "```"):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    lines = text.strip().splitlines()
    first_words = [
        line.strip().split()[0]
        for line in lines
        if line.strip() and not line.strip().startswith("--")
    ]
    if any(w in _LEAN_KEYWORDS for w in first_words):
        return text.strip()

    return ""


def _parse_axiom_stdout(stdout: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Parses `#print axioms <thm>` lines from Lean compiler stdout."""
    axiom_report: Dict[str, List[str]] = {}
    unauthorized_axioms: Dict[str, List[str]] = {}
    for line in (stdout or "").splitlines():
        line_s = line.strip()
        m_none = _AXIOM_LINE_NO_DEPS_RE.match(line_s)
        if m_none:
            axiom_report[m_none.group(1)] = []
            continue
        m_deps = _AXIOM_LINE_DEPS_RE.match(line_s)
        if m_deps:
            thm_name = m_deps.group(1)
            raw_deps = [a.strip() for a in m_deps.group(2).split(",") if a.strip()]
            axiom_report[thm_name] = raw_deps
            bad = [a for a in raw_deps if a not in ALLOWED_CORE_AXIOMS]
            if bad:
                unauthorized_axioms[thm_name] = bad
    return axiom_report, unauthorized_axioms


def run_lean_with_optional_lake(
    code: str,
    timeout: float = 90.0,
    lake_dir: Optional[Path] = None,
    lean_bin: Optional[Path] = None,
) -> LeanRunResult:
    """Runs Lean 4 code via standalone `run_lean` (default generic local Lean 4 installation)
    or via `lake env lean` if `lake_dir` is explicitly provided.
    """
    active_lake = lake_dir or (
        Path(os.environ["LEAN_LAKE_DIR"]) if os.environ.get("LEAN_LAKE_DIR") else None
    )
    if active_lake and active_lake.is_dir() and (
        (active_lake / "lakefile.lean").is_file() or (active_lake / "lakefile.toml").is_file()
    ):
        lake_bin = Path.home() / ".elan/bin/lake"
        if not lake_bin.is_file():
            lake_bin = Path("lake")
        start_t = time.perf_counter()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", dir=str(active_lake), delete=False, encoding="utf-8"
        ) as tmpf:
            tmp_path = Path(tmpf.name)
            tmpf.write(code.strip() + "\n")
        try:
            proc = subprocess.run(
                [str(lake_bin), "env", "lean", str(tmp_path)],
                cwd=str(active_lake),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            exec_time = round(time.perf_counter() - start_t, 3)
            combined = f"{proc.stdout}\n{proc.stderr}"
            errors = [l.strip() for l in combined.splitlines() if "error:" in l]
            warnings = [l.strip() for l in combined.splitlines() if "warning:" in l]
            unsolved = [l.strip() for l in combined.splitlines() if "unsolved goals" in l.lower()]
            clean_src = strip_lean_comments(code)
            has_sorry = bool(
                re.search(r"\b(sorry|admit|axiom|constant|unsafe|implemented_by|extern)\b", clean_src)
                or re.search(r"declaration uses\s+['\"]?sorry['\"]?", combined, re.IGNORECASE)
            )
            success = (proc.returncode == 0) and not errors and not unsolved and not has_sorry
            return LeanRunResult(
                success=success,
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                errors=errors,
                warnings=warnings,
                unsolved_goals=unsolved,
                has_sorry=has_sorry,
                execution_time=exec_time,
                lean_version=get_lean_version(lean_bin),
            )
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    return run_lean(code, timeout=timeout, lean_bin=lean_bin)


def verify_lean4_kernel(
    candidate_text_or_code: str,
    required_theorems: Optional[List[str]] = None,
    timeout: float = 90.0,
    lake_dir: Optional[Path] = None,
    lean_bin: Optional[Path] = None,
    require_zero_warnings: bool = True,
    require_imports_or_structures: bool = True,
) -> LeanKernelValidationResult:
    """Deterministic 5-Gate Lean 4 Compiler & Kernel Axiom Validator.

    1. Extracts Lean 4 code (if wrapped in markdown fences or \\begin{proof}).
    2. Extracts all theorem and definition names.
    3. Appends `#print axioms <thm>` for every theorem to inspect kernel-level axiom dependencies.
    4. Runs the Lean 4 compiler (`run_lean_with_optional_lake`).
    5. Verifies all 5 gates:
       - Gate 1: 0 compiler errors, 0 unsolved goals.
       - Gate 2: 0 `sorry`/`admit` and 0 compiler warnings (when `require_zero_warnings=True`).
       - Gate 3: 0 unauthorized axioms in `#print axioms` (only `propext`, `Classical.choice`, `Quot.sound` allowed).
       - Gate 4: 0 semantic tautology violations (`audit_lean_semantic_depth`).
       - Gate 5: All `required_theorems` (if specified) are present and proved.
    """
    code = extract_lean_code(candidate_text_or_code)
    if not code and candidate_text_or_code:
        stripped = re.sub(r"^\\begin\{proof\}|\\end\{proof\}$", "", candidate_text_or_code.strip()).strip()
        code = extract_lean_code(stripped) or (
            stripped if any(k in stripped for k in ("theorem ", "lemma ", "def ", "import ")) else ""
        )

    if not code or not code.strip():
        return LeanKernelValidationResult(
            passed=False,
            verdict="REJECT",
            status="NO_LEAN_CODE",
            lean_code="",
            issues_summary="No valid Lean 4 source code block (```lean ... ```) found in candidate output.",
        )

    clean_code = code.strip()
    theorems, defs = extract_lean_declarations(clean_code)

    req_list = [r.strip() for r in (required_theorems or []) if r.strip()]
    missing_reqs = [r for r in req_list if r not in theorems and r not in defs]

    axiom_check_lines = [f"#print axioms {thm}" for thm in theorems]
    instrumented_code = clean_code
    if axiom_check_lines:
        instrumented_code = (
            clean_code
            + "\n\n-- Automated LeanProver Kernel Axiom Audit\n"
            + "\n".join(axiom_check_lines)
            + "\n"
        )

    run_res = run_lean_with_optional_lake(
        instrumented_code, timeout=timeout, lake_dir=lake_dir, lean_bin=lean_bin
    )
    axiom_report, unauthorized_axioms = _parse_axiom_stdout(run_res.stdout or "")

    sem_violations = audit_lean_semantic_depth(
        clean_code, require_imports_or_structures=require_imports_or_structures
    )
    if not theorems:
        sem_violations.append("Lean file declares zero `theorem` or `lemma` statements.")

    issues: List[str] = []
    if not run_res.success or run_res.errors:
        for err in run_res.errors[:15]:
            issues.append(f"Lean Compiler Error: {err}")
        if not run_res.errors and run_res.stderr.strip():
            issues.append(f"Lean stderr: {run_res.stderr.strip()[:600]}")
    if run_res.unsolved_goals:
        for ug in run_res.unsolved_goals[:5]:
            issues.append(f"Unsolved Goal: {ug}")
    if run_res.has_sorry:
        issues.append("Unsound shortcut (`sorry`, `admit`, or forbidden `axiom`/`constant` keyword) detected.")
    if unauthorized_axioms:
        for thm_n, bad_ax in unauthorized_axioms.items():
            issues.append(
                f"Kernel Axiom Violation in `{thm_n}`: depends on unauthorized axioms {bad_ax} "
                f"(only {sorted(ALLOWED_CORE_AXIOMS)} are permitted)."
            )
    if require_zero_warnings and run_res.warnings:
        for w in run_res.warnings[:5]:
            issues.append(f"Lean Compiler Warning: {w}")
    if sem_violations:
        for sv in sem_violations:
            issues.append(f"Semantic Depth Violation: {sv}")
    if missing_reqs:
        issues.append(
            f"Missing required target declarations: {missing_reqs}. Found theorems: {theorems}, defs: {defs}."
        )

    passed = (
        run_res.success
        and len(run_res.errors) == 0
        and len(run_res.unsolved_goals) == 0
        and not run_res.has_sorry
        and len(unauthorized_axioms) == 0
        and (not require_zero_warnings or len(run_res.warnings) == 0)
        and len(sem_violations) == 0
        and len(missing_reqs) == 0
        and len(theorems) > 0
    )

    if passed:
        status = "CERTIFIED_HIT"
        verdict = "VALID"
        issues_summary = "none"
    elif run_res.has_sorry:
        status = "SORRY_DETECTED"
        verdict = "REJECT"
        issues_summary = " | ".join(issues)
    elif unauthorized_axioms:
        status = "AXIOM_VIOLATION"
        verdict = "REJECT"
        issues_summary = " | ".join(issues)
    elif not run_res.success or run_res.errors or run_res.unsolved_goals:
        status = "COMPILE_ERROR"
        verdict = "REJECT"
        issues_summary = " | ".join(issues)
    elif missing_reqs:
        status = "MISSING_THEOREMS"
        verdict = "REJECT"
        issues_summary = " | ".join(issues)
    else:
        status = "SEMANTIC_VIOLATION"
        verdict = "REJECT"
        issues_summary = " | ".join(issues)

    # Identify theorems verified by `#print axioms` with strictly pure core axioms
    verified_thms = [
        thm for thm in theorems
        if thm in axiom_report and thm not in unauthorized_axioms
    ]
    scores = compute_lean_audit_scores(
        passed=passed,
        status=status,
        theorems_found=theorems,
        required_theorems=req_list,
        missing_required_theorems=missing_reqs,
        verified_theorems=verified_thms,
        compiler_errors=run_res.errors,
        compiler_warnings=run_res.warnings,
        unsolved_goals=run_res.unsolved_goals,
        semantic_violations=sem_violations,
        unauthorized_axioms=unauthorized_axioms,
        has_sorry=run_res.has_sorry,
    )

    return LeanKernelValidationResult(
        passed=passed,
        verdict=verdict,
        status=status,
        lean_code=clean_code,
        theorems_found=theorems,
        definitions_found=defs,
        missing_required_theorems=missing_reqs,
        verified_theorems=verified_thms,
        axiom_report=axiom_report,
        unauthorized_axioms=unauthorized_axioms,
        compiler_errors=run_res.errors,
        compiler_warnings=run_res.warnings,
        unsolved_goals=run_res.unsolved_goals,
        semantic_violations=sem_violations,
        has_sorry=run_res.has_sorry,
        execution_time=run_res.execution_time,
        lean_version=run_res.lean_version,
        compiler_output=f"{run_res.stdout}\n{run_res.stderr}".strip(),
        issues_summary=issues_summary,
        validity_score=scores["validity"],
        rigor_score=scores["rigor"],
        completeness_score=scores["completeness"],
        boundary_cases_score=scores["boundary_cases"],
        compiler_health_score=scores["compiler_health"],
    )


def format_lean_progress_bar(score: float, width: int = 20) -> str:
    """Renders a visual ASCII/Unicode progress bar for Lean audit scores (0.00 to 1.00)."""
    clamped = max(0.0, min(1.0, float(score)))
    filled = int(round(clamped * width))
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}] {clamped:.2f}"


def compute_lean_audit_scores(
    *,
    passed: bool,
    status: str,
    theorems_found: List[str],
    required_theorems: List[str],
    missing_required_theorems: List[str],
    verified_theorems: List[str],
    compiler_errors: List[str],
    compiler_warnings: List[str],
    unsolved_goals: List[str],
    semantic_violations: List[str],
    unauthorized_axioms: Dict[str, List[str]],
    has_sorry: bool,
) -> Dict[str, float]:
    """Computes continuous, multi-dimensional audit scores (0.00 - 1.00) from Lean 4 kernel diagnostics.

    Axes:
      - `compiler_health`: Smooth inverse decay over compiler error and unsolved-goal counts.
      - `completeness`: Milestone ratio of declared and kernel-verified helper/target theorems.
      - `rigor`: Axiomatic purity (`#print axioms`), absence of `sorry`/`admit`, and semantic depth.
      - `boundary_cases`: Linter cleanliness, warning-free elaboration, and complete goal closure.
      - `validity`: Weighted composite score (strictly capped at 0.95 until 100% kernel certification).
    """
    if status == "NO_LEAN_CODE":
        return {
            "validity": 0.0,
            "rigor": 0.0,
            "completeness": 0.0,
            "boundary_cases": 0.0,
            "compiler_health": 0.0,
        }

    if passed:
        return {
            "validity": 1.0,
            "rigor": 1.0,
            "completeness": 1.0,
            "boundary_cases": 1.0,
            "compiler_health": 1.0,
        }

    # 1. Compiler Health (0.15 to 1.00)
    num_errors = len(compiler_errors) + len(unsolved_goals)
    num_warnings = len(compiler_warnings)
    if num_errors == 0 and num_warnings == 0:
        compiler_health = 1.0
    elif num_errors == 0:
        compiler_health = max(0.85, round(1.0 - 0.02 * num_warnings, 2))
    else:
        compiler_health = max(0.15, round(0.85 / (1.0 + 0.35 * num_errors), 2))

    # 2. Completeness (0.00 to 1.00)
    if required_theorems:
        req_total = max(1, len(required_theorems))
        present_ratio = max(0.0, (req_total - len(missing_required_theorems)) / req_total)
        verified_req = sum(1 for r in required_theorems if r in verified_theorems)
        verified_ratio = verified_req / req_total
        extra_helpers = max(0, len(verified_theorems) - verified_req)
        helper_bonus = min(0.25, 0.08 * extra_helpers)
        completeness = min(0.95, round(0.35 * present_ratio + 0.65 * verified_ratio + helper_bonus, 2))
    else:
        if not theorems_found:
            completeness = 0.0
        else:
            verified_ratio = len(verified_theorems) / len(theorems_found)
            completeness = min(0.95, round(0.30 + 0.65 * verified_ratio, 2))

    # 3. Rigor (0.10 to 1.00)
    if not theorems_found:
        rigor = 0.10
    else:
        rigor = 1.0
        if has_sorry:
            rigor = min(rigor, 0.45)
        if unauthorized_axioms:
            rigor = min(rigor, 0.35)
        if semantic_violations:
            rigor -= 0.25 * len(semantic_violations)
        rigor = max(0.10, round(rigor, 2))

    # 4. Boundary Cases & Linter Cleanliness (0.15 to 0.95)
    b_score = 1.0 - 0.15 * len(unsolved_goals) - 0.02 * num_warnings - 0.20 * len(missing_required_theorems)
    if num_errors > 0:
        b_score -= min(0.50, 0.10 * num_errors)
    boundary_cases = max(0.15, min(0.95, round(b_score, 2)))

    # 5. Composite Validity (capped at 0.95 until `passed == True`)
    raw_validity = 0.45 * compiler_health + 0.30 * completeness + 0.25 * rigor
    validity = max(0.10, min(0.95, round(raw_validity, 2)))

    return {
        "validity": validity,
        "rigor": rigor,
        "completeness": completeness,
        "boundary_cases": boundary_cases,
        "compiler_health": compiler_health,
    }


def judge_lean_proof(
    candidate: str,
    problem_statement: Optional[str] = None,
    timeout: float = 60.0,
    lean_bin: Optional[Path] = None,
) -> Dict[str, Any]:
    """Judges a candidate solution by verifying its formal Lean 4 proof and `#print axioms` purity.

    Args:
        candidate: Candidate proof response (either markdown containing ```lean blocks or raw Lean code).
        problem_statement: Optional problem specification (for problem context).
        timeout: Maximum seconds allowed for Lean verification.
        lean_bin: Optional override for Lean binary path.

    Returns:
        Dictionary conforming to `LeanJudgeRecord` schema:
        {"verdict": "1"|"0", "passed": bool, "status": str, "axiom_report": dict, ...}
    """
    code = extract_lean_code(candidate)
    if not code:
        rec = LeanJudgeRecord(
            verdict="0",
            passed=False,
            status="NO_LEAN_CODE",
            has_sorry=False,
            errors=["No valid Lean 4 code found in candidate response."],
            execution_time=0.0,
        )
        return rec.to_dict()

    clean_code = code.strip()
    code_preview = clean_code[:300] + ("..." if len(clean_code) > 300 else "")
    theorems, defs = extract_lean_declarations(clean_code)

    axiom_check_lines = [f"#print axioms {thm}" for thm in theorems]
    instrumented_code = clean_code
    if axiom_check_lines:
        instrumented_code = (
            clean_code
            + "\n\n-- Automated Kernel Axiom Audit\n"
            + "\n".join(axiom_check_lines)
            + "\n"
        )

    run_res: LeanRunResult = run_lean(instrumented_code, timeout=timeout, lean_bin=lean_bin)
    axiom_report, unauthorized_axioms = _parse_axiom_stdout(run_res.stdout or "")

    if run_res.has_sorry:
        status = "SORRY_DETECTED"
        passed = False
        verdict = "0"
    elif unauthorized_axioms:
        status = "AXIOM_VIOLATION"
        passed = False
        verdict = "0"
    elif run_res.returncode == -2:
        status = "TIMEOUT"
        passed = False
        verdict = "0"
    elif not run_res.success:
        status = "FAIL"
        passed = False
        verdict = "0"
    else:
        status = "PASS"
        passed = True
        verdict = "1"

    rec = LeanJudgeRecord(
        verdict=verdict,
        passed=passed,
        status=status,
        has_sorry=run_res.has_sorry,
        errors=run_res.errors,
        warnings=run_res.warnings,
        unsolved_goals=run_res.unsolved_goals,
        theorems_found=theorems,
        definitions_found=defs,
        axiom_report=axiom_report,
        unauthorized_axioms=unauthorized_axioms,
        compiler_output=f"{run_res.stdout}\n{run_res.stderr}".strip(),
        execution_time=run_res.execution_time,
        lean_version=run_res.lean_version,
        extracted_code_preview=code_preview,
    )
    return rec.to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description="Lean 4 Proof Judge for Prime Math / TCS-Bench")
    parser.add_argument("file", nargs="?", help="Path to Lean source file or candidate json/text")
    parser.add_argument("--timeout", type=float, default=60.0, help="Verification timeout in seconds")
    args = parser.parse_args()

    if not args.file:
        version = get_lean_version()
        print(f"Lean 4 Judge available. Detected Lean: {version or 'NOT FOUND'}")
        return 0

    path = Path(args.file)
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    content = path.read_text(encoding="utf-8")
    result = judge_lean_proof(content, timeout=args.timeout)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
