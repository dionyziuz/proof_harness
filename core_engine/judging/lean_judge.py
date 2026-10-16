"""Lean 4 Proof Judge for Prime Math and TCS-Bench.

Evaluates candidate formal proofs using the Lean 4 compiler and type checker rather than LLM judges.
Guarantees 100% mathematical soundness by verifying:
1. Exact compilation via the Lean 4 compiler.
2. Complete discharge of all goals (zero unsolved goals).
3. Prohibition of unsound axioms and shortcuts ('sorry', 'admit').
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core_engine.lean_runner import LeanRunResult, get_lean_version, run_lean

logger = logging.getLogger("core_engine.judging.lean_judge")

# Patterns to extract Lean code from markdown responses
_LEAN_BLOCK_RE = re.compile(r"```(?:lean|lean4)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
_LEAN_KEYWORDS = {"theorem", "lemma", "def", "inductive", "structure", "open", "import", "example"}


@dataclass
class LeanJudgeRecord:
    """Structured verdict from Lean 4 formal verification."""
    verdict: str  # "1" (Pass) or "0" (Fail)
    passed: bool
    status: str  # "PASS", "FAIL", "SORRY_DETECTED", "NO_LEAN_CODE", "TIMEOUT", "ERROR"
    has_sorry: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    unsolved_goals: List[str] = field(default_factory=list)
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
            "compiler_output": self.compiler_output,
            "execution_time": self.execution_time,
            "lean_version": self.lean_version,
            "extracted_code_preview": self.extracted_code_preview,
        }


def extract_lean_code(text: str) -> str:
    """Extracts Lean 4 source code from candidate markdown response."""
    blocks = _LEAN_BLOCK_RE.findall(text)
    if blocks:
        # Join all extracted Lean code blocks
        return "\n\n".join(b.strip() for b in blocks if b.strip())

    # If no fenced code blocks, check if raw text resembles Lean code
    lines = text.strip().splitlines()
    first_words = [line.strip().split()[0] for line in lines if line.strip() and not line.strip().startswith("--")]
    if any(w in _LEAN_KEYWORDS for w in first_words):
        return text.strip()

    return ""


def judge_lean_proof(
    candidate: str,
    problem_statement: Optional[str] = None,
    timeout: float = 60.0,
    lean_bin: Optional[Path] = None,
) -> Dict[str, Any]:
    """Judges a candidate solution by verifying its formal Lean 4 proof.

    Args:
        candidate: Candidate proof response (either markdown containing ```lean blocks or raw Lean code).
        problem_statement: Optional problem specification (for problem context).
        timeout: Maximum seconds allowed for Lean verification.
        lean_bin: Optional override for Lean binary path.

    Returns:
        Dictionary conforming to judge record schema:
        {"verdict": "1"|"0", "passed": bool, "status": str, ...}
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

    code_preview = code[:300] + ("..." if len(code) > 300 else "")

    run_res: LeanRunResult = run_lean(code, timeout=timeout, lean_bin=lean_bin)

    if run_res.has_sorry:
        status = "SORRY_DETECTED"
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
