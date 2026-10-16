"""Lightweight Lean 4 Runner for Prime Math.

Executes Lean 4 source code via the local Lean 4 binary, capturing compiler diagnostics,
proof verification status, syntax errors, and detecting unsound 'sorry' / 'admit' shortcuts.
Strictly local and compliant with offline / zero-internet benchmarking protocols.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("prime_math.lean_runner")

_DEFAULT_LEAN_PATHS = [
    Path(os.path.expanduser("~/.elan/bin/lean")),
    Path("/usr/local/bin/lean"),
    Path("/usr/bin/lean"),
]

# Patterns for diagnostics
_ERROR_RE = re.compile(r"(?:error:\s*|:\s*error:\s*)(.+)", re.MULTILINE)
_WARNING_RE = re.compile(r"(?:warning:\s*|:\s*warning:\s*)(.+)", re.MULTILINE)
_UNSOLVED_GOALS_RE = re.compile(r"unsolved goals\b", re.IGNORECASE)
_SORRY_WARNING_RE = re.compile(r"declaration uses\s+['\"]?sorry['\"]?", re.IGNORECASE)
_SORRY_CODE_RE = re.compile(r"\b(sorry|admit)\b")
_FORBIDDEN_AXIOM_CODE_RE = re.compile(r"\b(axiom|constant|unsafe|partial|implemented_by|extern)\b")


@dataclass
class LeanRunResult:
    """Detailed outcome of a Lean 4 compilation / verification run."""
    success: bool
    stdout: str
    stderr: str
    returncode: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    unsolved_goals: List[str] = field(default_factory=list)
    has_sorry: bool = False
    execution_time: float = 0.0
    lean_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "errors": self.errors,
            "warnings": self.warnings,
            "unsolved_goals": self.unsolved_goals,
            "has_sorry": self.has_sorry,
            "execution_time": self.execution_time,
            "lean_version": self.lean_version,
        }


def find_lean_binary() -> Optional[Path]:
    """Locates the Lean 4 executable."""
    env_bin = os.environ.get("LEAN_BIN") or os.environ.get("TCS_LEAN_BIN")
    if env_bin:
        p = Path(env_bin)
        if p.is_file() and os.access(p, os.X_OK):
            return p

    for p in _DEFAULT_LEAN_PATHS:
        if p.is_file() and os.access(p, os.X_OK):
            return p

    which_lean = shutil.which("lean")
    if which_lean:
        return Path(which_lean)

    return None


def get_lean_version(lean_bin: Optional[Path] = None) -> Optional[str]:
    """Returns the version string of the installed Lean compiler."""
    binary = lean_bin or find_lean_binary()
    if not binary:
        return None
    try:
        proc = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception as exc:
        logger.debug("Failed to query Lean version: %s", exc)
    return None


def run_lean(
    code: str,
    timeout: float = 60.0,
    lean_bin: Optional[Path] = None,
    extra_args: Optional[List[str]] = None,
    memory_limit_mb: int = 4096,
    max_heartbeats: int = 500000,
) -> LeanRunResult:
    """Executes a Lean 4 code snippet and checks for compilation/proof validity.

    Args:
        code: Lean 4 source code.
        timeout: Maximum seconds allowed for Lean verification.
        lean_bin: Optional path to the Lean binary.
        extra_args: Optional list of additional flags passed to Lean.
        memory_limit_mb: Maximum memory in megabytes for Lean compiler (default 4GB).
        max_heartbeats: Maximum memory allocation heartbeats per task.

    Returns:
        LeanRunResult containing diagnostics, exit status, and 'sorry' checks.
    """
    binary = lean_bin or find_lean_binary()
    if not binary:
        return LeanRunResult(
            success=False,
            stdout="",
            stderr="Lean binary not found. Please ensure Lean 4 is installed via elan.",
            returncode=-1,
            errors=["Lean binary not found on system PATH or ~/.elan/bin/lean"],
            execution_time=0.0,
        )

    clean_code = code.strip()
    if not clean_code:
        return LeanRunResult(
            success=False,
            stdout="",
            stderr="Empty Lean code provided.",
            returncode=-1,
            errors=["Empty Lean code"],
            execution_time=0.0,
        )

    # Strip single-line comments before scanning for 'sorry' or forbidden axiom/unsafe tokens
    code_no_comments = "\n".join(line.split("--")[0] for line in clean_code.splitlines())
    has_sorry_in_source = bool(_SORRY_CODE_RE.search(code_no_comments))
    forbidden_match = _FORBIDDEN_AXIOM_CODE_RE.search(code_no_comments)

    start_t = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="prime_lean_") as tmpdir:
        src_path = Path(tmpdir) / "Proof.lean"
        src_path.write_text(clean_code, encoding="utf-8")

        cmd = [str(binary), f"-M{memory_limit_mb}", f"-T{max_heartbeats}", str(src_path)]
        if extra_args:
            cmd.extend(extra_args)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=tmpdir,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
                proc.communicate()
                exec_time = round(time.perf_counter() - start_t, 3)
                return LeanRunResult(
                    success=False,
                    stdout="",
                    stderr=f"Lean compilation timed out after {timeout} seconds.",
                    returncode=-2,
                    errors=[f"Timeout ({timeout}s exceeded)"],
                    execution_time=exec_time,
                    lean_version=get_lean_version(binary),
                )
        except Exception as exc:
            exec_time = round(time.perf_counter() - start_t, 3)
            return LeanRunResult(
                success=False,
                stdout="",
                stderr=str(exc),
                returncode=-3,
                errors=[str(exc)],
                execution_time=exec_time,
                lean_version=get_lean_version(binary),
            )

    exec_time = round(time.perf_counter() - start_t, 3)
    combined_output = f"{stdout}\n{stderr}"

    # Extract diagnostic messages
    errors: List[str] = []
    warnings: List[str] = []
    unsolved_goals: List[str] = []

    for line in combined_output.splitlines():
        line_s = line.strip()
        if "error:" in line_s:
            errors.append(line_s)
        elif "warning:" in line_s:
            warnings.append(line_s)
        if _UNSOLVED_GOALS_RE.search(line_s):
            unsolved_goals.append(line_s)

    has_sorry_warning = bool(_SORRY_WARNING_RE.search(combined_output))
    if forbidden_match:
        errors.append(f"Forbidden unsound keyword '{forbidden_match.group(1)}' detected in Lean source.")
    has_sorry = has_sorry_in_source or has_sorry_warning or bool(forbidden_match)

    # Proof is only considered successful if:
    # 1. Lean exited with code 0
    # 2. No compiler errors
    # 3. No unsolved goals left
    # 4. No 'sorry' or 'admit' shortcuts used
    success = (returncode == 0) and (len(errors) == 0) and (not has_sorry) and (len(unsolved_goals) == 0)

    return LeanRunResult(
        success=success,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        errors=errors,
        warnings=warnings,
        unsolved_goals=unsolved_goals,
        has_sorry=has_sorry,
        execution_time=exec_time,
        lean_version=get_lean_version(binary),
    )
