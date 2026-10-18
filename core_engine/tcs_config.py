"""Externalized Configuration for Prime Math Prover & Pipeline.

Loads configuration from `.env` located in the project root or environment variables.
Provides typed access to compactor thresholds, subagent token budgets, timeouts,
and APU kernel execution constraints without modifying Python source code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


def _bool_env(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _int_env(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def _float_env(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


def _str_env(key: str, default: str) -> str:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip()


# ---------------------------------------------------------------------------
# Single TCS judge pass threshold (validity >= threshold -> pass).
# Every judging default, pipeline default and metric label derives from this value.
# Override with TCS_PIPELINE_JUDGE_THRESHOLD (environment or project .env).
# ---------------------------------------------------------------------------
DEFAULT_TCS_PASS_THRESHOLD = 0.50
if load_dotenv is not None and DEFAULT_ENV_PATH.exists():
    load_dotenv(dotenv_path=DEFAULT_ENV_PATH, override=False)
TCS_PASS_THRESHOLD: float = _float_env("TCS_PIPELINE_JUDGE_THRESHOLD", DEFAULT_TCS_PASS_THRESHOLD)

# Half-width of the borderline band centred on the pass threshold (used by adaptive judging).
TCS_BORDERLINE_BAND: float = 0.20

# Default number of independent TCS Judge replicates (median is the reported validity).
TCS_JUDGE_REPLICATES: int = 3


@dataclass
class TCSConfig:
    # Context Compactor
    compactor_max_context_tokens: int = 48000
    compactor_reserve_tokens: int = 16000
    compactor_keep_recent_turns: int = 2

    # Prover Loop
    solver_max_iterations: int = 8
    solver_enable_thinking: bool = True
    solver_thinking_budget: int = 24576  # matches load_tcs_config() default
    solver_temperature: float = 0.2
    solver_timeout_seconds: float = 3600.0

    @property
    def prover_max_iterations(self) -> int:
        return self.solver_max_iterations

    @property
    def prover_enable_thinking(self) -> bool:
        return self.solver_enable_thinking

    @property
    def prover_thinking_budget(self) -> int:
        return self.solver_thinking_budget

    @property
    def prover_temperature(self) -> float:
        return self.solver_temperature

    @property
    def prover_timeout_seconds(self) -> float:
        return self.solver_timeout_seconds

    # APU Kernel & Guardrails
    kernel_timeout_seconds: float = 12.0
    combinatorial_ceiling_n: int = 7

    # Subagents: Context Architect
    architect_max_tokens: int = 8192
    architect_thinking_budget: int = 2048
    architect_timeout_seconds: float = 180.0

    # Subagents: Computational Falsifier
    falsifier_max_tokens: int = 4096
    falsifier_thinking_budget: int = 1536
    falsifier_timeout_seconds: float = 120.0

    # Subagents: Adversarial Skeptic
    skeptic_max_tokens: int = 2048
    skeptic_thinking_budget: int = 1024
    skeptic_timeout_seconds: float = 90.0

    # Subagents: Publication Scribe
    scribe_max_tokens: int = 65536
    scribe_thinking_budget: int = 16384
    scribe_timeout_seconds: float = 360.0

    # High-Potential Overtime Gate
    enable_overtime: bool = True
    overtime_iterations: int = 4
    overtime_validity_threshold: float = 0.35
    overtime_seconds_per_iter: float = 450.0

    # Subagents: Proof Auditor
    auditor_max_tokens: int = 16384
    auditor_thinking_budget: int = 12288
    auditor_timeout_seconds: float = 360.0
    auditor_validity_threshold: float = 0.70
    proof_mode: str = "auto"  # "auto", "tcs", or "computational"
    auditor_kernel_timeout_seconds: float = 15.0

    # TurboVec Memory Retrieval
    turbovec_min_score: float = 0.68
    turbovec_limit: int = 5

    # Lean 4 Judge & Runner Defaults
    lean_timeout_seconds: float = 60.0

    # Continuous Pipeline Defaults
    pipeline_workers: int = 2
    pipeline_judge_config: str = "tcs"  # headline judge = TCS Judge (matches load_tcs_config())
    pipeline_judge_threshold: float = TCS_PASS_THRESHOLD
    pipeline_niceness: int = 15
    pipeline_judge_thinking: bool = True
    pipeline_judge_adaptive: bool = True


_CACHED_CONFIG: Optional[TCSConfig] = None


def load_tcs_config(env_path: Optional[Path] = None, reload: bool = False) -> TCSConfig:
    """Loads configuration strictly from .env located in the project root."""
    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None and not reload:
        return _CACHED_CONFIG

    target_env = env_path or DEFAULT_ENV_PATH
    if target_env.exists() and load_dotenv is not None:
        load_dotenv(dotenv_path=target_env, override=False)
    else:
        if load_dotenv is not None:
            load_dotenv(override=False)

    cfg = TCSConfig(
        compactor_max_context_tokens=_int_env("TCS_COMPACTOR_MAX_CONTEXT_TOKENS", 48000),
        compactor_reserve_tokens=_int_env("TCS_COMPACTOR_RESERVE_TOKENS", 16000),
        compactor_keep_recent_turns=_int_env("TCS_COMPACTOR_KEEP_RECENT_TURNS", 2),

        solver_max_iterations=_int_env("PROVER_MAX_ITERATIONS", _int_env("TCS_PROVER_MAX_ITERATIONS", _int_env("TCS_SOLVER_MAX_ITERATIONS", 8))),
        solver_enable_thinking=_bool_env("PROVER_ENABLE_THINKING", _bool_env("TCS_PROVER_ENABLE_THINKING", _bool_env("TCS_SOLVER_ENABLE_THINKING", True))),
        solver_thinking_budget=_int_env("PROVER_THINKING_BUDGET", _int_env("TCS_PROVER_THINKING_BUDGET", _int_env("TCS_SOLVER_THINKING_BUDGET", 24576))),
        solver_temperature=_float_env("PROVER_TEMPERATURE", _float_env("TCS_PROVER_TEMPERATURE", _float_env("TCS_SOLVER_TEMPERATURE", 0.2))),
        solver_timeout_seconds=_float_env("PROVER_TIMEOUT_SECONDS", _float_env("TCS_PROVER_TIMEOUT_SECONDS", _float_env("TCS_SOLVER_TIMEOUT_SECONDS", 3600.0))),

        kernel_timeout_seconds=_float_env("TCS_KERNEL_TIMEOUT_SECONDS", 12.0),
        combinatorial_ceiling_n=_int_env("TCS_COMBINATORIAL_CEILING_N", 7),

        architect_max_tokens=_int_env("TCS_ARCHITECT_MAX_TOKENS", 8192),
        architect_thinking_budget=_int_env("TCS_ARCHITECT_THINKING_BUDGET", 2048),
        architect_timeout_seconds=_float_env("TCS_ARCHITECT_TIMEOUT_SECONDS", 180.0),

        falsifier_max_tokens=_int_env("TCS_FALSIFIER_MAX_TOKENS", 4096),
        falsifier_thinking_budget=_int_env("TCS_FALSIFIER_THINKING_BUDGET", 1536),
        falsifier_timeout_seconds=_float_env("TCS_FALSIFIER_TIMEOUT_SECONDS", 120.0),

        skeptic_max_tokens=_int_env("TCS_SKEPTIC_MAX_TOKENS", 2048),
        skeptic_thinking_budget=_int_env("TCS_SKEPTIC_THINKING_BUDGET", 1024),
        skeptic_timeout_seconds=_float_env("TCS_SKEPTIC_TIMEOUT_SECONDS", 90.0),

        scribe_max_tokens=_int_env("TCS_SCRIBE_MAX_TOKENS", 65536),
        scribe_thinking_budget=_int_env("TCS_SCRIBE_THINKING_BUDGET", 16384),
        scribe_timeout_seconds=_float_env("TCS_SCRIBE_TIMEOUT_SECONDS", 360.0),

        enable_overtime=_bool_env("TCS_ENABLE_OVERTIME", True),
        overtime_iterations=_int_env("TCS_OVERTIME_ITERATIONS", 4),
        overtime_validity_threshold=_float_env("TCS_OVERTIME_VALIDITY_THRESHOLD", 0.35),
        overtime_seconds_per_iter=_float_env("TCS_OVERTIME_SECONDS_PER_ITER", 450.0),

        auditor_max_tokens=_int_env("TCS_AUDITOR_MAX_TOKENS", 16384),
        auditor_thinking_budget=_int_env("TCS_AUDITOR_THINKING_BUDGET", 12288),
        auditor_timeout_seconds=_float_env("TCS_AUDITOR_TIMEOUT_SECONDS", 360.0),
        auditor_validity_threshold=_float_env("TCS_AUDITOR_VALIDITY_THRESHOLD", 0.70),
        proof_mode=_str_env("TCS_PROOF_MODE", "auto"),
        auditor_kernel_timeout_seconds=_float_env("TCS_AUDITOR_KERNEL_TIMEOUT_SECONDS", 15.0),

        turbovec_min_score=_float_env("TCS_TURBOVEC_MIN_SCORE", 0.68),
        turbovec_limit=_int_env("TCS_TURBOVEC_LIMIT", 5),

        lean_timeout_seconds=_float_env("TCS_LEAN_TIMEOUT_SECONDS", 60.0),

        pipeline_workers=_int_env("TCS_PIPELINE_WORKERS", 2),
        pipeline_judge_config=_str_env("TCS_PIPELINE_JUDGE_CONFIG", "tcs"),
        pipeline_judge_threshold=_float_env("TCS_PIPELINE_JUDGE_THRESHOLD", TCS_PASS_THRESHOLD),
        pipeline_niceness=_int_env("TCS_PIPELINE_NICENESS", 15),
        pipeline_judge_thinking=_bool_env("TCS_PIPELINE_JUDGE_THINKING", True),
        pipeline_judge_adaptive=_bool_env("TCS_PIPELINE_JUDGE_ADAPTIVE", True),
    )
    _CACHED_CONFIG = cfg
    return cfg


# Semantic aliases for universal engine configuration
ProverConfig = TCSConfig
load_prover_config = load_tcs_config

