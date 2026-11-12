"""Backward-compatible configuration shim forwarding to `core_engine.prover_config`."""

from __future__ import annotations

from core_engine.prover_config import (
    BORDERLINE_BAND,
    DEFAULT_ENV_PATH,
    DEFAULT_PASS_THRESHOLD,
    DEFAULT_TCS_PASS_THRESHOLD,
    JUDGE_REPLICATES,
    PASS_THRESHOLD,
    PROJECT_ROOT,
    ProverConfig,
    TCS_BORDERLINE_BAND,
    TCS_JUDGE_REPLICATES,
    TCS_PASS_THRESHOLD,
    TCSConfig,
    _bool_env,
    _first_env,
    _float_env,
    _int_env,
    _str_env,
    load_prover_config,
    load_tcs_config,
)

__all__ = [
    "BORDERLINE_BAND",
    "DEFAULT_ENV_PATH",
    "DEFAULT_PASS_THRESHOLD",
    "DEFAULT_TCS_PASS_THRESHOLD",
    "JUDGE_REPLICATES",
    "PASS_THRESHOLD",
    "PROJECT_ROOT",
    "ProverConfig",
    "TCS_BORDERLINE_BAND",
    "TCS_JUDGE_REPLICATES",
    "TCS_PASS_THRESHOLD",
    "TCSConfig",
    "load_prover_config",
    "load_tcs_config",
]
