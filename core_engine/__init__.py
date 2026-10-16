"""Prime Math: Agentic mathematical reasoning and persistent Python execution harness.

Built around a persistent Python execution kernel with symbolic solvers (SymPy, Z3,
mpmath) and a closed-loop CoALA decision cycle powered by OpenAI-compatible thinking models.
"""

__version__ = "0.1.0"

from .kernel import PersistentMathKernel, ExecutionResult
from .solver import MathSolver, SolveResult
from .prover import MathProver
from .prover_types import (
    ProofResult,
    TCSSolveResult,
    CandidatePool,
    CandidateRecord,
    AuditResult,
    ProverThresholds,
)
from .prover_kernel import TCSCompliantKernel, AuditorKernel
from .tcs_config import (
    TCSConfig,
    load_tcs_config,
    ProverConfig,
    load_prover_config,
)
from .falsifier import MathFalsifier, FalsificationResult

from .discoverer import MathDiscoverer, DiscoveryResult
from .config import PrimeMathConfig, load_config
from .llm_client import TunnelThinkingClient
from .memory import CoALAMemoryManager, TurboVecUnavailableError
from .remediation import (
    RemediationEngine,
    RemediationTier,
    Tier2Rubric,
    ProblemDiagnostic,
    RemediationReport,
)
from .prompts import (
    MATH_SYSTEM_PROMPT,
    MATH_PROVER_SYSTEM_PROMPT,
    MATH_FALSIFIER_SYSTEM_PROMPT,
    MATH_DISCOVERER_SYSTEM_PROMPT,
    SHARED_MATH_ENVIRONMENT,
    SHARED_COGNITIVE_DISCIPLINE,
)

from .lean_runner import run_lean, LeanRunResult, find_lean_binary, get_lean_version
from .judging import (
    judge_lean_proof,
    LeanJudgeRecord,
    judge_tcs_proof,
    judge_proof,
    build_a_prompt,
    build_b_prompt,
)

__all__ = [
    "PersistentMathKernel",
    "ExecutionResult",
    "MathSolver",
    "SolveResult",
    "MathProver",
    "ProofResult",
    "TCSSolveResult",
    "CandidatePool",
    "CandidateRecord",
    "AuditResult",
    "ProverThresholds",
    "TCSCompliantKernel",
    "AuditorKernel",
    "MathFalsifier",
    "FalsificationResult",
    "MathDiscoverer",
    "DiscoveryResult",
    "PrimeMathConfig",
    "load_config",
    "TCSConfig",
    "load_tcs_config",
    "ProverConfig",
    "load_prover_config",
    "TunnelThinkingClient",
    "CoALAMemoryManager",
    "TurboVecUnavailableError",
    "RemediationEngine",
    "RemediationTier",
    "Tier2Rubric",
    "ProblemDiagnostic",
    "RemediationReport",
    "MATH_SYSTEM_PROMPT",
    "MATH_PROVER_SYSTEM_PROMPT",
    "MATH_FALSIFIER_SYSTEM_PROMPT",
    "MATH_DISCOVERER_SYSTEM_PROMPT",
    "SHARED_MATH_ENVIRONMENT",
    "SHARED_COGNITIVE_DISCIPLINE",
    "run_lean",
    "LeanRunResult",
    "find_lean_binary",
    "get_lean_version",
    "judge_lean_proof",
    "LeanJudgeRecord",
    "judge_tcs_proof",
    "judge_proof",
    "build_a_prompt",
    "build_b_prompt",
]


