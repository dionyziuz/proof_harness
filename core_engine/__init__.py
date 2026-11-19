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
    ProverSolveResult,
    TCSSolveResult,
    CandidatePool,
    CandidateRecord,
    AuditResult,
    ProverThresholds,
)
from .prover_kernel import ProverCompliantKernel, SandboxedMathKernel, TCSCompliantKernel, AuditorKernel
from .prover_config import (
    ProverConfig,
    load_prover_config,
    TCSConfig,
    load_tcs_config,
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
from .lean_prover import (
    ALLOWED_CORE_AXIOMS,
    LeanCompliantKernel,
    LeanKernel,
    LeanKernelValidationResult,
    LeanProver,
    extract_lean_declarations,
    strip_lean_comments,
    verify_lean4_kernel,
)
from .lean_harness_guards import (
    check_signature_drift,
    classify_lean_failure_triage,
    extract_declaration_signatures,
    extract_library_index,
    format_compact_library_toc,
    format_computational_witness_briefing,
    format_obstructions_briefing,
    load_computational_witnesses,
    load_obstructions,
    record_cycle_and_check_stall,
    register_obstruction,
    retrieve_cached_arxiv_briefing,
    save_computational_witness,
    search_library_signatures,
    snapshot_and_verify_library_signatures,
    write_library_index_markdown,
)
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
    "LeanProver",
    "LeanKernel",
    "LeanCompliantKernel",
    "LeanKernelValidationResult",
    "verify_lean4_kernel",
    "ALLOWED_CORE_AXIOMS",
    "extract_lean_declarations",
    "strip_lean_comments",
    "ProofResult",
    "ProverSolveResult",
    "TCSSolveResult",
    "CandidatePool",
    "CandidateRecord",
    "AuditResult",
    "ProverThresholds",
    "ProverCompliantKernel",
    "SandboxedMathKernel",
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
    "check_signature_drift",
    "classify_lean_failure_triage",
    "extract_declaration_signatures",
    "extract_library_index",
    "format_compact_library_toc",
    "format_computational_witness_briefing",
    "format_obstructions_briefing",
    "load_computational_witnesses",
    "load_obstructions",
    "record_cycle_and_check_stall",
    "register_obstruction",
    "retrieve_cached_arxiv_briefing",
    "save_computational_witness",
    "search_library_signatures",
    "snapshot_and_verify_library_signatures",
    "write_library_index_markdown",
]


