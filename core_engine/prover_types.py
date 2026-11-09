"""Prover Data Structures, Result Envelopes, and Candidate Pool Management.

Houses all data classes, state tracking records, and candidate pools for the
MathProver and TCS-Bench verification pipelines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .kernel import ExecutionResult

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "prover_thinking.log"


@dataclass
class AuditResult:
    """Outcome of an empirical or academic proof referee audit."""
    verdict: str  # "VALID", "REJECT", or "INCONCLUSIVE"
    issues: str
    validity: float = 0.0
    rigor: float = 0.0
    completeness: float = 0.0
    boundary_cases: float = 0.0

    @property
    def structural_validity(self) -> float:
        """Mean of structural sub-scores (rigor, completeness, boundary_cases) when available."""
        subs = [s for s in (self.rigor, self.completeness, self.boundary_cases) if s > 0.0]
        if not subs:
            return self.validity
        return max(self.validity, sum(subs) / len(subs))

    @property
    def has_substantive_structure(self) -> bool:
        """True when the referee's structural sub-scores indicate a substantive partial proof
        that should undergo surgical repair rather than a destructive clean-slate hard pivot.
        Requires genuine mathematical validity (validity >= 0.30 or structural_validity >= 0.40
        or rigor >= 0.45) in addition to structural components, preventing superficial formatting
        from blocking pivots on fundamentally flawed reductions.
        """
        if self.validity < 0.30 and self.structural_validity < 0.40 - 1e-6 and self.rigor < 0.45 - 1e-6:
            return False
        return self.completeness >= 0.35 or self.rigor >= 0.35

    @property
    def is_boundary_defect(self) -> bool:
        """True when the defect is an isolated boundary/degenerate/small-n/constant-factor issue
        while the main reduction has substantive rigor/completeness.
        """
        return is_boundary_or_local_defect(
            self.issues, self.rigor, self.completeness, self.boundary_cases
        )

    def __iter__(self):
        return iter((self.verdict, self.issues, self.validity))


@dataclass
class CandidateRecord:
    """Historical record of an individual candidate proof draft and audit evaluation."""
    proof: str
    validity: float
    verdict: str
    issues: str
    turn: int
    rigor: float = 0.0
    completeness: float = 0.0
    boundary_cases: float = 0.0
    stagnant_revisions: int = 0

    @property
    def structural_validity(self) -> float:
        """Mean of structural sub-scores (rigor, completeness, boundary_cases) when available."""
        subs = [s for s in (self.rigor, self.completeness, self.boundary_cases) if s > 0.0]
        if not subs:
            return self.validity
        return max(self.validity, sum(subs) / len(subs))

    @property
    def has_substantive_structure(self) -> bool:
        """True when structural sub-scores indicate a substantive partial proof."""
        if self.validity < 0.30 and self.structural_validity < 0.40 - 1e-6 and self.rigor < 0.45 - 1e-6:
            return False
        return self.completeness >= 0.35 or self.rigor >= 0.35

    @property
    def is_boundary_defect(self) -> bool:
        """True when the defect is an isolated boundary/degenerate/small-n/constant-factor issue."""
        return is_boundary_or_local_defect(
            self.issues, self.rigor, self.completeness, self.boundary_cases
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proof": self.proof,
            "validity": float(self.validity),
            "verdict": self.verdict,
            "issues": self.issues,
            "turn": int(self.turn),
            "rigor": float(self.rigor),
            "completeness": float(self.completeness),
            "boundary_cases": float(self.boundary_cases),
            "stagnant_revisions": int(self.stagnant_revisions),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateRecord":
        return cls(
            proof=str(data.get("proof", "")),
            validity=float(data.get("validity", 0.0)),
            verdict=str(data.get("verdict", "")),
            issues=str(data.get("issues", "")),
            turn=int(data.get("turn", 0)),
            rigor=float(data.get("rigor", 0.0)),
            completeness=float(data.get("completeness", 0.0)),
            boundary_cases=float(data.get("boundary_cases", 0.0)),
            stagnant_revisions=int(data.get("stagnant_revisions", 0)),
        )


@dataclass(frozen=True)
class ProverThresholds:
    """Single source of truth for proof-search policy thresholds (used by MathProver)."""
    early_exit_validity: float = 0.90       # Stop exploring only when the internal referee is near-certain
    turn_1_early_exit_validity: float = 0.85 # Two-pass confirmed zero-shot draft on Turn 1 with no issues can early-exit at 0.85
    audit_valid_floor: float = 0.80         # Internal referee VALID requires validity >= this (unless verdict text VALID >= 0.75)
    audit_verdict_floor: float = 0.75       # Referee explicitly says VALID and scores at least this
    pivot_threshold: float = 0.30           # Below this after repeated stagnant rejections -> abandon reduction path
    hard_pivot_floor: float = 0.05          # At/below this -> abandon immediately
    restore_margin: float = 0.10            # Restore best earlier candidate if current is worse by this margin
    judge_pass_threshold: float = 0.50      # Benchmark passing threshold for the TCS referee


THRESHOLDS = ProverThresholds()

# Issue phrases that mean the internal referee found a fatal flaw, regardless of the numeric score.
# Deliberately specific: bare "contradiction"/"circular" would match proofs by contradiction or circular graphs.
FATAL_ISSUE_KEYWORDS = (
    "fatal",
    "false claim",
    "is false",
    "invalid deduction",
    "completely missing",
    "circular reasoning",
    "circular argument",
    "self-contradictory",
)

_NEGATED_FATAL_RE = re.compile(
    r"\b(?:no|not|non|without|nothing|never|"
    r"rather\s+than(?:\s+(?:a|an|any))?|"
    r"instead\s+of(?:\s+(?:a|an|any))?|"
    r"far\s+from(?:\s+(?:a|an|any))?|"
    r"not\s+constitute(?:\s+(?:a|an|any))?)[\s-]+(?:\w+[\s-]+){0,2}fatal\b",
    re.IGNORECASE,
)


def has_fatal_issue(issues: str) -> bool:
    """True if the referee's issue text reports a fatal flaw (negation-aware: 'no fatal flaws' or 'rather than a fatal flaw' is not fatal)."""
    if not issues:
        return False
    text = _NEGATED_FATAL_RE.sub(" ", issues.lower())
    return any(kw in text for kw in FATAL_ISSUE_KEYWORDS)


_BOUNDARY_ISSUE_RE = re.compile(
    r"\b(?:"
    r"boundary\s+case|boundary\s+regime|base\s+case|degenerate\s+case|edge\s+case|"
    r"equality\s+case|small[\s-]+n\b|for\s+small\s+[a-z]\b|"
    r"when\s+[pkndrm]\s*=\s*[012]\b|for\s+[pkndrm]\s*=\s*[012]\b|"
    r"\b[pkndrm]\s*=\s*1\s+boundary|sound\s+for\s+[a-z]\s*(?:>=|\\ge|\\geq|>)\s*[12]|"
    r"main\s+[^.]{0,60}\bsound\b|directionally\s+correct"
    r")",
    re.IGNORECASE,
)


def is_boundary_or_local_defect(
    issues: str,
    rigor: float = 0.0,
    completeness: float = 0.0,
    boundary_cases: float = 0.0,
) -> bool:
    """Classifies whether an audit rejection is a Tier-2 localized boundary/degenerate/small-n
    hole on top of a viable main argument, rather than a Tier-1 fatal structural dead end.
    """
    has_boundary_kw = bool(issues and _BOUNDARY_ISSUE_RE.search(issues))
    # Subscore signature: solid rigor/completeness on the main reduction but low boundary_cases score
    subscore_boundary_gap = (
        (rigor >= 0.45 or completeness >= 0.45)
        and 0.0 < boundary_cases <= 0.25
        and max(rigor, completeness) >= boundary_cases + 0.20
    )
    if subscore_boundary_gap:
        return True
    if has_boundary_kw and (rigor >= 0.30 or completeness >= 0.30 or (rigor == 0.0 and completeness == 0.0)):
        return True
    return False


from .candidate_pool import CandidatePool, CandidatePoolManager


@dataclass
class ProofResult:
    """Complete result envelope for a theorem proving session."""
    problem: str = ""
    task_id: str = ""
    candidate_proof: Optional[str] = None
    verified: bool = False
    iterations: int = 0
    reasoning_traces: List[str] = field(default_factory=list)
    code_history: List[str] = field(default_factory=list)
    execution_history: List[ExecutionResult] = field(default_factory=list)
    retrieved_heuristics: List[str] = field(default_factory=list)
    learned_heuristics: List[str] = field(default_factory=list)
    blueprint: Optional[str] = None
    verified_facts: List[str] = field(default_factory=list)
    falsified_alerts: List[str] = field(default_factory=list)
    final_response: str = ""
    log_file: str = str(DEFAULT_LOG_PATH)
    duration_s: float = 0.0
    error: Optional[str] = None
    # Compatibility attributes with theorem proving workflows
    conjecture: str = ""
    status: str = "INCONCLUSIVE"  # "PROVEN", "DISPROVEN", "INCONCLUSIVE", "completed", "error"
    theorem: str = ""
    proof: str = ""
    verified_invariants: List[str] = field(default_factory=list)
    implementation: Optional[str] = None
    research_briefing: Optional[str] = None
    audit_verdict: Optional[str] = None
    audit_validity: Optional[float] = None
    lean_verdict: Optional[str] = None

    def __post_init__(self):
        if not self.conjecture and self.problem:
            self.conjecture = self.problem
        if not self.problem and self.conjecture:
            self.problem = self.conjecture
        if not self.proof and self.candidate_proof:
            self.proof = self.candidate_proof
        if not self.candidate_proof and self.proof:
            self.candidate_proof = self.proof
        if not self.theorem and self.task_id:
            self.theorem = self.task_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "task_id": self.task_id,
            "candidate_proof": self.candidate_proof,
            "verified": bool(self.verified),
            "iterations": int(self.iterations),
            "reasoning_traces": list(self.reasoning_traces),
            "code_history": list(self.code_history),
            "execution_history": [ex.to_dict() for ex in self.execution_history],
            "retrieved_heuristics": list(self.retrieved_heuristics),
            "learned_heuristics": list(self.learned_heuristics),
            "blueprint": self.blueprint,
            "verified_facts": list(self.verified_facts),
            "falsified_alerts": list(self.falsified_alerts),
            "final_response": self.final_response,
            "log_file": str(self.log_file),
            "duration_s": float(self.duration_s or 0.0),
            "error": self.error,
            "conjecture": self.conjecture,
            "status": self.status,
            "theorem": self.theorem,
            "proof": self.proof,
            "verified_invariants": list(self.verified_invariants),
            "implementation": self.implementation,
            "research_briefing": self.research_briefing,
            "audit_verdict": self.audit_verdict,
            "audit_validity": self.audit_validity,
            "lean_verdict": self.lean_verdict,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProofResult":
        return cls(
            problem=str(data.get("problem", "")),
            task_id=str(data.get("task_id", "")),
            candidate_proof=data.get("candidate_proof"),
            verified=bool(data.get("verified", False)),
            iterations=int(data.get("iterations", 0)),
            reasoning_traces=list(data.get("reasoning_traces") or []),
            code_history=list(data.get("code_history") or []),
            execution_history=[
                ExecutionResult.from_dict(ex) for ex in (data.get("execution_history") or [])
            ],
            retrieved_heuristics=list(data.get("retrieved_heuristics") or []),
            learned_heuristics=list(data.get("learned_heuristics") or []),
            blueprint=data.get("blueprint"),
            verified_facts=list(data.get("verified_facts") or []),
            falsified_alerts=list(data.get("falsified_alerts") or []),
            final_response=str(data.get("final_response", "")),
            log_file=str(data.get("log_file", DEFAULT_LOG_PATH)),
            duration_s=float(data.get("duration_s") or 0.0),
            error=data.get("error"),
            conjecture=str(data.get("conjecture", "")),
            status=str(data.get("status", "INCONCLUSIVE")),
            theorem=str(data.get("theorem", "")),
            proof=str(data.get("proof", "")),
            verified_invariants=list(data.get("verified_invariants") or []),
            implementation=data.get("implementation"),
            research_briefing=data.get("research_briefing"),
            audit_verdict=data.get("audit_verdict"),
            audit_validity=data.get("audit_validity"),
            lean_verdict=data.get("lean_verdict"),
        )


TCSSolveResult = ProofResult

__all__ = [
    "AuditResult",
    "CandidateRecord",
    "ProverThresholds",
    "CandidatePool",
    "CandidatePoolManager",
    "ProofResult",
    "TCSSolveResult",
    "DEFAULT_LOG_PATH",
    "has_fatal_issue",
    "is_boundary_or_local_defect",
]
