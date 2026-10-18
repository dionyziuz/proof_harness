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
        Requires genuine mathematical validity (validity >= 0.30 or structural_validity >= 0.40)
        in addition to structural components, preventing superficial formatting from blocking pivots
        on fundamentally flawed reductions.
        """
        if self.validity < 0.30 and self.structural_validity < 0.40 - 1e-6:
            return False
        return self.completeness >= 0.35 or self.rigor >= 0.35

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
        if self.validity < 0.30 and self.structural_validity < 0.40 - 1e-6:
            return False
        return self.completeness >= 0.35 or self.rigor >= 0.35

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


class CandidatePool:
    """Maintains and arbitrates candidate mathematical proofs across exploration,
    overtime, and synthesis passes. Enforces Occam's razor: equal validity favors
    the tighter/more concise proof, preventing verbose degradation.
    """

    def __init__(self):
        self._best: Optional[CandidateRecord] = None
        self._history: List[CandidateRecord] = []

    @property
    def best(self) -> Optional[CandidateRecord]:
        return self._best

    @property
    def best_validity(self) -> float:
        return self._best.validity if self._best else 0.0

    def offer(
        self,
        proof: str,
        validity: float,
        verdict: str,
        issues: str,
        turn: int,
        rigor: float = 0.0,
        completeness: float = 0.0,
        boundary_cases: float = 0.0,
        audit_result: Optional[AuditResult] = None,
    ) -> bool:
        """Evaluates a new proof draft. Returns True if this draft becomes the new best candidate."""
        if not proof or not proof.strip():
            return False

        if audit_result is not None:
            rigor = rigor or audit_result.rigor
            completeness = completeness or audit_result.completeness
            boundary_cases = boundary_cases or audit_result.boundary_cases

        record = CandidateRecord(
            proof=proof,
            validity=validity,
            verdict=verdict,
            issues=issues,
            turn=turn,
            rigor=rigor,
            completeness=completeness,
            boundary_cases=boundary_cases,
        )
        self._history.append(record)

        if self._best is None:
            self._best = record
            return True

        # 1. Same-proof re-audit demotion: if the identical proof text is re-audited on a later turn
        #    and rejected with concrete issues, update the stale score/verdict so a single-turn
        #    false positive (e.g. 0.82 VALID later rejected at 0.30 REJECT) does not freeze the pool.
        has_real_issues = bool(issues and issues.strip().lower() != "none")
        best_has_real_issues = bool(
            self._best.issues and self._best.issues.strip().lower() != "none"
        )
        if (
            verdict == "REJECT"
            and has_real_issues
            and proof.strip() == self._best.proof.strip()
        ):
            record.stagnant_revisions = self._best.stagnant_revisions + 1
            self._best = record
            for hist_rec in self._history[:-1]:
                if hist_rec.proof.strip() != proof.strip() and hist_rec.validity > self._best.validity:
                    self._best = hist_rec
            return True

        # 2. Cleanly verified candidate (VALID with no issues) supersedes an incumbent
        #    that is either REJECT or carries unresolved referee issues from a post-restore audit.
        if (
            verdict == "VALID"
            and not has_real_issues
            and (self._best.verdict != "VALID" or best_has_real_issues)
        ):
            self._best = record
            return True

        # Noise-Band Occam's Razor (epsilon = 0.03) for strong candidates (>= 0.70):
        # Single-pass LLM auditor scores fluctuate by ~0.03. Within this noise band:
        # 1. A candidate that is >25% more concise than the incumbent wins even if its raw
        #    score is up to 0.03 lower.
        # 2. A candidate that bloats length by >50% over a substantial incumbent (>= 1500 chars)
        #    cannot displace the incumbent for a marginal <= 0.03 bump.
        noise_band = 0.03
        if validity >= 0.70 and self._best.validity >= 0.70:
            diff = validity - self._best.validity
            if abs(diff) <= noise_band:
                if len(proof) < int(0.75 * len(self._best.proof)):
                    self._best = record
                    return True
                if diff > 0 and len(self._best.proof) >= 1500 and len(proof) > int(1.50 * len(self._best.proof)):
                    return False

        # Structural arbitration among boolean-gate-capped / rejected drafts (validity <= 0.45):
        # 1. Protect a high-rigor draft (e.g., rigor=0.75, structural=0.55 capped at validity=0.20)
        #    from being overwritten by a hand-waving draft (e.g., rigor=0.35, structural=0.37)
        #    that happened to hit a slightly higher boolean-gate cap (validity=0.25).
        # 2. When two rejected drafts hit the same boolean-gate cap (e.g. 0.20 == 0.20 or 0.35 == 0.35),
        #    prefer the draft with higher structural_validity (or the newer surgical revision on ties).
        both_capped_reject = (
            verdict != "VALID"
            and self._best.verdict != "VALID"
            and validity <= 0.45
            and self._best.validity <= 0.45
        )
        has_subscores = (
            record.structural_validity > record.validity
            or self._best.structural_validity > self._best.validity
        )
        if both_capped_reject and has_subscores:
            if (
                self._best.structural_validity >= record.structural_validity + 0.08
                and self._best.rigor >= record.rigor + 0.15
                and validity <= self._best.validity + 0.05
            ):
                self._best.stagnant_revisions += 1
                return False
            if (
                record.structural_validity > self._best.structural_validity + 0.01
                and (
                    validity >= self._best.validity
                    or (
                        validity >= self._best.validity - 0.05
                        and record.rigor >= self._best.rigor + 0.25
                        and record.structural_validity >= self._best.structural_validity + 0.12
                    )
                )
            ):
                self._best = record
                return True
            if (
                abs(validity - self._best.validity) <= 0.01
                and abs(record.structural_validity - self._best.structural_validity) <= 0.02
                and turn >= self._best.turn
                and len(proof) >= int(0.70 * len(self._best.proof))
            ):
                record.stagnant_revisions = self._best.stagnant_revisions + 1
                self._best = record
                return True

        # Strict improvement
        if validity > self._best.validity:
            self._best = record
            return True

        # Equal validity (when no structural sub-scores differentiate the drafts):
        # For passing/viable candidates (validity >= 0.50), Occam's razor favors the tighter proof.
        if validity == self._best.validity and validity >= 0.50 and len(proof) < len(self._best.proof):
            if verdict != "VALID":
                record.stagnant_revisions = self._best.stagnant_revisions + 1
            self._best = record
            return True

        if self._best.verdict != "VALID":
            self._best.stagnant_revisions += 1
        return False

    def demote_current_best(self, max_validity: float = 0.10) -> bool:
        """Demotes the current best candidate's validity and structural sub-scores when an
        empirical/REPL counterexample confirms a mathematical falsehood in the incumbent draft,
        ensuring subsequent repaired revisions can supersede it.
        """
        if self._best is not None and self._best.validity > max_validity:
            self._best.validity = max_validity
            self._best.rigor = min(self._best.rigor, max_validity)
            self._best.completeness = min(self._best.completeness, max_validity)
            return True
        return False

    def prune_abandoned(self, max_validity_floor: float = 0.25, force: bool = False) -> Optional[CandidateRecord]:
        """Prunes the current best candidate if it belongs to an abandoned/refuted reduction
        (i.e. validity < max_validity_floor and either force=True on hard pivot or lacks substantive structure).
        Returns the pruned candidate if one was removed.
        """
        if self._best and self._best.validity < max_validity_floor and (force or not self._best.has_substantive_structure):
            pruned = self._best
            self._best = None
            return pruned
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best": self._best.to_dict() if self._best is not None else None,
            "history": [r.to_dict() for r in self._history],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidatePool":
        pool = cls()
        if not data:
            return pool
        pool._history = [CandidateRecord.from_dict(r) for r in (data.get("history") or [])]
        best_data = data.get("best")
        if best_data:
            pool._best = CandidateRecord.from_dict(best_data)
        return pool


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
    "ProofResult",
    "TCSSolveResult",
    "DEFAULT_LOG_PATH",
]
