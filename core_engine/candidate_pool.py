"""Candidate Pool Manager for Prime Math Prover.

Maintains and arbitrates candidate mathematical proofs across exploration,
overtime, and synthesis passes. Enforces Occam's razor: equal validity favors
the tighter/more concise proof, preventing verbose degradation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .prover_types import (
    AuditResult,
    CandidateRecord,
    THRESHOLDS,
)

logger = logging.getLogger("prime_math.candidate_pool")


class CandidatePool:
    """Maintains and arbitrates candidate mathematical proofs across exploration,
    overtime, and synthesis passes. Enforces Occam's razor: equal validity favors
    the tighter/more concise proof, preventing verbose degradation.
    """

    def __init__(self):
        self._best: Optional[CandidateRecord] = None
        self._history: List[CandidateRecord] = []
        self._archived: List[CandidateRecord] = []

    @property
    def best(self) -> Optional[CandidateRecord]:
        return self._best

    @property
    def best_validity(self) -> float:
        return self._best.validity if self._best else 0.0

    @property
    def history(self) -> List[CandidateRecord]:
        return list(self._history)

    @property
    def archived(self) -> List[CandidateRecord]:
        return list(self._archived)

    @property
    def is_stagnant_rejected(self) -> bool:
        """True when the current best candidate is rejected and has stagnated for >= 2 revisions."""
        return bool(
            self._best
            and self._best.verdict != "VALID"
            and getattr(self._best, "stagnant_revisions", 0) >= 2
        )

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

    def has_viable_foundation(self, rejection_count: int = 0) -> bool:
        """Determines if the pool's current best candidate represents a viable
        mathematical foundation suitable for surgical revision rather than abandoning
        via a Clean-Slate Hard Pivot.
        """
        if self._best is None:
            return False
        best_stagnant = getattr(self._best, "stagnant_revisions", 0)
        return bool(
            self._best.verdict == "VALID"
            or self.best_validity >= THRESHOLDS.pivot_threshold
            or getattr(self._best, "is_boundary_defect", False)
            or (
                self._best.has_substantive_structure
                and self._best.structural_validity >= 0.40
                and rejection_count <= 2
                and best_stagnant < 2
            )
        )

    def should_restore_for_revision(
        self,
        current_validity: float,
        current_structural: float,
        turn: int,
        max_turns: int,
        rejection_count: int = 0,
        overtime_triggered_now: bool = False,
    ) -> bool:
        """Determines if pool.best should supersede the current turn's lower-scoring candidate
        for the upcoming revision cycle.
        """
        if self._best is None:
            return False

        stagnant_rejected_loop = (
            self._best.verdict != "VALID"
            and getattr(self._best, "stagnant_revisions", 0) >= 2
        )
        can_restore_best = bool(
            self._best.verdict == "VALID"
            or self.best_validity >= THRESHOLDS.audit_verdict_floor
            or not stagnant_rejected_loop
        )
        if not can_restore_best:
            return False

        has_viable_quality = (
            self.best_validity >= THRESHOLDS.judge_pass_threshold
            or (self._best.structural_validity >= 0.45 and rejection_count <= 2)
        )
        if not has_viable_quality:
            return False

        return (
            overtime_triggered_now
            or (turn >= max_turns and (current_validity < self.best_validity or current_structural < self._best.structural_validity))
            or (self.best_validity > current_validity + THRESHOLDS.restore_margin)
            or (self._best.structural_validity > current_structural + THRESHOLDS.restore_margin and self.best_validity >= current_validity)
        )

    def can_deep_stagnation_pivot(
        self,
        rejection_count: int,
        stagnation_pivot_turns: int,
        turns_remaining: int,
        is_improving: bool,
        boundary_defect: bool = False,
        overtime_triggered_now: bool = False,
        audit_inconclusive: bool = False,
    ) -> bool:
        """Determines whether deep stagnation triggers a Clean-Slate Hard Pivot."""
        if audit_inconclusive or overtime_triggered_now or boundary_defect or turns_remaining < 2:
            return False
        best_stagnant = getattr(self._best, "stagnant_revisions", 0) if self._best else 0
        if rejection_count < stagnation_pivot_turns or best_stagnant < max(2, stagnation_pivot_turns - 2) or is_improving:
            return False
        return self._best is None or (
            self._best.verdict != "VALID"
            and self.best_validity < THRESHOLDS.audit_verdict_floor
        )

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
        (i.e. validity < max_validity_floor and lacks substantive structure or boundary-only protection).
        Never wipes out a candidate whose only defect is a Tier-2 boundary/degenerate case
        unless validity <= hard_pivot_floor (0.05).
        Returns the pruned candidate if one was removed.
        """
        if self._best and self._best.validity < max_validity_floor:
            if self._best.validity > THRESHOLDS.hard_pivot_floor and self._best.is_boundary_defect:
                return None
            if force or not self._best.has_substantive_structure:
                pruned = self._best
                self._best = None
                return pruned
        return None

    def archive_for_pivot(self) -> Optional[CandidateRecord]:
        """Archives the current branch's best candidate into cross-pivot memory and clears
        the active branch best so a fresh Clean-Slate Hard Pivot can explore a new proof
        architecture from scratch without losing the best prior draft if later pivots fail.
        """
        if self._best is not None:
            archived = self._best
            self._archived.append(archived)
            self._best = None
            self._history.clear()
            return archived
        return None

    def restore_best_from_archive(self) -> Optional[CandidateRecord]:
        """Restores the highest-scoring candidate across all pivot branches when exploratory
        turns finish without an active candidate that surpasses earlier archived branches.
        """
        if not self._archived:
            return self._best
        candidates = list(self._archived)
        if self._best is not None:
            candidates.append(self._best)
        best_overall = max(
            candidates,
            key=lambda c: (
                1 if c.verdict == "VALID" else 0,
                round(c.validity, 4),
                round(c.structural_validity, 4),
                -len(c.proof),
            ),
        )
        self._best = best_overall
        return self._best

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best": self._best.to_dict() if self._best else None,
            "history": [r.to_dict() for r in self._history],
            "archived": [r.to_dict() for r in self._archived],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidatePool":
        pool = cls()
        if not data:
            return pool
        pool._history = [CandidateRecord.from_dict(r) for r in (data.get("history") or [])]
        pool._archived = [CandidateRecord.from_dict(r) for r in (data.get("archived") or [])]
        best_data = data.get("best")
        if best_data:
            pool._best = CandidateRecord.from_dict(best_data)
        return pool


CandidatePoolManager = CandidatePool

__all__ = ["CandidatePool", "CandidatePoolManager"]
