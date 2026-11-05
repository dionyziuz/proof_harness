"""Tests for Aletheia-inspired audit enhancements.

Tests:
1. Deterministic Singularity & Zero-Denominator Pre-Check.
2. Literature Citation Explication Gate.
3. Structured Counterexample Witness Recording in MathStateMachine.
4. Counterexample Invariant Gate blocking refuted premises.
5. Checkpoint & Rollback preservation of counterexample witnesses.
"""

import pytest
from core_engine.prover_auditor import ProverAuditorMixin
from core_engine.prover_types import AuditResult
from core_engine.state_machine import MathStateMachine, StateInvariantViolation


class MockAuditor(ProverAuditorMixin):
    pass


def test_detect_singularities_unguarded():
    auditor = MockAuditor()
    proof_unguarded = r"""
    \begin{proof}
    Consider the sequence defined by $a_n = \frac{n^2 + 5}{n - 1}$.
    Since $n^2 + 5 > 0$, the sequence is strictly positive.
    \end{proof}
    """
    singularities = auditor._detect_singularities_and_zero_denominators(proof_unguarded, "Target Problem: For all n >= 1")
    assert len(singularities) >= 1
    assert any(s["symbol"] == "n" and s["vanishing_root"] == 1 for s in singularities)


def test_detect_singularities_guarded():
    auditor = MockAuditor()
    proof_guarded = r"""
    \begin{proof}
    For $n = 1$, we explicitly set $a_1 = 0$.
    For $n \ge 2$, we define $a_n = \frac{n^2 + 5}{n - 1}$.
    Since $n \ge 2$, the denominator is strictly positive and non-zero.
    \end{proof}
    """
    singularities = auditor._detect_singularities_and_zero_denominators(proof_guarded, "Target Problem: For all n >= 1")
    assert len(singularities) == 0


def test_detect_unverified_citations():
    auditor = MockAuditor()
    proof_unverified = r"""
    \begin{proof}
    By \cite{Galambos1976}, the prime factors satisfy $p_j > p_1 \cdots p_{j-1}$.
    Therefore the theorem holds.
    \end{proof}
    """
    flagged = auditor._detect_unverified_citations(proof_unverified)
    assert len(flagged) >= 1

    proof_verified = r"""
    \begin{proof}
    By \cite{Galambos1976}, a sequence of prime factors satisfies the bound under the condition of asymptotic density.
    We explicitly verify that the hypotheses of the theorem are satisfied by our choice of $p_j$.
    Therefore the theorem holds.
    \end{proof}
    """
    flagged_ok = auditor._detect_unverified_citations(proof_verified)
    assert len(flagged_ok) == 0


def test_state_machine_counterexample_recording():
    sm = MathStateMachine()
    sm.record_lemma("Lemma 1: For all k, R(k) <= 2^k")
    assert len(sm._verified_lemmas) == 1

    # Record counterexample discovered by AuditorKernel
    witness = {"k": 3, "actual_value": 9, "bound": 8}
    record = sm.record_counterexample(
        claim="R(k) <= 2^k",
        witness=witness,
        source="AuditorKernel",
        error="AssertionError: actual 9 > bound 8 at k=3"
    )

    assert record["claim"] == "R(k) <= 2^k"
    assert sm.has_counterexample("R(k) <= 2^k")
    # Verify the refuted lemma was automatically revoked from verified_lemmas
    assert len(sm._verified_lemmas) == 0

    # Verify status report displays the counterexample
    status_rep = sm.status()
    assert "Refuted Claims & Counterexamples" in status_rep
    assert "Witness:" in status_rep


def test_state_machine_counterexample_blocks_assertion():
    sm = MathStateMachine()
    sm.register("G", "graph", properties=["bipartite"])

    # Record counterexample for 'connected'
    sm.record_counterexample(
        claim="connected",
        witness={"components": 2},
        source="AuditorKernel",
        error="Graph has 2 components"
    )

    # Trying to assert 'connected' must raise StateInvariantViolation
    with pytest.raises(StateInvariantViolation) as excinfo:
        sm.assert_property("G", "connected")
    assert "was REFUTED by AuditorKernel" in str(excinfo.value)


def test_state_machine_checkpoint_and_rollback_with_counterexamples():
    sm = MathStateMachine()
    sm.register("n", "integer", properties=["positive"])
    sm.checkpoint("base")

    # Add counterexample
    sm.record_counterexample("n is even", witness={"n": 3})
    assert sm.has_counterexample("n is even")

    # Rollback to base checkpoint (which had no counterexamples)
    sm.rollback("base")
    assert not sm.has_counterexample("n is even")


def test_state_machine_serialization_with_counterexamples():
    sm = MathStateMachine()
    sm.register("X", "random_variable")
    sm.record_counterexample("X is gaussian", witness={"kurtosis": 5.2})

    d = sm.to_dict()
    assert "counterexample_witnesses" in d
    assert len(d["counterexample_witnesses"]) == 1

    sm2 = MathStateMachine()
    sm2.load_state_dict(d)
    assert sm2.has_counterexample("X is gaussian")
    assert len(sm2.get_counterexamples()) == 1


def test_audit_result_carries_witness():
    res = AuditResult(
        verdict="REJECT",
        issues="Assertion failure at n=1",
        validity=0.20,
        counterexample_witness={"n": 1, "value": 0}
    )
    assert res.counterexample_witness == {"n": 1, "value": 0}
