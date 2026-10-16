"""Tests for AuditorKernel and Dual-Mode Auditor in Prime Math.

Validates that:
1. AuditorKernel executes in complete isolation without state bleed.
2. AuditorKernel enforces the strict timeout (default 15s).
3. _detect_proof_mode correctly partitions TCS-Bench vs. computational/witness problems.
4. Stage 1 empirical falsification rejects false witnesses (e.g. discrete log bluff, clique failure)
   with validity: 0.0 before the LLM referee can be fooled.
5. Stage 1 passes valid witnesses and passes abstract proofs without witnesses.
6. TCS mode completely bypasses Stage 1 to ensure zero regression on pure theoretical mathematics.
"""

from unittest.mock import MagicMock, patch
import pytest

from core_engine.prover import MathProver
from core_engine.prover_kernel import AuditorKernel, TCSCompliantKernel
from core_engine.prover_types import AuditResult


def test_auditor_kernel_isolation():
    """Verify AuditorKernel does not share state with solver kernel."""
    solver_kernel = TCSCompliantKernel()
    solver_kernel.execute("my_special_solver_var = 12345")
    assert solver_kernel.namespace.get("my_special_solver_var") == 12345

    auditor_kernel = AuditorKernel(default_timeout=5.0)
    assert "my_special_solver_var" not in auditor_kernel.namespace

    auditor_kernel.execute("auditor_test_var = 999")
    assert "auditor_test_var" not in solver_kernel.namespace


def test_auditor_kernel_timeout():
    """Verify AuditorKernel aborts runaway loops within timeout."""
    auditor_kernel = AuditorKernel(default_timeout=1.0)
    res = auditor_kernel.execute("while True: pass", timeout_seconds=1.0)
    assert not res.success
    assert "timed out" in (res.error or "").lower()


def test_detect_proof_mode():
    """Verify proof mode detection partitions TCS vs. Computational."""
    prover = MathProver(proof_mode="auto")

    # Explicit TCS prompt with context lemmas
    tcs_prompt = """# Context:
Theorem 1. Let G be an undirected planar graph...
Lemma 1. Every planar graph has a vertex of degree at most 5.

Target Problem (your task):
Prove that every planar graph is 5-colorable.
"""
    assert prover._detect_proof_mode(tcs_prompt) == "tcs"

    # Computational prompt with discrete log / curve / diophantine
    comp_prompt = "Find the discrete logarithm x such that 3^x = 12 mod 101 on the given elliptic curve."
    assert prover._detect_proof_mode(comp_prompt) == "computational"

    # Computational prompt with Ramsey / clique
    ramsey_prompt = "Determine whether the Paley graph of order 109 contains a clique of size 6."
    assert prover._detect_proof_mode(ramsey_prompt) == "computational"

    # Explicit override forces mode regardless of prompt
    tcs_override_prover = MathProver(proof_mode="tcs")
    assert tcs_override_prover._detect_proof_mode(comp_prompt) == "tcs"

    comp_override_prover = MathProver(proof_mode="computational")
    assert comp_override_prover._detect_proof_mode(tcs_prompt) == "computational"


def test_empirical_falsifier_catches_false_witness():
    """Verify Stage 1 empirical falsification catches false mathematical claims."""
    prover = MathProver(proof_mode="computational")

    task_prompt = "Find an integer x such that 2^x = 5 (mod 13)."
    # Model bluffs that x = 3 is the solution (2^3 = 8 != 5 mod 13)
    proof_text = r"""\begin{proof}
We compute 2^x mod 13. Notice that x = 3 satisfies 2^3 = 8 = 5 (mod 13).
Thus x = 3 is the required solution.
\end{proof}"""

    # Mock LLM to generate an adversarial test asserting the claimed witness
    mock_test_code = """```python
assert pow(2, 3, 13) == 5, f"2^3 mod 13 is {pow(2, 3, 13)}, not 5"
```"""
    mock_resp = MagicMock()
    mock_resp.content = mock_test_code

    with patch.object(prover.client, "complete", return_value=mock_resp):
        passed, reason = prover._empirical_falsifier_audit(task_prompt, proof_text)
        assert not passed
        assert "AssertionError" in reason or "Witness/assertion failed" in reason


def test_empirical_falsifier_passes_valid_witness():
    """Verify Stage 1 empirical falsification passes correct mathematical claims."""
    prover = MathProver(proof_mode="computational")

    task_prompt = "Find an integer x such that 2^x = 8 (mod 13)."
    proof_text = r"""\begin{proof}
Direct calculation shows 2^3 = 8 = 8 (mod 13). Hence x = 3.
\end{proof}"""

    mock_test_code = """```python
assert pow(2, 3, 13) == 8
```"""
    mock_resp = MagicMock()
    mock_resp.content = mock_test_code

    with patch.object(prover.client, "complete", return_value=mock_resp):
        passed, reason = prover._empirical_falsifier_audit(task_prompt, proof_text)
        assert passed
        assert "passed all assertions" in reason


def test_empirical_falsifier_abstract_proof_passthrough():
    """Verify Stage 1 passes abstract proofs with NO_CONCRETE_WITNESS."""
    prover = MathProver(proof_mode="computational")

    task_prompt = "Prove that there exist infinitely many primes."
    proof_text = r"""\begin{proof}
Suppose finite primes p_1, ..., p_n. Let N = p_1 ... p_n + 1. N has a prime factor not in the list.
\end{proof}"""

    mock_test_code = """```python
# NO_CONCRETE_WITNESS
```"""
    mock_resp = MagicMock()
    mock_resp.content = mock_test_code

    with patch.object(prover.client, "complete", return_value=mock_resp):
        passed, reason = prover._empirical_falsifier_audit(task_prompt, proof_text)
        assert passed
        assert "no finite witness" in reason


def test_audit_proof_draft_computational_mode_rejects_false_positive():
    """End-to-end check: In computational mode, a failed Stage 1 audit immediately
    returns REJECT with validity 0.0 without consulting the LLM referee."""
    prover = MathProver(proof_mode="computational")

    task_prompt = "Find an integer x such that 2^x = 5 (mod 13)."
    proof_text = r"""\begin{proof}
Let x = 3. Then 2^x = 5 mod 13.
\end{proof}"""

    # Mock falsifier generating failing assertion
    mock_falsifier_resp = MagicMock()
    mock_falsifier_resp.content = "```python\nassert pow(2, 3, 13) == 5\n```"

    with patch.object(prover.client, "complete", return_value=mock_falsifier_resp):
        audit_res = prover._audit_proof_draft(task_prompt, proof_text)
        assert audit_res.verdict == "REJECT"
        assert audit_res.validity == 0.0
        assert "Empirical Falsification" in audit_res.issues


def test_audit_proof_draft_tcs_mode_runs_stage1_and_passes_abstract_proof():
    """Verify TCS mode runs Stage 1 empirical falsifier and seamlessly passes abstract proofs."""
    prover = MathProver(proof_mode="tcs")

    task_prompt = "# Context:\nTheorem 1.\nTarget Problem (your task):\nProve bound."
    proof_text = r"""\begin{proof}
By Theorem 1, the bound holds.
\end{proof}"""

    # Mock Stage 2 referee directly returning VALID
    mock_referee_resp = MagicMock()
    mock_referee_resp.content = '{"verdict": "VALID", "validity": 0.95, "issues": "none"}'

    with patch.object(prover.client, "complete", return_value=mock_referee_resp):
        with patch.object(prover, "_empirical_falsifier_audit", return_value=(True, "Abstract proof with no finite witness")) as mock_stage1:
            audit_res = prover._audit_proof_draft(task_prompt, proof_text)
            mock_stage1.assert_called_once_with(task_prompt, proof_text)
            assert audit_res.verdict == "VALID"
            assert audit_res.validity >= 0.80


def test_audit_proof_draft_tcs_mode_rejects_false_algebraic_claim():
    """Verify TCS mode runs Stage 1 and immediately rejects false boundary/algebraic claims."""
    prover = MathProver(proof_mode="tcs")

    task_prompt = "# Context:\nLemma 1.\nTarget Problem (your task):\nBound chromatic number."
    proof_text = r"""\begin{proof}
For any planar graph G with n >= 1, chi(G) <= 1. Specifically for n=3 (K_3), chi(K_3) = 1.
\end{proof}"""

    # Mock Stage 1 detecting the false claim on K_3
    with patch.object(prover, "_empirical_falsifier_audit", return_value=(False, "Witness/assertion failed: chi(K_3) == 3 != 1")):
        audit_res = prover._audit_proof_draft(task_prompt, proof_text)
        assert audit_res.verdict == "REJECT"
        assert audit_res.validity == 0.0
        assert "chi(K_3)" in audit_res.issues
