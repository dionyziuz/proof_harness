"""Tests for Lean 4 runner, judge, and kernel integration."""

import pytest
from core_engine.lean_runner import find_lean_binary, get_lean_version, run_lean
from core_engine.judging import (
    judge_lean_proof,
    extract_lean_code,
    verify_lean4_kernel,
    ALLOWED_CORE_AXIOMS,
)
from core_engine.lean_prover import LeanKernel, LeanProver
from core_engine.kernel import PersistentMathKernel
from core_engine.prover_kernel import TCSCompliantKernel


def test_lean_binary_and_version():
    binary = find_lean_binary()
    assert binary is not None, "Lean binary should be discovered via elan or PATH"
    version = get_lean_version(binary)
    assert version is not None
    assert "Lean" in version


def test_run_lean_valid():
    code = """
theorem nat_id (n : Nat) : n = n := rfl
"""
    res = run_lean(code)
    assert res.success is True
    assert res.returncode == 0
    assert len(res.errors) == 0
    assert res.has_sorry is False


def test_run_lean_syntax_error():
    code = """
theorem bad : 1 = 2 := by rfl
"""
    res = run_lean(code)
    assert res.success is False
    assert len(res.errors) > 0


def test_run_lean_detects_sorry():
    code = """
theorem cheated (a b : Nat) : a + b = b + a := by
  sorry
"""
    res = run_lean(code)
    assert res.has_sorry is True
    assert res.success is False


def test_judge_lean_proof_markdown():
    markdown = """
Here is the formal theorem:
```lean
def square (n : Nat) : Nat := n * n
theorem square_zero : square 0 = 0 := rfl
```
Hope this is helpful!
"""
    rec = judge_lean_proof(markdown)
    assert rec["passed"] is True
    assert rec["verdict"] == "1"
    assert rec["status"] == "PASS"


def test_judge_lean_proof_rejects_sorry():
    markdown = """
```lean
theorem unproven (n : Nat) : n = 0 := by sorry
```
"""
    rec = judge_lean_proof(markdown)
    assert rec["passed"] is False
    assert rec["verdict"] == "0"
    assert rec["status"] == "SORRY_DETECTED"
    assert rec["has_sorry"] is True


def test_judge_lean_proof_no_code():
    text = "Just informal handwaving without any Lean code."
    rec = judge_lean_proof(text)
    assert rec["passed"] is False
    assert rec["status"] == "NO_LEAN_CODE"


def test_kernel_lean_check_integration():
    kernel = PersistentMathKernel()
    py_code = """
proof = "theorem add_comm_test (a b : Nat) : a + b = b + a := Nat.add_comm a b"
result = lean_check(proof)
"""
    res = kernel.execute(py_code)
    assert res.success is True
    assert kernel.namespace["result"]["success"] is True
    assert kernel.namespace["result"]["has_sorry"] is False


def test_tcs_compliant_kernel_allows_lean():
    kernel = TCSCompliantKernel()
    py_code = """
proof = "theorem id_test (x : Nat) : x = x := rfl"
result = lean_check(proof)
"""
    res = kernel.execute(py_code, timeout_seconds=30.0)
    assert res.success is True
    assert kernel.namespace["result"]["success"] is True


def test_verify_lean4_kernel_5gate_and_print_axioms():
    code = """
import Std

def doubleNat (n : Nat) : Nat := n + n

theorem doubleNat_pos (n : Nat) (h : n > 0) : doubleNat n >= 2 := by
  unfold doubleNat
  omega
"""
    val = verify_lean4_kernel(code, required_theorems=["doubleNat", "doubleNat_pos"])
    assert val.passed is True
    assert val.verdict == "VALID"
    assert val.status == "CERTIFIED_HIT"
    assert "doubleNat_pos" in val.theorems_found
    assert "doubleNat_pos" in val.axiom_report
    assert set(val.axiom_report["doubleNat_pos"]).issubset(ALLOWED_CORE_AXIOMS)


def test_verify_lean4_kernel_rejects_custom_axiom():
    code = """
import Std

def myVal (n : Nat) : Nat := n + 1

axiom fake_axiom : ∀ n : Nat, myVal n = 0

theorem cheated_thm (n : Nat) : myVal n = 0 := fake_axiom n
"""
    val = verify_lean4_kernel(code, required_theorems=["cheated_thm"])
    assert val.passed is False
    assert val.verdict == "REJECT"


def test_lean_kernel_repl_integration():
    kernel = LeanKernel(required_theorems=["sq_zero"])
    py_code = """
lean_src = '''
import Std
def sq (n : Nat) : Nat := n * n
theorem sq_zero : sq 0 = 0 := rfl
'''
res = verify_lean4_kernel(lean_src)
"""
    exec_res = kernel.execute(py_code, timeout_seconds=30.0)
    assert exec_res.success is True
    assert kernel.last_verified_lean_result is not None
    assert kernel.last_verified_lean_result.passed is True

