"""Unit tests for the 3-Tier Post-Evaluation Remediation Engine."""

import json
from unittest.mock import MagicMock

import pytest

from core_engine.remediation import (
    ProblemDiagnostic,
    RemediationEngine,
    RemediationTier,
    Tier2Rubric,
    extract_json_payload,
)


def test_extract_json_payload():
    # 1. Plain JSON
    assert extract_json_payload('{"key": "value"}') == {"key": "value"}

    # 2. Markdown fenced JSON
    md = """Here is the diagnosis:
```json
{
  "target_tier": "TIER_3_PROCEDURAL",
  "root_cause": "Missed boundary condition"
}
```
Hope this helps!"""
    data = extract_json_payload(md)
    assert data is not None
    assert data["target_tier"] == "TIER_3_PROCEDURAL"
    assert data["root_cause"] == "Missed boundary condition"

    # 3. JSON with <think> reasoning tags
    with_think = """<think>Let me ponder the failure...
I should propose a tier 1 fix.</think>
```json
{"target_tier": "TIER_1_HARNESS", "root_cause": "Kernel timeout"}
```"""
    data2 = extract_json_payload(with_think)
    assert data2 is not None
    assert data2["target_tier"] == "TIER_1_HARNESS"

    # 4. Invalid text
    assert extract_json_payload("No JSON here whatsoever") is None


def test_tier2_rubric_gate_rerouting():
    """Verifies that non-universal or algorithm-specific rules proposed for Tier 2 are programmatically rerouted to Tier 3."""
    mock_client = MagicMock()
    # Model proposes Tier 2, but fails the universality gate (e.g. domain specific to Richardson extrapolation)
    mock_client.complete.return_value.content = json.dumps({
        "root_cause": "Asymptotic extrapolation error in computing densities",
        "target_tier": "TIER_2_ORCHESTRATION",
        "tier2_rubric": {
            "is_universal": False,  # Fails gate 1!
            "is_behavioral_posture": False,
            "no_degradation_risk": False,
            "procedural_insufficient": False,
            "rationale": "Applies only to asymptotic limits with logarithmic decay.",
            "verdict": "REJECT_TO_TIER3",
        },
        "tier2_rule": "Apply Richardson extrapolation when computing densities.",
        "tier3_heuristic": None,
    })

    mock_memory = MagicMock()
    engine = RemediationEngine(client=mock_client, memory=mock_memory)

    telemetry = {
        "problem_id": 818,
        "title": "Asymptotic Limits",
        "statement": "Find the limit d_infty...",
        "expected_answer": "12345",
        "predicted_answer": "99999",
        "failure_category": "WRONG_ANSWER",
        "turns_count": 12,
        "last_error": "None",
        "log_excerpt": "...",
    }

    diagnostic = engine.diagnose_failure(telemetry)

    # Must be intercepted by programmatic gate and converted to Tier 3
    assert diagnostic.target_tier == RemediationTier.TIER_3_PROCEDURAL.value
    assert diagnostic.rerouted_to_tier3 is True
    assert diagnostic.tier3_heuristic == "Apply Richardson extrapolation when computing densities."
    assert diagnostic.tier2_rule is None


def test_tier2_rubric_gate_pass():
    """Verifies that a genuine cognitive posture invariant passing all 4 gates is accepted for Tier 2."""
    mock_client = MagicMock()
    mock_client.complete.return_value.content = json.dumps({
        "root_cause": "Model guessed answer on turn 1 without executing code across multiple domains",
        "target_tier": "TIER_2_ORCHESTRATION",
        "tier2_rubric": {
            "is_universal": True,
            "is_behavioral_posture": True,
            "no_degradation_risk": True,
            "procedural_insufficient": True,
            "rationale": "Mandate to never guess without code execution is universal across all math.",
            "verdict": "PASS",
        },
        "tier2_rule": "Never emit final boxed answer without executing verified Python code.",
        "tier3_heuristic": None,
    })

    mock_memory = MagicMock()
    engine = RemediationEngine(client=mock_client, memory=mock_memory)

    telemetry = {
        "problem_id": 836,
        "title": "April Fools",
        "statement": "Calculate something...",
        "expected_answer": "0",
        "predicted_answer": "aprifoolsjoke",
        "failure_category": "UNVERIFIED_GUESS",
        "turns_count": 1,
        "last_error": "None",
        "log_excerpt": "...",
    }

    diagnostic = engine.diagnose_failure(telemetry)

    assert diagnostic.target_tier == RemediationTier.TIER_2_ORCHESTRATION.value
    assert diagnostic.rerouted_to_tier3 is False
    assert diagnostic.tier2_rule is not None
    assert "Never emit final boxed answer" in diagnostic.tier2_rule


def test_tier1_harness_diagnosis():
    """Verifies that harness/execution issues produce concrete Tier 1 code diagnostics."""
    mock_client = MagicMock()
    mock_client.complete.return_value.content = json.dumps({
        "root_cause": "Process execution timed out after 300 seconds due to infinite generator",
        "target_tier": "TIER_1_HARNESS",
        "tier1_patch": {
            "file": "core_engine/kernel.py",
            "issue": "Kernel default execution timeout needs adaptive bounds",
            "suggested_fix": "Increase timeout or add early loop cancellation",
        },
        "tier2_rubric": None,
        "tier2_rule": None,
        "tier3_heuristic": None,
    })

    mock_memory = MagicMock()
    engine = RemediationEngine(client=mock_client, memory=mock_memory)

    telemetry = {
        "problem_id": 74,
        "title": "Digit Factorial Chains",
        "statement": "...",
        "expected_answer": "402",
        "predicted_answer": None,
        "failure_category": "TIMEOUT_OR_BLOWUP",
        "turns_count": 5,
        "last_error": "TimeoutError: Execution exceeded 300s",
        "log_excerpt": "...",
    }

    diagnostic = engine.diagnose_failure(telemetry)

    assert diagnostic.target_tier == RemediationTier.TIER_1_HARNESS.value
    assert diagnostic.tier1_patch is not None
    assert diagnostic.tier1_patch["file"] == "core_engine/kernel.py"


def test_apply_remediation_persists_to_turbovec():
    """Verifies that apply_remediation persists Tier 3 heuristics to TurboVec via CoALAMemoryManager."""
    mock_memory = MagicMock()
    mock_memory.add_procedural_heuristic.return_value = True

    engine = RemediationEngine(memory=mock_memory)

    diagnostic = ProblemDiagnostic(
        problem_id=850,
        title="Fractions and Powers",
        root_cause="Missed boundary anomaly at n=1",
        target_tier=RemediationTier.TIER_3_PROCEDURAL.value,
        tier3_heuristic="For exponential vs polynomial inequalities 2^n > n^k, always test n=1 as a boundary anomaly.",
    )

    result = engine.apply_remediation(diagnostic, auto_apply_tier3=True, dry_run=False)

    assert result.persisted_to_turbovec is True
    assert "PERSISTED to TurboVec" in result.persisted_status
    mock_memory.add_procedural_heuristic.assert_called_once()


def test_apply_remediation_dry_run():
    """Verifies that dry_run skips persistence."""
    mock_memory = MagicMock()
    engine = RemediationEngine(memory=mock_memory)

    diagnostic = ProblemDiagnostic(
        problem_id=850,
        title="Fractions and Powers",
        root_cause="Missed boundary anomaly",
        target_tier=RemediationTier.TIER_3_PROCEDURAL.value,
        tier3_heuristic="Always check n=1.",
    )

    result = engine.apply_remediation(diagnostic, auto_apply_tier3=True, dry_run=True)

    assert result.persisted_to_turbovec is False
    assert "DRY_RUN" in result.persisted_status
    mock_memory.add_procedural_heuristic.assert_not_called()


def test_apply_tier1_patch_security_boundary(tmp_path):
    """Verifies that patch application rejects files outside core_engine/ or remediation.py."""
    mock_client = MagicMock()
    engine = RemediationEngine(client=mock_client)

    telemetry = {"last_error": "None"}

    # 1. Locked remediation engine itself
    d_locked = ProblemDiagnostic(
        problem_id=1,
        title="Test",
        root_cause="Bug in remediation",
        target_tier=RemediationTier.TIER_1_HARNESS.value,
        tier1_patch={"file": "core_engine/remediation.py", "issue": "x", "suggested_fix": "y"},
    )
    ok, msg = engine.apply_tier1_patch(d_locked, telemetry)
    assert not ok
    assert "locked and cannot modify itself" in msg

    # 2. File outside core_engine/
    d_outside = ProblemDiagnostic(
        problem_id=1,
        title="Test",
        root_cause="Bug in outside file",
        target_tier=RemediationTier.TIER_1_HARNESS.value,
        tier1_patch={"file": "non_existent_file_xyz_123.py", "issue": "x", "suggested_fix": "y"},
    )
    ok2, msg2 = engine.apply_tier1_patch(d_outside, telemetry)
    assert not ok2
    assert "does not exist in core_engine/" in msg2


def test_apply_tier1_patch_execution_and_rollback(monkeypatch, tmp_path):
    """Verifies patch application, AST check, and rollback when pytest fails."""
    from core_engine import remediation as rem_module

    # Create a temporary test harness file inside core_engine package
    test_harness_file = rem_module.CORE_ENGINE_PKG_DIR / "_test_dummy_harness.py"
    try:
        test_harness_file.write_text("def compute():\n    return 41\n", encoding="utf-8")

        mock_client = MagicMock()
        engine = RemediationEngine(client=mock_client)
        telemetry = {"last_error": "AssertionError"}

        diag = ProblemDiagnostic(
            problem_id=1,
            title="Dummy",
            root_cause="Returns 41 instead of 42",
            target_tier=RemediationTier.TIER_1_HARNESS.value,
            tier1_patch={
                "file": "_test_dummy_harness.py",
                "issue": "Off by one",
                "suggested_fix": "Return 42",
            },
        )

        # 1. Successful patch with pytest mock passing
        mock_client.complete.return_value.content = json.dumps({
            "target_content": "    return 41\n",
            "replacement_content": "    return 42\n",
        })

        with monkeypatch.context() as m:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            m.setattr(rem_module.subprocess, "run", lambda *args, **kwargs: mock_proc)

            ok, msg = engine.apply_tier1_patch(diag, telemetry, dry_run=False)
            assert ok is True
            assert "all tests passed" in msg
            assert "return 42" in test_harness_file.read_text(encoding="utf-8")

        # 2. Regression rollback when pytest fails
        mock_client.complete.return_value.content = json.dumps({
            "target_content": "    return 42\n",
            "replacement_content": "    return 999\n",
        })

        with monkeypatch.context() as m:
            mock_fail_proc = MagicMock()
            mock_fail_proc.returncode = 1
            mock_fail_proc.stdout = "FAILED tests/test_dummy.py"
            mock_fail_proc.stderr = ""
            m.setattr(rem_module.subprocess, "run", lambda *args, **kwargs: mock_fail_proc)

            ok_fail, msg_fail = engine.apply_tier1_patch(diag, telemetry, dry_run=False)
            assert ok_fail is False
            assert "ROLLBACK" in msg_fail
            # Verifies rollback to original content before the failed patch!
            assert "return 42" in test_harness_file.read_text(encoding="utf-8")

    finally:
        if test_harness_file.exists():
            test_harness_file.unlink()


def test_apply_tier1_patch_complexity_parking(monkeypatch, tmp_path):
    """Verifies that large or complex Tier 1 patches (>10 lines) are parked and logged for AGY review rather than auto-applied."""
    from core_engine import remediation as rem_module

    test_harness_file = rem_module.CORE_ENGINE_PKG_DIR / "_test_complex_dummy.py"
    parked_file = tmp_path / "test_parked.jsonl"
    monkeypatch.setattr(rem_module, "PARKED_ISSUES_FILE", parked_file)

    # 15 lines of code
    multiline_body = "\n".join([f"    x_{i} = {i}" for i in range(15)])
    test_harness_file.write_text(f"def complex_fn():\n{multiline_body}\n    return 0\n", encoding="utf-8")

    try:
        mock_client = MagicMock()
        engine = rem_module.RemediationEngine(
            client=mock_client,
            max_tier1_lines_target=10,
            max_tier1_lines_replacement=15,
        )
        telemetry = {"last_error": "ComplexArchitectureError"}

        diag = ProblemDiagnostic(
            problem_id=99,
            title="Complex Issue Test",
            root_cause="Architecture overhaul needed",
            target_tier=RemediationTier.TIER_1_HARNESS.value,
            tier1_patch={
                "file": "_test_complex_dummy.py",
                "issue": "Needs 15-line overhaul",
                "suggested_fix": "Rewrite block",
            },
        )

        # Proposed patch modifies 15 lines (>10 lines allowed)
        mock_client.complete.return_value.content = json.dumps({
            "target_content": multiline_body + "\n",
            "replacement_content": "    return 100\n",
        })

        ok, msg = engine.apply_tier1_patch(diag, telemetry, dry_run=False)

        # Must be rejected from automated application and parked!
        assert ok is False
        assert "Exceeds automated change threshold" in msg
        assert diag.is_parked is True
        assert "TIER_1_PARKED_COMPLEX" in diag.persisted_status

        # Original file MUST NOT have been modified
        assert "x_14 = 14" in test_harness_file.read_text(encoding="utf-8")

        # Parked issue MUST be logged to PARKED_ISSUES_FILE for paired AGY review
        assert parked_file.exists()
        log_lines = parked_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(log_lines) == 1
        entry = json.loads(log_lines[0])
        assert entry["problem_id"] == 99
        assert entry["target_file"] == "core_engine/_test_complex_dummy.py"
        assert entry["complexity"]["target_lines"] == 15
        assert "Exceeds automated change threshold" in entry["parked_reason"]

    finally:
        if test_harness_file.exists():
            test_harness_file.unlink()


def test_diagnose_complex_tier1_preemptively_parked(monkeypatch, tmp_path):
    """Verifies that diagnosis flagged as COMPLEX is preemptively parked without file write."""
    from core_engine import remediation as rem_module

    parked_file = tmp_path / "test_parked_preemptive.jsonl"
    monkeypatch.setattr(rem_module, "PARKED_ISSUES_FILE", parked_file)

    mock_client = MagicMock()
    mock_client.complete.return_value.content = json.dumps({
        "root_cause": "Full architectural rewrite required for multiprocess shared memory",
        "target_tier": "TIER_1_HARNESS",
        "tier1_patch": {
            "file": "core_engine/kernel.py",
            "issue": "Kernel process isolation overhaul",
            "suggested_fix": "Implement new IPC channel and socket server",
            "complexity": "COMPLEX",
        },
        "tier2_rubric": None,
        "tier2_rule": None,
        "tier3_heuristic": None,
    })

    engine = rem_module.RemediationEngine(client=mock_client)
    telemetry = {
        "problem_id": 50,
        "title": "IPC Overhaul",
        "statement": "...",
        "expected_answer": "1",
        "predicted_answer": None,
        "failure_category": "HARNESS_CRASH",
        "turns_count": 1,
        "last_error": "BrokenPipeError",
        "log_excerpt": "...",
    }

    diag = engine.diagnose_failure(telemetry)

    assert diag.target_tier == RemediationTier.TIER_1_HARNESS.value
    assert diag.is_parked is True
    assert "TIER_1_PARKED_COMPLEX" in diag.persisted_status
    assert parked_file.exists()


def test_is_valid_universal_heuristic():
    """Verifies that is_valid_universal_heuristic approves universal heuristics and rejects empirical/problem-bound ones."""
    from core_engine.remediation import is_valid_universal_heuristic

    # 1. Valid universal mathematical heuristics
    valid_1 = "When computing Dirichlet characters modulo q, verify orthogonality relations before summing over residue classes."
    ok, msg = is_valid_universal_heuristic(valid_1)
    assert ok is True
    assert "Valid" in msg

    valid_2 = "For exponential vs polynomial inequalities 2^n > n^k, always test n=1 as a boundary anomaly."
    ok, msg = is_valid_universal_heuristic(valid_2)
    assert ok is True

    valid_3 = "When analyzing prime orders ord_p(a) > ord_p(b) for coprime bases, decompose into Frobenius conjugacy classes."
    ok, msg = is_valid_universal_heuristic(valid_3)
    assert ok is True

    # 2. Reject trivial / placeholder
    assert is_valid_universal_heuristic("NONE")[0] is False
    assert is_valid_universal_heuristic("N/A")[0] is False
    assert is_valid_universal_heuristic("no reflection")[0] is False
    assert is_valid_universal_heuristic("Review numerical precision")[0] is False

    # 3. Reject empirical extrapolation / sieving hacks
    assert is_valid_universal_heuristic("Use Richardson extrapolation to estimate the limit from counts.")[0] is False
    assert is_valid_universal_heuristic("Apply Aitken's delta-squared process to accelerate limits.")[0] is False
    assert is_valid_universal_heuristic("Sieve primes up to 10^7 and compute the empirical density.")[0] is False
    assert is_valid_universal_heuristic("Sample checkpoints at 10^5, 10^6 and extrapolate the limit to infinity.")[0] is False
    assert is_valid_universal_heuristic("Approximate infinite density by sieving up to large N.")[0] is False

    # 4. Reject problem-specific benchmark / ID bindings
    assert is_valid_universal_heuristic("Review constraints and boundary conditions for Problem 1.")[0] is False
    assert is_valid_universal_heuristic("For FrontierMath problem #42, check modular conditions.")[0] is False
    assert is_valid_universal_heuristic("Check boundary values for problem 850.", problem_id=850)[0] is False

    # 5. Reject problem-specific title and variable bindings
    assert is_valid_universal_heuristic(
        "In Prime Order Density for 2 vs 3, test small primes first.",
        problem_title="Prime Order Density for 2 vs 3",
    )[0] is False
    assert is_valid_universal_heuristic("When ord_p(2) > ord_p(3), use Kummer field extensions.")[0] is False


def test_apply_remediation_rejects_non_universal_heuristic():
    """Verifies that apply_remediation programmatically rejects empirical or problem-specific heuristics from TurboVec."""
    from core_engine.remediation import is_valid_universal_heuristic

    mock_memory = MagicMock()
    engine = RemediationEngine(memory=mock_memory)

    diagnostic = ProblemDiagnostic(
        problem_id=1,
        title="Prime Order Density for 2 vs 3",
        root_cause="Empirical sieving could not reach asymptotic limit",
        target_tier=RemediationTier.TIER_3_PROCEDURAL.value,
        tier3_heuristic="Sieve primes up to 10^7 and use Richardson extrapolation to extrapolate the limit.",
    )

    result = engine.apply_remediation(diagnostic, auto_apply_tier3=True, dry_run=False)

    assert result.persisted_to_turbovec is False
    assert "TIER_3_REJECTED_NON_UNIVERSAL" in result.persisted_status
    assert result.tier3_rejected_reason is not None
    assert "Richardson extrapolation" in result.tier3_rejected_reason
    mock_memory.add_procedural_heuristic.assert_not_called()


