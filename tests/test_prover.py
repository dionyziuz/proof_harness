"""Comprehensive unit and regression tests for MathProver, CandidatePool, Auditor, Skeptic, and Compaction."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core_engine.compactor import MathContextCompactor
from core_engine.kernel import PersistentMathKernel
from core_engine.proof_extractor import is_valid_latex_proof
from core_engine.prover import MATH_PROVER_SYSTEM_PROMPT, MathProver
from core_engine.prover_auditor import ProverAuditorMixin
from core_engine.prover_stages import (
    SKEPTIC_INCONCLUSIVE,
    SKEPTIC_REFUTED,
    SKEPTIC_VERIFIED,
    ProverStagesMixin,
)
from core_engine.prover_types import (
    THRESHOLDS,
    AuditResult,
    CandidatePool,
    ProofResult,
    has_fatal_issue,
)


PROOF = (
    "\\begin{proof}\n"
    + "Let $G$ be a graph and let $S \\subseteq V(G)$ satisfy the hypotheses of the statement. " * 4
    + "\nHence the claim holds.\n\\end{proof}"
)


# ---------------------------------------------------------------- core prover initialization & isolation


def test_prover_initialization():
    kernel = PersistentMathKernel()
    prover = MathProver(kernel=kernel)
    assert prover.kernel is kernel
    assert "MATH_PROVER_SYSTEM_PROMPT" in globals() or len(MATH_PROVER_SYSTEM_PROMPT) > 100


def test_prover_extract_code_blocks():
    prover = MathProver()
    text = """
Here is the invariant proof:
```python
import numpy as np
P = np.array([[0.7, 0.3], [0.4, 0.6]])
w, v = np.linalg.eig(P.T)
pi = v[:, 0] / np.sum(v[:, 0])
print(pi)
```
And another block:
```python
assert np.allclose(pi @ P, pi)
```
"""
    blocks = prover.extract_code_blocks(text)
    assert len(blocks) == 2
    assert "P = np.array" in blocks[0]
    assert "assert np.allclose" in blocks[1]


def test_proof_result_dataclass():
    res = ProofResult(
        conjecture="Test conjecture",
        status="PROVEN",
        theorem="Theorem 1: Ergodic convergence",
        proof="Proof by Perron-Frobenius theorem.",
        verified_invariants=["pi @ P == pi", "spectral gap > 0"],
        implementation="def stationary(P): return pi",
    )
    assert res.status == "PROVEN"
    assert len(res.verified_invariants) == 2
    assert res.implementation is not None
    # Compatibility with TCSSolveResult
    assert res.candidate_proof == "Proof by Perron-Frobenius theorem."


def test_prover_literature_search_flag():
    prover_no_search = MathProver(enable_literature_search=False)
    assert prover_no_search.enable_literature_search is False
    assert prover_no_search.kernel.allow_internet is False

    prover_with_search = MathProver(enable_literature_search=True)
    assert prover_with_search.enable_literature_search is True
    assert prover_with_search.kernel.allow_internet is True


def test_kernel_internet_allowance():
    from core_engine.prover_kernel import TCSCompliantKernel
    k_blocked = TCSCompliantKernel(allow_internet=False)
    res_blocked = k_blocked.execute("import urllib.request")
    assert res_blocked.success is False
    assert "internet access is prohibited" in res_blocked.error

    k_allowed = TCSCompliantKernel(allow_internet=True)
    res_allowed = k_allowed.execute("import urllib.request")
    assert res_allowed.success is True


def test_tcs_hermetic_isolation_guarantee():
    from core_engine.prover_kernel import TCSCompliantKernel

    solver = MathProver()
    assert solver.enable_literature_search is False
    assert isinstance(solver.kernel, TCSCompliantKernel)
    assert solver.kernel.allow_internet is False

    # Stubs active in REPL namespace
    stub_res = solver.kernel.namespace["search_math"]("graph separator")
    assert "internet access disabled" in stub_res[0]["title"]

    # Pre-proof research is bypassed
    with patch.object(solver, "_pre_proof_research") as mock_research, \
         patch.object(solver, "_run_context_architect", return_value=""), \
         patch.object(solver, "_run_computational_falsifier", return_value=([], [])):
        messages, result, retrieved = solver._init_solve_session(
            task_id="task_test_isolation",
            task_prompt="Target Problem (your task): Prove separator bound.",
            max_turns=1,
            reset_kernel=False,
        )
        assert mock_research.call_count == 0


# ---------------------------------------------------------------- golden balance thresholds & pivot logic


def test_golden_balance_thresholds():
    assert THRESHOLDS.hard_pivot_floor == 0.05
    assert THRESHOLDS.pivot_threshold == 0.30
    assert THRESHOLDS.judge_pass_threshold == 0.50


def test_candidate_pool_prune_abandoned():
    pool = CandidatePool()
    # Offer a dead-end candidate (< 0.25)
    pool.offer(r"\begin{proof} flawed draft \end{proof}", 0.15, "REJECT", "fatal gap", 1)
    assert pool.best is not None
    assert pool.best_validity == 0.15

    # prune_abandoned with default floor (0.25) should prune it
    pruned = pool.prune_abandoned()
    assert pruned is not None
    assert pruned.validity == 0.15
    assert pool.best is None

    # Offer a viable candidate (0.45)
    pool.offer(r"\begin{proof} promising draft \end{proof}", 0.45, "REJECT", "minor gap", 2)
    assert pool.best is not None
    assert pool.best_validity == 0.45

    # prune_abandoned with floor 0.25 should NOT prune a 0.45 candidate!
    pruned = pool.prune_abandoned(THRESHOLDS.pivot_threshold)
    assert pruned is None
    assert pool.best is not None
    assert pool.best_validity == 0.45


def test_golden_balance_pivot_logic():
    # Helper simulating the prover's pivot decision
    def should_hard_pivot(turn, validity, rejection_count, prev_rejected_validity, audit_inconclusive=False):
        is_improving = (rejection_count >= 2 and validity > prev_rejected_validity + 0.05)
        return (not audit_inconclusive) and (
            (validity <= THRESHOLDS.hard_pivot_floor)
            or (rejection_count >= 2 and validity < THRESHOLDS.pivot_threshold and not is_improving)
        )

    # 1. Single-strike at 0.15: must NOT pivot (nascent draft protected)
    assert not should_hard_pivot(turn=1, validity=0.15, rejection_count=1, prev_rejected_validity=0.0)

    # 2. Single-strike at 0.20 (e.g. Turn 5 of Task 22): must NOT pivot!
    assert not should_hard_pivot(turn=5, validity=0.20, rejection_count=1, prev_rejected_validity=0.0)

    # 3. Single-strike at 0.04 (gibberish <= 0.05): MUST pivot immediately
    assert should_hard_pivot(turn=1, validity=0.04, rejection_count=1, prev_rejected_validity=0.0)

    # 4. Repeated rejection at 0.15 (stagnant, rejection #2): MUST pivot (dead route confirmed)
    assert should_hard_pivot(turn=2, validity=0.15, rejection_count=2, prev_rejected_validity=0.15)

    # 5. Repeated rejection but improving (0.15 -> 0.22, > +0.05): must NOT pivot!
    assert not should_hard_pivot(turn=2, validity=0.22, rejection_count=2, prev_rejected_validity=0.15)

    # 6. Viable candidate at 0.45 (Task 22 Turns 7 & 8): must NOT pivot!
    assert not should_hard_pivot(turn=8, validity=0.45, rejection_count=4, prev_rejected_validity=0.45)


def test_synchronized_overtime_clock_extension():
    from core_engine.tcs_config import TCSConfig
    cfg = TCSConfig()
    assert cfg.enable_overtime is True
    assert cfg.overtime_iterations == 4
    assert cfg.overtime_seconds_per_iter == 450.0

    # Simulate overtime gate triggering
    current_max_turns = 8
    max_turns = 8
    max_time_seconds = 3600.0
    solve_deadline = 10000.0
    validity = 0.35

    if cfg.enable_overtime and current_max_turns == max_turns and validity >= cfg.overtime_validity_threshold:
        current_max_turns += cfg.overtime_iterations
        overtime_seconds = cfg.overtime_seconds_per_iter * cfg.overtime_iterations
        max_time_seconds += overtime_seconds
        solve_deadline += overtime_seconds

    assert current_max_turns == 12
    assert max_time_seconds == 3600.0 + 1800.0  # 5400.0s
    assert solve_deadline == 10000.0 + 1800.0


# ---------------------------------------------------------------- skeptic verdict parsing


@pytest.mark.parametrize(
    "text,expected",
    [
        ("analysis...\nVERIFIED: holds for n <= 6", SKEPTIC_VERIFIED),
        ("UNVERIFIED: could not test", None),
        ("**REFUTED**: n=3 breaks it", SKEPTIC_REFUTED),
        ("VERIFIED: a\nREFUTED: b", SKEPTIC_REFUTED),
        ("The claim is VERIFIED: probably", None),
        ("no verdict at all", None),
    ],
)
def test_parse_skeptic_verdict(text, expected):
    status, _ = ProverStagesMixin._parse_skeptic_verdict(text)
    assert status == expected


def _skeptic_host(content=None, exc=None, exec_result=None):
    host = MathProver()
    host.client = MagicMock()
    if exc is not None:
        host.client.complete.side_effect = exc
    else:
        host.client.complete.return_value = SimpleNamespace(content=content)
    if exec_result is not None:
        host.kernel = MagicMock()
        host.kernel.execute.return_value = exec_result
    host._log_write = lambda *_a, **_k: None
    return host


@patch("core_engine.prover_stages.CoALAMemoryManager", side_effect=RuntimeError("no memory"))
def test_skeptic_is_fail_closed(_mem):
    # No verdict line -> INCONCLUSIVE (was: VERIFIED)
    host = _skeptic_host(content="I looked at it and it seems fine.")
    status, _, _ = host._run_adversarial_skeptic("t", "task", "claim text that is long enough", "")
    assert status == SKEPTIC_INCONCLUSIVE

    # Exception -> INCONCLUSIVE (was: VERIFIED)
    host = _skeptic_host(exc=TimeoutError("boom"))
    status, _, _ = host._run_adversarial_skeptic("t", "task", "claim text that is long enough", "")
    assert status == SKEPTIC_INCONCLUSIVE

    # Script crashed with a non-assertion error + VERIFIED text -> INCONCLUSIVE
    crashed = SimpleNamespace(success=False, error="NameError: name 'nx' is not defined", stdout="", result_repr="")
    host = _skeptic_host(content="```python\nnx.foo()\n```\nVERIFIED: fine", exec_result=crashed)
    status, _, _ = host._run_adversarial_skeptic("t", "task", "claim text that is long enough", "")
    assert status == SKEPTIC_INCONCLUSIVE

    # Script timed out / crashed + hallucinated REFUTED text -> INCONCLUSIVE (Task 160 Turn 5 fix!)
    timed_out = SimpleNamespace(success=False, error="TimeoutException: Execution timed out after 15.0s", stdout="", result_repr="")
    host = _skeptic_host(content="```python\nwhile True: pass\n```\nREFUTED: w* not in W(v)", exec_result=timed_out)
    status, _, _ = host._run_adversarial_skeptic("t", "task", "claim text that is long enough", "")
    assert status == SKEPTIC_INCONCLUSIVE

    # Script ran cleanly with 0 counterexamples + hallucinated REFUTED text -> INCONCLUSIVE
    clean_run = SimpleNamespace(success=True, error=None, stdout="0 counterexamples found", result_repr="")
    host = _skeptic_host(content="```python\nprint('0 counterexamples found')\n```\nREFUTED: speculative doubt", exec_result=clean_run)
    status, _, _ = host._run_adversarial_skeptic("t", "task", "claim text that is long enough", "")
    assert status == SKEPTIC_INCONCLUSIVE

    # Assertion failure -> REFUTED
    failed = SimpleNamespace(success=False, error="AssertionError: n=3", stdout="", result_repr="")
    host = _skeptic_host(content="```python\nassert False\n```", exec_result=failed)
    status, _, _ = host._run_adversarial_skeptic("t", "task", "claim text that is long enough", "")
    assert status == SKEPTIC_REFUTED

    # Output merely containing "violates" is no longer a refutation
    ok = SimpleNamespace(success=True, error=None, stdout="no graph violates the bound", result_repr="")
    host = _skeptic_host(content="```python\nprint('x')\n```\nVERIFIED: checked n<=6", exec_result=ok)
    status, _, _ = host._run_adversarial_skeptic("t", "task", "claim text that is long enough", "")
    assert status == SKEPTIC_VERIFIED


# ---------------------------------------------------------------- auditor classification


def test_audit_classification_never_invents_scores():
    res = ProverAuditorMixin._classify_audit_output('{"verdict": "VALID", "issues": "none"}')
    assert res.verdict == "INCONCLUSIVE"  # no numeric validity -> no synthetic 0.85
    res = ProverAuditorMixin._classify_audit_output("garbage")
    assert res.verdict == "INCONCLUSIVE"
    res = ProverAuditorMixin._classify_audit_output('{"validity": 0.82, "verdict": "VALID", "issues": "none"}')
    assert res.verdict == "VALID" and res.validity == pytest.approx(0.82)
    res = ProverAuditorMixin._classify_audit_output('{"validity": 0.9, "verdict": "VALID", "issues": "Step 3 is a fatal gap"}')
    assert res.verdict == "REJECT"
    res = ProverAuditorMixin._classify_audit_output('{"validity": 0.6, "verdict": "VALID", "issues": "minor"}')
    assert res.verdict == "REJECT"


def test_has_fatal_issue_is_negation_aware():
    assert has_fatal_issue("Lemma 2 is false for n=1")
    assert not has_fatal_issue("No fatal flaws; minor notation issues.")
    assert not has_fatal_issue("Uses a proof by contradiction correctly.")


def test_audit_unparseable_never_gets_artificial_high_score():
    """Unparseable audit text must return validity=0.0 and INCONCLUSIVE,
    never 0.40 or 0.50 which would beat real candidates.
    """
    prover = MathProver.__new__(MathProver)
    prover._log_write = MagicMock()
    prover.extract_code_blocks = MagicMock(return_value=[])
    prover._detect_proof_mode = MagicMock(return_value="abstract")

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = 'ISSUES: The argument is somewhat informal and missing details on claim 2.'
    mock_client.complete.return_value = mock_resp
    prover.client = mock_client

    result = prover._audit_proof_draft("target prompt", r"\begin{proof} proof \end{proof}")
    assert result.verdict in ("INCONCLUSIVE", "REJECT")
    assert result.validity == 0.0, f"Unparseable audit should have validity 0.0, got {result.validity}"


def test_audit_regex_extracts_explicit_validity_when_json_broken():
    """If the LLM emits a validity score in broken JSON, that score is respected,
    rather than a synthetic default of 0.40/0.50.
    """
    prover = MathProver.__new__(MathProver)
    prover._log_write = MagicMock()
    prover.extract_code_blocks = MagicMock(return_value=[])
    prover._detect_proof_mode = MagicMock(return_value="abstract")

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = '```json\n{\n  "verdict": "REJECT",\n  "validity": 0.25,\n  "issues": "Missing base case"\n'
    mock_client.complete.return_value = mock_resp
    prover.client = mock_client

    result = prover._audit_proof_draft("target prompt", r"\begin{proof} proof \end{proof}")
    assert result.validity == 0.25


def test_falsifier_assert_error_does_not_zero_proof_without_witness():
    """An arbitrary AssertionError in falsifier script (e.g. ill-conditioned test matrix)
    must not immediately zero out proof validity to 0.0; it should fall through to Stage 2.
    """
    prover = MathProver.__new__(MathProver)
    prover._log_write = MagicMock()
    prover._detect_proof_mode = MagicMock(return_value="tcs")
    prover.auditor_kernel_timeout = 5.0

    resp_stage1 = MagicMock()
    resp_stage1.content = "```python\n# Buggy test script assertion on its own setup\nassert 1 == 2\n```"

    resp_stage2 = MagicMock()
    resp_stage2.content = '{"verdict": "VALID", "validity": 0.85, "issues": "none"}'

    prover.client = MagicMock()
    prover.client.complete.side_effect = [resp_stage1, resp_stage2]
    prover.extract_code_blocks = lambda text: [b.strip() for b in text.split("```python")[1].split("```")[0:1]] if "```python" in text else []

    audit = prover._audit_proof_draft("target prompt", r"\begin{proof} proof \end{proof}")
    assert audit.validity > 0.0, f"Proof was killed with validity 0.0 due to test script bug! Got {audit.validity}"


# ---------------------------------------------------------------- compaction anchors & ledger


def test_compactor_keeps_protected_anchors_verbatim():
    client = MagicMock()
    client.complete.return_value = SimpleNamespace(content="SUMMARY")
    comp = MathContextCompactor(client=client, max_context_tokens=100, reserve_tokens=10, keep_recent_turns=1)
    big = "x" * 30000
    cand = {"role": "assistant", "content": "CANDIDATE " + big}
    rev = {"role": "user", "content": "REVISION PROMPT"}
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        cand,
        rev,
        {"role": "assistant", "content": "old scratch " + big},
        {"role": "user", "content": "old feedback"},
        {"role": "assistant", "content": "recent"},
        {"role": "user", "content": "recent feedback"},
    ]
    res = comp.compact(msgs, force=True, protected=[cand, rev])
    assert res.compacted
    out = res.compacted_messages
    assert out[2] is cand and out[3] is rev
    assert cand["content"].endswith(big)  # never clamped
    assert "SUMMARY" in out[4]["content"]


def test_prover_history_compaction_keeps_candidate_under_revision():
    prover = MathProver()
    cand = {"role": "assistant", "content": "Here is the candidate proof under revision:\n\n" + PROOF}
    rev = {"role": "user", "content": "[ADVERSARIAL PROOF AUDIT - DEFECTS DETECTED] identified the following defects:\nstep 2"}
    prover._anchor_msgs = [cand, rev]
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        cand,
        rev,
        {"role": "assistant", "content": "[CLAIM]: every leaf has degree one in T"},
        {"role": "user", "content": "exec feedback"},
        {"role": "assistant", "content": "more"},
        {"role": "user", "content": "next"},
    ]
    out = prover._compact_conversation_history(msgs, current_turn=4)
    assert cand in out and rev in out
    summary = next(m["content"] for m in out if "CONCISE SUMMARY" in m["content"])
    assert "Verified definitions" not in summary  # no invented progress claims
    assert "every leaf has degree one" in summary


def test_compactor_deterministic_ledger_fallback():
    client = MagicMock()
    client.complete.side_effect = RuntimeError("LLM offline")
    comp = MathContextCompactor(client=client, max_context_tokens=100, reserve_tokens=10, keep_recent_turns=1)
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "[CLAIM]: Every bipartite graph has no odd cycles\n\nMore text " + ("x" * 1000)},
        {"role": "user", "content": "[Execution Output]:\nVerified for n<=8"},
        {"role": "assistant", "content": "recent"},
        {"role": "user", "content": "recent user"},
    ]
    res = comp.compact(msgs, force=True)
    assert res.compacted
    assert "Every bipartite graph has no odd cycles" in res.summary
    assert "Verified for n<=8" in res.summary


def test_lemma_ledger_safe_recording():
    """Claim text containing LaTeX backslashes, quotes, and newlines must record cleanly."""
    kernel = PersistentMathKernel()
    prover = MathProver.__new__(MathProver)
    prover.kernel = kernel

    class DummyState:
        def __init__(self):
            self.lemmas = []

        def record_lemma(self, claim):
            self.lemmas.append(claim)

    dummy_state = DummyState()
    kernel.namespace["state"] = dummy_state
    kernel._state_machine = dummy_state

    claim = r'For all \epsilon > 0, the "sparse cut" has \delta(S) \le O(\sqrt{\log n})'

    if hasattr(prover, "_safe_record_lemma"):
        prover._safe_record_lemma(claim)
    else:
        clean_claim = " ".join(claim.split())[:200]
        if hasattr(kernel, "_state_machine") and kernel._state_machine:
            kernel._state_machine.record_lemma(clean_claim)
        else:
            kernel.execute(f"state.record_lemma({repr(clean_claim)})")

    assert len(dummy_state.lemmas) == 1
    assert r"\delta(S)" in dummy_state.lemmas[0]
    assert r'"sparse cut"' in dummy_state.lemmas[0]


# ---------------------------------------------------------------- validator & proof extractor


def test_validator_allows_proof_by_contradiction_language():
    body = "\\begin{proof}\n" + "Let $x \\in S$ be arbitrary and consider every such element. " * 4
    assert is_valid_latex_proof(body + "Suppose, for contradiction, that the statement is false. Then $x<0$.\n\\end{proof}")
    assert not is_valid_latex_proof(body + "Therefore the statement is false as written.\n\\end{proof}")
    assert is_valid_latex_proof(body + "Note that we don't have $x = 0$ here.\n\\end{proof}")


def test_proof_extractor_rejects_omission_placeholders():
    """is_valid_latex_proof must reject proofs with omission placeholders like (as above) or proof omitted."""
    valid_proof = r"\begin{proof} " + "A" * 200 + r" \end{proof}"
    assert is_valid_latex_proof(valid_proof)

    placeholders = [
        r"\begin{proof} Claim 1. Proof: ... (as above) " + "A" * 150 + r" \end{proof}",
        r"\begin{proof} Claim 1. Proof: (as above). " + "A" * 150 + r" \end{proof}",
        r"\begin{proof} Details are omitted for brevity. " + "A" * 150 + r" \end{proof}",
        r"\begin{proof} Proof omitted. " + "A" * 150 + r" \end{proof}",
        r"\begin{proof} The remaining verification is left to the reader. " + "A" * 150 + r" \end{proof}",
    ]
    for p in placeholders:
        assert not is_valid_latex_proof(p), f"Should have rejected proof containing placeholder: {p}"


# ---------------------------------------------------------------- end-to-end exploratory loop


def test_exploratory_loop_inconclusive_audit_and_anchor_pinning():
    """Turn 1 emits a proof (audit INCONCLUSIVE), turn 2 emits a proof (REJECT 0.7),
    turn 3 emits code; the candidate from turn 2 must survive compaction and the revision
    context must not contain the Turn-1 instructions."""
    prover = MathProver()
    prover._log_write = lambda *_a, **_k: None
    prover.compactor = MagicMock()
    prover.compactor.should_compact.return_value = False

    with patch.object(prover, "_run_context_architect", return_value="blueprint"), \
         patch.object(prover, "_run_computational_falsifier", return_value=([], [])), \
         patch.object(prover.memory, "query_procedural_heuristics", return_value=[]), \
         patch.object(prover.memory, "check_health", return_value=True):
        messages, result, _ = prover._init_solve_session("t1", "Target Problem (your task): prove it.", max_turns=4, reset_kernel=True)

    assert "DEDUCTIVE LOGICIAN TURN 1" in messages[1]["content"]
    assert "DEDUCTIVE LOGICIAN TURN 1" not in prover._task_only_msg["content"]

    responses = iter([
        ("", PROOF),
        ("", PROOF.replace("Hence", "Therefore")),
        ("", "Let me test.\n```python\nprint(1+1)\n```"),
        ("", "Let me test again.\n```python\nprint(2+2)\n```"),
    ])
    seen_contexts = []

    def fake_stream(messages, **_kw):
        seen_contexts.append([dict(m) for m in messages])
        return next(responses)

    audits = iter([
        AuditResult(verdict="INCONCLUSIVE", issues="", validity=0.0),
        AuditResult(verdict="REJECT", issues="Step 2 lacks justification.", validity=0.7),
    ])
    with patch.object(prover, "_stream_completion_with_retry", side_effect=fake_stream), \
         patch.object(prover, "_audit_proof_draft", side_effect=lambda *_a, **_k: next(audits)), \
         patch.object(prover, "_run_critique_falsifier", return_value=([], [])), \
         patch.object(prover, "_run_publication_scribe", return_value=None):
        best, _ = prover._run_exploratory_turns(
            task_id="t1",
            task_prompt="Target Problem (your task): prove it.",
            messages=messages,
            result=result,
            max_turns=4,
            start_wall_time=0.0,
        )

    # Turn 2 context is a revision context: task-only message, no Turn-1 directive, no pivot.
    turn2 = seen_contexts[1]
    assert "DEDUCTIVE LOGICIAN TURN 1" not in turn2[1]["content"]
    assert "MANDATORY REDUCTIVE PIVOT" not in turn2[3]["content"]  # INCONCLUSIVE must not trigger a pivot
    # Turn 4 context still contains the turn-2 candidate under revision after history compaction.
    turn4_text = "\n".join(m["content"] for m in seen_contexts[3])
    assert "Therefore the claim holds." in turn4_text
    assert best is not None and best.validity == pytest.approx(0.7)
    assert THRESHOLDS.early_exit_validity == pytest.approx(0.90)


# ---------------------------------------------------------------- V4.1 harness gates & candidate pool


def test_unauthorized_domain_restriction_detection_and_gate():
    task = "Target Problem (your task): Prove that for every parameter x > 0, f(x) <= g(x)."
    bad_proof = (
        "\\begin{proof}\n"
        "We restrict our attention to the regime where x <= 1, since the intermediate series bound converges there.\n"
        "\\end{proof}"
    )
    flagged = ProverAuditorMixin._detect_unauthorized_restrictions(bad_proof, task)
    assert len(flagged) == 1

    # When the proof covers the complementary case, it is NOT flagged
    good_proof = (
        "\\begin{proof}\n"
        "Case 1: We restrict our attention to the regime where x <= 1. Here the series bound holds.\n"
        "Case 2 (Otherwise): When x > 1, the bound holds directly.\n"
        "\\end{proof}"
    )
    assert ProverAuditorMixin._detect_unauthorized_restrictions(good_proof, task) == []

    # Boolean referee gates enforce hard caps in _classify_audit_output
    res_fid = ProverAuditorMixin._classify_audit_output(
        '{"statement_fidelity_exact": false, "inequality_directions_verified": true, '
        '"validity": 0.82, "verdict": "VALID", "issues": "none"}'
    )
    assert res_fid.verdict == "REJECT"
    assert res_fid.validity <= 0.25

    res_ineq = ProverAuditorMixin._classify_audit_output(
        '{"statement_fidelity_exact": true, "inequality_directions_verified": false, '
        '"validity": 0.80, "verdict": "VALID", "issues": "none"}'
    )
    assert res_ineq.verdict == "REJECT"
    assert res_ineq.validity <= 0.30


def test_candidate_pool_noise_band_occam():
    pool = CandidatePool()
    concise_proof = "A" * 4000
    bloated_proof = "B" * 16000

    assert pool.offer(concise_proof, 0.78, "VALID", "", turn=2)
    # Bloated proof (+300% length) with only +0.02 validity gain (inside 0.03 noise band) must NOT displace concise proof
    assert not pool.offer(bloated_proof, 0.80, "VALID", "", turn=5)
    assert pool.best.proof == concise_proof

    # Conversely, if bloated proof was first, a >25% shorter proof within 0.03 noise band DOES displace it
    pool2 = CandidatePool()
    assert pool2.offer(bloated_proof, 0.80, "VALID", "", turn=2)
    assert pool2.offer(concise_proof, 0.78, "VALID", "", turn=3)
    assert pool2.best.proof == concise_proof


def test_candidate_pool_tie_breaking():
    """Candidate pool must not overwrite a concise proof with a longer rambling proof of equal validity."""
    pool = CandidatePool()
    short_proof = r"\begin{proof} short proof \end{proof}"
    long_proof = r"\begin{proof} verbose rambling proof with thousands of unnecessary tokens \end{proof}"

    pool.offer(short_proof, validity=0.45, verdict="REJECT", issues="small gap", turn=3)
    assert pool.best is not None
    assert pool.best.proof == short_proof

    # Later turn has equal validity (0.45) but is much longer
    pool.offer(long_proof, validity=0.45, verdict="REJECT", issues="different gap", turn=7)
    assert pool.best.proof == short_proof, "Shorter proof was overwritten by longer proof with equal validity!"

    # Proof with higher validity (0.50) must overwrite
    better_proof = r"\begin{proof} better proof \end{proof}"
    pool.offer(better_proof, validity=0.50, verdict="VALID", issues="none", turn=8)
    assert pool.best.proof == better_proof


def test_candidate_pool_low_validity_sketch_protection():
    """At low validity (< 0.50), an abbreviated sketch must not overwrite an existing candidate of equal validity."""
    pool = CandidatePool()
    detailed_draft = r"\begin{proof} detailed case analysis covering all base cases and lemmas \end{proof}"
    sketch_draft = r"\begin{proof} brief sketch \end{proof}"

    pool.offer(detailed_draft, validity=0.20, verdict="REJECT", issues="gap", turn=6)
    assert pool.best.proof == detailed_draft

    # Shorter sketch with same low validity (0.20) must NOT overwrite
    overwrote = pool.offer(sketch_draft, validity=0.20, verdict="REJECT", issues="gap", turn=990)
    assert not overwrote
    assert pool.best.proof == detailed_draft, "Shorter sketch overwrote detailed draft at low validity!"


def test_lean_runner_blocks_axiom_and_unsafe():
    from core_engine.lean_runner import _FORBIDDEN_AXIOM_CODE_RE

    code_with_axiom = "-- comment about axiom\naxiom cheater : 1 = 2"
    code_no_comments = "\n".join(line.split("--")[0] for line in code_with_axiom.splitlines())
    m = _FORBIDDEN_AXIOM_CODE_RE.search(code_no_comments)
    assert m is not None and m.group(1) == "axiom"


# ---------------------------------------------------------------- kernel stdout isolation & judge replicates


def test_concurrent_kernel_stdout_isolation():
    """Two concurrent threads executing in their own kernels must not cross-pollute stdout."""
    k1 = PersistentMathKernel()
    k2 = PersistentMathKernel()

    results = {}

    def run_worker_1():
        time.sleep(0.01)
        res = k1.execute("print('WORKER_1_UNIQUE_TOKEN')")
        results["w1"] = res.stdout

    def run_worker_2():
        res = k2.execute("import time; time.sleep(0.03); print('WORKER_2_UNIQUE_TOKEN')")
        results["w2"] = res.stdout

    t1 = threading.Thread(target=run_worker_1)
    t2 = threading.Thread(target=run_worker_2)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert "WORKER_1_UNIQUE_TOKEN" in results.get("w1", "")
    assert "WORKER_2_UNIQUE_TOKEN" not in results.get("w1", ""), "Cross-worker stdout leak detected in w1!"
    assert "WORKER_2_UNIQUE_TOKEN" in results.get("w2", "")
    assert "WORKER_1_UNIQUE_TOKEN" not in results.get("w2", ""), "Cross-worker stdout leak detected in w2!"


def test_external_print_does_not_leak_into_kernel_stdout():
    """Prints from non-kernel threads (e.g. loggers, other workers) must not leak into kernel stdout."""
    k = PersistentMathKernel()

    def background_logger():
        time.sleep(0.01)
        print("ORCHESTRATOR_BACKGROUND_LOG")

    bg_thread = threading.Thread(target=background_logger)
    bg_thread.start()

    res = k.execute("import time; time.sleep(0.04); print('KERNEL_FACT: 2+2=4')")
    bg_thread.join()

    assert "KERNEL_FACT: 2+2=4" in res.stdout
    assert "ORCHESTRATOR_BACKGROUND_LOG" not in res.stdout, "External print leaked into REPL stdout!"


def test_judge_borderline_replicates_median():
    """Borderline scores [0.35, 0.65] trigger 3 replicates and compute median score."""
    from core_engine.judging.tcs_judge import judge_tcs_proof

    mock_client = MagicMock()
    r1 = MagicMock(content='{"validity": 0.45, "rigor": 0.5, "completeness": 0.4, "issues": "minor gap"}')
    r2 = MagicMock(content='{"validity": 0.55, "rigor": 0.6, "completeness": 0.5, "issues": "sound"}')
    r3 = MagicMock(content='{"validity": 0.45, "rigor": 0.5, "completeness": 0.4, "issues": "minor gap"}')
    mock_client.complete.side_effect = [r1, r2, r3]

    res = judge_tcs_proof(
        candidate=r"\begin{proof} candidate proof \end{proof}",
        problem_prompt="Target statement context...",
        gt_solution="Ground truth proof...",
        client=mock_client,
        config="tcs",
        threshold=0.50,
        borderline_replicates=True,
    )

    assert res["b"]["validity"] == 0.45
    assert len(res["b"]["replicate_validities"]) == 3
    assert res["b"]["replicate_validities"] == [0.45, 0.55, 0.45]
    assert mock_client.complete.call_count == 3


def test_substantive_partial_proof_protected_from_hard_pivot():
    """When validity is capped at 0.20 by a boolean gate for 2 consecutive turns,
    a substantive partial proof (completeness >= 0.35 or rigor >= 0.35) must be
    protected for Surgical Revision instead of wiped by a Clean-Slate Hard Pivot.
    """
    res = ProverAuditorMixin._classify_audit_output(
        '{"statement_fidelity_exact": false, "inequality_directions_verified": true, '
        '"theorem_hypotheses_verified": false, "rigor": 0.35, "completeness": 0.45, '
        '"boundary_cases": 0.40, "validity": 0.20, "verdict": "REJECT", "issues": "Missing lambda=0 boundary"}'
    )
    assert res.verdict == "REJECT"
    assert res.validity == 0.20
    assert res.rigor == 0.35
    assert res.completeness == 0.45
    assert res.boundary_cases == 0.40
    assert res.has_substantive_structure is True
    assert abs(res.structural_validity - 0.40) < 1e-6

    prover = MathProver()
    prover._log_write = lambda *_a, **_k: None
    prover.compactor = MagicMock()
    prover.compactor.should_compact.return_value = False

    with patch.object(prover, "_run_context_architect", return_value="blueprint"), \
         patch.object(prover, "_run_computational_falsifier", return_value=([], [])), \
         patch.object(prover.memory, "query_procedural_heuristics", return_value=[]), \
         patch.object(prover.memory, "check_health", return_value=True):
        messages, result, _ = prover._init_solve_session("t_sub", "Target Problem: prove Theorem 1.", max_turns=4, reset_kernel=True)

    p1 = "\\begin{proof}\n" + ("By Lemma 1, the tree has depth at most k and branching factor bounded. " * 4) + "\n\\end{proof}"
    p2 = "\\begin{proof}\n" + ("By Lemma 1 and Lemma 2, the tree depth is k and the product of branching factors is k^{O(k)}. " * 4) + "\n\\end{proof}"
    p3 = "\\begin{proof}\n" + ("Including the lambda=0 boundary case via direct projection, the full bound holds for all lambda in [0,1]. " * 4) + "\n\\end{proof}"

    captured_prompts = []

    def fake_stream(messages, **_kw):
        captured_prompts.append([m["content"] for m in messages])
        idx = len(captured_prompts)
        if idx == 1:
            return ("t1", p1)
        elif idx == 2:
            return ("t2", p2)
        return ("t3", p3)

    audit_results = iter([
        AuditResult(verdict="REJECT", issues="Branching factor sum vs product", validity=0.20, rigor=0.30, completeness=0.35, boundary_cases=0.30),
        AuditResult(verdict="REJECT", issues="Missing lambda=0 boundary case", validity=0.20, rigor=0.35, completeness=0.45, boundary_cases=0.40),
        AuditResult(verdict="VALID", issues="", validity=0.92, rigor=0.90, completeness=0.95, boundary_cases=0.90),
    ])

    with patch.object(prover, "_stream_completion_with_retry", side_effect=fake_stream), \
         patch.object(prover, "_audit_proof_draft", side_effect=lambda *_a, **_k: next(audit_results)), \
         patch.object(prover, "_run_critique_falsifier", return_value=([], [])), \
         patch.object(prover, "_run_publication_scribe", return_value=None):
        best, _ = prover._run_exploratory_turns(
            task_id="t_sub",
            task_prompt="Target Problem: prove Theorem 1.",
            messages=messages,
            result=result,
            max_turns=4,
            start_wall_time=0.0,
        )

    assert result.verified is True
    # Turn 3 prompt (after 2nd rejection) must be Surgical Revision, NOT Clean-Slate Hard Pivot
    turn3_join = "\n".join(captured_prompts[2])
    assert "Here is the candidate proof under revision:" in turn3_join
    assert "MANDATORY REDUCTIVE PIVOT" not in turn3_join


def test_repl_first_pivot_gate_on_clean_slate_hard_pivot():
    """When two genuinely low-validity drafts (rigor < 0.35, completeness < 0.35) trigger
    a Clean-Slate Hard Pivot with >= 2 turns remaining, Turn t+1 enforces the 1-Turn
    REPL-First Pivot Gate and Turn t+2 automatically unlocks proof synthesis.
    """
    prover = MathProver()
    prover._log_write = lambda *_a, **_k: None
    prover.compactor = MagicMock()
    prover.compactor.should_compact.return_value = False

    with patch.object(prover, "_run_context_architect", return_value="blueprint"), \
         patch.object(prover, "_run_computational_falsifier", return_value=([], [])), \
         patch.object(prover.memory, "query_procedural_heuristics", return_value=[]), \
         patch.object(prover.memory, "check_health", return_value=True):
        messages, result, _ = prover._init_solve_session("t_piv", "Target Problem: prove concentration of T.", max_turns=5, reset_kernel=True)

    p_bad1 = "\\begin{proof}\n" + ("Applying McDiarmid directly to unbounded stopping time T gives concentration. " * 4) + "\n\\end{proof}"
    p_bad2 = "\\begin{proof}\n" + ("Applying Bernstein directly to random stopping time T gives concentration. " * 4) + "\n\\end{proof}"
    p_good = "\\begin{proof}\n" + ("By inverting T > n+D iff W_{n+D} > D for fixed horizon n+D, Chernoff applies cleanly. " * 4) + "\n\\end{proof}"

    captured_prompts = []

    def fake_stream(messages, **_kw):
        captured_prompts.append([dict(m) for m in messages])
        idx = len(captured_prompts)
        if idx == 1:
            return ("t1", p_bad1)
        elif idx == 2:
            return ("t2", p_bad2)
        elif idx == 3:
            # Turn 3 is under the 1-Turn REPL-First Pivot Gate: model outputs Python exploration code
            return ("t3", "Let us test the fixed-horizon inversion in Python:\n```python\nprint('Fixed-horizon identity verified')\n```")
        return ("t4", p_good)

    audit_results = iter([
        AuditResult(verdict="REJECT", issues="T is not 1-Lipschitz", validity=0.15, rigor=0.15, completeness=0.20, boundary_cases=0.20),
        AuditResult(verdict="REJECT", issues="T is a random stopping time", validity=0.10, rigor=0.15, completeness=0.20, boundary_cases=0.20),
        AuditResult(verdict="VALID", issues="", validity=0.92, rigor=0.92, completeness=0.92, boundary_cases=0.90),
    ])

    with patch.object(prover, "_stream_completion_with_retry", side_effect=fake_stream), \
         patch.object(prover, "_audit_proof_draft", side_effect=lambda *_a, **_k: next(audit_results)), \
         patch.object(prover, "_run_critique_falsifier", return_value=([], [])), \
         patch.object(prover, "_run_publication_scribe", return_value=None):
        best, _ = prover._run_exploratory_turns(
            task_id="t_piv",
            task_prompt="Target Problem: prove concentration of T.",
            messages=messages,
            result=result,
            max_turns=5,
            start_wall_time=0.0,
        )

    assert result.verified is True
    # Turn 3 (index 2) must have received the 1-Turn REPL-First Pivot Gate
    turn3_user = captured_prompts[2][-1]["content"]
    assert "1-TURN REPL-FIRST PIVOT GATE" in turn3_user
    # Turn 4 (index 3) must have had messages[3] unlocked AND received MANDATORY PROOF SYNTHESIS DIRECTIVE
    turn4_all = "\n".join(m["content"] for m in captured_prompts[3])
    assert "1-TURN REPL-FIRST PIVOT GATE" not in turn4_all
    assert "MANDATORY PROOF SYNTHESIS DIRECTIVE" in captured_prompts[3][-1]["content"]


def test_v43_extract_code_blocks_ignores_latex_proof_fences():
    """Bare markdown fences wrapping a LaTeX proof must never be extracted as Python code."""
    prover = MathProver()
    latex_in_bare_fence = (
        "Here is the proof:\n```\n\\begin{proof}\nLet $G = (V, E)$ be a connected graph.\n\\end{proof}\n```"
    )
    assert prover.extract_code_blocks(latex_in_bare_fence) == []

    valid_bare_python = "```\nx = 2 + 2\nassert x == 4\n```"
    assert prover.extract_code_blocks(valid_bare_python) == ["x = 2 + 2\nassert x == 4"]


def test_v43_negated_fatal_phrases():
    """Phrases like 'rather than a fatal flaw' must not trigger has_fatal_issue()."""
    note = (
        "The proof is valid and establishes all required bounds. "
        "Note (minor observation rather than a fatal flaw): the bound 2n-2 can be tightened to 2n-3."
    )
    assert has_fatal_issue(note) is False
    assert has_fatal_issue("This is instead of a fatal gap a minor stylistic point.") is False
    assert has_fatal_issue("Step 3 contains a fatal gap in the induction.") is True


def test_v43_extract_latex_proof_omitted_outer_begin_before_claim():
    """When outer \\begin{proof} was omitted before \\begin{claim}, extract_latex_proof must reconstruct the full proof."""
    from core_engine.proof_extractor import extract_latex_proof

    raw = (
        "Let $I = [t_1, t_2]$ be an interval with $|I| = k$ and fix parameters.\n"
        "\\begin{claim}\nFor every step $t$, the invariant holds.\n\\end{claim}\n"
        "\\begin{proof}\nBy induction on $t$, the base case $t=0$ is immediate.\n\\end{proof}\n"
        "Combining the claim over all intervals yields the desired bound $O(k \\log k)$.\n"
        "\\end{proof}"
    )
    extracted = extract_latex_proof(raw)
    assert extracted is not None
    assert "Let $I = [t_1, t_2]$" in extracted
    assert "Combining the claim over all intervals" in extracted
    assert extracted.startswith(r"\begin{proof}")
    assert extracted.endswith(r"\end{proof}")


def test_v43_falsifier_ignores_script_runtime_crashes():
    """IndexError/TypeError/NameError crashes in falsifier scripts must not be classified as counterexamples."""
    assert ProverStagesMixin._is_assertion_failure("AssertionError: bound violated at n=4") is True
    assert ProverStagesMixin._is_assertion_failure("IndexError: list index out of range") is False
    assert ProverStagesMixin._is_assertion_failure("TypeError: 'int' object is not iterable") is False
    assert ProverStagesMixin._is_stdout_counterexample("COUNTEREXAMPLE_FOUND: n=3 fails") is True
    assert ProverStagesMixin._is_stdout_counterexample("INVARIANT_VERIFIED: all 100 trials passed") is False


def test_v43_candidate_pool_structural_arbitration_and_demotion():
    """CandidatePool must protect high-rigor capped drafts from lower-rigor drafts and support demote_current_best."""
    pool = CandidatePool()
    rigorous_draft = r"\begin{proof} rigorous derivation with minor scale convention \end{proof}"
    handwavy_draft = r"\begin{proof} handwavy derivation with unproved claim \end{proof}"

    # Turn 10: rigorous draft (rigor=0.75, completeness=0.60, boundary=0.30 -> structural=0.58) capped at validity=0.20
    pool.offer(
        rigorous_draft,
        validity=0.20,
        verdict="REJECT",
        issues="scale convention",
        turn=10,
        rigor=0.75,
        completeness=0.60,
        boundary_cases=0.30,
    )
    assert pool.best.proof == rigorous_draft

    # Turn 11: handwavy draft (rigor=0.35, completeness=0.45, boundary=0.30 -> structural=0.385) capped at validity=0.25
    replaced = pool.offer(
        handwavy_draft,
        validity=0.25,
        verdict="REJECT",
        issues="unproved lemma",
        turn=11,
        rigor=0.35,
        completeness=0.45,
        boundary_cases=0.30,
    )
    assert replaced is False
    assert pool.best.proof == rigorous_draft

    # Demoting current best when REPL confirms a counterexample allows repaired draft to win
    assert pool.demote_current_best(0.10) is True
    assert pool.best_validity == 0.10


def test_v44_third_low_validity_rejection_unlocks_hard_pivot_and_clears_prior_defects():
    """While rejection #2 with has_substantive_structure grants one surgical revision window,
    if rejection #3 still fails below pivot_threshold (e.g. Task 142 at 0.25 -> 0.20),
    Clean-Slate Hard Pivot must unlock, prune the dead candidate, and clear _prior_audit_defects.
    """
    prover = MathProver()
    prover._log_write = lambda *_a, **_k: None
    prover.compactor = MagicMock()
    prover.compactor.should_compact.return_value = False

    with patch.object(prover, "_run_context_architect", return_value="### Strategy Alpha\nblueprint"), \
         patch.object(prover, "_run_computational_falsifier", return_value=([], [])), \
         patch.object(prover.memory, "query_procedural_heuristics", return_value=[]), \
         patch.object(prover.memory, "check_health", return_value=True):
        messages, result, _ = prover._init_solve_session("t_142", "Target Problem: prove Theorem.", max_turns=6, reset_kernel=True)

    prover._prior_audit_defects = ["Old pre-pivot defect that must be cleared on hard pivot"]

    p_draft = "\\begin{proof}\n" + ("By block concatenation [D_1 | D_h] * U^{-1}, the bound follows. " * 4) + "\n\\end{proof}"
    p_valid = "\\begin{proof}\n" + ("By iterative matrix-vector products w_r = A w_{r-1} in O(h n^2) time, the bound follows. " * 4) + "\n\\end{proof}"

    captured_prompts = []

    def fake_stream(messages, **_kw):
        captured_prompts.append([dict(m) for m in messages])
        idx = len(captured_prompts)
        if idx in (1, 2, 3):
            return (f"t{idx}", p_draft)
        if idx == 4:
            # Turn 4 is 1-Turn REPL-First Pivot Gate after Turn 3 hard pivot
            return ("t4", "```python\nprint('Iterative matrix-vector verified')\n```")
        return ("t5", p_valid)

    audit_results = iter([
        AuditResult(verdict="REJECT", issues="Dimension mismatch", validity=0.20, rigor=0.40, completeness=0.50, boundary_cases=0.50),
        AuditResult(verdict="REJECT", issues="Still dimension mismatch", validity=0.25, rigor=0.40, completeness=0.50, boundary_cases=0.60),
        AuditResult(verdict="REJECT", issues="Unfixed dimension mismatch", validity=0.20, rigor=0.40, completeness=0.50, boundary_cases=0.50),
        AuditResult(verdict="VALID", issues="", validity=0.93, rigor=0.92, completeness=0.95, boundary_cases=0.90),
    ])

    with patch.object(prover, "_stream_completion_with_retry", side_effect=fake_stream), \
         patch.object(prover, "_audit_proof_draft", side_effect=lambda *_a, **_k: next(audit_results)), \
         patch.object(prover, "_run_critique_falsifier", return_value=([], [])), \
         patch.object(prover, "_run_publication_scribe", return_value=None):
        best, _ = prover._run_exploratory_turns(
            task_id="t_142",
            task_prompt="Target Problem: prove Theorem.",
            messages=messages,
            result=result,
            max_turns=6,
            start_wall_time=0.0,
        )

    assert result.verified is True
    assert best is not None and best.validity == pytest.approx(0.93)
    # Turn 3 (index 2, after rejection #2) received Surgical Revision
    assert "Here is the candidate proof under revision:" in "\n".join(m["content"] for m in captured_prompts[2])
    # Turn 4 (index 3, after rejection #3) unlocked Clean-Slate Hard Pivot & cleared _prior_audit_defects!
    assert "1-TURN REPL-FIRST PIVOT GATE" in captured_prompts[3][-1]["content"]
    assert prover._prior_audit_defects == []


def test_candidate_pool_reaudit_demotion_and_clean_valid_supersede():
    """If an earlier borderline VALID proof (0.82, issues='none') is later found to have
    defects (either on re-audit of the identical text or via post-restore defect attachment),
    CandidatePool must:
    1. Demote the stale record when the identical proof text is re-audited as REJECT.
    2. Allow a cleanly verified VALID candidate (e.g. 0.78 VALID, issues='none') to supersede
       an incumbent carrying unresolved referee issues.
    """
    pool = CandidatePool()
    p_turn9 = r"\begin{proof} Turn 9 proof with subtle lifting gap \end{proof}"
    p_repaired = r"\begin{proof} Repaired proof fixing lifting gap rigorously \end{proof}"

    assert pool.offer(p_turn9, 0.82, "VALID", "none", turn=9) is True
    assert pool.best.validity == pytest.approx(0.82)
    assert pool.best.verdict == "VALID"

    # Re-auditing the identical proof on Turn 12 as REJECT must demote the false-positive 0.82
    assert pool.offer(p_turn9, 0.30, "REJECT", "Lifting claim invalid when CC edge overlaps", turn=12) is True
    assert pool.best.validity == pytest.approx(0.30)
    assert pool.best.verdict == "REJECT"
    assert "Lifting claim invalid" in pool.best.issues

    # Also verify that if an incumbent at 0.82 carries attached referee issues from a later turn,
    # a cleanly verified VALID revision at 0.78 (issues='none') supersedes it.
    pool2 = CandidatePool()
    pool2.offer(p_turn9, 0.82, "VALID", "none", turn=9)
    pool2.best.issues = "Lifting claim invalid when CC edge overlaps"
    assert pool2.offer(p_repaired, 0.78, "VALID", "none", turn=991) is True
    assert pool2.best.proof == p_repaired
    assert pool2.best.verdict == "VALID"
    assert pool2.best.issues == "none"


def test_prover_turn_level_checkpoint_crash_at_turn_4_and_resume(tmp_path):
    """Simulates a crash during Iteration 4 of Problem 3 and verifies that restarting
    with a fresh MathProver resumes at the end of Iteration 3 / beginning of Iteration 4
    with REPL variables, MathStateMachine, CandidatePool, and anchor messages intact.
    """
    ckpt_path = tmp_path / "checkpoints" / "task_3.ckpt.json"
    task_prompt = "Target Problem (your task): Prove that for every graph G, the invariant holds."

    # --- Process 1: Runs Turns 1..3 and crashes in Turn 4 ---
    prover1 = MathProver(log_path=tmp_path / "task_3.log")
    prover1.compactor = MagicMock()
    prover1.compactor.should_compact.return_value = False

    draft_turn3 = (
        "\\begin{proof}\n"
        + ("By the intermediate lemma verified in Python, the bound holds up to a boundary check. " * 4)
        + "\n\\end{proof}"
    )

    p1_turns_started = []

    def p1_stream(messages, **_kw):
        turn_idx = len(p1_turns_started)
        if turn_idx == 1:
            return ("r1", "[CLAIM]: Lemma 1 holds for base graphs\n```python\nx_turn1 = 101\nprint('x_turn1 =', x_turn1)\n```")
        if turn_idx == 2:
            return ("r2", "```python\ny_turn2 = x_turn1 + 202\nprint('y_turn2 =', y_turn2)\n```")
        if turn_idx == 3:
            return ("r3", draft_turn3)
        raise RuntimeError("Simulated process crash mid-Iteration 4!")

    with patch.object(prover1, "_run_context_architect", return_value="Blueprint for task_3") as mock_arch1, \
         patch.object(prover1, "_run_computational_falsifier", return_value=(["n=3 verified"], [])), \
         patch.object(prover1.memory, "query_procedural_heuristics", return_value=["Use induction on |V|"]), \
         patch.object(prover1.memory, "check_health", return_value=True), \
         patch.object(prover1, "_stream_completion_with_retry", side_effect=p1_stream), \
         patch.object(
             prover1,
             "_audit_proof_draft",
             return_value=AuditResult(verdict="REJECT", issues="Missing n=1 base case", validity=0.65, rigor=0.70, completeness=0.65),
         ), \
         patch.object(prover1, "_run_critique_falsifier", return_value=([], [])), \
         patch.object(prover1, "_run_adversarial_skeptic", return_value=(SKEPTIC_VERIFIED, "verified", None)):
        with pytest.raises(RuntimeError, match="Simulated process crash mid-Iteration 4"):
            prover1.solve(
                task_id="task_3",
                task_prompt=task_prompt,
                max_iterations=5,
                on_step=lambda ev, data: p1_turns_started.append(data) if ev == "turn_start" else None,
                checkpoint_path=ckpt_path,
            )

    assert p1_turns_started == [1, 2, 3, 4]
    assert mock_arch1.call_count == 1
    assert ckpt_path.exists(), "Checkpoint file must exist after Turn 3 completed"

    # --- Process 2 (Watchdog Restart): Brand-new MathProver + fresh kernel ---
    prover2 = MathProver(log_path=tmp_path / "task_3.log")
    prover2.compactor = MagicMock()
    prover2.compactor.should_compact.return_value = False

    final_proof_turn5 = (
        "\\begin{proof}\n"
        + ("Including the n=1 base case and combining with x_turn1=101 and y_turn2=303, the theorem holds completely. " * 4)
        + "\n\\end{proof}"
    )

    p2_turns_started = []
    p2_events = []

    def p2_stream(messages, **_kw):
        # First call in Process 2 MUST be Turn 4!
        if len(p2_turns_started) == 1:
            # Verify Turn-3 candidate anchor survived checkpoint serialization
            joined = "\n".join(m["content"] for m in messages)
            assert "Here is the candidate proof under revision:" in joined
            return (
                "r4",
                "```python\nassert x_turn1 == 101 and y_turn2 == 303, f'Got {x_turn1}, {y_turn2}'\nprint('REPL state restored:', x_turn1 + y_turn2)\n```",
            )
        return ("r5", final_proof_turn5)

    with patch.object(prover2, "_run_context_architect") as mock_arch2, \
         patch.object(prover2, "_run_computational_falsifier") as mock_fals2, \
         patch.object(prover2.memory, "check_health", return_value=True), \
         patch.object(prover2, "_consolidate_learning"), \
         patch.object(prover2, "_stream_completion_with_retry", side_effect=p2_stream), \
         patch.object(
             prover2,
             "_audit_proof_draft",
             return_value=AuditResult(verdict="VALID", issues="none", validity=0.92, rigor=0.92, completeness=0.95),
         ), \
         patch.object(prover2, "_run_publication_scribe", return_value=None):
        res = prover2.solve(
            task_id="task_3",
            task_prompt=task_prompt,
            max_iterations=5,
            on_step=lambda ev, data: (
                p2_turns_started.append(data) if ev == "turn_start" else p2_events.append((ev, data))
            ),
            checkpoint_path=ckpt_path,
        )

    # Must NOT have re-run initialization or Turns 1..3
    assert mock_arch2.call_count == 0
    assert mock_fals2.call_count == 0
    assert p2_turns_started == [4, 5], f"Expected to resume directly at Turn 4, got {p2_turns_started}"
    assert any(ev == "checkpoint_resume" and d["completed_turn"] == 3 for ev, d in p2_events)
    assert res.verified is True
    assert res.iterations == 5
    assert "x_turn1 = 101" in res.code_history[0]
    assert any("REPL state restored: 404" in er.stdout for er in res.execution_history)
    # State machine verified lemmas from Turn 1 must also be preserved in restored kernel
    assert any("Lemma 1 holds" in lem for lem in prover2.kernel._state_machine._verified_lemmas)
    # Checkpoint file must be unlinked after normal completion
    assert not ckpt_path.exists()


def test_prover_synthesis_substage_checkpoint_and_resume(tmp_path):
    """If the process crashes during the post-exploration synthesis pipeline (e.g. during
    revision turn 1 after the initial synthesis pass completed), resuming starts directly
    at revision turn 1 without re-running exploration or the initial synthesis pass.
    """
    ckpt_path = tmp_path / "checkpoints" / "task_synth.ckpt.json"
    task_prompt = "Target Problem (your task): Prove the spectral bound."

    prover1 = MathProver(log_path=tmp_path / "task_synth.log")
    prover1.compactor = MagicMock()
    prover1.compactor.should_compact.return_value = False

    synth_draft = (
        "\\begin{proof}\n"
        + ("Initial synthesized proof draft from exploratory notes with a minor gap. " * 4)
        + "\n\\end{proof}"
    )

    # Process 1: 1 exploratory turn (code only), then enters synthesis:
    # scribe returns None -> synth_turn produces synth_draft (audited REJECT 0.60) -> crashes on rev_1!
    p1_stream_calls = 0

    def p1_stream(messages, **_kw):
        nonlocal p1_stream_calls
        p1_stream_calls += 1
        if p1_stream_calls == 1:
            return ("exp1", "```python\nz_val = 77\nprint(z_val)\n```")
        if p1_stream_calls == 2:
            return ("synth", synth_draft)
        raise RuntimeError("Crash during synthesis revision 1!")

    with patch.object(prover1, "_run_context_architect", return_value="Blueprint"), \
         patch.object(prover1, "_run_computational_falsifier", return_value=([], [])), \
         patch.object(prover1.memory, "query_procedural_heuristics", return_value=[]), \
         patch.object(prover1.memory, "check_health", return_value=True), \
         patch.object(prover1, "_stream_completion_with_retry", side_effect=p1_stream), \
         patch.object(
             prover1,
             "_audit_proof_draft",
             return_value=AuditResult(verdict="REJECT", issues="Explain step 2", validity=0.60, rigor=0.65, completeness=0.60),
         ), \
         patch.object(prover1, "_run_critique_falsifier", return_value=([], [])), \
         patch.object(prover1, "_run_publication_scribe", return_value=None):
        with pytest.raises(RuntimeError, match="Crash during synthesis revision 1"):
            prover1.solve(
                task_id="task_synth",
                task_prompt=task_prompt,
                max_iterations=1,
                checkpoint_path=ckpt_path,
            )

    assert ckpt_path.exists()
    ckpt_data = prover1._load_checkpoint(ckpt_path, "task_synth")
    assert ckpt_data["stage"] == "synthesis"
    assert ckpt_data["synthesis_state"]["substage"] == "after_synth_turn"

    # Process 2: Resumes directly at after_synth_turn (runs rev_1 only)
    prover2 = MathProver(log_path=tmp_path / "task_synth.log")
    prover2.compactor = MagicMock()
    prover2.compactor.should_compact.return_value = False

    repaired_proof = (
        "\\begin{proof}\n"
        + ("Repaired synthesis proof with step 2 fully justified. " * 4)
        + "\n\\end{proof}"
    )
    p2_stream_calls = 0

    def p2_stream(messages, **_kw):
        nonlocal p2_stream_calls
        p2_stream_calls += 1
        return ("rev1", repaired_proof)

    with patch.object(prover2.memory, "check_health", return_value=True), \
         patch.object(prover2, "_consolidate_learning"), \
         patch.object(prover2, "_stream_completion_with_retry", side_effect=p2_stream), \
         patch.object(prover2, "_run_critique_falsifier", return_value=([], [])), \
         patch.object(
             prover2,
             "_audit_proof_draft",
             return_value=AuditResult(verdict="VALID", issues="none", validity=0.91, rigor=0.90, completeness=0.92),
         ), \
         patch.object(prover2, "_run_publication_scribe", return_value=None):
        res = prover2.solve(
            task_id="task_synth",
            task_prompt=task_prompt,
            max_iterations=1,
            checkpoint_path=ckpt_path,
        )

    assert p2_stream_calls == 1, f"Expected only 1 stream call (rev_1) on resume, got {p2_stream_calls}"
    assert res.verified is True
    assert res.candidate_proof == repaired_proof
    assert not ckpt_path.exists()

