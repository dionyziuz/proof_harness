"""Unit tests for core_engine.judging package (Lean 4 and TCS judges)."""

import pytest
from core_engine.judging import (
    extract_lean_code,
    judge_lean_proof,
    judge_tcs_proof,
    judge_proof,
    build_a_prompt,
    build_b_prompt,
    parse_a_vote,
    parse_b_json,
    clamp01,
    extract_target,
    A1_INSTRUCTIONS,
)


def test_tcs_prompt_builders():
    prompt = "Context: Let G be a graph.\nTarget Problem (your task): Prove G is 3-colorable."
    target = extract_target(prompt)
    assert target == "Prove G is 3-colorable."

    gt = "Proof: We induct on vertices..."
    student = "Proof: Consider coloring..."

    prompt_a = build_a_prompt(prompt, target, gt, student)
    assert A1_INSTRUCTIONS in prompt_a
    assert "'student_answer':" in prompt_a
    assert student in prompt_a

    prompt_b = build_b_prompt(target, gt, student)
    assert "mathematical referee" in prompt_b
    assert target in prompt_b


def test_tcs_parsers():
    # Test vote parser
    assert parse_a_vote("1") == "1"
    assert parse_a_vote("0") == "0"
    assert parse_a_vote("The verdict is 1") == "1"
    assert parse_a_vote("Verdict: 0.") == "0"
    assert parse_a_vote("inconclusive text") is None

    # Test B JSON parser
    raw_json = '{"rigor": 0.95, "completeness": 1.0, "boundary_cases": 0.9, "confidence": 0.8, "validity": 0.95, "issues": "none"}'
    parsed = parse_b_json(raw_json)
    assert parsed is not None
    assert parsed["validity"] == 0.95
    assert parsed["rigor"] == 0.95

    # Test B JSON parser with markdown fences
    fenced_json = f"```json\n{raw_json}\n```"
    parsed_fenced = parse_b_json(fenced_json)
    assert parsed_fenced is not None
    assert parsed_fenced["validity"] == 0.95

    # Test clamp01
    assert clamp01(1.5) == 1.0
    assert clamp01(-0.2) == 0.0
    assert clamp01(0.75) == 0.75
    assert clamp01("invalid") is None


def test_tcs_empty_candidate():
    res = judge_tcs_proof(
        candidate="",
        problem_prompt="Target Problem (your task): x = y",
        gt_solution="x = y because...",
    )
    assert res["status"] == "no_proof"
    assert res["verdict"] == "0"


def test_unified_judge_auto_lean():
    lean_doc = """
```lean
theorem test_id (x : Nat) : x = x := rfl
```
"""
    res = judge_proof(lean_doc)
    assert res["mode"] == "lean"
    assert res["passed"] is True
    assert res["verdict"] == "1"


def test_unified_judge_auto_tcs_empty():
    text_doc = "This is a latex proof without ground truth."
    res = judge_proof(text_doc)
    assert res["mode"] == "tcs"
    assert res["passed"] is False


# ---------------------------------------------------------------------------
# Config-B parsing: truncated JSON recovery and explicit parse_error status
# ---------------------------------------------------------------------------
import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock

from core_engine.judging import B_PARSE_ERROR, B_PARSE_OK, B_PARSE_RECOVERED, parse_b_validity
from core_engine.llm_client import LLMResponse

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _resp(content, finish_reason="stop"):
    return LLMResponse(content=content, finish_reason=finish_reason)


def _b(v):
    return _resp('{"rigor": 0.5, "completeness": 0.5, "boundary_cases": 0.5, "confidence": 0.5, '
                 f'"validity": {v}, "issues": "issue at {v}"}}')


def test_parse_b_full_json_status_ok():
    v, status, parsed = parse_b_validity('{"validity": 0.0, "issues": "fatal gap"}')
    assert v == 0.0 and status == B_PARSE_OK  # a REAL zero is distinguishable from a parse failure
    assert parsed["issues"] == "fatal gap"


def test_parse_b_recovers_validity_from_truncated_json():
    truncated = '{"rigor": 0.8, "completeness": 0.7, "boundary_cases": 0.6, "confidence": 0.9, "validity": 0.65, "issues": "The proof of Lemma 2 neglects the case wh'
    v, status, parsed = parse_b_validity(truncated)
    assert v == 0.65
    assert status == B_PARSE_RECOVERED
    assert parsed["rigor"] == 0.8
    assert parsed["issues"].startswith("The proof of Lemma 2")


def test_parse_b_recovers_from_unclosed_fence():
    v, status, _ = parse_b_validity('```json\n{"validity": 0.42,\n "issues": "x')
    assert v == 0.42 and status == B_PARSE_RECOVERED


def test_parse_b_number_cut_at_end_is_not_trusted():
    # '0.' could be the prefix of '0.85'; must not be read as validity 0.0
    v, status, _ = parse_b_validity('{"rigor": 0.8, "validity": 0.')
    assert v is None and status == B_PARSE_ERROR


def test_parse_b_null_validity_is_parse_error_not_typeerror():
    v, status, parsed = parse_b_validity('{"validity": null, "issues": "could not decide"}')
    assert v is None and status == B_PARSE_ERROR
    assert parsed["parse_status"] == B_PARSE_ERROR


def test_parse_b_garbage_is_parse_error():
    assert parse_b_validity("I think the proof is fine.") == (None, B_PARSE_ERROR, None)
    assert parse_b_json("no json here") is None


# ---------------------------------------------------------------------------
# Replicated judging
# ---------------------------------------------------------------------------
def _judge(client, **kw):
    kw.setdefault("threshold", 0.50)
    return judge_tcs_proof(
        candidate=r"\begin{proof} p \end{proof}",
        problem_prompt="Context\nTarget Problem (your task): prove it",
        gt_solution="gt",
        client=client,
        config="tcs",
        **kw,
    )


def test_replicates_median_mean_spread_pass_probability():
    client = MagicMock()
    client.complete.side_effect = [_b(0.4), _b(0.7), _b(0.6)]
    res = _judge(client)  # default k = 3
    b = res["b"]
    assert client.complete.call_count == 3
    assert b["replicate_validities"] == [0.4, 0.7, 0.6]
    assert b["validity"] == 0.6 and res["validity"] == 0.6
    assert b["validity_mean"] == pytest.approx(0.5667, abs=1e-4)
    assert b["validity_spread"] == pytest.approx(0.3)
    assert b["pass_probability"] == pytest.approx(0.6667, abs=1e-4)
    assert res["pass_probability"] == b["pass_probability"]
    assert b["verdict"] == "1" and res["verdict"] == "1"
    assert res["status"] == "judged"
    assert b["scores"]["issues"] == "issue at 0.6"  # representative replicate = closest to median


def test_replicates_still_run_when_first_call_fails_to_parse():
    client = MagicMock()
    client.complete.side_effect = [
        _resp('{"rigor": 0.9, "validity": 0.', finish_reason="length"),  # truncated
        _resp("still garbage"),                                          # repair retry fails too
        _b(0.8),
        _b(0.3),
    ]
    res = _judge(client, max_tokens=8192)
    b = res["b"]
    assert client.complete.call_count == 4
    assert b["replicate_statuses"][0] == B_PARSE_ERROR
    assert b["n_parse_errors"] == 1
    assert b["replicate_validities"] == [0.8, 0.3]
    assert b["validity"] == pytest.approx(0.55)
    assert b["pass_probability"] == 0.5
    assert res["status"] == "judged"
    # The repair retry: thinking disabled, larger max_tokens.
    retry_kwargs = client.complete.call_args_list[1].kwargs
    assert retry_kwargs["enable_thinking"] is False
    assert retry_kwargs["max_tokens"] > 8192


def test_length_truncation_triggers_repair_retry_and_uses_it():
    client = MagicMock()
    client.complete.side_effect = [_resp('{"validity": 0.2, "issues": "cut', finish_reason="length"), _b(0.9)]
    res = _judge(client, replicates=1, borderline_replicates=False)
    assert client.complete.call_count == 2
    assert res["b"]["replicate_validities"] == [0.9]
    assert res["b"]["replicate_retried"] == [True]


def test_all_replicates_unparseable_gives_judge_parse_error():
    client = MagicMock()
    client.complete.side_effect = [_resp("nope")] * 6
    res = _judge(client)
    assert res["status"] == "judge_parse_error"
    assert res["validity"] is None and res["b"]["validity"] is None
    assert res["b"]["verdict"] is None and res["pass_probability"] is None
    assert res["b"]["n_parse_errors"] == 3


def test_adaptive_stops_on_decisive_first_call_and_extends_on_borderline():
    client = MagicMock()
    client.complete.side_effect = [_b(0.95)]
    res = _judge(client, adaptive=True)
    assert client.complete.call_count == 1 and res["b"]["n_replicates"] == 1

    client = MagicMock()
    client.complete.side_effect = [_b(0.55), _b(0.2), _b(0.65)]
    res = _judge(client, adaptive=True)  # 0.55 within 0.50 +/- 0.20
    assert client.complete.call_count == 3
    assert res["b"]["borderline_band"] == [0.3, 0.7]


def test_default_threshold_is_tcs_pass_threshold(monkeypatch):
    from core_engine import tcs_config

    monkeypatch.setattr(tcs_config, "TCS_PASS_THRESHOLD", 0.50)
    client = MagicMock()
    client.complete.side_effect = [_b(0.6)] * 3
    res = judge_tcs_proof(candidate="proof", problem_prompt="p", gt_solution="g", client=client)
    assert res["threshold"] == 0.50
    assert res["verdict"] == "1"  # 0.6 would have failed under the old 0.70 default


def test_tcs_config_defaults_consistent():
    from core_engine.tcs_config import TCS_PASS_THRESHOLD, TCSConfig, load_tcs_config

    d = TCSConfig()
    loaded = load_tcs_config(reload=True)
    assert d.pipeline_judge_config in ("tcs", "b")
    assert d.solver_thinking_budget == 24576
    assert d.pipeline_judge_threshold == TCS_PASS_THRESHOLD
    assert loaded.pipeline_judge_threshold == TCS_PASS_THRESHOLD


# ---------------------------------------------------------------------------
# Bench runner: judge_one delegates to judge_tcs_proof
# ---------------------------------------------------------------------------
def _bench_module():
    sys.path.insert(0, str(_SCRIPTS))
    import run_tcs_bench_judge
    return run_tcs_bench_judge


def _bench_args(**kw):
    base = dict(config="b", threshold=0.50, temp_a=0.3, temp_b=0.1, max_tokens=4096, replicates=3, adaptive=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_judge_one_null_validity_is_parse_error_not_typeerror():
    mod = _bench_module()
    client = MagicMock()
    client.complete.side_effect = [_resp('{"validity": null, "issues": "?"}')] * 6
    rec = {"task_id": "task_001", "candidate_proof": "proof text"}
    out = mod.judge_one(rec, {"prompt": "Target Problem (your task): t", "ground_truth_solution": "g"},
                        _bench_args(), client, True, 30.0)
    assert out["status"] == "judge_parse_error"
    assert out["b"]["validity"] is None


def test_judge_one_records_replicate_fields():
    mod = _bench_module()
    client = MagicMock()
    client.complete.side_effect = [_b(0.7), _b(0.45), _b(0.6)]
    rec = {"task_id": "task_002", "candidate_proof": "proof text"}
    out = mod.judge_one(rec, {"prompt": "p", "ground_truth_solution": "g"}, _bench_args(), client, True, 30.0)
    assert out["status"] == "judged"
    assert out["b"]["validity"] == 0.6
    assert out["replicate_validities"] == [0.7, 0.45, 0.6]
    assert out["pass_probability"] == pytest.approx(0.6667, abs=1e-4)
    assert client.complete.call_count == 3
    assert mod.TARGET_RE is __import__("core_engine.judging", fromlist=["TARGET_RE"]).TARGET_RE


# ---------------------------------------------------------------------------
# Pipeline aggregation: None is never 0, parse errors excluded from denominators
# ---------------------------------------------------------------------------
def _pipeline_module():
    sys.path.insert(0, str(_SCRIPTS))
    import run_tcs_continuous_pipeline
    return run_tcs_continuous_pipeline


def test_pipeline_entry_keeps_none_validity():
    mod = _pipeline_module()
    judge = {"status": "judge_parse_error", "b": {"validity": None, "verdict": None}}
    e = mod.build_tracker_entry(1, "t1", {"status": "completed", "candidate_proof": "x"}, judge, 0.50)
    assert e["judge_b_validity"] is None
    assert e["judge_parse_error"] is True and e["scored"] is False
    assert e["pass_b"] is False

    # Legacy record shape (status 'judged' but validity missing) is also a parse error.
    e2 = mod.build_tracker_entry(2, "t2", {}, {"status": "judged", "b": {"validity": None}}, 0.50)
    assert e2["judge_b_validity"] is None and e2["judge_parse_error"] is True


def test_pipeline_summary_excludes_parse_errors_and_lean_from_headline(tmp_path):
    mod = _pipeline_module()
    card = mod.PipelineScorecard(total_target=4, solve_dir=tmp_path, threshold=0.50)
    solve = {"status": "completed", "candidate_proof": "x"}
    card.record_judgment(1, "t1", solve, {"status": "judged", "b": {"validity": 0.8, "pass_probability": 1.0}})
    card.record_judgment(2, "t2", solve, {"status": "judged", "b": {"validity": 0.2, "pass_probability": 0.0}})
    card.record_judgment(3, "t3", solve, {"status": "judge_parse_error", "b": {"validity": None}})
    s = card.record_judgment(4, "t4", solve, {"status": "judged", "b": {"validity": 0.45},
                                               "a": {"verdict": "1", "ones": 4},
                                               "lean": {"passed": True, "verdict": "1"}})
    assert s["total_checked"] == 4
    assert s["total_scored"] == 3
    assert s["judge_parse_errors"] == 1
    assert s["judge_b"]["passes"] == 1
    assert s["judge_b"]["pass_percentage"] == pytest.approx(33.33, abs=0.01)
    assert s["proven"]["passes"] == 1  # alias of headline; no OR with A / Lean
    assert s["combined_either"]["passes"] == 2  # secondary: A OR B
    assert s["lean"]["passes"] == 1 and s["lean"]["experimental"] is True
    assert s["directionally_sound"]["passes"] == 2  # 0.8 and 0.45 (>= 0.40)
    assert "0.50" in s["judge_b"]["rule"] and "0.50" in s["headline_label"]
    assert s["judge_b_expected_pass_rate"] == 50.0
    assert (tmp_path / "pipeline_metrics.json").exists()
    rows = [r for r in s["recent_tasks"] if r["task_id"] == "t3"]
    assert rows[0]["judge_b_validity"] is None
