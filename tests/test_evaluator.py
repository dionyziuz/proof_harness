from core_engine.evaluator import normalize_answer, answers_match


def test_normalize_answer():
    assert normalize_answer("233168") == "233168"
    assert normalize_answer(" 233,168 ") == "233168"
    assert normalize_answer(r"\text{233168}") == "233168"
    assert normalize_answer(r"\mathbf{233168}") == "233168"
    assert normalize_answer("$233168$") == "233168"
    assert normalize_answer("0.464399") == "0.464399"


def test_answers_match():
    assert answers_match("233168", "233168")
    assert answers_match(r"\text{233,168}", "233168")
    assert answers_match(r"\boxed{906609}", "906609")
    assert answers_match("0.4643991", "0.464399")
    assert not answers_match("123", "456")
    assert not answers_match(None, "123")


def test_evaluator_defaults_and_harness_version():
    from core_engine.evaluator import EulerEvaluator, HARNESS_VERSION
    from pathlib import Path

    evaluator = EulerEvaluator()
    assert evaluator.enable_web_research is False
    assert evaluator.harness_version == "2.0"
    assert HARNESS_VERSION == "2.0"


def test_smart_cache_filters_mismatched_harness_version(tmp_path, monkeypatch):
    import json
    from core_engine.evaluator import EulerEvaluator

    eval_dir = tmp_path / "eval_results"
    eval_dir.mkdir()

    # Old legacy run (no harness_version)
    old_file = eval_dir / "eval_euler_old.json"
    with open(old_file, "w") as f:
        json.dump({
            "records": [{"problem_id": 1, "is_correct": True, "title": "P1", "expected_answer": "10", "predicted_answer": "10"}]
        }, f)

    # Run with mismatched harness version (e.g. 1.0)
    v1_file = eval_dir / "eval_euler_v1.json"
    with open(v1_file, "w") as f:
        json.dump({
            "harness_version": "1.0",
            "records": [{"problem_id": 2, "is_correct": True, "title": "P2", "expected_answer": "20", "predicted_answer": "20"}]
        }, f)

    # Current run matching harness version 2.0
    v2_file = eval_dir / "eval_euler_v2.json"
    with open(v2_file, "w") as f:
        json.dump({
            "harness_version": "2.0",
            "records": [{"problem_id": 3, "is_correct": True, "title": "P3", "expected_answer": "30", "predicted_answer": "30"}]
        }, f)

    monkeypatch.setattr("core_engine.evaluator.DEFAULT_EVAL_DIR", eval_dir)
    evaluator = EulerEvaluator(harness_version="2.0")
    passed = evaluator.get_passed_problem_records()

    # Problem 1 (legacy) and Problem 2 (v1.0) must be rejected; Problem 3 (v2.0) must be loaded
    assert 1 not in passed
    assert 2 not in passed
    assert 3 in passed
    assert passed[3].predicted_answer == "30"

