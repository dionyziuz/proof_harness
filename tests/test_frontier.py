import pytest
from core_engine.evaluator_frontier import FrontierMathEvaluator, FRONTIER_DATASET_PATH


def test_frontier_dataset_loads():
    evaluator = FrontierMathEvaluator()
    assert len(evaluator.problems) == 5
    for pid in [1, 2, 3, 4, 5]:
        assert pid in evaluator.problems
        p = evaluator.problems[pid]
        assert "tier" in p
        assert "answer" in p
        assert len(p["answer"]) > 0


def test_frontier_tier_filtering():
    evaluator = FrontierMathEvaluator()
    tier1_ids = [pid for pid, p in evaluator.problems.items() if p["tier"] == 1]
    assert len(tier1_ids) == 1
    assert tier1_ids[0] == 5

    tier4_ids = [pid for pid, p in evaluator.problems.items() if p["tier"] == 4]
    assert len(tier4_ids) == 2
    assert set(tier4_ids) == {1, 2}
