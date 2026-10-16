import pytest
from core_engine.memory import CoALAMemoryManager, TurboVecUnavailableError


def test_turbovec_health():
    mgr = CoALAMemoryManager()
    assert mgr.check_health() is True


def test_turbovec_offline_raises_error():
    # Test strict requirement: NO silent fallback, must raise TurboVecUnavailableError
    mgr = CoALAMemoryManager(base_url="http://127.0.0.1:99999")
    with pytest.raises(TurboVecUnavailableError):
        mgr.check_health()


def test_query_and_add_procedural():
    mgr = CoALAMemoryManager()
    rule = "Always factor polynomials using sp.factor() before computing roots in SymPy."
    added = mgr.add_procedural_heuristic(rule)
    # Could be True or False (if duplicate), but shouldn't raise
    assert isinstance(added, bool)

    results = mgr.query_procedural_heuristics("polynomial factorization", limit=3)
    assert isinstance(results, list)
    assert len(results) > 0


def test_record_episode():
    mgr = CoALAMemoryManager()
    success = mgr.record_episode(
        problem="Test problem: x^2 = 4",
        code_blocks=["solve(Eq(x**2, 4), x)"],
        verified=True,
        answer="2, -2",
        critique="Simple quadratic equation.",
    )
    assert success is True
