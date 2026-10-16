import pytest
from core_engine.kernel import PersistentMathKernel
from core_engine.research import oeis_lookup, search_math, read_paper


def test_oeis_lookup_fibonacci():
    results = oeis_lookup([1, 1, 2, 3, 5, 8, 13], max_results=2)
    assert isinstance(results, list)
    assert len(results) > 0
    top = results[0]
    assert "A000045" in top["id"] or "fibonacci" in top["name"].lower()


def test_kernel_research_prelude():
    kernel = PersistentMathKernel()
    res = kernel.execute("""
seq = [1, 2, 4, 8, 16]
found = oeis_lookup(seq, max_results=1)
len(found) > 0 and 'A000079' in found[0]['id']
""")
    assert res.success
    assert res.result_repr == "True" or res.result_value is True


def test_search_math():
    results = search_math("Euler totient function", count=2)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "title" in results[0]
    assert "url" in results[0]


def test_read_paper():
    text = read_paper("1511.06862", max_chars=500)
    assert isinstance(text, str)
    assert len(text) > 50
    assert "Diophantine" in text or "Title" in text or "URL" in text
