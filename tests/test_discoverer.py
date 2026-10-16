import pytest
from core_engine.discoverer import MathDiscoverer, DiscoveryResult
from core_engine.kernel import PersistentMathKernel


def test_discoverer_initialization():
    kernel = PersistentMathKernel()
    discoverer = MathDiscoverer(kernel=kernel)
    assert discoverer.kernel is kernel


def test_discovery_result_dataclass():
    res = DiscoveryResult(
        input_data="[1, 2, 5, 14, 42]",
        status="DISCOVERED",
        law="Catalan Numbers",
        conjecture="C_n = 1/(n+1) * binom(2n, n)",
        evidence="Exact match for terms n=0..4, residual = 0",
        implementation="def catalan(n): return math.comb(2*n, n) // (n + 1)",
    )
    assert res.status == "DISCOVERED"
    assert res.law == "Catalan Numbers"
    assert "C_n" in res.conjecture
    assert res.implementation is not None


def test_sequence_parsing():
    discoverer = MathDiscoverer()
    
    # Python list literal
    res1 = discoverer._try_parse_sequence("[1, 2, 5, 14, 42]")
    assert res1 == [1, 2, 5, 14, 42]

    # Comma-separated integers
    res2 = discoverer._try_parse_sequence("1, 4, 9, 16, 25, 36")
    assert res2 == [1, 4, 9, 16, 25, 36]

    # Comma-separated floats
    res3 = discoverer._try_parse_sequence("0.5, 0.25, 0.125, 0.0625")
    assert res3 == [0.5, 0.25, 0.125, 0.0625]

    # General text without numerical sequence
    res4 = discoverer._try_parse_sequence("Find the invariant of matrix P")
    assert res4 is None


def test_discoverer_code_sanitization():
    discoverer = MathDiscoverer()
    text = """
```python
import numpy as np
diffs = np.diff([1, 4, 9, 16, 25])
print(diffs)
```
And invalid code block from thinking:
```python
what if we use sympy.rsolve?
```
"""
    raw_blocks = discoverer.extract_code_blocks(text)
    assert len(raw_blocks) == 2

    valid = []
    for b in raw_blocks:
        try:
            compile(b, "<string>", "exec")
            valid.append(b)
        except SyntaxError:
            pass
    assert len(valid) == 1
    assert "import numpy as np" in valid[0]
