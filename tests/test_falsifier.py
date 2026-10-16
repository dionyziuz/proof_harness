import pytest
from core_engine.falsifier import MathFalsifier, FalsificationResult
from core_engine.kernel import PersistentMathKernel


def test_falsifier_initialization():
    kernel = PersistentMathKernel()
    falsifier = MathFalsifier(kernel=kernel)
    assert falsifier.kernel is kernel


def test_falsification_result_dataclass():
    res = FalsificationResult(
        claim="For all n, n^2 + n + 41 is prime",
        status="FALSIFIED",
        counterexample="n = 40 gives 40^2 + 40 + 41 = 40*41 + 41 = 41^2, composite",
        explanation="Fails at n = 40.",
        implementation="assert not isprime(40**2 + 40 + 41)",
    )
    assert res.status == "FALSIFIED"
    assert "n = 40" in res.counterexample
    assert res.implementation is not None


def test_falsifier_code_sanitization():
    falsifier = MathFalsifier()
    text = """
```python
x = 10
y = 20
assert x + y == 30
```
And broken thinking block:
```python
from scipy import linprog? wait no.
```
"""
    raw_blocks = falsifier.extract_code_blocks(text)
    assert len(raw_blocks) == 2

    # Verify only syntactically valid blocks can be compiled
    valid = []
    for b in raw_blocks:
        try:
            compile(b, "<string>", "exec")
            valid.append(b)
        except SyntaxError:
            pass
    assert len(valid) == 1
    assert "assert x + y == 30" in valid[0]
