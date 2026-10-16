import pytest
from core_engine.kernel import PersistentMathKernel, TimeoutException


def test_kernel_persistence():
    k = PersistentMathKernel()
    res1 = k.execute("val = 12345")
    assert res1.success
    assert res1.error is None

    res2 = k.execute("val * 2")
    assert res2.success
    assert res2.result_value == 24690 or res2.result_repr == "24690"


def test_kernel_sympy_prelude():
    k = PersistentMathKernel()
    code = """
eq = Eq(x**2 - 9, 0)
sol = solve(eq, x)
sol
"""
    res = k.execute(code)
    assert res.success
    assert res.result_repr == "list(len=2, preview=[-3, 3])" or "[-3, 3]" in str(res.result_repr)


def test_kernel_z3_prelude():
    k = PersistentMathKernel()
    code = """
s = Solver()
v = Int('v')
s.add(v > 50, v < 60, v % 7 == 0)
s.check() == sat and s.model()[v].as_long()
"""
    res = k.execute(code)
    assert res.success
    assert res.result_repr == "56"


def test_kernel_timeout():
    k = PersistentMathKernel(default_timeout=1.0)
    code = """
while True:
    pass
"""
    res = k.execute(code)
    assert not res.success
    assert "timed out" in res.error.lower()


def test_kernel_error_sanitization():
    k = PersistentMathKernel()
    code = "1 / 0"
    res = k.execute(code)
    assert not res.success
    assert "ZeroDivisionError" in res.error


def test_kernel_reset():
    k = PersistentMathKernel()
    k.execute("my_custom_var = 999")
    assert "my_custom_var" in k.get_user_variables()

    k.reset()
    assert "my_custom_var" not in k.get_user_variables()
