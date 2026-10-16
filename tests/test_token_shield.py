from core_engine.token_shield import shield_output, sanitize_traceback, summarize_value


def test_shield_output_short():
    short = "Hello Math World"
    assert shield_output(short) == short


def test_shield_output_long_lines():
    lines = [f"Line {i}" for i in range(200)]
    output = "\n".join(lines)
    shielded = shield_output(output, max_lines=40, head_lines=10, tail_lines=10)
    assert "truncated by TokenShield" in shielded
    assert "Line 0" in shielded
    assert "Line 199" in shielded


def test_summarize_value():
    assert summarize_value([1, 2, 3]) == "list(len=3, preview=[1, 2, 3])"
    assert summarize_value("short") == "'short'"
