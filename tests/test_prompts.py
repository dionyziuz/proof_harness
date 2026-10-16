import pytest
from core_engine.prompts import (
    MATH_SYSTEM_PROMPT,
    MATH_PROVER_SYSTEM_PROMPT,
    SHARED_MATH_ENVIRONMENT,
    SHARED_COGNITIVE_DISCIPLINE,
    build_solver_system_prompt,
    build_prover_system_prompt,
)


def test_prompts_contain_shared_foundation():
    # Both solver and prover prompts must inherit the shared environment and discipline
    for prompt in [MATH_SYSTEM_PROMPT, MATH_PROVER_SYSTEM_PROMPT]:
        assert "Division of Cognitive Labor" in prompt
        assert "Think to Formulate, Python to Compute" in prompt
        assert "NEVER attempt manual polynomial expansions" in prompt
        assert "SymPy" in prompt
        assert "Z3-Solver" in prompt
        assert "mpmath" in prompt
        assert "persistent Python REPL" in prompt


def test_solver_prompt_specialization():
    assert "Rapid First-Turn Execution" in MATH_SYSTEM_PROMPT
    assert r"\boxed{" in MATH_SYSTEM_PROMPT
    assert "Euler-Gauss Protocol" in MATH_SYSTEM_PROMPT
    assert "<theorem>" not in MATH_SYSTEM_PROMPT


def test_prover_prompt_specialization():
    assert "<theorem>" in MATH_PROVER_SYSTEM_PROMPT
    assert "<proof>" in MATH_PROVER_SYSTEM_PROMPT
    assert "<invariants>" in MATH_PROVER_SYSTEM_PROMPT
    assert "<implementation>" in MATH_PROVER_SYSTEM_PROMPT
    assert "Rigorous Formalization" in MATH_PROVER_SYSTEM_PROMPT
    assert r"\boxed{" not in MATH_PROVER_SYSTEM_PROMPT


def test_prompt_builder_consistency():
    assert build_solver_system_prompt() == MATH_SYSTEM_PROMPT
    assert build_prover_system_prompt() == MATH_PROVER_SYSTEM_PROMPT
