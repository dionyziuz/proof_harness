from core_engine.solver import MathSolver


def test_extract_code_blocks():
    solver = MathSolver()
    md = """
Here is my plan:
```python
x = 10
y = 20
print(x + y)
```
And now another:
```py
import math
math.sqrt(16)
```
"""
    blocks = solver.extract_code_blocks(md)
    assert len(blocks) == 2
    assert "print(x + y)" in blocks[0]
    assert "math.sqrt(16)" in blocks[1]


def test_extract_boxed_answer():
    solver = MathSolver()
    text = r"Therefore, the solution to the equation is \boxed{35350}."
    assert solver.extract_boxed_answer(text) == "35350"

    nested = r"The roots are \boxed{\frac{1 + \sqrt{5}}{2}}."
    assert solver.extract_boxed_answer(nested) == r"\frac{1 + \sqrt{5}}{2}"


def test_session_serialization(tmp_path):
    from core_engine.session import SolverSession
    save_file = tmp_path / "test_session.json"

    session = SolverSession(problem="Find the sum of all multiples of 3 or 5 below 10.")
    session.turn = 1
    session.result.code_history.append("x = 3 + 5\nprint(x)")
    session.save(save_file)

    loaded = SolverSession.load(save_file)
    assert loaded.problem == session.problem
    assert loaded.turn == 1
    assert "x" in loaded.kernel.namespace
    assert loaded.kernel.namespace["x"] == 8


def test_two_tier_fast_composer():
    from unittest.mock import MagicMock

    mock_fast = MagicMock()
    mock_fast.fast_cfg.enabled = True
    mock_fast.synthesize_code.return_value = "print(42)"

    mock_client = MagicMock()
    mock_client.complete.return_value.content = "VERIFIED\nCalculated correctly."

    # mock memory: keep unit tests from writing to the production TurboVec collections
    solver = MathSolver(client=mock_client, fast_composer=mock_fast, memory=MagicMock())
    res = solver.solve("What is the answer to life?")
    assert res.boxed_answer == "42"
    assert res.iterations == 1
    assert res.verified is True
    assert res.self_verified is True


def test_two_tier_fallback():
    from unittest.mock import MagicMock

    mock_fast = MagicMock()
    mock_fast.fast_cfg.enabled = True
    mock_fast.synthesize_code.return_value = None  # fails to synthesize

    mock_client = MagicMock()
    mock_client.config.max_context_window = 131072
    mock_client.config.max_tokens = 32768
    mock_client.stream_complete.side_effect = lambda *args, **kwargs: iter([("content", "The answer is \\boxed{999}")])

    solver = MathSolver(client=mock_client, fast_composer=mock_fast, memory=MagicMock())
    res = solver.solve("Some hard problem")
    assert res.boxed_answer == "999"


def test_is_repetitive_loop():
    from core_engine.llm_client import is_repetitive_loop

    assert not is_repetitive_loop("This is normal mathematical reasoning with varied sentences.")
    assert is_repetitive_loop("Also T? T? T? T? T? T? T? T? T? T? T? T?")
    assert is_repetitive_loop("step\nstep\nstep\nstep\nstep\nstep\nstep\nstep\nstep\nstep\nstep\nstep\n")
    assert not is_repetitive_loop("short text")


def test_fast_composer_nested_backticks():
    from core_engine.llm_client import FastComposerClient

    client = FastComposerClient()
    raw_output = "```python\n```python\nimport math\nprint(123)\n```\n```"
    extracted = client._extract_code(raw_output)
    assert extracted is not None
    assert "```" not in extracted
    assert "print(123)" in extracted


def test_extract_boxed_answer_variants():
    solver = MathSolver()
    assert solver.extract_boxed_answer(r"Answer is \boxed{9811}") == "9811"
    assert solver.extract_boxed_answer(r"\boxed 625243878951") == "625243878951"
    assert solver.extract_boxed_answer(r"\boxed{\text{3814708984376}}") == "3814708984376"


def test_extract_answer_from_text():
    solver = MathSolver()
    assert solver.extract_answer_from_text("Therefore, the answer is 9811.") == "9811"
    assert solver.extract_answer_from_text("Final Answer: 625243878951") == "625243878951"
    assert solver.extract_answer_from_text(r"So \boxed{3814708984376}") == "3814708984376"


def test_extract_best_answer_from_history():
    from core_engine.kernel import ExecutionResult
    from core_engine.solver import SolveResult

    solver = MathSolver()
    res = SolveResult(
        problem="Test",
        answer=None,
        boxed_answer=None,
        verified=False,
        iterations=1,
        execution_history=[
            ExecutionResult(
                success=True,
                stdout="9811 is prime: True\n9811 mod 7 = 4\nAnswer: 9811\n",
                stderr="",
                result_value=None,
                result_repr=None,
                error=None,
                execution_time=0.1,
            )
        ],
    )
    ans = solver._extract_best_answer_from_history(res)
    assert ans == "9811"


def test_extract_best_answer_sanitizes_timings_and_scaled_metrics():
    from core_engine.kernel import ExecutionResult
    from core_engine.solver import SolveResult

    solver = MathSolver()

    # Problem asking for floor of scaled metric
    res = SolveResult(
        problem=r"Compute \lfloor 10^6 d_\infty \rfloor.",
        answer=None,
        boxed_answer=None,
        verified=False,
        iterations=1,
        execution_history=[
            ExecutionResult(
                success=True,
                stdout=(
                    "Found 1990 primes up to 17320\n"
                    "  progress: 200000000/300000000, d=0.3652685461, t=5.2s\n"
                    "range [200000000, 300000000]: d=0.3676084222  1e6*d=367608.4222  total_time=273.0s\n"
                ),
                stderr="",
                result_value=None,
                result_repr=None,
                error=None,
                execution_time=273.0,
            )
        ],
    )
    ans = solver._extract_best_answer_from_history(res)
    # MUST extract 367608 from 1e6*d=367608.4222, and MUST NOT extract 273 from total_time=273.0s
    assert ans == "367608"

    # Only timing diagnostics without any valid answer must return None
    res_timing_only = SolveResult(
        problem=r"Compute \lfloor 10^6 d_\infty \rfloor.",
        answer=None,
        boxed_answer=None,
        verified=False,
        iterations=1,
        execution_history=[
            ExecutionResult(
                success=True,
                stdout="Found 1990 primes up to 17320\nprogress: 200000000/300000000\ntotal_time=273.0s\n",
                stderr="",
                result_value=None,
                result_repr=None,
                error=None,
                execution_time=273.0,
            )
        ],
    )
    assert solver._extract_best_answer_from_history(res_timing_only) is None


def test_extract_best_answer_density_autoscaling_and_sample_sanitization():
    from core_engine.kernel import ExecutionResult
    from core_engine.solver import SolveResult

    solver = MathSolver()

    # Raw density with sample count and timing in stdout
    res = SolveResult(
        problem=r"Definitions: Let S_x be primes ... d_x = |S_x|/\pi(x). Compute \lfloor 10^6 d_\infty \rfloor.",
        answer=None,
        boxed_answer=None,
        verified=False,
        iterations=1,
        execution_history=[
            ExecutionResult(
                success=True,
                stdout="x=2*10^8: d_x=0.3676859071 (n=11078937, time=239.7s)\n",
                stderr="",
                result_value=None,
                result_repr=None,
                error=None,
                execution_time=239.7,
            )
        ],
    )
    ans = solver._extract_best_answer_from_history(res)
    # MUST extract 367685 from d_x=0.3676859071 * 10^6, and MUST NOT extract 11078937 or 239
    assert ans == "367685"


def test_extract_best_answer_floor_with_parentheses():
    from core_engine.kernel import ExecutionResult
    from core_engine.solver import SolveResult

    solver = MathSolver()

    res = SolveResult(
        problem=r"Compute \lfloor 10^6 d_\infty \rfloor.",
        answer=None,
        boxed_answer=None,
        verified=False,
        iterations=1,
        execution_history=[
            ExecutionResult(
                success=True,
                stdout=(
                    "Richardson (last 2): d_inf = 0.3676874488\n"
                    "  floor(10^6 * d_inf) = 367687\n"
                    "Average of last 4: 0.3677609195\n"
                    "  floor(10^6 * avg) = 367760\n"
                ),
                stderr="",
                result_value=None,
                result_repr=None,
                error=None,
                execution_time=1.0,
            )
        ],
    )
    ans = solver._extract_best_answer_from_history(res)
    # MUST extract the latest floor line (367760 or 367687)
    assert ans in ("367760", "367687")


def test_pre_solve_academic_research():
    from unittest.mock import MagicMock, patch

    mock_client = MagicMock()
    q_resp = MagicMock()
    q_resp.content = "<query>monodromy reducible polynomial</query>\n<query>Chebotarev density Artin symbol</query>"
    q_resp.reasoning = None

    synth_resp = MagicMock()
    synth_resp.content = "GOVERNING DEFINITIONS:\nMonodromy group action on fibers.\nCANONICAL STRATEGY:\nCompute branch loci."
    synth_resp.reasoning = None

    mock_client.complete.side_effect = [q_resp, synth_resp]

    mock_fast = MagicMock()
    mock_fast.generate_research_queries.return_value = []  # test thinking model fallback

    with patch("core_engine.solver.search_math") as mock_search:
        mock_search.return_value = [
            {"title": "Monodromy Theorem - Wikipedia", "snippet": "In algebraic topology, monodromy is the study of how objects behave..."},
        ]
        solver = MathSolver(client=mock_client, fast_composer=mock_fast, enable_pre_solve_research=True)
        briefing = solver._conduct_pre_solve_research("Let f(x) be a polynomial with branch locus...")

        assert briefing is not None
        assert "Monodromy group action on fibers" in briefing
        assert mock_search.called
        assert mock_client.complete.call_count == 2


import pytest


@pytest.mark.asyncio
async def test_pre_solve_academic_research_async():
    from unittest.mock import MagicMock, patch, AsyncMock

    mock_client = MagicMock()
    synth_resp = MagicMock()
    synth_resp.content = "GOVERNING DEFINITIONS:\nKlein quartic curve C: x^3 y + y^3 z + z^3 x = 0.\nCANONICAL STRATEGY:\nEvaluate L-polynomial over F_q."
    synth_resp.reasoning = None

    mock_client.acomplete = AsyncMock(return_value=synth_resp)

    mock_fast = MagicMock()
    mock_fast.agenerate_research_queries = AsyncMock(return_value=["Klein quartic zeta function Frobenius"])

    with patch("core_engine.solver.search_math") as mock_search:
        mock_search.return_value = [
            {"title": "Klein Quartic - MathWorld", "snippet": "The Klein quartic is a compact Riemann surface of genus 3 with automorphism group PSL(2, 7)..."},
        ]
        solver = MathSolver(client=mock_client, fast_composer=mock_fast, enable_pre_solve_research=True)
        briefing = await solver._aconduct_pre_solve_research("Find the number of rational points on Klein quartic over F_5^18")

        assert briefing is not None
        assert "Klein quartic curve" in briefing
        assert mock_search.called
        assert mock_fast.agenerate_research_queries.called


def test_post_solve_verification_wrong_triggers_revision():
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.config.max_context_window = 131072
    mock_client.config.max_tokens = 32768
    turns = iter([
        iter([("content", "```python\nprint(2 + 2)\n```")]),
        iter([("content", "The answer is \\boxed{999}")]),
        iter([("content", "The answer is \\boxed{888}")]),
    ])
    mock_client.stream_complete.side_effect = lambda *a, **k: next(turns)
    critic = MagicMock()
    critic.content = "WRONG: 999 is not the sum; independent recomputation gives 4."
    critic.reasoning = None
    mock_client.complete.return_value = critic

    mock_fast = MagicMock()
    mock_fast.fast_cfg.enabled = False

    solver = MathSolver(client=mock_client, fast_composer=mock_fast, memory=MagicMock())
    res = solver.solve("What is 2 + 2?")
    assert res.boxed_answer == "888"
    assert res.answer == "888"
    assert res.iterations == 3
    assert mock_client.stream_complete.call_count == 3


def test_post_solve_verification_inconclusive_keeps_original():
    from unittest.mock import MagicMock

    def _run(critic_text, boxed):
        mock_client = MagicMock()
        mock_client.config.max_context_window = 131072
        mock_client.config.max_tokens = 32768
        mock_client.stream_complete.side_effect = lambda *a, **k: iter([
            ("content", f"```python\nprint(7)\n```\nThe answer is \\boxed{{{boxed}}}"),
        ])
        critic = MagicMock()
        critic.content = critic_text
        critic.reasoning = None
        mock_client.complete.return_value = critic

        mock_fast = MagicMock()
        mock_fast.fast_cfg.enabled = False

        solver = MathSolver(client=mock_client, fast_composer=mock_fast, memory=MagicMock())
        return solver.solve("What is 7?"), mock_client

    # Explicit INCONCLUSIVE keeps the original boxed answer
    res, mock_client = _run("INCONCLUSIVE", "999")
    assert res.boxed_answer == "999"
    assert res.iterations == 1
    assert mock_client.stream_complete.call_count == 1

    # Garbage critic output also parses as INCONCLUSIVE (conservative) and keeps the answer
    res2, mock_client2 = _run("I think the answer might be wrong because the steps look off.", "777")
    assert res2.boxed_answer == "777"
    assert mock_client2.stream_complete.call_count == 1


def test_stall_detection_strategy_reset():
    from core_engine.kernel import ExecutionResult

    solver = MathSolver()
    err = ExecutionResult(
        success=False,
        stdout="",
        stderr="",
        result_value=None,
        result_repr=None,
        error="NameError: name 'x' is not defined",
        execution_time=0.1,
    )
    fb1 = solver._execution_error_feedback(err)
    fb2 = solver._execution_error_feedback(err)
    fb3 = solver._execution_error_feedback(err)

    assert "[Execution Error]:" in fb1
    assert "[STALLED ROUTE]" not in fb1
    assert "[STALLED ROUTE]" not in fb2
    assert "[STALLED ROUTE]" in fb3
    assert "fundamentally different strategy" in fb3
    assert len(solver._error_signatures) == 3


def test_stall_window_clears_on_success_through_solve():
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.config.max_context_window = 131072
    mock_client.config.max_tokens = 32768
    turns = iter([
        iter([("content", "```python\n1/0\n```")]),
        iter([("content", "```python\nprint(1)\n```")]),
        iter([("content", "```python\n1/0\n```")]),
        iter([("content", "The answer is \\boxed{1}")]),
    ])
    mock_client.stream_complete.side_effect = lambda *a, **k: next(turns)
    critic = MagicMock()
    critic.content = "VERIFIED"
    critic.reasoning = None
    mock_client.complete.return_value = critic

    mock_fast = MagicMock()
    mock_fast.fast_cfg.enabled = False
    solver = MathSolver(client=mock_client, fast_composer=mock_fast, memory=MagicMock())
    res = solver.solve("What is the value of 1?")

    assert res.boxed_answer == "1"
    assert res.iterations == 4
    # The successful print(1) turn cleared the window; without the clear this would be 3
    assert len(solver._error_signatures) == 1
    assert "[STALLED ROUTE]" not in solver.log_path.read_text()


def test_fast_composer_falsified_escalates_with_critique():
    from unittest.mock import MagicMock

    mock_fast = MagicMock()
    mock_fast.fast_cfg.enabled = True
    mock_fast.synthesize_code.return_value = "print(123)"

    mock_client = MagicMock()
    mock_client.config.max_context_window = 131072
    mock_client.config.max_tokens = 32768
    critic1 = MagicMock()
    critic1.content = "WRONG: 123 does not satisfy the problem constraints."
    critic1.reasoning = None
    critic2 = MagicMock()
    critic2.content = "VERIFIED"
    critic2.reasoning = None
    # First complete call is the Tier 1 cross-model critique; later calls (post-solve
    # verification, learning consolidation) see a verifying critic.
    mock_client.complete.side_effect = (
        lambda *a, **k: critic1 if mock_client.complete.call_count == 1 else critic2
    )
    mock_client.stream_complete.side_effect = lambda *a, **k: iter([
        ("content", "```python\nprint(456)\n```\nThe answer is \\boxed{456}"),
    ])

    solver = MathSolver(client=mock_client, fast_composer=mock_fast, memory=MagicMock())
    res = solver.solve("Find the magic number.")
    assert res.boxed_answer == "456"
    assert res.iterations == 1
    assert mock_client.stream_complete.call_count == 1

    # The Deep Orchestrator prompt must carry the falsified-route critique
    deep_msgs = mock_client.stream_complete.call_args[0][0]
    user_contents = [m["content"] for m in deep_msgs if m["role"] == "user"]
    assert any("123" in c and "falsified" in c.lower() for c in user_contents)


def test_fast_composer_inconclusive_escalates_to_deep_orchestrator():
    from unittest.mock import MagicMock

    mock_fast = MagicMock()
    mock_fast.fast_cfg.enabled = True
    mock_fast.synthesize_code.return_value = "print(100)"

    mock_client = MagicMock()
    mock_client.config.max_context_window = 131072
    mock_client.config.max_tokens = 32768
    critic1 = MagicMock()
    critic1.content = "INCONCLUSIVE"
    critic1.reasoning = None
    critic2 = MagicMock()
    critic2.content = "VERIFIED"
    critic2.reasoning = None
    mock_client.complete.side_effect = (
        lambda *a, **k: critic1 if mock_client.complete.call_count == 1 else critic2
    )
    mock_client.stream_complete.side_effect = lambda *a, **k: iter([
        ("content", "```python\nprint(200)\n```\nThe answer is \\boxed{200}"),
    ])

    solver = MathSolver(client=mock_client, fast_composer=mock_fast, memory=MagicMock())
    res = solver.solve("Some problem where fast composer is inconclusive.")
    assert res.boxed_answer == "200"
    assert res.iterations == 1
    # Must have called deep orchestrator stream_complete because inconclusive was rejected for early exit
    assert mock_client.stream_complete.call_count == 1


def test_consolidate_learning_gated_by_is_correct():
    from unittest.mock import MagicMock
    from core_engine.solver import SolveResult

    mock_client = MagicMock()
    mock_client.complete.return_value.content = (
        "When solving linear recurrence relations, apply matrix exponentiation instead of iterative stepping."
    )
    mock_memory = MagicMock()
    mock_memory.add_procedural_heuristic.return_value = True

    solver = MathSolver(client=mock_client, memory=mock_memory)

    res_wrong = SolveResult(
        problem="Project Euler Problem 999 (Test): ...",
        answer="123",
        boxed_answer="123",
        verified=True,
        iterations=2,
        code_history=["print(123)"],
    )

    # 1. is_correct=False -> No heuristic added, record unverified episode
    solver.consolidate_learning("Project Euler Problem 999 (Test): ...", res_wrong, is_correct=False)
    assert mock_memory.add_procedural_heuristic.call_count == 0
    assert mock_memory.record_episode.call_count == 1
    assert mock_memory.record_episode.call_args[1]["verified"] is False
    assert len(res_wrong.learned_heuristics) == 0

    # 2. is_correct=None -> No heuristic added
    mock_memory.reset_mock()
    solver.consolidate_learning("Project Euler Problem 999 (Test): ...", res_wrong, is_correct=None)
    assert mock_memory.add_procedural_heuristic.call_count == 0
    assert mock_memory.record_episode.call_count == 1
    assert mock_memory.record_episode.call_args[1]["verified"] is False

    # 3. is_correct=True -> Heuristic validated and committed
    mock_memory.reset_mock()
    res_correct = SolveResult(
        problem="Project Euler Problem 999 (Test): ...",
        answer="456",
        boxed_answer="456",
        verified=True,
        iterations=2,
        code_history=["print(456)"],
    )
    solver.consolidate_learning("Project Euler Problem 999 (Test): ...", res_correct, is_correct=True)
    assert mock_memory.add_procedural_heuristic.call_count == 1
    assert mock_memory.record_episode.call_count == 1
    assert mock_memory.record_episode.call_args[1]["verified"] is True
    assert len(res_correct.learned_heuristics) == 1
    assert "linear recurrence relations" in res_correct.learned_heuristics[0]


def test_solve_passes_exclude_problem_id_to_procedural_query():
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.config.max_context_window = 131072
    mock_client.config.max_tokens = 32768
    mock_client.stream_complete.side_effect = lambda *a, **k: iter([
        ("content", "The answer is \\boxed{777}"),
    ])
    mock_client.complete.return_value.content = "VERIFIED"

    mock_fast = MagicMock()
    mock_fast.fast_cfg.enabled = False

    mock_memory = MagicMock()
    mock_memory.query_procedural_heuristics.return_value = []

    solver = MathSolver(
        client=mock_client, fast_composer=mock_fast, memory=mock_memory, enable_pre_solve_research=False
    )
    problem_text = "Project Euler Problem 123 (Testing Prime Search):\nFind the nth prime."
    res = solver.solve(problem_text)
    assert res.boxed_answer == "777"

    # Verify query_procedural_heuristics was called with exclude_problem_id="123"
    assert mock_memory.query_procedural_heuristics.call_count == 1
    kall = mock_memory.query_procedural_heuristics.call_args
    assert kall[1]["exclude_problem_id"] == "123"

