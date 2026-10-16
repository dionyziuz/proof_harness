import pytest
from core_engine.mcp_server import (
    run_math_code,
    reset_math_session,
    inspect_math_session,
    get_recent_thinking_log,
    server,
)


@pytest.mark.asyncio
async def test_mcp_tools_list():
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]
    assert "solve_math_problem" in tool_names
    assert "run_math_code" in tool_names
    assert "reset_math_session" in tool_names
    assert "inspect_math_session" in tool_names
    assert "get_recent_thinking_log" in tool_names
    assert "search_math_heuristics" in tool_names
    assert "record_math_heuristic" in tool_names


@pytest.mark.asyncio
async def test_mcp_run_math_code():
    await reset_math_session()
    res = await run_math_code("test_num = 49\nsp.isprime(test_num)")
    assert "SUCCESS" in res
    assert "False" in res

    inspect_res = await inspect_math_session()
    assert "test_num" in inspect_res


@pytest.mark.asyncio
async def test_mcp_thinking_log():
    log_res = await get_recent_thinking_log(lines=10)
    assert isinstance(log_res, str)
