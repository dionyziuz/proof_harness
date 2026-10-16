from unittest.mock import MagicMock
from core_engine.compactor import (
    MathContextCompactor,
    estimate_tokens,
    estimate_messages_tokens,
)
from core_engine.llm_client import LLMResponse


def test_estimate_tokens():
    text = "Hello world mathematical proof"
    tok = estimate_tokens(text)
    assert tok > 0


def test_should_compact_threshold():
    compactor = MathContextCompactor(max_context_tokens=100, reserve_tokens=20, keep_recent_turns=1)
    messages = [
        {"role": "system", "content": "You are Prime Math"},
        {"role": "user", "content": "Problem"},
        {"role": "assistant", "content": "Turn 1" * 10},
        {"role": "user", "content": "Feedback 1" * 10},
        {"role": "assistant", "content": "Turn 2" * 10},
        {"role": "user", "content": "Feedback 2" * 10},
    ]
    # Tokens will exceed 80
    assert compactor.should_compact(messages)


def test_compaction_execution():
    mock_client = MagicMock()
    mock_client.complete.return_value = LLMResponse(
        content="### 1. Core Goal\nSolve x^2 = 2\n### 2. Verified Milestones\nTested x=1.4",
        reasoning="",
        model="test-model",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        raw_response={},
    )

    compactor = MathContextCompactor(
        client=mock_client,
        max_context_tokens=1000,
        reserve_tokens=200,
        keep_recent_turns=1,
    )

    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Problem statement"},
        {"role": "assistant", "content": "Older turn 1"},
        {"role": "user", "content": "Older feedback 1"},
        {"role": "assistant", "content": "Recent turn 2"},
        {"role": "user", "content": "Recent feedback 2"},
    ]

    res = compactor.compact(messages, force=True)
    assert res.compacted is True
    assert res.summary is not None
    assert len(res.compacted_messages) == 6
    assert res.compacted_messages[0]["content"] == "System prompt"
    assert res.compacted_messages[1]["content"] == "Problem statement"
    assert "[MATHEMATICAL CONTEXT COMPACTION SUMMARY]" in res.compacted_messages[2]["content"]
    assert res.compacted_messages[4]["content"] == "Recent turn 2"
    assert res.compacted_messages[5]["content"] == "Recent feedback 2"


import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_async_compaction_execution():
    mock_client = MagicMock()
    mock_client.acomplete = AsyncMock(return_value=LLMResponse(
        content="### 1. Core Goal\nSolve x^2 = 2\n### 2. Verified Milestones\nTested x=1.4",
        reasoning="",
        model="test-model",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        raw_response={},
    ))

    compactor = MathContextCompactor(
        client=mock_client,
        max_context_tokens=1000,
        reserve_tokens=200,
        keep_recent_turns=1,
    )

    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Problem statement"},
        {"role": "assistant", "content": "Older turn 1"},
        {"role": "user", "content": "Older feedback 1"},
        {"role": "assistant", "content": "Recent turn 2"},
        {"role": "user", "content": "Recent feedback 2"},
    ]

    res = await compactor.acompact(messages, force=True)
    assert res.compacted is True
    assert res.summary is not None
    assert len(res.compacted_messages) == 6
    assert res.compacted_messages[0]["content"] == "System prompt"
    assert res.compacted_messages[1]["content"] == "Problem statement"
    assert "[MATHEMATICAL CONTEXT COMPACTION SUMMARY]" in res.compacted_messages[2]["content"]
    assert res.compacted_messages[4]["content"] == "Recent turn 2"
    assert res.compacted_messages[5]["content"] == "Recent feedback 2"

