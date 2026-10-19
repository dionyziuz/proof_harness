"""Unit tests for core_engine.llm_client (no network: httpx.MockTransport)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import httpx
import pytest

import core_engine.llm_client as llm
from core_engine.llm_client import (
    ContextOverflowError,
    LLMResponse,
    LLMStreamError,
    TunnelThinkingClient,
)

_REAL_CLIENT = httpx.Client
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _cfg(**over: Any) -> SimpleNamespace:
    base = dict(
        api_key="test-key",
        base_url="https://llm.test/v1",
        model="test-model",
        max_tokens=1024,
        temperature=0.2,
        timeout_seconds=5.0,
        top_p=0.95,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        thinking_token_budget=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _sse(*chunks: Any, done: bool = True) -> bytes:
    lines = []
    for c in chunks:
        lines.append("data: " + (c if isinstance(c, str) else json.dumps(c)))
        lines.append("")
    if done:
        lines.append("data: [DONE]")
        lines.append("")
    return ("\n".join(lines) + "\n").encode()


class _Recorder:
    """Installs a MockTransport into every httpx.Client/AsyncClient created by llm_client."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]):
        self.calls: List[httpx.Request] = []
        self.payloads: List[Dict[str, Any]] = []
        self.client_kwargs: List[Dict[str, Any]] = []

        def wrapped(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            try:
                self.payloads.append(json.loads(request.content or b"{}"))
            except ValueError:
                self.payloads.append({})
            return handler(request)

        transport = httpx.MockTransport(wrapped)

        def make_client(*args: Any, **kwargs: Any) -> httpx.Client:
            self.client_kwargs.append(dict(kwargs))
            kwargs.pop("verify", None)
            return _REAL_CLIENT(*args, transport=transport, **kwargs)

        def make_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            self.client_kwargs.append(dict(kwargs))
            kwargs.pop("verify", None)
            return _REAL_ASYNC_CLIENT(*args, transport=transport, **kwargs)

        monkeypatch.setattr(llm.httpx, "Client", make_client)
        monkeypatch.setattr(llm.httpx, "AsyncClient", make_async_client)
        monkeypatch.setattr(llm, "compute_backoff", lambda attempt, *a, **k: 0.0)


def _stream_response(body: bytes, status: int = 200) -> httpx.Response:
    return httpx.Response(status, content=body, headers={"content-type": "text/event-stream"})


# ---------------------------------------------------------------------------
# finish_reason / usage capture in streaming mode
# ---------------------------------------------------------------------------
def test_stream_collect_captures_finish_reason_and_usage(monkeypatch):
    body = _sse(
        {"choices": [{"delta": {"reasoning": "think "}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": '{"validity": 0.'}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "7"}, "finish_reason": "length"}]},
        {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33}},
    )
    rec = _Recorder(monkeypatch, lambda req: _stream_response(body))
    client = TunnelThinkingClient(config=_cfg())

    resp = client.complete([{"role": "user", "content": "hi"}], max_tokens=64)

    assert isinstance(resp, LLMResponse)
    assert resp.content == '{"validity": 0.7'
    assert resp.reasoning == "think"
    assert resp.finish_reason == "length"
    assert (resp.prompt_tokens, resp.completion_tokens, resp.total_tokens) == (11, 22, 33)
    assert len(rec.calls) == 1
    assert rec.payloads[0]["stream"] is True
    assert rec.payloads[0]["stream_options"] == {"include_usage": True}


def test_non_stream_parse_response_exposes_finish_reason(monkeypatch):
    data = {
        "model": "m",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    _Recorder(monkeypatch, lambda req: httpx.Response(200, json=data))
    client = TunnelThinkingClient(config=_cfg())
    resp = client.complete([{"role": "user", "content": "hi"}], stream_collect=False)
    assert resp.finish_reason == "stop"
    assert resp.content == "ok"
    assert resp.total_tokens == 3


# ---------------------------------------------------------------------------
# In-stream SSE error events
# ---------------------------------------------------------------------------
def test_sse_error_event_mid_stream_raises(monkeypatch):
    body = _sse(
        {"choices": [{"delta": {"content": "partial"}}]},
        {"error": {"message": "engine crashed", "type": "InternalServerError", "code": 500}},
    )
    rec = _Recorder(monkeypatch, lambda req: _stream_response(body))
    client = TunnelThinkingClient(config=_cfg())

    got: List[Any] = []
    with pytest.raises(LLMStreamError) as ei:
        for ev in client.stream_complete([{"role": "user", "content": "hi"}]):
            got.append(ev)
    assert got == [("content", "partial")]
    assert ei.value.status_code == 500
    assert len(rec.calls) == 1  # never retried after tokens were yielded


def test_sse_error_event_non_retryable_is_not_retried(monkeypatch):
    body = _sse({"error": {"message": "invalid parameter 'foo'", "type": "BadRequestError", "code": 400}})
    rec = _Recorder(monkeypatch, lambda req: _stream_response(body))
    client = TunnelThinkingClient(config=_cfg())
    with pytest.raises(LLMStreamError):
        client.complete([{"role": "user", "content": "hi"}])
    assert len(rec.calls) == 1


def test_sse_error_event_retryable_is_retried_then_succeeds(monkeypatch):
    bodies = [
        _sse({"object": "error", "message": "temporarily overloaded", "code": 503}),
        _sse({"choices": [{"delta": {"content": "fine"}, "finish_reason": "stop"}]}),
    ]
    rec = _Recorder(monkeypatch, lambda req: _stream_response(bodies[min(len(rec.calls) - 1, 1)]))
    client = TunnelThinkingClient(config=_cfg())
    resp = client.complete([{"role": "user", "content": "hi"}], max_retries=3)
    assert resp.content == "fine"
    assert resp.finish_reason == "stop"
    assert len(rec.calls) == 2


# ---------------------------------------------------------------------------
# Retry policy: non-retryable 4xx vs retryable 5xx/429
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_no_retry_on_non_retryable_4xx_stream_collect(monkeypatch, status):
    rec = _Recorder(monkeypatch, lambda req: httpx.Response(status, json={"error": "nope"}))
    client = TunnelThinkingClient(config=_cfg())
    with pytest.raises(httpx.HTTPStatusError) as ei:
        client.complete([{"role": "user", "content": "hi"}], max_retries=5)
    assert ei.value.response.status_code == status
    assert len(rec.calls) == 1


def test_no_retry_on_400_non_stream(monkeypatch):
    rec = _Recorder(monkeypatch, lambda req: httpx.Response(400, json={"error": "bad field"}))
    client = TunnelThinkingClient(config=_cfg())
    with pytest.raises(httpx.HTTPStatusError):
        client.complete([{"role": "user", "content": "hi"}], stream_collect=False, max_retries=5)
    assert len(rec.calls) == 1


@pytest.mark.parametrize("status", [408, 429, 503, 524])
def test_retry_on_transient_status(monkeypatch, status):
    ok = _sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})

    def handler(req: httpx.Request) -> httpx.Response:
        if len(rec.calls) == 1:
            return httpx.Response(status, text="transient")
        return _stream_response(ok)

    rec = _Recorder(monkeypatch, handler)
    client = TunnelThinkingClient(config=_cfg())
    resp = client.complete([{"role": "user", "content": "hi"}], max_retries=3)
    assert resp.content == "ok"
    assert len(rec.calls) == 2


def test_retry_on_network_error(monkeypatch):
    ok = _sse({"choices": [{"delta": {"content": "ok"}}]})

    def handler(req: httpx.Request) -> httpx.Response:
        if len(rec.calls) == 1:
            raise httpx.ConnectError("boom", request=req)
        return _stream_response(ok)

    rec = _Recorder(monkeypatch, handler)
    client = TunnelThinkingClient(config=_cfg())
    assert client.complete([{"role": "user", "content": "hi"}], max_retries=3).content == "ok"
    assert len(rec.calls) == 2


def test_programming_errors_are_not_retried(monkeypatch):
    rec = _Recorder(monkeypatch, lambda req: _stream_response(_sse()))
    client = TunnelThinkingClient(config=_cfg())

    def boom(*a, **k):
        raise KeyError("bug")
        yield  # pragma: no cover

    monkeypatch.setattr(client, "_stream_events", boom)
    with pytest.raises(KeyError):
        client.complete([{"role": "user", "content": "hi"}], max_retries=5)


def test_acomplete_no_retry_on_400(monkeypatch):
    rec = _Recorder(monkeypatch, lambda req: httpx.Response(400, json={"error": "bad"}))
    client = TunnelThinkingClient(config=_cfg())
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.acomplete([{"role": "user", "content": "hi"}], max_retries=5))
    assert len(rec.calls) == 1


def test_astream_complete_captures_content_and_raises_on_error(monkeypatch):
    body = _sse({"choices": [{"delta": {"content": "a"}}]}, {"error": {"message": "x", "code": 500}})
    _Recorder(monkeypatch, lambda req: _stream_response(body))
    client = TunnelThinkingClient(config=_cfg())

    async def run() -> List[Any]:
        out = []
        async for ev in client.astream_complete([{"role": "user", "content": "hi"}]):
            out.append(ev)
        return out

    with pytest.raises(LLMStreamError):
        asyncio.run(run())


# ---------------------------------------------------------------------------
# ContextOverflowError
# ---------------------------------------------------------------------------
_OVERFLOW_BODY = {
    "object": "error",
    "message": "This model's maximum context length is 32768 tokens. However, you requested 40000 tokens.",
    "type": "BadRequestError",
    "code": 400,
}


@pytest.mark.parametrize("stream_collect", [True, False])
def test_context_overflow_detected_and_not_retried(monkeypatch, stream_collect):
    rec = _Recorder(monkeypatch, lambda req: httpx.Response(400, json=_OVERFLOW_BODY))
    client = TunnelThinkingClient(config=_cfg())
    with pytest.raises(ContextOverflowError) as ei:
        client.complete([{"role": "user", "content": "x"}], stream_collect=stream_collect, max_retries=5)
    assert isinstance(ei.value, RuntimeError)
    assert "400" in str(ei.value)  # legacy string checks keep working
    assert ei.value.status_code == 400
    assert len(rec.calls) == 1


def test_context_overflow_via_stream_complete(monkeypatch):
    body = {"error": {"message": "decoder prompt is longer than the maximum model length of 32768"}}
    _Recorder(monkeypatch, lambda req: httpx.Response(400, json=body))
    client = TunnelThinkingClient(config=_cfg())
    with pytest.raises(ContextOverflowError):
        list(client.stream_complete([{"role": "user", "content": "x"}]))


def test_context_overflow_in_stream_event(monkeypatch):
    body = _sse({"error": {"message": "max_model_len exceeded: too many tokens", "code": 400}})
    _Recorder(monkeypatch, lambda req: _stream_response(body))
    client = TunnelThinkingClient(config=_cfg())
    with pytest.raises(ContextOverflowError) as ei:
        client.complete([{"role": "user", "content": "x"}])
    assert "400" in str(ei.value)


def test_plain_400_is_not_context_overflow(monkeypatch):
    _Recorder(monkeypatch, lambda req: httpx.Response(400, json={"error": "temperature must be >= 0"}))
    client = TunnelThinkingClient(config=_cfg())
    with pytest.raises(httpx.HTTPStatusError) as ei:
        client.complete([{"role": "user", "content": "x"}])
    assert not isinstance(ei.value, ContextOverflowError)


def test_context_overflow_error_str_always_contains_400():
    assert "400" in str(ContextOverflowError("context length exceeded"))


# ---------------------------------------------------------------------------
# TLS + payload builder
# ---------------------------------------------------------------------------
def test_tls_verify_defaults_true_and_env_override(monkeypatch):
    monkeypatch.setattr(llm, "TLS_VERIFY", None)
    monkeypatch.delenv("PRIME_MATH_TLS_VERIFY", raising=False)
    assert llm.get_tls_verify() is True
    monkeypatch.setenv("PRIME_MATH_TLS_VERIFY", "false")
    assert llm.get_tls_verify() is False
    monkeypatch.setenv("PRIME_MATH_TLS_VERIFY", "1")
    assert llm.get_tls_verify() is True


def test_requests_use_tls_verify_setting(monkeypatch):
    monkeypatch.setattr(llm, "TLS_VERIFY", None)
    monkeypatch.delenv("PRIME_MATH_TLS_VERIFY", raising=False)
    rec = _Recorder(monkeypatch, lambda req: _stream_response(_sse({"choices": [{"delta": {"content": "ok"}}]})))
    TunnelThinkingClient(config=_cfg()).complete([{"role": "user", "content": "hi"}])
    assert rec.client_kwargs and all(kw.get("verify") is True for kw in rec.client_kwargs)


def test_build_payload_thinking_budget_semantics():
    client = TunnelThinkingClient(config=_cfg(thinking_token_budget=2048))
    msgs = [{"role": "user", "content": "q"}]

    p = client._build_payload(msgs, None, None, None, None)
    assert p["max_tokens"] == 1024 and p["temperature"] == 0.2
    assert p["thinking_token_budget"] == 2048 and p["thinking_budget_tokens"] == 2048
    assert "chat_template_kwargs" not in p and "stream" not in p

    p = client._build_payload(msgs, 99, 0.0, True, 512, stream=True)
    assert p["max_tokens"] == 99 and p["temperature"] == 0.0
    assert p["chat_template_kwargs"] == {"enable_thinking": True}
    assert p["thinking_token_budget"] == 512 and p["thinking_budget_tokens"] == 512
    assert p["stream"] is True

    p = client._build_payload(msgs, None, None, False, 512)
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert "thinking_token_budget" not in p and "thinking_budget_tokens" not in p


def test_build_payload_reasoning_effort_and_disable_thinking():
    cfg = _cfg(vllm_params=False, reasoning_effort="xhigh", reasoning_effort_disabled="minimal")
    client = TunnelThinkingClient(config=cfg)
    msgs = [{"role": "user", "content": "q"}]

    # Default / enable_thinking=True -> xhigh
    p_on = client._build_payload(msgs, 16384, 0.1, True, 12288)
    assert p_on["reasoning_effort"] == "xhigh"
    assert "thinking_budget_tokens" not in p_on
    assert "chat_template_kwargs" not in p_on

    # enable_thinking=False -> switches to minimal
    p_off = client._build_payload(msgs, 4096, 0.1, False, None)
    assert p_off["reasoning_effort"] == "minimal"


@pytest.mark.parametrize("stream_collect", [True, False])
def test_complete_retries_on_hidden_reasoning_length_truncation(monkeypatch, stream_collect):
    cfg = _cfg(max_tokens=65536, vllm_params=False, reasoning_effort="xhigh")
    client = TunnelThinkingClient(config=cfg)
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content.decode())
        calls.append(body)
        if len(calls) == 1:
            if stream_collect:
                return _stream_response(_sse({"choices": [{"delta": {}, "finish_reason": "length"}]}))
            return httpx.Response(200, json={
                "choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": None}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 16384, "total_tokens": 16394},
            })
        if stream_collect:
            return _stream_response(_sse(
                {"choices": [{"delta": {"content": '{"validity": 0.95}'}, "finish_reason": "stop"}]},
            ))
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": '{"validity": 0.95}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110},
        })

    _Recorder(monkeypatch, handler)
    resp = client.complete([{"role": "user", "content": "audit"}], max_tokens=16384, stream_collect=stream_collect)
    assert resp.content == '{"validity": 0.95}'
    assert len(calls) == 2
    assert calls[0]["reasoning_effort"] == "xhigh"
    assert calls[0]["max_tokens"] == 16384
    assert calls[1]["reasoning_effort"] == "high"
    assert calls[1]["max_tokens"] == 32768

