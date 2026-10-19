"""LLM Client for OpenAI-Compatible Thinking Models.

Interfaces with local or remote OpenAI-compatible thinking models (e.g. Qwen / vLLM).
Extracts native reasoning traces and clean mathematical responses, with full streaming
support to enable live thinking logs.

Error semantics (shared by every request path):
- HTTP 408/429/5xx and transport/network errors are retried with exponential backoff.
- Other 4xx (400/401/403/404/422, ...) are raised immediately (never retried).
- HTTP 400 responses whose body reports a context-length overflow raise
  ``ContextOverflowError`` (a ``RuntimeError`` whose ``str()`` contains ``"400"``).
- In-stream SSE ``{"error": ...}`` events raise ``LLMStreamError`` (or
  ``ContextOverflowError``) instead of being silently skipped.
- TLS certificate verification is controlled by ``PRIME_MATH_TLS_VERIFY``
  (default: verify). Values: true/false, or a path to a CA bundle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import ssl
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple, Union

import httpx

from .config import PrimeMathConfig, load_config

logger = logging.getLogger("prime_math.llm")

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}

# Phrases vLLM / OpenAI-compatible servers use when the prompt + max_tokens exceed the window.
_CONTEXT_OVERFLOW_RE = re.compile(
    r"context[ _-]?length|context window|maximum context|max_model_len|maximum model length"
    r"|too many tokens|prompt is too long|input is too long|exceeds the maximum",
    re.IGNORECASE,
)


class LLMStreamError(RuntimeError):
    """An error event delivered inside an SSE stream (``data: {"error": ...}``)."""

    def __init__(self, message: str, status_code: Optional[int] = None, payload: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class ContextOverflowError(RuntimeError):
    """HTTP 400 caused by the request exceeding the model's context window.

    ``str(exc)`` always contains ``"400"`` so legacy ``"400" in str(exc)`` checks keep working;
    new callers should use ``isinstance(exc, ContextOverflowError)``.
    """

    def __init__(self, message: str, status_code: int = 400, body: str = "", response: Any = None):
        if "400" not in message:
            message = f"HTTP 400: {message}"
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.response = response


def is_context_overflow_message(text: Optional[str]) -> bool:
    """True if an error body/message describes a context-length overflow."""
    return bool(text) and bool(_CONTEXT_OVERFLOW_RE.search(str(text)))


def is_retryable_status(status_code: int) -> bool:
    """408/429/5xx (and Cloudflare 52x) are transient; every other 4xx is not."""
    if status_code in NON_RETRYABLE_STATUS_CODES:
        return False
    return status_code in RETRYABLE_STATUS_CODES or 500 <= status_code < 600


def is_retryable_error(exc: BaseException) -> bool:
    """Classifies an exception raised by a request path as transient (retry) or fatal."""
    if isinstance(exc, ContextOverflowError):
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return is_retryable_status(exc.response.status_code)
    if isinstance(exc, LLMStreamError):
        return exc.status_code is None or is_retryable_status(exc.status_code)
    # httpx.RequestError covers TransportError (timeouts, network, remote-protocol errors).
    return isinstance(exc, (httpx.RequestError, ConnectionResetError, OSError))


def _parse_tls_verify(value: Optional[str]) -> Union[bool, ssl.SSLContext]:
    if value is None or not value.strip():
        return True
    v = value.strip()
    low = v.lower()
    if low in ("0", "false", "no", "off"):
        return False
    if low in ("1", "true", "yes", "on"):
        return True
    if os.path.isfile(v):
        return ssl.create_default_context(cafile=v)
    logger.warning(f"PRIME_MATH_TLS_VERIFY={v!r} is neither a boolean nor a CA bundle path; verifying TLS.")
    return True


# Module-level override (tests or callers may set this). None -> read PRIME_MATH_TLS_VERIFY at call time
# (lazily, so values loaded from .env by load_config() are honoured).
TLS_VERIFY: Optional[Union[bool, ssl.SSLContext]] = None


def get_tls_verify() -> Union[bool, ssl.SSLContext]:
    """Returns the httpx ``verify=`` setting for tunnel requests (default True)."""
    if TLS_VERIFY is not None:
        return TLS_VERIFY
    return _parse_tls_verify(os.getenv("PRIME_MATH_TLS_VERIFY"))


def _stream_usage_enabled() -> bool:
    val = os.getenv("PRIME_MATH_STREAM_USAGE")
    if val is None:
        return True
    return val.strip().lower() in ("1", "true", "yes", "on")


def _status_from_error_obj(err: Any) -> Optional[int]:
    if not isinstance(err, dict):
        return None
    for key in ("code", "status", "status_code"):
        val = err.get(key)
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.strip().isdigit():
            return int(val.strip())
    etype = str(err.get("type") or "")
    if "BadRequest" in etype or "InvalidRequest" in etype or "invalid_request" in etype:
        return 400
    return None


def _extract_stream_error(chunk: Dict[str, Any]) -> Optional[RuntimeError]:
    """Builds the exception for an in-stream SSE error event, or None if the chunk is normal."""
    if "error" in chunk and chunk["error"]:
        err = chunk["error"]
    elif chunk.get("object") == "error":
        err = chunk
    else:
        return None
    if isinstance(err, dict):
        message = str(err.get("message") or err.get("detail") or err)
    else:
        message = str(err)
    status = _status_from_error_obj(err)
    if (status in (None, 400)) and is_context_overflow_message(message):
        return ContextOverflowError(f"HTTP 400 (in-stream) context overflow: {message[:1000]}", body=message)
    code_str = f" (code {status})" if status is not None else ""
    return LLMStreamError(f"LLM stream error event{code_str}: {message[:1000]}", status_code=status, payload=chunk)


def _http_error_body(response: httpx.Response) -> str:
    try:
        return response.text or ""
    except Exception:
        return ""


def _raise_http_error(response: httpx.Response) -> None:
    """Raises ContextOverflowError / httpx.HTTPStatusError for an error response (body already read)."""
    if not response.is_error:
        return
    body = _http_error_body(response)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if response.status_code == 400 and is_context_overflow_message(body):
            raise ContextOverflowError(
                f"HTTP 400 context overflow: {body[:1000]}", status_code=400, body=body, response=response
            ) from exc
        raise


def _raise_for_stream_status(response: httpx.Response) -> None:
    if response.is_error:
        try:
            response.read()
        except Exception:
            pass
        _raise_http_error(response)


async def _araise_for_stream_status(response: httpx.Response) -> None:
    if response.is_error:
        try:
            await response.aread()
        except Exception:
            pass
        _raise_http_error(response)


def compute_backoff(attempt: int, base: float = 3.0, max_wait: float = 60.0) -> float:
    """Computes exponential backoff delay with random jitter."""
    return min(max_wait, base * (2 ** attempt)) + random.uniform(1.0, 3.0)


def is_repetitive_loop(text: str, min_repeats: int = 10) -> bool:
    """Detects if the trailing text is caught in an infinite repetitive token loop (e.g. 'T? T? T?...')."""
    if len(text) < 30:
        return False
    tail = text[-200:]
    divider_chars = {'-', '=', '*', '_', '~', '#', '.', ' '}
    for pat_len in range(1, 20):
        pattern = tail[-pat_len:]
        p_stripped = pattern.strip()
        if not p_stripped:
            continue
        if set(p_stripped).issubset(divider_chars):
            continue
        effective_repeats = max(min_repeats, 25 if pat_len <= 2 else min_repeats)
        if len(tail) >= pat_len * effective_repeats and tail.endswith(pattern * effective_repeats):
            return True
    return False


@dataclass
class LLMResponse:
    content: str
    reasoning: Optional[str] = None
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw_response: Dict[str, Any] = field(default_factory=dict)
    # "stop", "length" (hit max_tokens), "loop_guard" (client-side repetition cut-off), or None if unknown.
    finish_reason: Optional[str] = None


class _SSEStreamParser:
    """Incremental parser for OpenAI-compatible SSE chat streams.

    Yields (type, delta) events, records finish_reason/usage into ``meta``, raises on in-stream
    error events, and applies the repetitive-reasoning loop guard.
    """

    def __init__(self, meta: Dict[str, Any]):
        self.meta = meta
        self.recent_reasoning = ""
        self.suppress_reasoning = False
        self.suppressed_tokens = 0

    def feed(self, raw_line: str) -> Tuple[List[Tuple[str, str]], bool]:
        """Returns (events, stop)."""
        line = raw_line.strip()
        if not line or not line.startswith("data: "):
            return [], False
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            return [], True
        try:
            chunk = json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            return [], False
        if not isinstance(chunk, dict):
            return [], False

        err = _extract_stream_error(chunk)
        if err is not None:
            raise err

        usage = chunk.get("usage")
        if isinstance(usage, dict) and usage:
            self.meta["usage"] = usage
        choices = chunk.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return [], False
        choice = choices[0]
        if choice.get("finish_reason"):
            self.meta["finish_reason"] = choice["finish_reason"]
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            return [], False

        events: List[Tuple[str, str]] = []
        r_text = delta.get("reasoning") or delta.get("reasoning_content")
        if r_text:
            if not self.suppress_reasoning:
                self.recent_reasoning += r_text
                if len(self.recent_reasoning) > 200:
                    self.recent_reasoning = self.recent_reasoning[-200:]
                if is_repetitive_loop(self.recent_reasoning):
                    logger.warning("Repetitive reasoning loop detected; muting reasoning stream and awaiting code content.")
                    self.suppress_reasoning = True
                else:
                    events.append(("reasoning", r_text))
            else:
                self.suppressed_tokens += 1
                if self.suppressed_tokens > 10000:
                    logger.warning("Excessive repetitive tokens (>10000) with no content; terminating stream.")
                    self.meta["finish_reason"] = self.meta.get("finish_reason") or "loop_guard"
                    return events, True
        content = delta.get("content")
        if content:
            events.append(("content", content))
        return events, False


def _split_think_tags(reasoning: Optional[str], content: str) -> Tuple[Optional[str], str]:
    if not reasoning and "<think>" in content and "</think>" in content:
        match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return reasoning, content


class TunnelThinkingClient:
    """Client for querying an OpenAI-compatible thinking model endpoint."""

    def __init__(self, config: Optional[PrimeMathConfig] = None):
        self.config = config or load_config()
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        self.chat_url = f"{self.config.base_url.rstrip('/')}/chat/completions"

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int],
        temperature: Optional[float],
        enable_thinking: Optional[bool],
        thinking_budget_tokens: Optional[int],
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Builds the chat-completions payload (single source of truth for all request paths)."""
        # vLLM extensions (repetition_penalty, chat_template_kwargs, thinking budgets)
        # are only sent when cfg.vllm_params is true; strict OpenAI-compatible
        # endpoints reject them with 400 unknown-parameter errors.
        use_vllm = bool(getattr(self.config, "vllm_params", True))
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "top_p": getattr(self.config, "top_p", 0.95),
            "frequency_penalty": getattr(self.config, "frequency_penalty", 0.0),
        }
        if use_vllm:
            payload["repetition_penalty"] = getattr(self.config, "repetition_penalty", 1.0)
        if stream:
            payload["stream"] = True
            if _stream_usage_enabled():
                # Ask the server to append a final usage chunk (OpenAI / vLLM stream_options).
                payload["stream_options"] = {"include_usage": True}
        if enable_thinking is not None and use_vllm:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        budget = (
            thinking_budget_tokens
            if thinking_budget_tokens is not None
            else getattr(self.config, "thinking_token_budget", None)
        )
        if use_vllm and budget is not None and (enable_thinking is True or enable_thinking is None):
            payload["thinking_token_budget"] = budget
            payload["thinking_budget_tokens"] = budget
        # Meta-native reasoning effort: explicit opt-in, sent whenever configured
        # (independent of the vLLM-params flag).
        # - When enable_thinking is False -> reasoning_effort_disabled ("minimal", ~14 tokens).
        # - When use_vllm is False (endpoints like api.meta.ai that do not support
        #   thinking_budget_tokens) and the caller specifies a lightweight subagent
        #   thinking_budget_tokens (e.g. Skeptic=1024, Falsifier=1536, Architect=2048),
        #   scale reasoning_effort down to "low" / "medium" so helper script subagents
        #   don't burn 15,000 xhigh reasoning tokens (~3 min) per helper call.
        effort = getattr(self.config, "reasoning_effort", None)
        if effort:
            if enable_thinking is False:
                payload["reasoning_effort"] = getattr(self.config, "reasoning_effort_disabled", None) or "minimal"
            elif not use_vllm and thinking_budget_tokens is not None and thinking_budget_tokens <= 2048:
                payload["reasoning_effort"] = "low"
            elif not use_vllm and thinking_budget_tokens is not None and thinking_budget_tokens <= 6144:
                payload["reasoning_effort"] = "medium"
            else:
                payload["reasoning_effort"] = effort
        return payload

    @staticmethod
    def _step_down_effort(effort: Optional[str]) -> str:
        order = ["max", "xhigh", "high", "medium", "low", "minimal"]
        if not effort or effort not in order:
            return "minimal"
        idx = order.index(effort)
        return order[min(idx + 1, len(order) - 1)]

    def _timeout(self, timeout_seconds: Optional[float]) -> float:
        return timeout_seconds or self.config.timeout_seconds

    def _stream_events(
        self,
        payload: Dict[str, Any],
        timeout: float,
        max_retries: int,
        meta: Dict[str, Any],
    ) -> Generator[Tuple[str, str], None, None]:
        """Synchronous SSE stream with retry (only before the first yielded token)."""
        for attempt in range(max_retries):
            yielded_any = False
            meta.clear()
            parser = _SSEStreamParser(meta)
            try:
                with httpx.Client(timeout=timeout, verify=get_tls_verify()) as client:
                    with client.stream("POST", self.chat_url, headers=self.headers, json=payload) as response:
                        _raise_for_stream_status(response)
                        for line in response.iter_lines():
                            events, stop = parser.feed(line)
                            for ev in events:
                                yielded_any = True
                                yield ev
                            if stop:
                                break
                return
            except Exception as exc:
                if yielded_any:
                    logger.error(f"Stream interrupted mid-generation: {exc}")
                    raise
                if not is_retryable_error(exc):
                    raise
                if attempt == max_retries - 1:
                    logger.error(f"Exhausted all {max_retries} stream retries on tunnel LLM request: {exc}")
                    raise
                wait = compute_backoff(attempt)
                logger.warning(
                    f"Tunnel LLM stream request hit {type(exc).__name__} ({exc}); "
                    f"retrying with exponential backoff in {wait:.1f}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait)

    async def _astream_events(
        self,
        payload: Dict[str, Any],
        timeout: float,
        max_retries: int,
        meta: Dict[str, Any],
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """Asynchronous SSE stream with retry (only before the first yielded token)."""
        for attempt in range(max_retries):
            yielded_any = False
            meta.clear()
            parser = _SSEStreamParser(meta)
            try:
                async with httpx.AsyncClient(timeout=timeout, verify=get_tls_verify()) as client:
                    async with client.stream("POST", self.chat_url, headers=self.headers, json=payload) as response:
                        await _araise_for_stream_status(response)
                        async for line in response.aiter_lines():
                            events, stop = parser.feed(line)
                            for ev in events:
                                yielded_any = True
                                yield ev
                            if stop:
                                break
                return
            except Exception as exc:
                if yielded_any:
                    logger.error(f"Async stream interrupted mid-generation: {exc}")
                    raise
                if not is_retryable_error(exc):
                    raise
                if attempt == max_retries - 1:
                    logger.error(f"Exhausted all {max_retries} async stream retries on tunnel LLM request: {exc}")
                    raise
                wait = compute_backoff(attempt)
                logger.warning(
                    f"Async tunnel LLM stream request hit {type(exc).__name__} ({exc}); "
                    f"retrying with exponential backoff in {wait:.1f}s (attempt {attempt + 1}/{max_retries})..."
                )
                await asyncio.sleep(wait)

    def _response_from_stream(self, reasoning_chunks: List[str], content_chunks: List[str],
                              meta: Dict[str, Any]) -> LLMResponse:
        reasoning = "".join(reasoning_chunks).strip() or None
        content = "".join(content_chunks).strip()
        reasoning, content = _split_think_tags(reasoning, content)
        usage = meta.get("usage") or {}
        return LLMResponse(
            content=content,
            reasoning=reasoning,
            model=self.config.model,
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            completion_tokens=usage.get("completion_tokens", 0) or 0,
            total_tokens=usage.get("total_tokens", 0) or 0,
            raw_response={"streamed": True, "finish_reason": meta.get("finish_reason"), "usage": usage},
            finish_reason=meta.get("finish_reason"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: int = 5,
        enable_thinking: Optional[bool] = None,
        thinking_budget_tokens: Optional[int] = None,
        stream_collect: bool = True,
    ) -> LLMResponse:
        """Synchronous chat completion request with exponential backoff retry.
        When stream_collect is True (default), tokens are streamed over SSE to prevent
        Cloudflare HTTP 524 timeouts on long reasoning generations (>100s).
        Non-retryable errors (4xx other than 408/429, context overflow) are raised immediately.
        """
        timeout = self._timeout(timeout_seconds)

        if stream_collect:
            payload = self._build_payload(messages, max_tokens, temperature, enable_thinking,
                                          thinking_budget_tokens, stream=True)
            retried_empty_length = False
            for attempt in range(max_retries):
                meta: Dict[str, Any] = {}
                try:
                    reasoning_chunks: List[str] = []
                    content_chunks: List[str] = []
                    for typ, delta in self._stream_events(payload, timeout, 1, meta):
                        if typ == "reasoning":
                            reasoning_chunks.append(delta)
                        elif typ == "content":
                            content_chunks.append(delta)
                    resp = self._response_from_stream(reasoning_chunks, content_chunks, meta)
                    if (
                        not retried_empty_length
                        and resp.finish_reason == "length"
                        and not resp.content
                        and not resp.reasoning
                        and payload.get("reasoning_effort")
                    ):
                        retried_empty_length = True
                        cur_max = int(payload.get("max_tokens") or 16384)
                        new_max = min(int(getattr(self.config, "max_tokens", 65536) or 65536), max(cur_max * 2, 32768))
                        new_effort = self._step_down_effort(payload.get("reasoning_effort"))
                        logger.warning(
                            f"Reasoning exhausted max_tokens={cur_max} with empty content; "
                            f"retrying with max_tokens={new_max}, reasoning_effort={new_effort!r}."
                        )
                        payload["max_tokens"] = new_max
                        payload["reasoning_effort"] = new_effort
                        continue
                    return resp
                except Exception as exc:
                    if not is_retryable_error(exc):
                        raise
                    if attempt == max_retries - 1:
                        logger.error(f"Exhausted all {max_retries} retries on stream_collect in complete: {exc}")
                        raise
                    wait = compute_backoff(attempt)
                    logger.warning(
                        f"Stream collection in complete hit {type(exc).__name__} ({exc}); "
                        f"retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait)
            raise RuntimeError("Unexpected exit from retry loop in complete (stream_collect)")

        payload = self._build_payload(messages, max_tokens, temperature, enable_thinking, thinking_budget_tokens)
        retried_empty_length = False
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=timeout, verify=get_tls_verify()) as client:
                    response = client.post(self.chat_url, headers=self.headers, json=payload)
                    _raise_http_error(response)
                    data = response.json()
                    resp = self._parse_response(data)
                    if (
                        not retried_empty_length
                        and resp.finish_reason == "length"
                        and not resp.content
                        and not resp.reasoning
                        and payload.get("reasoning_effort")
                    ):
                        retried_empty_length = True
                        cur_max = int(payload.get("max_tokens") or 16384)
                        new_max = min(int(getattr(self.config, "max_tokens", 65536) or 65536), max(cur_max * 2, 32768))
                        new_effort = self._step_down_effort(payload.get("reasoning_effort"))
                        logger.warning(
                            f"Reasoning exhausted max_tokens={cur_max} with empty content; "
                            f"retrying with max_tokens={new_max}, reasoning_effort={new_effort!r}."
                        )
                        payload["max_tokens"] = new_max
                        payload["reasoning_effort"] = new_effort
                        continue
                    return resp
            except Exception as exc:
                if not is_retryable_error(exc):
                    raise
                if attempt == max_retries - 1:
                    logger.error(f"Exhausted all {max_retries} retries on tunnel LLM request: {exc}")
                    raise
                wait = compute_backoff(attempt)
                logger.warning(
                    f"Tunnel LLM request hit {type(exc).__name__} ({exc}); "
                    f"retrying with exponential backoff in {wait:.1f}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait)
        raise RuntimeError("Unexpected exit from retry loop in complete")

    async def acomplete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: int = 5,
        enable_thinking: Optional[bool] = None,
        thinking_budget_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Asynchronous chat completion request with exponential backoff retry."""
        payload = self._build_payload(messages, max_tokens, temperature, enable_thinking, thinking_budget_tokens)
        timeout = self._timeout(timeout_seconds)

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout, verify=get_tls_verify()) as client:
                    response = await client.post(self.chat_url, headers=self.headers, json=payload)
                    _raise_http_error(response)
                    data = response.json()
                    return self._parse_response(data)
            except Exception as exc:
                if not is_retryable_error(exc):
                    raise
                if attempt == max_retries - 1:
                    logger.error(f"Exhausted all {max_retries} retries on async tunnel LLM request: {exc}")
                    raise
                wait = compute_backoff(attempt)
                logger.warning(
                    f"Async tunnel LLM request hit {type(exc).__name__} ({exc}); "
                    f"retrying with exponential backoff in {wait:.1f}s (attempt {attempt + 1}/{max_retries})..."
                )
                await asyncio.sleep(wait)
        raise RuntimeError("Unexpected exit from retry loop in acomplete")

    async def astream_complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: int = 5,
        enable_thinking: Optional[bool] = None,
        thinking_budget_tokens: Optional[int] = None,
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """Asynchronously streams (type, delta_text) with exponential backoff retry and loop detection."""
        payload = self._build_payload(messages, max_tokens, temperature, enable_thinking,
                                      thinking_budget_tokens, stream=True)
        meta: Dict[str, Any] = {}
        async for ev in self._astream_events(payload, self._timeout(timeout_seconds), max_retries, meta):
            yield ev

    def stream_complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: int = 5,
        enable_thinking: Optional[bool] = None,
        thinking_budget_tokens: Optional[int] = None,
    ) -> Generator[Tuple[str, str], None, None]:
        """Synchronously streams (type, delta_text) with exponential backoff retry and loop detection."""
        payload = self._build_payload(messages, max_tokens, temperature, enable_thinking,
                                      thinking_budget_tokens, stream=True)
        meta: Dict[str, Any] = {}
        yield from self._stream_events(payload, self._timeout(timeout_seconds), max_retries, meta)

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"No choices returned from LLM endpoint: {data}")

        choice = choices[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or msg.get("reasoning_content")

        # If reasoning wasn't in message.reasoning, check for <think> tags in content
        reasoning, content = _split_think_tags(reasoning, content)

        usage = data.get("usage", {}) or {}
        return LLMResponse(
            content=content.strip(),
            reasoning=reasoning.strip() if reasoning else None,
            model=data.get("model", self.config.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            raw_response=data,
            finish_reason=choice.get("finish_reason"),
        )


class FastComposerClient:
    """Client for querying the fast local vLLM composer model (e.g. Gemma 4 on LAN)."""

    def __init__(self, config: Optional[PrimeMathConfig] = None):
        cfg = config or load_config()
        self.fast_cfg = cfg.fast_composer
        self.headers = {
            "Authorization": f"Bearer {self.fast_cfg.api_key}",
            "Content-Type": "application/json",
        }
        self.chat_url = f"{self.fast_cfg.base_url.rstrip('/')}/chat/completions"

    def synthesize_code(self, problem: str, timeout_seconds: Optional[float] = None) -> Optional[str]:
        """Generates executable Python code to directly solve a math problem."""
        if not self.fast_cfg.enabled:
            return None

        timeout = timeout_seconds or self.fast_cfg.timeout_seconds
        payload = {
            "model": self.fast_cfg.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a fast, precise mathematical code synthesizer. "
                        "Output ONLY executable Python code enclosed in ```python ... ``` blocks to compute the exact numerical answer to the problem. "
                        "Write compact, direct code with ZERO comments or explanations to maximize speed and prevent truncation. "
                        "The full problem text and any attached tables/files are already preloaded in the Python variable `PROBLEM_TEXT`. "
                        "Access `PROBLEM_TEXT` directly in your code (e.g. `lines = PROBLEM_TEXT.splitlines()`); NEVER copy or transcribe large data blocks. "
                        "You MUST print the final numerical answer with print(). "
                        "Never output explanations, never write commentary outside the code block."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Write Python code using standard libraries, SymPy, or NumPy to solve this math problem and print the exact answer:\n\n{problem}"
                    ),
                },
            ],
            "temperature": self.fast_cfg.temperature,
            "max_tokens": self.fast_cfg.max_tokens,
        }

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(self.chat_url, json=payload, headers=self.headers)
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    return None
                text = choices[0].get("message", {}).get("content", "")
                return self._extract_code(text)
        except Exception as e:
            logger.debug(f"FastComposer synthesize_code hit exception: {e}")
            return None

    async def asynthesize_code(self, problem: str, timeout_seconds: Optional[float] = None) -> Optional[str]:
        """Asynchronously generates executable Python code to directly solve a math problem."""
        if not self.fast_cfg.enabled:
            return None

        timeout = timeout_seconds or self.fast_cfg.timeout_seconds
        payload = {
            "model": self.fast_cfg.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a fast, precise mathematical code synthesizer. "
                        "Output ONLY executable Python code enclosed in ```python ... ``` blocks to compute the exact numerical answer to the problem. "
                        "Write compact, direct code with ZERO comments or explanations to maximize speed and prevent truncation. "
                        "The full problem text and any attached tables/files are already preloaded in the Python variable `PROBLEM_TEXT`. "
                        "Access `PROBLEM_TEXT` directly in your code (e.g. `lines = PROBLEM_TEXT.splitlines()`); NEVER copy or transcribe large data blocks. "
                        "You MUST print the final numerical answer with print(). "
                        "Never output explanations, never write commentary outside the code block."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Write Python code using standard libraries, SymPy, or NumPy to solve this math problem and print the exact answer:\n\n{problem}"
                    ),
                },
            ],
            "temperature": self.fast_cfg.temperature,
            "max_tokens": self.fast_cfg.max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(self.chat_url, json=payload, headers=self.headers)
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    return None
                text = choices[0].get("message", {}).get("content", "")
                return self._extract_code(text)
        except Exception as e:
            logger.debug(f"FastComposer asynthesize_code hit exception: {e}")
            return None

    def generate_research_queries(self, problem: str, timeout_seconds: Optional[float] = None) -> List[str]:
        """Generates 1-2 targeted academic literature search queries using the fast local model."""
        if not self.fast_cfg.enabled:
            return []

        timeout = timeout_seconds or 10.0
        payload = {
            "model": self.fast_cfg.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a pure mathematics research assistant. "
                        "Analyze this mathematics problem and output 1-2 concise search queries to find underlying textbook concepts, "
                        "governing theorems, or literature lemmas in academic preprints (arXiv/journals).\n"
                        "CRITICAL RULES:\n"
                        "1. Enclose each query in <query>...</query> tags.\n"
                        "2. Output ONLY standard academic English keywords (e.g. <query>multiplicative order density Artin conjecture Kummer extensions</query>).\n"
                        "3. Do NOT output raw LaTeX math, variable names, dollar signs ($), or inequality symbols (<, >).\n"
                        "4. Target keywords that uncover the exact governing analytic evaluation formula (e.g. Euler products, character sums, generating functions).\n"
                        "5. Do NOT include conversational commentary, headings, or author names."
                    ),
                },
                {"role": "user", "content": f"Problem:\n{problem[:1500]}"},
            ],
            "temperature": 0.1,
            "max_tokens": 150,
        }

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(self.chat_url, json=payload, headers=self.headers)
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    return []
                text = choices[0].get("message", {}).get("content", "")
                raw_queries = re.findall(r"<query>(.*?)</query>", text, re.IGNORECASE)
                if not raw_queries:
                    raw_queries = [ln.strip() for ln in text.splitlines() if ln.strip()]

                clean = []
                for q in raw_queries:
                    cq = q.strip().strip("-*0123456789. \"'`")
                    if 4 <= len(cq) <= 80 and not any(cq.lower().startswith(p) for p in ["we need", "problem", "query", "user asks"]):
                        clean.append(cq)
                return clean[:2]
        except Exception as e:
            logger.debug(f"FastComposer generate_research_queries failed: {e}")
            return []

    async def agenerate_research_queries(self, problem: str, timeout_seconds: Optional[float] = None) -> List[str]:
        """Asynchronously generates 1-2 targeted academic literature search queries."""
        if not self.fast_cfg.enabled:
            return []

        timeout = timeout_seconds or 10.0
        payload = {
            "model": self.fast_cfg.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a pure mathematics research assistant. "
                        "Analyze this mathematics problem and output 1-2 concise search queries to find underlying textbook concepts, "
                        "governing theorems, or literature lemmas in academic preprints (arXiv/journals).\n"
                        "CRITICAL RULES:\n"
                        "1. Enclose each query in <query>...</query> tags.\n"
                        "2. Output ONLY standard academic English keywords (e.g. <query>multiplicative order density Artin conjecture Kummer extensions</query>).\n"
                        "3. Do NOT output raw LaTeX math, variable names, dollar signs ($), or inequality symbols (<, >).\n"
                        "4. Target keywords that uncover the exact governing analytic evaluation formula (e.g. Euler products, character sums, generating functions).\n"
                        "5. Do NOT include conversational commentary, headings, or author names."
                    ),
                },
                {"role": "user", "content": f"Problem:\n{problem[:1500]}"},
            ],
            "temperature": 0.1,
            "max_tokens": 150,
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(self.chat_url, json=payload, headers=self.headers)
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    return []
                text = choices[0].get("message", {}).get("content", "")
                raw_queries = re.findall(r"<query>(.*?)</query>", text, re.IGNORECASE)
                if not raw_queries:
                    raw_queries = [ln.strip() for ln in text.splitlines() if ln.strip()]

                clean = []
                for q in raw_queries:
                    cq = q.strip().strip("-*0123456789. \"'`")
                    if 4 <= len(cq) <= 80 and not any(cq.lower().startswith(p) for p in ["we need", "problem", "query", "user asks"]):
                        clean.append(cq)
                return clean[:2]
        except Exception as e:
            logger.debug(f"FastComposer agenerate_research_queries failed: {e}")
            return []

    def _extract_code(self, text: str) -> Optional[str]:
        lines = text.splitlines()
        code_lines: List[str] = []
        in_block = False
        saw_block = False

        for line in lines:
            s = line.strip()
            if s.startswith("```"):
                if not in_block:
                    in_block = True
                    saw_block = True
                else:
                    if s.startswith("```python") or s.startswith("```py"):
                        continue
                    in_block = False
                continue
            if in_block:
                code_lines.append(line)

        if saw_block and code_lines:
            code = "\n".join(code_lines).strip()
        else:
            cleaned = [l for l in lines if not l.strip().startswith("```")]
            code = "\n".join(cleaned).strip()

        if "def " in code or "print(" in code or "import " in code:
            return code
        return None

