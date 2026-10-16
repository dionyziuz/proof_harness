"""CoALA Tripartite Memory Manager for Prime Math backed strictly by TurboVec.

Implements procedural and episodic memory using the system-standard TurboVec
service on port 18792. Per system policy, NO local fallback is provided:
if TurboVec is offline, an explicit TurboVecUnavailableError is raised so the
service can be started by the operator.
"""

from __future__ import annotations

import datetime
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import httpx

from .heuristic_validation import (
    PROVENANCE_VERSION,
    is_valid_universal_heuristic,
    iter_payload_tags,
    normalize_id,
    parse_benchmark_header,
)

logger = logging.getLogger("prime_math.memory")

DEFAULT_TURBOVEC_URL = os.environ.get("TURBOVEC_BASE_URL", "http://127.0.0.1:18792")


class TurboVecUnavailableError(RuntimeError):
    """Raised when TurboVec vector search engine is unreachable or offline."""
    pass


class CoALAMemoryManager:
    """Manages CoALA Procedural and Episodic Memory for Prime Math via TurboVec."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 5.0, agent_id: str = "prime_math"):
        self.base_url = (base_url or DEFAULT_TURBOVEC_URL).rstrip("/")
        self.timeout = timeout
        self.agent_id = agent_id
        self._lock = threading.Lock()

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout)

    def _async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout)

    def check_health(self) -> bool:
        """Verifies TurboVec is online and responsive. Raises TurboVecUnavailableError if down."""
        try:
            with self._client() as client:
                resp = client.get(f"{self.base_url}/turbovec/status")
                if resp.status_code == 200:
                    return True
                raise TurboVecUnavailableError(
                    f"TurboVec at {self.base_url} returned HTTP {resp.status_code}: {resp.text}"
                )
        except httpx.RequestError as exc:
            raise TurboVecUnavailableError(
                f"TurboVec vector engine at {self.base_url} is offline or unreachable ({exc}). "
                "Please start or switch on the TurboVec service before running Prime Math."
            ) from exc

    async def acheck_health(self) -> bool:
        """Asynchronously verifies TurboVec is online. Raises TurboVecUnavailableError if down."""
        try:
            async with self._async_client() as client:
                resp = await client.get(f"{self.base_url}/turbovec/status")
                if resp.status_code == 200:
                    return True
                raise TurboVecUnavailableError(
                    f"TurboVec at {self.base_url} returned HTTP {resp.status_code}: {resp.text}"
                )
        except httpx.RequestError as exc:
            raise TurboVecUnavailableError(
                f"TurboVec vector engine at {self.base_url} is offline or unreachable ({exc}). "
                "Please start or switch on the TurboVec service before running Prime Math."
            ) from exc

    def query_procedural_heuristics(
        self,
        query: str,
        limit: int = 3,
        min_score: float = 0.50,
        exclude_problem_id: Optional[Any] = None,
        exclude_task_id: Optional[Any] = None,
    ) -> List[str]:
        """Queries TurboVec for domain-relevant mathematical heuristics and problem-solving rules."""
        self.check_health()
        col = f"{self.agent_id}_procedural"
        url = f"{self.base_url}/v1/memory/{col}/search"

        exclude_tags = set()
        for x in (exclude_problem_id, exclude_task_id):
            if x is not None and str(x).strip():
                exclude_tags.add(normalize_id(x))

        try:
            fetch_limit = max(limit * 4, 15)
            with self._client() as client:
                resp = client.get(url, params={"query": query, "limit": fetch_limit})
                if resp.status_code != 200:
                    raise TurboVecUnavailableError(
                        f"Failed to query TurboVec procedural memory ({resp.status_code}): {resp.text}"
                    )
                data = resp.json()
                results = data.get("results", [])

                rules = []
                for r in results:
                    score = r.get("score", r.get("similarity", 0.0))
                    if score >= min_score:
                        payload = r.get("payload", {})
                        if exclude_tags:
                            doc_tags = set(iter_payload_tags(payload, ["problem_id", "task_id", "pid"]))
                            if doc_tags & exclude_tags:
                                continue
                        text = (payload.get("text") or payload.get("content") or "").strip()
                        if text and text not in rules:
                            rules.append(text)
                return rules[:limit]
        except httpx.RequestError as exc:
            raise TurboVecUnavailableError(
                f"Connection to TurboVec failed during procedural query: {exc}"
            ) from exc

    async def aquery_procedural_heuristics(
        self,
        query: str,
        limit: int = 3,
        min_score: float = 0.50,
        exclude_problem_id: Optional[Any] = None,
        exclude_task_id: Optional[Any] = None,
    ) -> List[str]:
        """Asynchronously queries TurboVec for domain-relevant mathematical heuristics."""
        await self.acheck_health()
        col = f"{self.agent_id}_procedural"
        url = f"{self.base_url}/v1/memory/{col}/search"

        exclude_tags = set()
        for x in (exclude_problem_id, exclude_task_id):
            if x is not None and str(x).strip():
                exclude_tags.add(normalize_id(x))

        try:
            fetch_limit = max(limit * 4, 15)
            async with self._async_client() as client:
                resp = await client.get(url, params={"query": query, "limit": fetch_limit})
                if resp.status_code != 200:
                    raise TurboVecUnavailableError(
                        f"Failed to query TurboVec procedural memory ({resp.status_code}): {resp.text}"
                    )
                data = resp.json()
                results = data.get("results", [])

                rules = []
                for r in results:
                    score = r.get("score", r.get("similarity", 0.0))
                    if score >= min_score:
                        payload = r.get("payload", {})
                        if exclude_tags:
                            doc_tags = set(iter_payload_tags(payload, ["problem_id", "task_id", "pid"]))
                            if doc_tags & exclude_tags:
                                continue
                        text = (payload.get("text") or payload.get("content") or "").strip()
                        if text and text not in rules:
                            rules.append(text)
                return rules[:limit]
        except httpx.RequestError as exc:
            raise TurboVecUnavailableError(
                f"Connection to TurboVec failed during procedural query: {exc}"
            ) from exc

    def add_procedural_heuristic(
        self, rule: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Stores a new mathematical strategy or heuristic into TurboVec procedural memory.

        Enforces near-duplicate prevention (score > 0.90) to prevent rule clutter.
        """
        self.check_health()
        rule = rule.strip()
        if not rule:
            return False

        meta = dict(metadata or {})
        is_valid, reason = is_valid_universal_heuristic(
            rule,
            problem_title=meta.get("title"),
            problem_id=meta.get("problem_id"),
        )
        if not is_valid:
            logger.warning(f"Rejected non-universal procedural heuristic: {reason} ('{rule[:60]}')")
            return False

        col = f"{self.agent_id}_procedural"

        with self._lock:
            # Check for near duplicates (> 0.90)
            try:
                existing = self.query_procedural_heuristics(rule, limit=2, min_score=0.90)
                if existing:
                    logger.info(f"Skipping near-duplicate procedural rule: {rule[:60]}")
                    return False

                now = datetime.datetime.now()
                meta["type"] = "procedural"
                meta["scope"] = "global"
                meta["timestamp"] = now.isoformat()
                meta["provenance_version"] = PROVENANCE_VERSION

                payload = {
                    "content": rule,
                    "text": rule,
                    "title": f"heuristic_{int(now.timestamp())}",
                    "metadata": meta,
                    **meta,
                }

                with self._client() as client:
                    resp = client.post(f"{self.base_url}/v1/memory/{col}/add", json=payload)
                    if resp.status_code == 200:
                        logger.info(f"Persisted procedural rule to TurboVec ({col}): {rule[:60]}")
                        return True
                    raise TurboVecUnavailableError(
                        f"TurboVec returned {resp.status_code} while adding procedural rule: {resp.text}"
                    )
            except httpx.RequestError as exc:
                raise TurboVecUnavailableError(f"TurboVec error storing procedural rule: {exc}") from exc

    async def aadd_procedural_heuristic(
        self, rule: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Asynchronously stores a new mathematical strategy into TurboVec procedural memory."""
        await self.acheck_health()
        rule = rule.strip()
        if not rule:
            return False

        meta = dict(metadata or {})
        is_valid, reason = is_valid_universal_heuristic(
            rule,
            problem_title=meta.get("title"),
            problem_id=meta.get("problem_id"),
        )
        if not is_valid:
            logger.warning(f"Rejected non-universal procedural heuristic: {reason} ('{rule[:60]}')")
            return False

        col = f"{self.agent_id}_procedural"

        try:
            existing = await self.aquery_procedural_heuristics(rule, limit=2, min_score=0.90)
            if existing:
                logger.info(f"Skipping near-duplicate procedural rule: {rule[:60]}")
                return False

            now = datetime.datetime.now()
            meta["type"] = "procedural"
            meta["scope"] = "global"
            meta["timestamp"] = now.isoformat()
            meta["provenance_version"] = PROVENANCE_VERSION

            payload = {
                "content": rule,
                "text": rule,
                "title": f"heuristic_{int(now.timestamp())}",
                "metadata": meta,
                **meta,
            }

            async with self._async_client() as client:
                resp = await client.post(f"{self.base_url}/v1/memory/{col}/add", json=payload)
                if resp.status_code == 200:
                    logger.info(f"Persisted procedural rule to TurboVec ({col}): {rule[:60]}")
                    return True
                raise TurboVecUnavailableError(
                    f"TurboVec returned {resp.status_code} while adding procedural rule: {resp.text}"
                )
        except httpx.RequestError as exc:
            raise TurboVecUnavailableError(f"TurboVec error storing procedural rule: {exc}") from exc

    def record_episode(
        self,
        problem: str,
        code_blocks: List[str],
        verified: bool,
        answer: Optional[str],
        critique: str = "",
    ) -> bool:
        """Stores a completed math solution trajectory into TurboVec episodic memory."""
        self.check_health()
        col = f"{self.agent_id}_episodic"
        now = datetime.datetime.now()

        content = (
            f"Problem: {problem}\n"
            f"Verified: {verified}\n"
            f"Final Answer: {answer or 'None'}\n"
            f"Code Blocks Executed: {len(code_blocks)}\n"
            f"Post-Mortem Critique: {critique}"
        )

        parsed_pid, parsed_title = parse_benchmark_header(problem)
        meta = {
            "type": "episodic",
            "verified": verified,
            "turns": len(code_blocks),
            "timestamp": now.isoformat(),
            "provenance_version": PROVENANCE_VERSION,
        }
        if parsed_pid:
            meta["problem_id"] = parsed_pid
        if parsed_title:
            meta["title"] = parsed_title

        payload = {
            "content": content,
            "text": content,
            "title": f"episode_{int(now.timestamp())}",
            "metadata": meta,
            **meta,
        }

        try:
            with self._client() as client:
                resp = client.post(f"{self.base_url}/v1/memory/{col}/add", json=payload)
                return resp.status_code == 200
        except httpx.RequestError as exc:
            raise TurboVecUnavailableError(f"TurboVec error logging episode: {exc}") from exc

    async def arecord_episode(
        self,
        problem: str,
        code_blocks: List[str],
        verified: bool,
        answer: Optional[str],
        critique: str = "",
    ) -> bool:
        """Asynchronously stores a completed solution trajectory into TurboVec episodic memory."""
        await self.acheck_health()
        col = f"{self.agent_id}_episodic"
        now = datetime.datetime.now()

        content = (
            f"Problem: {problem}\n"
            f"Verified: {verified}\n"
            f"Final Answer: {answer or 'None'}\n"
            f"Code Blocks Executed: {len(code_blocks)}\n"
            f"Post-Mortem Critique: {critique}"
        )

        parsed_pid, parsed_title = parse_benchmark_header(problem)
        meta = {
            "type": "episodic",
            "verified": verified,
            "turns": len(code_blocks),
            "timestamp": now.isoformat(),
            "provenance_version": PROVENANCE_VERSION,
        }
        if parsed_pid:
            meta["problem_id"] = parsed_pid
        if parsed_title:
            meta["title"] = parsed_title

        payload = {
            "content": content,
            "text": content,
            "title": f"episode_{int(now.timestamp())}",
            "metadata": meta,
            **meta,
        }

        try:
            async with self._async_client() as client:
                resp = await client.post(f"{self.base_url}/v1/memory/{col}/add", json=payload)
                return resp.status_code == 200
        except httpx.RequestError as exc:
            raise TurboVecUnavailableError(f"TurboVec error logging episode: {exc}") from exc
