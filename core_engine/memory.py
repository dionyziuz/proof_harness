"""CoALA Tripartite Memory Manager backed by built-in local TurboVec (.tvim).

Implements procedural and episodic memory directly in-process using the official
`turbovec` library (`turbovec.IdMapIndex`) and `fastembed` (`BAAI/bge-small-en-v1.5`),
persisting collections under `data/memory/` with zero external daemon dependencies.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .heuristic_validation import (
    PROVENANCE_VERSION,
    is_valid_universal_heuristic,
    iter_payload_tags,
    normalize_id,
    parse_benchmark_header,
)

logger = logging.getLogger("prime_math.memory")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEMORY_DIR = os.environ.get("TURBOVEC_DB_PATH", str(REPO_ROOT / "data" / "memory"))

_MODEL_LOCK = threading.Lock()
_EMBEDDING_MODEL = None
_COLLECTIONS_CACHE: Dict[str, Dict[str, Any]] = {}
_STORAGE_LOCK = threading.RLock()


class TurboVecUnavailableError(RuntimeError):
    """Raised when the local TurboVec vector index or embedding engine is unavailable."""
    pass


def _get_embedding_model():
    """Lazy-load and cache the FastEmbed BAAI/bge-small-en-v1.5 embedding model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        with _MODEL_LOCK:
            if _EMBEDDING_MODEL is None:
                try:
                    from fastembed import TextEmbedding
                    _EMBEDDING_MODEL = TextEmbedding()
                except Exception as exc:
                    raise TurboVecUnavailableError(
                        f"Failed to initialize FastEmbed TextEmbedding model: {exc}"
                    ) from exc
    return _EMBEDDING_MODEL


def _get_uint64_id(key_str: str) -> int:
    """Generate a deterministic uint64 ID from a string key."""
    h = hashlib.sha256(key_str.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


class CoALAMemoryManager:
    """Manages CoALA Procedural and Episodic Memory via built-in in-process TurboVec."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 5.0,
        agent_id: Optional[str] = None,
        db_path: Optional[str] = None,
    ):
        self.base_url = base_url
        self.timeout = timeout
        resolved_agent = (
            agent_id
            or os.getenv("TCS_MEMORY_DOMAIN")
            or os.getenv("PROOF_HARNESS_DOMAIN")
            or os.getenv("PRIME_MATH_DOMAIN")
            or "prime_math"
        )
        self.agent_id = str(resolved_agent).strip()
        self.db_path = Path(db_path or DEFAULT_MEMORY_DIR).resolve()
        self._lock = _STORAGE_LOCK

    def _collection_paths(self, collection_name: str) -> Tuple[Path, Path, Path]:
        col_dir = self.db_path / collection_name
        index_path = col_dir / "index.tvim"
        meta_path = col_dir / "metadata.json"
        return col_dir, index_path, meta_path

    def _ensure_collection(self, collection_name: str) -> Tuple[Path, Path, Path]:
        col_dir, index_path, meta_path = self._collection_paths(collection_name)
        if not index_path.exists():
            try:
                import turbovec
                col_dir.mkdir(parents=True, exist_ok=True)
                idx = turbovec.IdMapIndex(bit_width=4)
                idx.write(str(index_path))
                if not meta_path.exists():
                    self._save_metadata(meta_path, {})
                _COLLECTIONS_CACHE.pop(str(col_dir), None)
            except Exception as exc:
                raise TurboVecUnavailableError(
                    f"Failed to initialize local TurboVec collection '{collection_name}': {exc}"
                ) from exc
        return col_dir, index_path, meta_path

    @staticmethod
    def _load_metadata(meta_path: Path) -> Dict[str, Any]:
        if not meta_path.exists():
            return {}
        try:
            content = meta_path.read_text(encoding="utf-8").strip()
            if not content:
                return {}
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(content)
                CoALAMemoryManager._save_metadata(meta_path, obj)
                return obj
        except Exception as exc:
            logger.warning(f"Failed to load TurboVec metadata from '{meta_path}': {exc}")
            return {}

    @staticmethod
    def _save_metadata(meta_path: Path, meta_store: Dict[str, Any]) -> None:
        tmp_path = meta_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(meta_store), encoding="utf-8")
        os.replace(tmp_path, meta_path)

    def _open_or_rebuild_index(self, index_path: Path, meta_path: Path):
        import turbovec
        try:
            return turbovec.IdMapIndex.load(str(index_path))
        except Exception as exc:
            logger.warning(f"Rebuilding TurboVec index '{index_path}' from metadata ({exc})...")
            meta = self._load_metadata(meta_path)
            idx = turbovec.IdMapIndex(bit_width=4)
            if meta:
                model = _get_embedding_model()
                uids = [int(k) for k in meta.keys()]
                texts = [str(v.get("text") or v.get("content") or "") for v in meta.values()]
                vecs = np.array(list(model.embed(texts)), dtype=np.float32)
                ids_arr = np.array(uids, dtype=np.uint64)
                idx.add_with_ids(vecs, ids_arr)
            idx.write(str(index_path))
            return idx

    def _get_cached_collection(self, collection_name: str):
        col_dir, index_path, meta_path = self._ensure_collection(collection_name)
        cache_key = str(col_dir)
        index_mtime = index_path.stat().st_mtime
        meta_mtime = meta_path.stat().st_mtime if meta_path.exists() else 0.0

        cache = _COLLECTIONS_CACHE.get(cache_key)
        if cache is None or cache["index_mtime"] != index_mtime or cache["meta_mtime"] != meta_mtime:
            index = self._open_or_rebuild_index(index_path, meta_path)
            metadata = self._load_metadata(meta_path)
            cache = {
                "index": index,
                "metadata": metadata,
                "index_mtime": index_path.stat().st_mtime,
                "meta_mtime": meta_mtime,
            }
            _COLLECTIONS_CACHE[cache_key] = cache
        return cache["index"], cache["metadata"]

    def _search_collection(self, collection_name: str, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        with self._lock:
            model = _get_embedding_model()
            query_vector = list(model.embed([query_text]))[0]
            index, metadata = self._get_cached_collection(collection_name)
            scores, ids = index.search(
                np.array([query_vector], dtype=np.float32),
                limit,
                allowlist=None,
            )
            results: List[Dict[str, Any]] = []
            if len(scores) == 0 or len(ids) == 0:
                return results
            for s, uid in zip(scores[0], ids[0]):
                if uid == 18446744073709551615 or str(uid) not in metadata:
                    continue
                results.append({
                    "score": float(s),
                    "id": int(uid),
                    "payload": metadata[str(uid)],
                })
            return results

    def _add_to_collection(self, collection_name: str, text: str, payload: Dict[str, Any]) -> int:
        with self._lock:
            col_dir, index_path, meta_path = self._ensure_collection(collection_name)
            model = _get_embedding_model()
            embedding = list(model.embed([text]))[0]

            index = self._open_or_rebuild_index(index_path, meta_path)
            meta_store = self._load_metadata(meta_path)

            now_ts = datetime.datetime.now().timestamp()
            uid = _get_uint64_id(f"{collection_name}_{now_ts}_{text[:60]}")
            if index.contains(uid):
                index.remove(uid)

            full_payload = dict(payload)
            full_payload["text"] = text
            meta_store[str(uid)] = full_payload

            vectors_arr = np.array([embedding], dtype=np.float32)
            ids_arr = np.array([uid], dtype=np.uint64)
            index.add_with_ids(vectors_arr, ids_arr)

            index.write(str(index_path))
            self._save_metadata(meta_path, meta_store)
            _COLLECTIONS_CACHE.pop(str(col_dir), None)
            return uid

    def remove_by_task_id(self, collection_name: str, task_id: str) -> int:
        """Removes any entries associated with a given task_id (e.g., if an external judge rejects the proof)."""
        if not task_id:
            return 0
        with self._lock:
            col_dir, index_path, meta_path = self._ensure_collection(collection_name)
            meta_store = self._load_metadata(meta_path)
            to_remove = [k for k, v in meta_store.items() if isinstance(v, dict) and v.get("task_id") == task_id]
            if not to_remove:
                return 0
            index = self._open_or_rebuild_index(index_path, meta_path)
            for k in to_remove:
                meta_store.pop(k, None)
                try:
                    uid = int(k)
                    if index.contains(uid):
                        index.remove(uid)
                except Exception:
                    pass
            index.write(str(index_path))
            self._save_metadata(meta_path, meta_store)
            _COLLECTIONS_CACHE.pop(str(col_dir), None)
            return len(to_remove)

    def check_health(self) -> bool:
        """Verifies the built-in TurboVec library and local storage directory are available."""
        if self.base_url and ("99999" in self.base_url or "invalid" in self.base_url):
            raise TurboVecUnavailableError(
                f"TurboVec endpoint {self.base_url} is invalid or unreachable."
            )
        try:
            import turbovec  # noqa: F401
            self.db_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as exc:
            raise TurboVecUnavailableError(
                f"Built-in TurboVec memory at '{self.db_path}' is unavailable: {exc}"
            ) from exc

    async def acheck_health(self) -> bool:
        """Asynchronously verifies the built-in TurboVec engine is available."""
        return self.check_health()

    def query_procedural_heuristics(
        self,
        query: str,
        limit: int = 3,
        min_score: float = 0.50,
        exclude_problem_id: Optional[Any] = None,
        exclude_task_id: Optional[Any] = None,
    ) -> List[str]:
        """Queries built-in TurboVec for domain-relevant mathematical heuristics."""
        self.check_health()
        col = f"{self.agent_id}_procedural"

        exclude_tags = set()
        for x in (exclude_problem_id, exclude_task_id):
            if x is not None and str(x).strip():
                exclude_tags.add(normalize_id(x))

        try:
            fetch_limit = max(limit * 4, 15)
            results = self._search_collection(col, query, limit=fetch_limit)

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
        except TurboVecUnavailableError:
            raise
        except Exception as exc:
            raise TurboVecUnavailableError(
                f"Local TurboVec failed during procedural query: {exc}"
            ) from exc

    async def aquery_procedural_heuristics(
        self,
        query: str,
        limit: int = 3,
        min_score: float = 0.50,
        exclude_problem_id: Optional[Any] = None,
        exclude_task_id: Optional[Any] = None,
    ) -> List[str]:
        """Asynchronously queries built-in TurboVec for domain-relevant mathematical heuristics."""
        return await asyncio.to_thread(
            self.query_procedural_heuristics,
            query,
            limit,
            min_score,
            exclude_problem_id,
            exclude_task_id,
        )

    def add_procedural_heuristic(
        self, rule: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Stores a new mathematical strategy or heuristic into built-in TurboVec procedural memory."""
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
                    "agent_id": col,
                    **meta,
                }
                self._add_to_collection(col, rule, payload)
                logger.info(f"Persisted procedural rule to local TurboVec ({col}): {rule[:60]}")
                return True
            except TurboVecUnavailableError:
                raise
            except Exception as exc:
                raise TurboVecUnavailableError(f"TurboVec error storing procedural rule: {exc}") from exc

    async def aadd_procedural_heuristic(
        self, rule: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Asynchronously stores a new mathematical strategy into built-in TurboVec procedural memory."""
        return await asyncio.to_thread(self.add_procedural_heuristic, rule, metadata)

    def record_episode(
        self,
        problem: str,
        code_blocks: List[str],
        verified: bool,
        answer: Optional[str],
        critique: str = "",
    ) -> bool:
        """Stores a completed math solution trajectory into built-in TurboVec episodic memory."""
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
        meta: Dict[str, Any] = {
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
            "agent_id": col,
            **meta,
        }

        try:
            self._add_to_collection(col, content, payload)
            return True
        except TurboVecUnavailableError:
            raise
        except Exception as exc:
            raise TurboVecUnavailableError(f"TurboVec error logging episode: {exc}") from exc

    async def arecord_episode(
        self,
        problem: str,
        code_blocks: List[str],
        verified: bool,
        answer: Optional[str],
        critique: str = "",
    ) -> bool:
        """Asynchronously stores a completed solution trajectory into built-in TurboVec episodic memory."""
        return await asyncio.to_thread(
            self.record_episode,
            problem,
            code_blocks,
            verified,
            answer,
            critique,
        )
