"""Redis-backed query-response cache for RAGFury."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

import redis.asyncio as redis


class QueryCache:
    """
    Async Redis cache for completed RAG responses.

    Cache entries are scoped to user + conversation so a cached response
    cannot accidentally cross conversation boundaries. The cache stores
    the complete structured response, including citations and metadata.

    Redis failures are deliberately propagated to the caller. The API
    layer should treat cache failures as non-fatal and continue to the
    RAG pipeline (cache fail-open).
    """

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 300,
        key_prefix: str = "ragfury:query_cache:v1",
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("Query-cache TTL must be greater than zero.")

        if not key_prefix:
            raise ValueError("Query-cache key prefix must not be empty.")

        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self.redis = redis.from_url(
            redis_url,
            decode_responses=True,
        )

    @staticmethod
    def _normalize(value: str) -> str:
        """Normalize user input without changing its semantic content."""

        return " ".join(value.strip().split()).casefold()

    def build_key(
        self,
        *,
        question: str,
        user_id: str,
        conversation_id: str,
    ) -> str:
        """Build a bounded, deterministic Redis key for one query scope."""

        payload = {
            "question": self._normalize(question),
            "user_id": self._normalize(user_id),
            "conversation_id": self._normalize(conversation_id),
        }

        digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        return f"{self.key_prefix}:{digest}"

    async def get(self, key: str) -> Dict[str, Any] | None:
        """Return a cached response, or ``None`` on a cache miss."""

        raw = await self.redis.get(key)

        if raw is None:
            return None

        value = json.loads(raw)

        if not isinstance(value, dict):
            raise ValueError("Cached query response must be a JSON object.")

        return value

    async def set(
        self,
        key: str,
        response: Dict[str, Any],
    ) -> None:
        """Store a complete successful query response with a bounded TTL."""

        await self.redis.set(
            key,
            json.dumps(
                response,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            ex=self.ttl_seconds,
        )

    async def delete(self, key: str) -> bool:
        """Delete one cached query response."""

        return bool(await self.redis.delete(key))

    async def clear(self) -> None:
        """Close the Redis connection pool used by this cache."""

        await self.redis.aclose()
