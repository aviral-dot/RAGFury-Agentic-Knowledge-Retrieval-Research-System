"""Production-grade Redis query cache for RAGFury."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from typing import Any

import redis.asyncio as redis


class QueryCache:
    """
    Redis-backed cache for RAG responses.

    Supports:

    1. Shared cache
       - Same knowledge-base query can be reused across users.

    2. User-scoped cache
       - User/conversation-dependent responses remain isolated.

    3. Distributed single-flight locking
       - Prevents multiple concurrent requests from executing
         the same expensive RAG computation simultaneously.
    """

    _RELEASE_LOCK_SCRIPT = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    else
        return 0
    end
    """

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 300,
        key_prefix: str = "ragfury:query_cache:v2",
        lock_prefix: str = "ragfury:query_lock:v1",
        lock_ttl_seconds: int = 60,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("Query-cache TTL must be greater than zero.")

        if lock_ttl_seconds <= 0:
            raise ValueError("Query-lock TTL must be greater than zero.")

        if not key_prefix:
            raise ValueError("Query-cache key prefix must not be empty.")

        if not lock_prefix:
            raise ValueError("Query-lock prefix must not be empty.")

        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self.lock_prefix = lock_prefix
        self.lock_ttl_seconds = lock_ttl_seconds

        self.redis = redis.from_url(
            redis_url,
            decode_responses=True,
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.strip().split()).casefold()

    # ------------------------------------------------------------------
    # Fingerprint
    # ------------------------------------------------------------------

    def _fingerprint(
        self,
        payload: dict[str, Any],
    ) -> str:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Shared cache
    # ------------------------------------------------------------------

    def build_shared_key(
        self,
        *,
        question: str,
        model_version: str,
        prompt_version: str,
        index_version: str,
    ) -> str:
        payload = {
            "scope": "shared",
            "question": self._normalize(question),
            "model_version": self._normalize(model_version),
            "prompt_version": self._normalize(prompt_version),
            "index_version": self._normalize(index_version),
        }

        digest = self._fingerprint(payload)

        return f"{self.key_prefix}:shared:{digest}"

    # ------------------------------------------------------------------
    # User-scoped cache
    # ------------------------------------------------------------------

    def build_user_key(
        self,
        *,
        question: str,
        user_id: str,
        conversation_id: str,
        model_version: str,
        prompt_version: str,
        index_version: str,
    ) -> str:
        payload = {
            "scope": "user",
            "question": self._normalize(question),
            "user_id": self._normalize(user_id),
            "conversation_id": self._normalize(conversation_id),
            "model_version": self._normalize(model_version),
            "prompt_version": self._normalize(prompt_version),
            "index_version": self._normalize(index_version),
        }

        digest = self._fingerprint(payload)

        return f"{self.key_prefix}:user:{digest}"

    # ------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------

    async def get(
        self,
        key: str,
    ) -> dict[str, Any] | None:
        raw = await self.redis.get(key)

        if raw is None:
            return None

        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Cached query response contains invalid JSON.") from exc

        if not isinstance(value, dict):
            raise ValueError("Cached query response must be a JSON object.")

        return value

    # ------------------------------------------------------------------
    # SET
    # ------------------------------------------------------------------

    async def set(
        self,
        key: str,
        response: dict[str, Any],
    ) -> None:
        await self.redis.set(
            key,
            json.dumps(
                response,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            ex=self.ttl_seconds,
        )

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------

    async def delete(
        self,
        key: str,
    ) -> bool:
        deleted = await self.redis.delete(key)

        return bool(deleted)

    # ------------------------------------------------------------------
    # Distributed lock
    # ------------------------------------------------------------------

    def build_lock_key(
        self,
        *,
        computation_key: str,
    ) -> str:
        digest = hashlib.sha256(computation_key.encode("utf-8")).hexdigest()

        return f"{self.lock_prefix}:{digest}"

    async def acquire_lock(
        self,
        key: str,
    ) -> str | None:
        token = uuid.uuid4().hex

        acquired = await self.redis.set(
            key,
            token,
            nx=True,
            ex=self.lock_ttl_seconds,
        )

        if acquired:
            return token

        return None

    async def release_lock(
        self,
        key: str,
        token: str,
    ) -> bool:
        released = await self.redis.eval(
            self._RELEASE_LOCK_SCRIPT,
            1,
            key,
            token,
        )

        return bool(released)

    # ------------------------------------------------------------------
    # Wait for another request
    # ------------------------------------------------------------------

    async def wait_for_result(
        self,
        key: str,
        *,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.05,
    ) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout_seconds

        while time.monotonic() < deadline:
            cached = await self.get(key)

            if cached is not None:
                return cached

            remaining = deadline - time.monotonic()

            if remaining <= 0:
                break

            await asyncio.sleep(
                min(
                    poll_interval_seconds,
                    remaining,
                )
            )

        return None

    # ------------------------------------------------------------------
    # Redis health
    # ------------------------------------------------------------------

    async def ping(self) -> bool:
        return bool(await self.redis.ping())

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await self.redis.aclose()
