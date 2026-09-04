"""Global Redis-backed rate limiter for RAGFury."""

from __future__ import annotations

import time
from dataclasses import dataclass

import redis.asyncio as redis


@dataclass(frozen=True)
class RateLimitResult:
    """Result returned by the global rate limiter."""

    allowed: bool
    limit: int
    remaining: int
    retry_after: int
    reset_after: int


class GlobalRateLimiter:
    """
    Redis-backed global sliding-window rate limiter.

    The limit applies to the entire RAGFury API, not to individual users.

    Example:
        30 requests / 60 seconds globally.
    """

    def __init__(
        self,
        redis_url: str,
        limit: int,
        window_seconds: int = 60,
        key: str = "ragfury:rate_limit:global",
    ) -> None:
        if limit <= 0:
            raise ValueError("Rate limit must be greater than zero.")

        if window_seconds <= 0:
            raise ValueError("Rate-limit window must be greater than zero.")

        self.limit = limit
        self.window_seconds = window_seconds
        self.key = key

        self.redis = redis.from_url(
            redis_url,
            decode_responses=True,
        )

        self._script = self.redis.register_script(
            """
            local key = KEYS[1]
            local now = tonumber(ARGV[1])
            local window_start = tonumber(ARGV[2])
            local limit = tonumber(ARGV[3])
            local window_seconds = tonumber(ARGV[4])
            local request_id = ARGV[5]

            -- Remove requests outside the current window.
            redis.call(
                "ZREMRANGEBYSCORE",
                key,
                0,
                window_start
            )

            -- Count requests currently inside the window.
            local current_count = redis.call(
                "ZCARD",
                key
            )

            -- Reject when the global limit has been reached.
            if current_count >= limit then
                local oldest = redis.call(
                    "ZRANGE",
                    key,
                    0,
                    0,
                    "WITHSCORES"
                )

                local retry_after = window_seconds

                if oldest[2] ~= nil then
                    retry_after = math.ceil(
                        tonumber(oldest[2])
                        + window_seconds
                        - now
                    )
                end

                if retry_after < 1 then
                    retry_after = 1
                end

                return {
                    0,
                    limit,
                    0,
                    retry_after
                }
            end

            -- Record this request.
            redis.call(
                "ZADD",
                key,
                now,
                request_id
            )

            -- Keep the Redis key from living forever.
            redis.call(
                "EXPIRE",
                key,
                window_seconds + 1
            )

            local new_count = current_count + 1
            local remaining = limit - new_count

            return {
                1,
                limit,
                remaining,
                window_seconds
            }
            """
        )

    async def check(
        self,
        request_id: str,
    ) -> RateLimitResult:
        """
        Check and consume one global request slot.

        Redis errors are intentionally allowed to propagate so the
        API can fail closed instead of bypassing rate limiting.
        """

        now = time.time()

        window_start = now - self.window_seconds

        result = await self._script(
            keys=[self.key],
            args=[
                str(now),
                str(window_start),
                str(self.limit),
                str(self.window_seconds),
                request_id,
            ],
        )

        allowed = bool(int(result[0]))
        limit = int(result[1])
        remaining = int(result[2])
        retry_after = int(result[3])

        return RateLimitResult(
            allowed=allowed,
            limit=limit,
            remaining=remaining,
            retry_after=retry_after,
            reset_after=retry_after,
        )

    async def close(self) -> None:
        """Close the Redis connection pool."""

        await self.redis.aclose()
