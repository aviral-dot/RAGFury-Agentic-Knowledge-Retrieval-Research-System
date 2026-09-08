"""Memory orchestration layer for Redis and Mem0."""

import asyncio
import logging
import time

from src.memory.memo_memory import Mem0Memory
from src.memory.redis_memory import RedisMemory
from src.utils.loggers import (
    configure_logging,
    get_logger,
    log_event,
)

configure_logging()

logger = get_logger(__name__)


class MemoryManager:
    """Coordinates short-term Redis and long-term Mem0 memory."""

    def __init__(self):
        """Initialize short-term and long-term memory backends."""

        self.redis = None
        self.mem0 = None

    def _get_redis(self) -> RedisMemory:
        """Lazily initialize and return Redis memory."""

        if self.redis is None:
            self.redis = RedisMemory()

        return self.redis

    def _get_mem0(self) -> Mem0Memory:
        """Lazily initialize and return Mem0 memory."""

        if self.mem0 is None:
            self.mem0 = Mem0Memory()

        return self.mem0

    # =========================================================
    # GET MEMORY CONTEXT
    # =========================================================

    async def get_context(
        self,
        user_id: str,
        conversation_id: str,
        query: str,
    ):
        """
        Retrieve short-term and long-term memory context.

        Redis and Mem0 perform their own detailed logging.
        This layer records only the orchestration lifecycle
        and aggregate counts.
        """

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="memory.context.started",
        )

        # -----------------------------------------------------
        # SHORT-TERM MEMORY
        # -----------------------------------------------------

        redis_start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.DEBUG,
            event="memory.context.redis.started",
        )

        try:
            recent_history = await asyncio.to_thread(
                self._get_redis().get_history,
                user_id=user_id,
                conversation_id=conversation_id,
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - redis_start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="memory.context.redis.failed",
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            logger.exception(
                "Failed to retrieve Redis conversation history",
            )

            raise

        redis_elapsed = (time.perf_counter() - redis_start_time) * 1000

        log_event(
            logger,
            level=logging.DEBUG,
            event="memory.context.redis.completed",
            history_count=len(recent_history),
            duration_ms=round(
                redis_elapsed,
                2,
            ),
        )

        # -----------------------------------------------------
        # LONG-TERM MEMORY
        # -----------------------------------------------------

        mem0_start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.DEBUG,
            event="memory.context.mem0.started",
        )

        long_term_memories = []

        try:
            long_term_memories = await asyncio.to_thread(
                self._get_mem0().search,
                user_id=user_id,
                query=query,
                limit=5,
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - mem0_start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="memory.context.mem0.failed",
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed,
                    2,
                ),
                fallback="empty_long_term_memory",
            )

            logger.exception(
                "Mem0 long-term memory unavailable; "
                "continuing without long-term memory",
            )

            long_term_memories = []

        mem0_elapsed = (time.perf_counter() - mem0_start_time) * 1000

        log_event(
            logger,
            level=logging.DEBUG,
            event="memory.context.mem0.completed",
            memory_count=len(long_term_memories),
            duration_ms=round(
                mem0_elapsed,
                2,
            ),
        )

        # -----------------------------------------------------
        # COMPLETED
        # -----------------------------------------------------

        total_elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="memory.context.completed",
            history_count=len(recent_history),
            memory_count=len(long_term_memories),
            duration_ms=round(
                total_elapsed,
                2,
            ),
        )

        return {
            "recent_history": recent_history,
            "long_term_memories": long_term_memories,
        }

    # =========================================================
    # SAVE TURN
    # =========================================================

    async def save_turn(
        self,
        user_id: str,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
    ):
        """
        Save a conversation turn to long-term Mem0 memory.

        The actual conversation content is never written
        to application logs.
        """

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.DEBUG,
            event="memory.turn.save.started",
        )

        try:
            redis_start_time = time.perf_counter()

            await asyncio.to_thread(
                self._get_redis().add_turn,
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=user_message,
                assistant_message=assistant_message,
            )

            redis_elapsed = (time.perf_counter() - redis_start_time) * 1000

            log_event(
                logger,
                level=logging.DEBUG,
                event="memory.turn.save.redis.completed",
                duration_ms=round(
                    redis_elapsed,
                    2,
                ),
            )

            result = await asyncio.to_thread(
                self._get_mem0().add,
                user_id=user_id,
                user_message=user_message,
                assistant_message=assistant_message,
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="memory.turn.save.failed",
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            logger.exception(
                "Failed to save conversation turn to long-term memory",
            )

            raise

        elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="memory.turn.save.completed",
            duration_ms=round(
                elapsed,
                2,
            ),
        )

        return result
