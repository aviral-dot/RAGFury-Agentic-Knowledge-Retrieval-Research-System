"""Memory orchestration layer for Redis and Mem0."""

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

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.DEBUG,
            event="memory.manager.initialization.started",
        )

        try:
            self.redis = RedisMemory()
            self.mem0 = Mem0Memory()

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="memory.manager.initialization.failed",
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            logger.exception(
                "Failed to initialize memory manager",
            )

            raise

        elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="memory.manager.initialization.completed",
            backends="redis+mem0",
            duration_ms=round(
                elapsed,
                2,
            ),
        )

    # =========================================================
    # GET MEMORY CONTEXT
    # =========================================================

    def get_context(
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
            recent_history = self.redis.get_history(
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

        try:
            long_term_memories = self.mem0.search(
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
            )

            logger.exception(
                "Failed to retrieve Mem0 long-term memories",
            )

            raise

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

    def save_turn(
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
            result = self.mem0.add(
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
