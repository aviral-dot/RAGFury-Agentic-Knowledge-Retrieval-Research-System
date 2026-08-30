"""Long-term semantic memory backed by Mem0 + Qdrant."""

import logging
import os
import time

from dotenv import load_dotenv
from mem0 import Memory

from src.utils.loggers import (
    configure_logging,
    get_logger,
    log_event,
)

load_dotenv()

configure_logging()

logger = get_logger(__name__)


class Mem0Memory:
    """Long-term semantic memory backed by Mem0 + Qdrant."""

    def __init__(self):
        """Initialize the Mem0 memory backend."""

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="memory.mem0.initialization.started",
        )

        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "openrouter/free",
                    "api_key": os.getenv("OPENROUTER_API_KEY"),
                    "openai_base_url": ("https://openrouter.ai/api/v1"),
                    "temperature": 0.1,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": ("ragfury_memories"),
                    "url": os.getenv(
                        "QDRANT_URL",
                        "http://localhost:6333",
                    ),
                    "embedding_model_dims": 768,
                },
            },
            "embedder": {
                "provider": "gemini",
                "config": {
                    "model": ("models/gemini-embedding-001"),
                    "embedding_dims": 768,
                    "api_key": os.getenv("GEMINI_API_KEY"),
                },
            },
        }

        try:
            self.memory = Memory.from_config(config)

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="memory.mem0.initialization.failed",
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            logger.exception(
                "Failed to initialize Mem0 memory backend",
            )

            raise

        elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="memory.mem0.initialization.completed",
            vector_store="qdrant",
            collection="ragfury_memories",
            embedding_dimensions=768,
            duration_ms=round(
                elapsed,
                2,
            ),
        )

    # =========================================================
    # SEARCH
    # =========================================================

    def search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ):
        """
        Search long-term semantic memory.

        Memory contents and query text are intentionally
        excluded from logs.
        """

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.DEBUG,
            event="memory.mem0.search.started",
            requested_limit=limit,
        )

        try:
            result = self.memory.search(
                query=query,
                filters={
                    "user_id": user_id,
                },
                limit=limit,
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="memory.mem0.search.failed",
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            logger.exception(
                "Mem0 memory search failed",
            )

            raise

        results = result.get(
            "results",
            [],
        )

        memories = [item["memory"] for item in results if item.get("memory")]

        elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="memory.mem0.search.completed",
            result_count=len(results),
            memory_count=len(memories),
            duration_ms=round(
                elapsed,
                2,
            ),
        )

        return memories

    # =========================================================
    # ADD MEMORY
    # =========================================================

    def add(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
    ):
        """
        Persist a conversation turn into long-term memory.

        Conversation contents are intentionally excluded
        from logs.
        """

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.DEBUG,
            event="memory.mem0.add.started",
        )

        try:
            result = self.memory.add(
                [
                    {
                        "role": "user",
                        "content": user_message,
                    },
                    {
                        "role": "assistant",
                        "content": assistant_message,
                    },
                ],
                user_id=user_id,
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="memory.mem0.add.failed",
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            logger.exception(
                "Failed to add conversation turn to Mem0",
            )

            raise

        elapsed = (time.perf_counter() - start_time) * 1000

        result_count = None

        if isinstance(
            result,
            dict,
        ):
            result_items = result.get("results")

            if isinstance(
                result_items,
                list,
            ):
                result_count = len(result_items)

        log_event(
            logger,
            level=logging.INFO,
            event="memory.mem0.add.completed",
            result_count=result_count,
            duration_ms=round(
                elapsed,
                2,
            ),
        )

        return result

    # =========================================================
    # GET ALL MEMORIES
    # =========================================================

    def get_all(
        self,
        user_id: str,
    ):
        """
        Retrieve all long-term memories for a user.

        Memory contents are intentionally excluded from logs.
        """

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.DEBUG,
            event="memory.mem0.get_all.started",
        )

        try:
            result = self.memory.get_all(user_id=user_id)

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="memory.mem0.get_all.failed",
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            logger.exception(
                "Failed to retrieve all Mem0 memories",
            )

            raise

        elapsed = (time.perf_counter() - start_time) * 1000

        memory_count = (
            len(result)
            if isinstance(
                result,
                list,
            )
            else None
        )

        log_event(
            logger,
            level=logging.INFO,
            event="memory.mem0.get_all.completed",
            memory_count=memory_count,
            duration_ms=round(
                elapsed,
                2,
            ),
        )

        return result
