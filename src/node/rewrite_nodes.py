"""Query rewriting nodes for the Agentic RAG workflow."""

import logging
import time

from src.state.rag_state import RAGState
from src.utils.loggers import (
    configure_logging,
    get_logger,
    log_event,
)

configure_logging()

logger = get_logger(__name__)


class RewriteNodes:
    """Nodes responsible for improving failed retrieval queries."""

    def __init__(
        self,
        llm,
    ):
        """
        Initialize query rewriting nodes.

        Args:
            llm:
                Chat model used for query rewriting.
        """

        self.llm = llm

        log_event(
            logger,
            level=logging.DEBUG,
            event="rewrite.nodes.initialized",
            component="RewriteNodes",
        )

    # =========================================================
    # QUERY REWRITING
    # =========================================================

    def rewrite_query(
        self,
        state: RAGState,
    ) -> dict:
        """
        Rewrite the user's question to improve document retrieval.
        """

        start_time = time.perf_counter()

        question = state["question"]

        retrieval_attempts = state.get(
            "retrieval_attempts",
            0,
        )

        rewrite_attempts = state.get(
            "reflection_attempts",
            0,
        )

        # -----------------------------------------------------
        # START
        # -----------------------------------------------------

        log_event(
            logger,
            level=logging.INFO,
            event="rag.rewrite.started",
            retrieval_attempt=retrieval_attempts,
            rewrite_attempt=rewrite_attempts + 1,
        )

        # -----------------------------------------------------
        # BUILD PROMPT
        # -----------------------------------------------------

        try:
            prompt = f"""
You are a query rewriting assistant for an Agentic RAG system.

The initial retrieval did not return sufficiently relevant
documents for the user's question.

Rewrite the question so that it is clearer, more specific,
and more suitable for semantic and keyword retrieval and use
different wording so as to retrieve the documents effectively.

Do not answer the question.

Original question:
{question}

Return only the improved search query.
"""

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="rag.rewrite.prompt_build.failed",
                error_type=type(exc).__name__,
                retrieval_attempt=retrieval_attempts,
                rewrite_attempt=rewrite_attempts + 1,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            logger.exception(
                "Failed to build query rewrite prompt",
            )

            raise

        # -----------------------------------------------------
        # LLM REWRITE
        # -----------------------------------------------------

        llm_start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.DEBUG,
            event="rag.rewrite.llm.started",
            retrieval_attempt=retrieval_attempts,
            rewrite_attempt=rewrite_attempts + 1,
        )

        try:
            response = self.llm.invoke(prompt)

        except Exception as exc:
            llm_elapsed = (time.perf_counter() - llm_start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="rag.rewrite.llm.failed",
                error_type=type(exc).__name__,
                retrieval_attempt=retrieval_attempts,
                rewrite_attempt=rewrite_attempts + 1,
                duration_ms=round(
                    llm_elapsed,
                    2,
                ),
            )

            logger.exception(
                "Query rewriting LLM invocation failed",
            )

            raise

        llm_elapsed = (time.perf_counter() - llm_start_time) * 1000

        # -----------------------------------------------------
        # RESPONSE VALIDATION
        # -----------------------------------------------------

        rewritten_question = getattr(
            response,
            "content",
            None,
        )

        if rewritten_question is None:
            log_event(
                logger,
                level=logging.ERROR,
                event="rag.rewrite.invalid_response",
                response_type=type(response).__name__,
                retrieval_attempt=retrieval_attempts,
                rewrite_attempt=rewrite_attempts + 1,
                llm_duration_ms=round(
                    llm_elapsed,
                    2,
                ),
            )

            raise ValueError("Query rewriting LLM returned a response without content.")

        rewritten_question = rewritten_question.strip()

        if not rewritten_question:
            log_event(
                logger,
                level=logging.ERROR,
                event="rag.rewrite.empty_response",
                retrieval_attempt=retrieval_attempts,
                rewrite_attempt=rewrite_attempts + 1,
                llm_duration_ms=round(
                    llm_elapsed,
                    2,
                ),
            )

            raise ValueError("Query rewriting LLM returned an empty query.")

        # -----------------------------------------------------
        # COMPLETED
        # -----------------------------------------------------

        total_elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="rag.rewrite.completed",
            retrieval_attempt=retrieval_attempts,
            rewrite_attempt=rewrite_attempts + 1,
            original_query_length=len(question),
            rewritten_query_length=len(rewritten_question),
            llm_duration_ms=round(
                llm_elapsed,
                2,
            ),
            duration_ms=round(
                total_elapsed,
                2,
            ),
        )

        return {
            "question": rewritten_question,
        }
