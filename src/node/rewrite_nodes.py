"""Query rewriting nodes for the Agentic RAG workflow."""

import logging
import time

from src.state.rag_state import RAGState
from src.utils.llm_observability import invoke_llm
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

    async def rewrite_query(
        self,
        state: RAGState,
    ) -> dict:
        """
        Rewrite the user's question to improve document retrieval.
        """

        start_time = time.perf_counter()

        question = state["question"]

        retrieved_docs = state.get(
            "retrieved_docs",
            [],
        )

        grade_reason = state.get(
            "grade_reason",
            "The retrieved documents were not sufficiently relevant.",
        )

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
        retrieved_context = "\n\n".join(
            [
                f"Candidate {index + 1}:\n{doc.page_content[:1500]}"
                for index, doc in enumerate(retrieved_docs[:5])
            ]
        )

        if not retrieved_context:
            retrieved_context = "No documents were retrieved."

        try:
            prompt = f"""
You are a query rewriting assistant for an Agentic RAG system.

The previous retrieval attempt did not return sufficiently
relevant documents.

Your task is to generate ONE improved search query for the
next retrieval attempt.

Previous search query:
{question}

Retrieved candidate documents:
{retrieved_context}

Grader reason:
{grade_reason}

Instructions:

1. Preserve the original information need.
2. Analyze the grader's reason to understand why retrieval failed.
3. Examine the retrieved candidates to understand what retrieval
   found and why those documents may be irrelevant.
4. If the retrieved documents are about the wrong topic, change
   the query so that it targets the correct topic.
5. If the query is too broad, make it more specific.
6. If useful terminology or synonyms are present in the retrieved
   documents, use them when appropriate.
7. Optimize the query for both semantic and keyword retrieval.
8. Do not answer the user's question.
9. Do not invent facts.
10. Return ONLY the improved search query.
11. Do not include explanations, numbering, labels, or quotation marks.

Improved search query:
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
            response = await invoke_llm(
                llm=self.llm,
                operation="rewrite",
                invoke=lambda: self.llm.ainvoke(prompt),
            )

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
