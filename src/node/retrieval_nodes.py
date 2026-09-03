"""LangGraph nodes for RAG workflow."""

import logging
import time

from langsmith import traceable

from src.state.rag_state import RAGState
from src.utils.loggers import (
    configure_logging,
    get_logger,
    log_event,
)

configure_logging()

logger = get_logger(__name__)


class RAGNodes:
    """Contains node functions for RAG workflow."""

    def __init__(
        self,
        retriever,
        llm,
    ):
        self.retriever = retriever
        self.llm = llm

        log_event(
            logger,
            level=logging.DEBUG,
            event="rag.nodes.initialized",
            component="RAGNodes",
        )

    @staticmethod
    def _serialize_retrieved_document(
        doc,
        rank: int,
    ) -> dict:
        """Return safe retrieval metadata for observability."""

        metadata = (
            getattr(
                doc,
                "metadata",
                {},
            )
            or {}
        )

        return {
            "rank": rank,
            "document_id": metadata.get("document_id"),
            "chunk_id": metadata.get("chunk_id"),
            "source": metadata.get("source"),
            "score": metadata.get("score"),
        }

    # =========================================================
    # RETRIEVAL
    # =========================================================
    @traceable(
        name="RAGFury Retrieval",
        run_type="retriever",
    )
    async def retrieve_docs(
        self,
        state: RAGState,
    ) -> RAGState:
        """
        Retrieve documents and validate them
        using the retrieval security guardrail.
        """

        question = state["question"]

        retrieval_attempts = (
            state.get(
                "retrieval_attempts",
                0,
            )
            + 1
        )

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="rag.retrieval.started",
            retrieval_attempt=retrieval_attempts,
        )

        # -----------------------------------------------------
        # RETRIEVE
        # -----------------------------------------------------

        try:
            docs = await self.retriever.ainvoke(question)

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="rag.retrieval.failed",
                retrieval_attempt=retrieval_attempts,
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            logger.exception(
                "Document retrieval failed",
            )

            raise

        retrieval_elapsed = (time.perf_counter() - start_time) * 1000

        document_count = len(docs)

        retrieval_results = [
            self._serialize_retrieved_document(
                doc,
                rank=index + 1,
            )
            for index, doc in enumerate(docs)
        ]

        log_event(
            logger,
            level=logging.INFO,
            event="rag.retrieval.completed",
            retrieval_attempt=retrieval_attempts,
            document_count=document_count,
            duration_ms=round(
                retrieval_elapsed,
                2,
            ),
        )

        # -----------------------------------------------------
        # EMPTY RETRIEVAL
        # -----------------------------------------------------

        if not docs:
            log_event(
                logger,
                level=logging.WARNING,
                event="rag.retrieval.empty",
                retrieval_attempt=retrieval_attempts,
            )

        total_elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="rag.retrieval.node.completed",
            retrieval_attempt=retrieval_attempts,
            document_count=document_count,
            duration_ms=round(
                total_elapsed,
                2,
            ),
        )

        return {
            "question": question,
            "retrieved_docs": docs,
            "retrieval_attempts": retrieval_attempts,
            "retrieval_metadata": {
                "attempt": retrieval_attempts,
                "document_count": document_count,
                "duration_ms": round(
                    retrieval_elapsed,
                    2,
                ),
                "results": retrieval_results,
            },
        }
