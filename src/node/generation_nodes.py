"""RAG answer generation nodes."""

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


class GenerationNodes:
    """Nodes responsible for generating answers from retrieved documents."""

    def __init__(
        self,
        llm,
    ):
        """
        Initialize generation nodes.

        Args:
            llm:
                Chat model used for answer generation.
        """

        self.llm = llm

        log_event(
            logger,
            level=logging.DEBUG,
            event="generation.nodes.initialized",
            component="GenerationNodes",
        )

    # =========================================================
    # ANSWER GENERATION
    # =========================================================
    @traceable(
        name="RAGFury Answer Generation",
        run_type="chain",
    )
    async def generate_answer(
        self,
        state: RAGState,
    ) -> dict:
        """
        Generate the final answer using retrieved documents.
        """

        start_time = time.perf_counter()

        question = state["question"]

        documents = state.get(
            "retrieved_docs",
            [],
        )

        document_count = len(documents)

        log_event(
            logger,
            level=logging.INFO,
            event="rag.generation.started",
            document_count=document_count,
        )

        # -----------------------------------------------------
        # BUILD CONTEXT
        # -----------------------------------------------------

        try:
            context = "\n\n".join(doc.page_content for doc in documents)

            prompt = f"""
You are a knowledgeable assistant answering a user's question
using the provided retrieved documents.

User question:
{question}

Retrieved documents:
{context}

Instructions:

1. Answer the user's question directly.
2. Use the retrieved documents as the primary source of information.
3. Do not invent information that is not supported by the documents.
4. If the documents do not contain enough information to answer
   the question, clearly say so.
5. Do not mention internal tools, agents, retrieval, grading,
   or workflow.
6. Keep the answer clear, concise, and useful.
"""

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="rag.generation.prompt_build.failed",
                document_count=document_count,
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            logger.exception(
                "Failed to build RAG generation prompt",
            )

            raise

        # -----------------------------------------------------
        # LLM INVOCATION
        # -----------------------------------------------------

        llm_start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.DEBUG,
            event="rag.generation.llm.started",
            document_count=document_count,
        )

        try:
            response = await self.llm.ainvoke(prompt)

        except Exception as exc:
            llm_elapsed = (time.perf_counter() - llm_start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="rag.generation.llm.failed",
                document_count=document_count,
                error_type=type(exc).__name__,
                duration_ms=round(
                    llm_elapsed,
                    2,
                ),
            )

            logger.exception(
                "RAG answer generation failed",
            )

            raise

        llm_elapsed = (time.perf_counter() - llm_start_time) * 1000

        # -----------------------------------------------------
        # RESPONSE VALIDATION
        # -----------------------------------------------------

        answer = getattr(
            response,
            "content",
            None,
        )

        if answer is None:
            log_event(
                logger,
                level=logging.ERROR,
                event="rag.generation.invalid_response",
                document_count=document_count,
                response_type=type(response).__name__,
                llm_duration_ms=round(
                    llm_elapsed,
                    2,
                ),
            )

            raise ValueError(
                "Answer generation LLM returned a response without content."
            )

        answer = answer.strip()

        if not answer:
            log_event(
                logger,
                level=logging.ERROR,
                event="rag.generation.empty_response",
                document_count=document_count,
                llm_duration_ms=round(
                    llm_elapsed,
                    2,
                ),
            )

            raise ValueError("Answer generation LLM returned an empty answer.")

        # -----------------------------------------------------
        # COMPLETED
        # -----------------------------------------------------

        total_elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="rag.generation.completed",
            document_count=document_count,
            answer_length=len(answer),
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
            "answer": answer,
        }
