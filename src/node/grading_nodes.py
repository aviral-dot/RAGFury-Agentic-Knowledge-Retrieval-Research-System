"""Document grading nodes for the Agentic RAG workflow."""

import logging
import time

from langsmith import traceable
from pydantic import BaseModel, Field

from src.state.rag_state import RAGState
from src.utils.loggers import (
    configure_logging,
    get_logger,
    log_event,
)

configure_logging()

logger = get_logger(__name__)


class DocumentGrade(BaseModel):
    """Structured output from the document relevance grader."""

    relevant: bool = Field(
        description=(
            "Whether the retrieved documents are relevant to the user's question."
        )
    )

    reason: str = Field(description=("Brief explanation for the relevance decision."))


class GradingNodes:
    """Nodes responsible for grading retrieved documents."""

    def __init__(
        self,
        llm,
    ):
        """
        Initialize grading nodes.

        Args:
            llm:
                Chat model used for document grading.
        """

        self.llm = llm

        self.grader = llm.with_structured_output(DocumentGrade, method="json_schema")

        log_event(
            logger,
            level=logging.DEBUG,
            event="grading.nodes.initialized",
            component="GradingNodes",
            structured_output=True,
        )

    # =========================================================
    # DOCUMENT GRADING
    # =========================================================
    @traceable(
        name="RAGFury Document Grading",
        run_type="chain",
    )
    async def grade_documents(
        self,
        state: RAGState,
    ) -> dict:
        """
        Grade whether retrieved documents are relevant
        to the user's question.
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
            event="rag.grading.started",
            document_count=document_count,
        )

        # -----------------------------------------------------
        # NO DOCUMENTS
        # -----------------------------------------------------

        if not documents:
            log_event(
                logger,
                level=logging.WARNING,
                event="rag.grading.skipped",
                reason="no_documents",
                document_count=0,
            )

            return {
                "document_relevance": False,
                "grade_reason": ("No documents were retrieved."),
            }

        # -----------------------------------------------------
        # BUILD CONTEXT
        # -----------------------------------------------------

        try:
            context = "\n\n".join(doc.page_content for doc in documents)

            prompt = f"""
You are a document relevance grader for a Retrieval-Augmented
Generation (RAG) system.

Your task is to determine whether the retrieved document content
is relevant and useful for answering the user's question.

User question:
{question}

Retrieved documents:
{context}

Grading rules:

1. Return relevant=True if ANY retrieved document contains
   information that directly or indirectly helps answer the question.

2. Return relevant=True when the document contains the answer
   or contains specific facts needed to answer the question.

3. Semantic matches count as relevant. Do not require exact
   keyword matches.

4. Minor spelling or grammar mistakes in the user's question
   must NOT affect relevance.

5. Do NOT require the entire source document to be retrieved.
   A single relevant chunk is sufficient.

6. Return relevant=False only when the retrieved documents
   genuinely do not contain useful information for answering
   the question.

User questions may use different wording from the document.
Judge based on meaning, not exact wording.

Return a brief reason explaining your decision.
"""

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="rag.grading.prompt_build.failed",
                document_count=document_count,
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            logger.exception(
                "Failed to build document grading prompt",
            )

            raise

        # -----------------------------------------------------
        # LLM GRADING
        # -----------------------------------------------------

        llm_start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.DEBUG,
            event="rag.grading.llm.started",
            document_count=document_count,
        )

        try:
            result = await self.grader.ainvoke(prompt)

        except Exception as exc:
            llm_elapsed = (time.perf_counter() - llm_start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="rag.grading.llm.failed",
                document_count=document_count,
                error_type=type(exc).__name__,
                duration_ms=round(
                    llm_elapsed,
                    2,
                ),
            )

            logger.exception(
                "Document relevance grading failed",
            )

            raise

        llm_elapsed = (time.perf_counter() - llm_start_time) * 1000

        # -----------------------------------------------------
        # STRUCTURED RESPONSE VALIDATION
        # -----------------------------------------------------

        if not isinstance(
            result,
            DocumentGrade,
        ):
            log_event(
                logger,
                level=logging.ERROR,
                event="rag.grading.invalid_response",
                document_count=document_count,
                response_type=type(result).__name__,
                llm_duration_ms=round(
                    llm_elapsed,
                    2,
                ),
            )

            raise ValueError("Document grader returned an invalid structured response.")

        # -----------------------------------------------------
        # GRADING RESULT
        # -----------------------------------------------------

        relevance = result.relevant

        total_elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="rag.grading.completed",
            document_count=document_count,
            relevant=relevance,
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
            "document_relevance": relevance,
            "grade_reason": result.reason,
        }
