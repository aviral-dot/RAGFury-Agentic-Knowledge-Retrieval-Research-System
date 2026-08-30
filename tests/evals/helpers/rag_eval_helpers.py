"""Helpers for RAG component evaluation."""

from typing import Any

from deepeval.dataset import get_current_golden
from deepeval.test_case import LLMTestCase
from deepeval.tracing import update_current_span

from src.node.retrieval_nodes import RAGNodes
from src.vectorstore.vectorstore import VectorStore


def documents_to_context(
    documents: list[Any],
) -> list[str]:
    """Convert LangChain Documents into retrieval context."""

    return [
        document.page_content
        for document in documents
        if getattr(
            document,
            "page_content",
            None,
        )
    ]


def build_rag_nodes() -> RAGNodes:
    """Build RAGNodes using the real hybrid retriever."""

    vector_store = VectorStore()

    vector_store.initialize()

    retriever = vector_store.get_retriever()

    return RAGNodes(
        retriever=retriever,
        llm=None,
    )


def update_retrieval_span(
    query: str,
    retrieved_context: list[str],
) -> None:
    """
    Register the retriever's component-level
    LLMTestCase on the current DeepEval span.
    """

    golden = get_current_golden()

    expected_output = None

    if golden is not None:
        expected_output = golden.expected_output

    update_current_span(
        test_case=LLMTestCase(
            input=query,
            actual_output=("\n\n".join(retrieved_context)),
            expected_output=expected_output,
            retrieval_context=retrieved_context,
        )
    )
