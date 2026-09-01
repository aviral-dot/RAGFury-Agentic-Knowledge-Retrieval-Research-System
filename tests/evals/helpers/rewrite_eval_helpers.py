"""Helpers for RAG query-rewrite evaluation."""

from langchain_core.documents import Document


def contexts_to_documents(
    contexts: list[str],
) -> list[Document]:
    """Convert evaluation contexts into LangChain Documents."""

    return [
        Document(
            page_content=context,
        )
        for context in contexts
    ]


def build_rewrite_state(
    question: str,
    retrieved_context: list[str],
    grade_reason: str,
) -> dict:
    """Build the state required by RewriteNodes."""

    return {
        "question": question,
        "retrieved_docs": contexts_to_documents(
            retrieved_context,
        ),
        "grade_reason": grade_reason,
        "retrieval_attempts": 1,
        "reflection_attempts": 0,
    }
