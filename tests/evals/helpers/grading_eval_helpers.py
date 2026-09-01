"""Helpers for RAG document-grader evaluation."""

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


def build_grader_state(
    question: str,
    retrieved_context: list[str],
) -> dict:
    """Build the state required by GradingNodes."""

    return {
        "question": question,
        "retrieved_docs": contexts_to_documents(
            retrieved_context,
        ),
    }
