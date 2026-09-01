"""Helpers for RAG answer-generation evaluation."""

from langchain_core.documents import Document

from src.node.generation_nodes import GenerationNodes


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


def build_generation_state(
    question: str,
    retrieved_context: list[str],
) -> dict:
    """Build the state required by GenerationNodes."""

    return {
        "question": question,
        "retrieved_docs": contexts_to_documents(
            retrieved_context,
        ),
    }


def build_generation_nodes(
    llm,
) -> GenerationNodes:
    """Build the real production generation node."""

    return GenerationNodes(
        llm=llm,
    )
