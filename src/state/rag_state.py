from typing import Any, List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from src.models.citation import CitationSource


class RAGState(TypedDict, total=False):
    request_id: str

    messages: List[BaseMessage]
    question: str

    user_id: str
    conversation_id: str

    chat_history: list
    relevant_memories: list

    next_step: str

    retrieved_docs: List[Document]
    citations: List[CitationSource]

    retrieval_attempts: int

    document_relevance: bool
    grade_reason: str

    answer: str

    retrieval_abstained: bool
    abstention_reason: str

    retrieval_metadata: dict[str, Any]
