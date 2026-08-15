

from typing import List, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document


class RAGState(TypedDict, total=False):

    messages: List[BaseMessage]

    question: str

    next_step: str

    retrieved_docs: List[Document]

    document_relevance: bool
    grade_reason: str

    answer: str

    reflection: str
    reflection_passed: bool

    retrieval_attempts: int
    reflection_attempts: int