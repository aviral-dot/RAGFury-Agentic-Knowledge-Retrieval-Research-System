"""API request and response schemas for RAGFury."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for the RAG query endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Ask your agent",
        examples=["How much sick leave can an employee take?"],
    )

    user_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description=("User identifier used for long-term Mem0 memory."),
    )

    conversation_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description=(
            "Conversation identifier used to maintain "
            "conversation continuity and map the request "
            "to a persistent LangGraph checkpoint thread. "
            "Generated automatically if not provided."
        ),
    )


class RetrievedDocument(BaseModel):
    """Serializable representation of a LangChain Document."""

    content: str

    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Response returned by RAGFury."""

    question: str

    answer: str

    # Generated/active conversation ID
    conversation_id: str

    next_step: Optional[str] = None

    documents: List[RetrievedDocument] = Field(default_factory=list)

    document_relevance: Optional[bool] = None

    grade_reason: Optional[str] = None

    reflection: Optional[str] = None

    reflection_passed: Optional[bool] = None

    retrieval_attempts: Optional[int] = None

    reflection_attempts: Optional[int] = None

    response_time: float


class HealthResponse(BaseModel):
    """Health status response."""

    status: str

    rag_initialized: bool


class SystemInfoResponse(BaseModel):
    """RAGFury system information."""

    name: str

    version: str

    rag_initialized: bool

    document_chunks: int
