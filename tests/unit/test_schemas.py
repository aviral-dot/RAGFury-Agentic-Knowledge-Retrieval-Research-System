import pytest
from pydantic import ValidationError

from api.schemas import (
    CitationResponse,
    ErrorDetail,
    ErrorResponse,
    FeedbackRequest,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrievedDocument,
    SystemInfoResponse,
)

# ---------------------------------------------------------------------------
# QueryRequest
# ---------------------------------------------------------------------------


def test_query_request_valid():
    request = QueryRequest(
        question="How many hours per week must a full-time employee work?",
        user_id="user-123",
    )

    assert request.question == (
        "How many hours per week must a full-time employee work?"
    )
    assert request.user_id == "user-123"
    assert request.conversation_id is None


def test_query_request_accepts_conversation_id():
    request = QueryRequest(
        question="What is the leave policy?",
        user_id="user-123",
        conversation_id="conversation-456",
    )

    assert request.conversation_id == "conversation-456"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"user_id": "user-123"},
        {"question": "What is the policy?"},
    ],
)
def test_query_request_missing_required_fields(payload):
    with pytest.raises(ValidationError):
        QueryRequest(**payload)


def test_query_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        QueryRequest(
            question="",
            user_id="user-123",
        )


def test_query_request_allows_whitespace_question():
    request = QueryRequest(
        question=" ",
        user_id="user-123",
    )

    assert request.question == " "


def test_query_request_question_too_long():
    with pytest.raises(ValidationError):
        QueryRequest(
            question="a" * 2001,
            user_id="user-123",
        )


def test_query_request_user_id_too_long():
    with pytest.raises(ValidationError):
        QueryRequest(
            question="What is the leave policy?",
            user_id="u" * 101,
        )


def test_query_request_conversation_id_too_long():
    with pytest.raises(ValidationError):
        QueryRequest(
            question="What is the leave policy?",
            user_id="user-123",
            conversation_id="c" * 101,
        )


# ---------------------------------------------------------------------------
# RetrievedDocument
# ---------------------------------------------------------------------------


def test_retrieved_document_valid(sample_document):
    document = RetrievedDocument(
        content=sample_document.page_content,
        metadata=sample_document.metadata,
    )

    assert document.content == sample_document.page_content
    assert document.metadata["source"] == "employee_handbook.pdf"
    assert document.metadata["chunk_id"] == ("employee_handbook_p1_chunk_001")


def test_retrieved_document_metadata_defaults_to_empty_dict():
    document = RetrievedDocument(content="Some content")

    assert document.metadata == {}


def test_retrieved_document_metadata_is_independent():
    first = RetrievedDocument(content="First")
    second = RetrievedDocument(content="Second")

    first.metadata["source"] = "document.pdf"

    assert second.metadata == {}


# ---------------------------------------------------------------------------
# CitationResponse
# ---------------------------------------------------------------------------


def test_citation_response_valid():
    citation = CitationResponse(
        citation_id="cite-1",
        source="employee_handbook.pdf",
        chunk_id="employee_handbook_p1_chunk_001",
        page=1,
    )

    assert citation.citation_id == "cite-1"
    assert citation.source == "employee_handbook.pdf"
    assert citation.chunk_id == "employee_handbook_p1_chunk_001"
    assert citation.page == 1


def test_citation_response_page_is_optional():
    citation = CitationResponse(
        citation_id="cite-1",
        source="employee_handbook.pdf",
        chunk_id="chunk-001",
    )

    assert citation.page is None


def test_citation_response_missing_required_fields():
    with pytest.raises(ValidationError):
        CitationResponse(
            citation_id="cite-1",
            source="employee_handbook.pdf",
        )


# ---------------------------------------------------------------------------
# QueryResponse
# ---------------------------------------------------------------------------


def test_query_response_minimal_valid():
    response = QueryResponse(
        question="What is the leave policy?",
        answer="Employees receive annual leave according to company policy.",
        request_id="request-123",
        conversation_id="conversation-123",
        response_time=0.42,
    )

    assert response.question == "What is the leave policy?"
    assert response.answer.startswith("Employees receive")
    assert response.request_id == "request-123"
    assert response.conversation_id == "conversation-123"
    assert response.response_time == 0.42

    assert response.citations == []
    assert response.documents == []


def test_query_response_with_citations_and_documents():
    response = QueryResponse(
        question="How many hours must a full-time employee work?",
        answer="At least 30 hours per week on average.",
        request_id="request-123",
        conversation_id="conversation-123",
        response_time=0.50,
        citations=[
            CitationResponse(
                citation_id="cite-1",
                source="employee_handbook.pdf",
                chunk_id="employee_handbook_p1_chunk_001",
                page=1,
            )
        ],
        documents=[
            RetrievedDocument(
                content="A full-time employee must work at least 30 hours.",
                metadata={
                    "source": "employee_handbook.pdf",
                    "page": 1,
                    "chunk_id": "employee_handbook_p1_chunk_001",
                },
            )
        ],
        document_relevance=True,
        grade_reason="The retrieved document directly answers the question.",
        reflection="The answer is supported by the retrieved document.",
        reflection_passed=True,
        retrieval_attempts=1,
        reflection_attempts=1,
        next_step="complete",
        run_id="run-123",
    )

    assert len(response.citations) == 1
    assert len(response.documents) == 1
    assert response.citations[0].citation_id == "cite-1"
    assert response.documents[0].metadata["page"] == 1
    assert response.document_relevance is True
    assert response.reflection_passed is True
    assert response.retrieval_attempts == 1
    assert response.reflection_attempts == 1
    assert response.run_id == "run-123"


def test_query_response_optional_fields_default_to_none():
    response = QueryResponse(
        question="Test question",
        answer="Test answer",
        request_id="request-123",
        conversation_id="conversation-123",
        response_time=0.1,
    )

    assert response.run_id is None
    assert response.next_step is None
    assert response.document_relevance is None
    assert response.grade_reason is None
    assert response.reflection is None
    assert response.reflection_passed is None
    assert response.retrieval_attempts is None
    assert response.reflection_attempts is None


# ---------------------------------------------------------------------------
# HealthResponse
# ---------------------------------------------------------------------------


def test_health_response():
    response = HealthResponse(
        status="healthy",
        rag_initialized=True,
    )

    assert response.status == "healthy"
    assert response.rag_initialized is True


# ---------------------------------------------------------------------------
# SystemInfoResponse
# ---------------------------------------------------------------------------


def test_system_info_response():
    response = SystemInfoResponse(
        name="RAGFury",
        version="0.1.0",
        rag_initialized=True,
        document_chunks=100,
    )

    assert response.name == "RAGFury"
    assert response.version == "0.1.0"
    assert response.rag_initialized is True
    assert response.document_chunks == 100


# ---------------------------------------------------------------------------
# ErrorResponse
# ---------------------------------------------------------------------------


def test_error_response():
    response = ErrorResponse(
        error=ErrorDetail(
            code="VALIDATION_ERROR",
            message="Invalid request.",
            request_id="request-123",
        )
    )

    assert response.error.code == "VALIDATION_ERROR"
    assert response.error.message == "Invalid request."
    assert response.error.request_id == "request-123"


def test_error_detail_missing_required_fields():
    with pytest.raises(ValidationError):
        ErrorDetail(
            code="ERROR",
            message="Something went wrong.",
        )


# ---------------------------------------------------------------------------
# FeedbackRequest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score",
    [0.0, 0.5, 1.0],
)
def test_feedback_request_valid_scores(score):
    request = FeedbackRequest(
        run_id="run-123",
        score=score,
    )

    assert request.run_id == "run-123"
    assert request.score == score
    assert request.comment is None


@pytest.mark.parametrize(
    "score",
    [-0.1, 1.1, 2.0, -1.0],
)
def test_feedback_request_rejects_invalid_scores(score):
    with pytest.raises(ValidationError):
        FeedbackRequest(
            run_id="run-123",
            score=score,
        )


def test_feedback_request_with_comment():
    request = FeedbackRequest(
        run_id="run-123",
        score=0.9,
        comment="Very useful answer.",
    )

    assert request.comment == "Very useful answer."


def test_feedback_request_requires_run_id():
    with pytest.raises(ValidationError):
        FeedbackRequest(score=0.8)
