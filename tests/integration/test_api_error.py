import pytest
from fastapi.testclient import TestClient

import api.main as main
from src.guardrails.exceptions import MaliciousDocumentError
from src.rate_limit.rate_limiter import RateLimitResult


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def allow_rate_limit(monkeypatch):
    async def fake_check(**kwargs):
        return RateLimitResult(
            allowed=True,
            limit=30,
            remaining=29,
            retry_after=60,
            reset_after=60,
        )

    monkeypatch.setattr(
        main.global_rate_limiter,
        "check",
        fake_check,
    )


@pytest.fixture
def allow_input_guardrail(monkeypatch):
    async def fake_check_input(question):
        return True

    monkeypatch.setattr(
        main,
        "check_input",
        fake_check_input,
    )


def test_query_returns_400_when_malicious_document_is_detected(
    client,
    monkeypatch,
    allow_rate_limit,
    allow_input_guardrail,
):
    monkeypatch.setattr(
        main.rag_service,
        "initialized",
        True,
    )

    async def fake_query(**kwargs):
        raise MaliciousDocumentError("Malicious document detected")

    monkeypatch.setattr(
        main.rag_service,
        "query",
        fake_query,
    )

    response = client.post(
        "/api/v1/query",
        json={
            "question": "What is the leave policy?",
            "user_id": "user-123",
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert "detail" in data
    assert data["detail"]["code"] == "MALICIOUS_DOCUMENT"
    assert data["detail"]["message"] == "The document failed security validation."
    assert data["detail"]["request_id"]

    # UUID4 hex request IDs are 32 characters.
    assert len(data["detail"]["request_id"]) == 32


def test_malicious_document_error_does_not_expose_internal_exception(
    client,
    monkeypatch,
    allow_rate_limit,
    allow_input_guardrail,
):
    monkeypatch.setattr(
        main.rag_service,
        "initialized",
        True,
    )

    async def fake_query(**kwargs):
        raise MaliciousDocumentError("SECRET_INTERNAL_SECURITY_DETAILS")

    monkeypatch.setattr(
        main.rag_service,
        "query",
        fake_query,
    )

    response = client.post(
        "/api/v1/query",
        json={
            "question": "Tell me about the documents.",
            "user_id": "user-123",
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert data["detail"]["code"] == "MALICIOUS_DOCUMENT"

    # Internal exception text must never be returned to the client.
    assert "SECRET_INTERNAL_SECURITY_DETAILS" not in response.text
    assert data["detail"]["message"] == ("The document failed security validation.")


def test_malicious_document_error_contains_request_id(
    client,
    monkeypatch,
    allow_rate_limit,
    allow_input_guardrail,
):
    monkeypatch.setattr(
        main.rag_service,
        "initialized",
        True,
    )

    async def fake_query(**kwargs):
        raise MaliciousDocumentError()

    monkeypatch.setattr(
        main.rag_service,
        "query",
        fake_query,
    )

    response = client.post(
        "/api/v1/query",
        json={
            "question": "What does the employee handbook say?",
            "user_id": "user-123",
        },
    )

    assert response.status_code == 400

    detail = response.json()["detail"]

    assert detail["request_id"]
    assert isinstance(detail["request_id"], str)
    assert len(detail["request_id"]) == 32
