import pytest
from fastapi.testclient import TestClient

import api.main as main
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


@pytest.fixture
def initialized_rag(monkeypatch):
    monkeypatch.setattr(
        main.rag_service,
        "initialized",
        True,
    )

    async def fake_query(**kwargs):
        return {
            "question": kwargs["question"],
            "answer": "The employee is entitled to annual leave.",
            "citations": [],
            "documents": [],
            "document_relevance": True,
            "grade_reason": "Relevant document found.",
            "reflection": None,
            "reflection_passed": True,
            "retrieval_attempts": 1,
            "reflection_attempts": 0,
            "next_step": None,
            "run_id": "run-test-123",
        }

    monkeypatch.setattr(
        main.rag_service,
        "query",
        fake_query,
    )


def test_query_returns_403_when_output_guardrail_blocks_response(
    client,
    monkeypatch,
    allow_rate_limit,
    allow_input_guardrail,
    initialized_rag,
):
    async def fake_check_output(answer):
        assert answer == "The employee is entitled to annual leave."
        return False

    monkeypatch.setattr(
        main,
        "check_output",
        fake_check_output,
    )

    response = client.post(
        "/api/v1/query",
        json={
            "question": "What is the leave policy?",
            "user_id": "user-123",
        },
    )

    assert response.status_code == 403

    data = response.json()

    assert data["detail"] == (
        "The generated response was blocked by the RAGFury safety policy."
    )


def test_query_returns_503_when_output_guardrail_fails(
    client,
    monkeypatch,
    allow_rate_limit,
    allow_input_guardrail,
    initialized_rag,
):
    async def fake_check_output(answer):
        raise RuntimeError("Output guardrail unavailable")

    monkeypatch.setattr(
        main,
        "check_output",
        fake_check_output,
    )

    response = client.post(
        "/api/v1/query",
        json={
            "question": "What is the leave policy?",
            "user_id": "user-123",
        },
    )

    assert response.status_code == 503

    data = response.json()

    assert data["detail"] == "Output security validation failed."


def test_output_guardrail_receives_generated_answer(
    client,
    monkeypatch,
    allow_rate_limit,
    allow_input_guardrail,
    initialized_rag,
):
    received = {}

    async def fake_check_output(answer):
        received["answer"] = answer
        return True

    monkeypatch.setattr(
        main,
        "check_output",
        fake_check_output,
    )

    response = client.post(
        "/api/v1/query",
        json={
            "question": "What is the leave policy?",
            "user_id": "user-123",
        },
    )

    assert response.status_code == 200

    assert received["answer"] == ("The employee is entitled to annual leave.")
