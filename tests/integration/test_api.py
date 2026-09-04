import pytest
from fastapi.testclient import TestClient

from api.main import app, rag_service

# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================
# ROOT
# ============================================================


def test_root_returns_api_information(client):
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["name"] == "RAGFury API"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"


# ============================================================
# HEALTH
# ============================================================


def test_health_returns_healthy_when_rag_is_initialized(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        rag_service,
        "initialized",
        True,
    )

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
        "rag_initialized": True,
    }


def test_health_returns_unhealthy_when_rag_is_not_initialized(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        rag_service,
        "initialized",
        False,
    )

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "unhealthy",
        "rag_initialized": False,
    }


# ============================================================
# SYSTEM INFO
# ============================================================


def test_system_info_returns_current_service_state(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        rag_service,
        "initialized",
        True,
    )

    monkeypatch.setattr(
        rag_service,
        "num_chunks",
        42,
    )

    response = client.get("/api/v1/info")

    assert response.status_code == 200

    data = response.json()

    assert data["name"] == "RAGFury"
    assert data["version"] == "1.0.0"
    assert data["rag_initialized"] is True
    assert data["document_chunks"] == 42


# ============================================================
# QUERY VALIDATION
# ============================================================


def test_query_rejects_empty_question(client):
    response = client.post(
        "/api/v1/query",
        json={
            "question": "",
            "user_id": "user-123",
        },
    )

    assert response.status_code == 422


def test_query_rejects_question_over_2000_characters(client):
    response = client.post(
        "/api/v1/query",
        json={
            "question": "a" * 2001,
            "user_id": "user-123",
        },
    )

    assert response.status_code == 422


def test_query_rejects_empty_user_id(client):
    response = client.post(
        "/api/v1/query",
        json={
            "question": "What is the leave policy?",
            "user_id": "",
        },
    )

    assert response.status_code == 422


def test_query_rejects_user_id_over_100_characters(client):
    response = client.post(
        "/api/v1/query",
        json={
            "question": "What is the leave policy?",
            "user_id": "u" * 101,
        },
    )

    assert response.status_code == 422


# ============================================================
# QUERY — RATE LIMIT
# ============================================================


def test_query_returns_429_when_rate_limit_exceeded(
    client,
    monkeypatch,
):
    async def fake_check(**kwargs):
        from src.rate_limit.rate_limiter import RateLimitResult

        return RateLimitResult(
            allowed=False,
            limit=30,
            remaining=0,
            retry_after=60,
            reset_after=60,
        )

    monkeypatch.setattr(
        "api.main.global_rate_limiter.check",
        fake_check,
    )

    response = client.post(
        "/api/v1/query",
        json={
            "question": "What is the leave policy?",
            "user_id": "user-123",
        },
    )

    assert response.status_code == 429

    data = response.json()

    assert data["detail"]["code"] == "GLOBAL_RATE_LIMIT_EXCEEDED"
    assert data["detail"]["request_id"]


def test_query_returns_503_when_rate_limiter_fails(
    client,
    monkeypatch,
):
    async def fake_check(**kwargs):
        raise RuntimeError("Redis unavailable")

    monkeypatch.setattr(
        "api.main.global_rate_limiter.check",
        fake_check,
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

    assert data["detail"]["code"] == "RATE_LIMIT_SERVICE_UNAVAILABLE"
    assert data["detail"]["request_id"]


# ============================================================
# QUERY — INPUT GUARDRAIL FAILURE
# ============================================================


def test_query_returns_503_when_input_guardrail_fails(
    client,
    monkeypatch,
):
    from src.rate_limit.rate_limiter import RateLimitResult

    async def fake_check(**kwargs):
        return RateLimitResult(
            allowed=True,
            limit=30,
            remaining=29,
            retry_after=60,
            reset_after=60,
        )

    async def fake_check_input(question):
        raise RuntimeError("Guardrail unavailable")

    monkeypatch.setattr(
        "api.main.global_rate_limiter.check",
        fake_check,
    )

    monkeypatch.setattr(
        "api.main.check_input",
        fake_check_input,
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

    assert data["detail"] == "Input security validation failed."


# ============================================================
# QUERY — INPUT GUARDRAIL BLOCK
# ============================================================


def test_query_returns_400_when_input_guardrail_blocks(
    client,
    monkeypatch,
):
    from src.rate_limit.rate_limiter import RateLimitResult

    async def fake_check(**kwargs):
        return RateLimitResult(
            allowed=True,
            limit=30,
            remaining=29,
            retry_after=60,
            reset_after=60,
        )

    async def fake_check_input(question):
        return False

    monkeypatch.setattr(
        "api.main.global_rate_limiter.check",
        fake_check,
    )

    monkeypatch.setattr(
        "api.main.check_input",
        fake_check_input,
    )

    response = client.post(
        "/api/v1/query",
        json={
            "question": "Malicious request",
            "user_id": "user-123",
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert data["detail"] == ("Your request was blocked by the RAGFury safety policy.")


# ============================================================
# HTTP METHOD VALIDATION
# ============================================================


def test_health_rejects_post(client):
    response = client.post("/health")

    assert response.status_code == 405


def test_root_rejects_post(client):
    response = client.post("/")

    assert response.status_code == 405
