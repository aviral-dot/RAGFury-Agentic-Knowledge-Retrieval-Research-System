import pytest
from fastapi.testclient import TestClient

import api.main as main


@pytest.fixture
def client():
    return TestClient(main.app)


class FakeLangSmithClient:
    def __init__(self):
        self.calls = []

    def create_feedback(self, **kwargs):
        self.calls.append(kwargs)


class FailingLangSmithClient:
    def create_feedback(self, **kwargs):
        raise RuntimeError("LangSmith unavailable")


def test_feedback_accepts_valid_score(client, monkeypatch):
    fake_client = FakeLangSmithClient()

    monkeypatch.setattr(
        main,
        "langsmith_client",
        fake_client,
    )

    response = client.post(
        "/api/v1/feedback",
        json={
            "run_id": "run-123",
            "score": 1.0,
            "comment": "Excellent answer",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "success"
    assert body["run_id"] == "run-123"

    assert len(fake_client.calls) == 1

    call = fake_client.calls[0]

    assert call["run_id"] == "run-123"
    assert call["key"] == "user-feedback"
    assert call["score"] == 1.0
    assert call["comment"] == "Excellent answer"


def test_feedback_accepts_score_zero(client, monkeypatch):
    fake_client = FakeLangSmithClient()

    monkeypatch.setattr(
        main,
        "langsmith_client",
        fake_client,
    )

    response = client.post(
        "/api/v1/feedback",
        json={
            "run_id": "run-123",
            "score": 0.0,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "success"
    assert body["run_id"] == "run-123"

    call = fake_client.calls[0]

    assert call["score"] == 0.0
    assert call["comment"] is None


def test_feedback_accepts_without_comment(client, monkeypatch):
    fake_client = FakeLangSmithClient()

    monkeypatch.setattr(
        main,
        "langsmith_client",
        fake_client,
    )

    response = client.post(
        "/api/v1/feedback",
        json={
            "run_id": "run-456",
            "score": 0.8,
        },
    )

    assert response.status_code == 200

    call = fake_client.calls[0]

    assert call["run_id"] == "run-456"
    assert call["score"] == 0.8
    assert call["comment"] is None


@pytest.mark.parametrize(
    "score",
    [
        -0.1,
        1.1,
        2.0,
        -1.0,
    ],
)
def test_feedback_rejects_invalid_score(client, score):
    response = client.post(
        "/api/v1/feedback",
        json={
            "run_id": "run-123",
            "score": score,
        },
    )

    assert response.status_code == 422


def test_feedback_requires_run_id(client):
    response = client.post(
        "/api/v1/feedback",
        json={
            "score": 0.8,
        },
    )

    assert response.status_code == 422


def test_feedback_requires_score(client):
    response = client.post(
        "/api/v1/feedback",
        json={
            "run_id": "run-123",
        },
    )

    assert response.status_code == 422


def test_feedback_handles_langsmith_failure(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "langsmith_client",
        FailingLangSmithClient(),
    )

    response = client.post(
        "/api/v1/feedback",
        json={
            "run_id": "run-123",
            "score": 0.5,
            "comment": "Test",
        },
    )

    assert response.status_code == 500

    body = response.json()

    assert body["detail"] == "Failed to submit feedback."
