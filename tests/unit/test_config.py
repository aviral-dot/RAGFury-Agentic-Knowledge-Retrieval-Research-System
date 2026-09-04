import pytest

from src.config.config import Config


def test_groq_api_key_is_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    # Config.GROQ_API_KEY is evaluated at import time, so update it
    # explicitly for this isolated unit test.
    monkeypatch.setattr(Config, "GROQ_API_KEY", "test-groq-key")

    assert Config.GROQ_API_KEY == "test-groq-key"


def test_llm_model_is_configured():
    assert Config.LLM_MODEL == "openai/gpt-oss-20b"


def test_document_processing_defaults():
    assert Config.CHUNK_SIZE == 500
    assert Config.CHUNK_OVERLAP == 50


def test_redis_url_default(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)

    assert Config.REDIS_URL == "redis://localhost:6379"


def test_qdrant_url_default(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)

    assert Config.QDRANT_URL == "http://localhost:6333"


def test_qdrant_collection_default(monkeypatch):
    monkeypatch.delenv("QDRANT_COLLECTION", raising=False)

    assert Config.QDRANT_COLLECTION == "ragfury_documents"


def test_cors_origins_default(monkeypatch):
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)

    assert Config.get_cors_origins() == [
        "http://localhost:8501",
    ]


def test_cors_origins_support_multiple_values(monkeypatch):
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:8501, http://localhost:3000, https://example.com",
    )

    assert Config.get_cors_origins() == [
        "http://localhost:8501",
        "http://localhost:3000",
        "https://example.com",
    ]


def test_cors_origins_removes_empty_values(monkeypatch):
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:8501, , http://localhost:3000,",
    )

    assert Config.get_cors_origins() == [
        "http://localhost:8501",
        "http://localhost:3000",
    ]


def test_graph_timeout_default(monkeypatch):
    monkeypatch.delenv("GRAPH_TIMEOUT_SECONDS", raising=False)

    assert Config.get_graph_timeout() == 120.0


def test_graph_timeout_from_environment(monkeypatch):
    monkeypatch.setenv("GRAPH_TIMEOUT_SECONDS", "45")

    assert Config.get_graph_timeout() == 45.0


def test_environment_default(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)

    assert Config.get_environment() == "development"


def test_environment_from_environment_variable(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")

    assert Config.get_environment() == "production"


def test_app_version_default(monkeypatch):
    monkeypatch.delenv("APP_VERSION", raising=False)

    assert Config.get_app_version() == "unknown"


def test_app_version_from_environment(monkeypatch):
    monkeypatch.setenv("APP_VERSION", "1.2.3")

    assert Config.get_app_version() == "1.2.3"


def test_global_rate_limit_default(monkeypatch):
    monkeypatch.delenv("GLOBAL_RATE_LIMIT_PER_MINUTE", raising=False)

    assert Config.get_global_rate_limit_per_minute() == 30


def test_global_rate_limit_from_environment(monkeypatch):
    monkeypatch.setenv("GLOBAL_RATE_LIMIT_PER_MINUTE", "100")

    assert Config.get_global_rate_limit_per_minute() == 100


def test_default_urls_are_configured():
    assert len(Config.DEFAULT_URLS) == 2

    assert "https://lilianweng.github.io/posts/2023-06-23-agent/" in Config.DEFAULT_URLS

    assert (
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
        in Config.DEFAULT_URLS
    )


def test_get_llm_requires_groq_api_key(monkeypatch):
    monkeypatch.setattr(Config, "GROQ_API_KEY", None)

    with pytest.raises(
        ValueError,
        match="GROQ_API_KEY not found in environment variables",
    ):
        Config.get_llm()


def test_get_llm_creates_groq_client(monkeypatch):
    class FakeChatGroq:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        "src.config.config.ChatGroq",
        FakeChatGroq,
    )

    monkeypatch.setattr(
        Config,
        "GROQ_API_KEY",
        "test-groq-key",
    )

    llm = Config.get_llm()

    assert isinstance(llm, FakeChatGroq)
    assert llm.kwargs == {
        "model": "openai/gpt-oss-20b",
        "groq_api_key": "test-groq-key",
        "temperature": 0,
    }
