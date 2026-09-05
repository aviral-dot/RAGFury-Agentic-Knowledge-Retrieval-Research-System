"""Configuration module for Agentic RAG system"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


class Config:
    """Configuration class for RAG system"""

    # API Key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Groq model
    LLM_MODEL = "openai/gpt-oss-20b"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    REDIS_URL = os.getenv(
        "REDIS_URL",
        "redis://localhost:6379",
    )

    QUERY_CACHE_TTL_SECONDS = int(
        os.getenv(
            "QUERY_CACHE_TTL_SECONDS",
            "300",
        )
    )

    QUERY_CACHE_KEY_PREFIX = os.getenv(
        "QUERY_CACHE_KEY_PREFIX",
        "ragfury:query_cache:v1",
    )

    QDRANT_URL = os.getenv(
        "QDRANT_URL",
        "http://localhost:6333",
    )

    QDRANT_COLLECTION = os.getenv(
        "QDRANT_COLLECTION",
        "ragfury_documents",
    )

    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    @classmethod
    def get_llm(cls):
        """Initialize and return the Groq LLM"""

        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        return ChatGroq(
            model=cls.LLM_MODEL, groq_api_key=cls.GROQ_API_KEY, temperature=0
        )

    @staticmethod
    def get_cors_origins() -> list[str]:
        value = os.getenv(
            "CORS_ALLOWED_ORIGINS",
            "http://localhost:8501",
        )

        return [origin.strip() for origin in value.split(",") if origin.strip()]

    @classmethod
    def get_graph_timeout(cls) -> float:

        return float(
            os.getenv(
                "GRAPH_TIMEOUT_SECONDS",
                "120",
            )
        )

    @classmethod
    def get_environment(cls) -> str:
        return os.getenv(
            "APP_ENV",
            "development",
        )

    @classmethod
    def get_app_version(cls) -> str:

        return os.getenv(
            "APP_VERSION",
            "unknown",
        )

    @classmethod
    def get_global_rate_limit_per_minute(cls) -> int:
        return int(
            os.getenv(
                "GLOBAL_RATE_LIMIT_PER_MINUTE",
                "30",
            )
        )
