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
    LLM_MODEL = "llama-3.1-8b-instant"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]

    @classmethod
    def get_llm(cls):
        """Initialize and return the Groq LLM"""

        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables."
            )

        return ChatGroq(
            model=cls.LLM_MODEL,
            groq_api_key=cls.GROQ_API_KEY,
            temperature=0
        )