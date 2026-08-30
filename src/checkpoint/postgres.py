"""PostgreSQL-backed LangGraph checkpointing for RAGFury."""

import os

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


def get_checkpoint_database_url() -> str:
    """
    Get the dedicated RAGFury checkpoint database URL.

    This database must be separate from the AgentFlow
    checkpoint database.
    """

    database_url = os.getenv("RAGFURY_CHECKPOINT_DATABASE_URL")

    if not database_url:
        raise RuntimeError("RAGFURY_CHECKPOINT_DATABASE_URL is not configured.")

    return database_url


def create_checkpointer(database_url: str):
    """
    Create the asynchronous PostgreSQL checkpointer.

    The returned object is an async context manager and
    should be managed by the FastAPI application lifespan.
    """

    return AsyncPostgresSaver.from_conn_string(database_url)
