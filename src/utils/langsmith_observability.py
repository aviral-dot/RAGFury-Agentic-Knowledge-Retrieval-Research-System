"""LangSmith observability helpers for RAGFury."""

from __future__ import annotations

import os
from typing import Any


def get_environment() -> str:
    return os.getenv(
        "APP_ENV",
        "development",
    )


def get_app_version() -> str:
    return os.getenv(
        "APP_VERSION",
        "unknown",
    )


def build_trace_metadata(
    *,
    request_id: str | None = None,
    user_id: str | None = None,
    conversation_id: str | None = None,
    thread_id: str | None = None,
    workflow: str | None = None,
) -> dict[str, Any]:
    """Build common metadata for LangSmith traces."""

    metadata: dict[str, Any] = {
        "environment": get_environment(),
        "app_version": get_app_version(),
        "service": "ragfury",
    }

    if request_id:
        metadata["request_id"] = request_id

    if user_id:
        metadata["user_id"] = user_id

    if conversation_id:
        metadata["conversation_id"] = conversation_id

    if thread_id:
        metadata["thread_id"] = thread_id

    if workflow:
        metadata["workflow"] = workflow

    return metadata


def build_trace_tags(
    *,
    workflow: str,
) -> list[str]:
    """Build standard LangSmith tags."""

    return [
        "ragfury",
        f"workflow:{workflow}",
        f"environment:{get_environment()}",
    ]
