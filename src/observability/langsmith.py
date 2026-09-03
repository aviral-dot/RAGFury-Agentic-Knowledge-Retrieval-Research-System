from typing import Any

SERVICE_NAME = "ragfury"
WORKFLOW_NAME = "agentic-rag"


def build_trace_metadata(
    *,
    request_id: str | None,
    user_id: str | None,
    conversation_id: str | None,
    environment: str,
    app_version: str,
) -> dict[str, Any]:

    return {
        "service": SERVICE_NAME,
        "workflow": WORKFLOW_NAME,
        "request_id": request_id,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "environment": environment,
        "app_version": app_version,
    }


def build_trace_tags(
    environment: str,
) -> list[str]:

    return [
        "ragfury",
        "agentic-rag",
        environment,
    ]
