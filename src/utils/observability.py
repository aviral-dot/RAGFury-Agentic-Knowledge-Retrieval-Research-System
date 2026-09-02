"""Request-scoped observability context."""

from contextvars import ContextVar
from typing import Optional

request_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "request_id",
    default=None,
)

user_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "user_id",
    default=None,
)

conversation_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "conversation_id",
    default=None,
)

node_ctx: ContextVar[Optional[str]] = ContextVar(
    "node",
    default=None,
)


def set_request_context(
    *,
    request_id: str,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> None:
    """Set request-scoped correlation information."""

    request_id_ctx.set(request_id)
    user_id_ctx.set(user_id)
    conversation_id_ctx.set(conversation_id)


def set_node_context(node: Optional[str]) -> None:
    """Set the currently executing graph node."""

    node_ctx.set(node)


def get_request_context() -> dict[str, Optional[str]]:
    """Return the current request-scoped context."""

    return {
        "request_id": request_id_ctx.get(),
        "user_id": user_id_ctx.get(),
        "conversation_id": conversation_id_ctx.get(),
        "node": node_ctx.get(),
    }


def clear_observability_context() -> None:
    """Clear request-scoped context."""

    request_id_ctx.set(None)
    user_id_ctx.set(None)
    conversation_id_ctx.set(None)
    node_ctx.set(None)
