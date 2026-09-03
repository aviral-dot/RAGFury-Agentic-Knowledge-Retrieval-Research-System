"""Utilities for preventing secrets from entering traces."""

from __future__ import annotations

from typing import Any

SENSITIVE_KEYS = {
    "password",
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "secret",
    "client_secret",
}


def sanitize_metadata(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Remove sensitive values from trace metadata."""

    result: dict[str, Any] = {}

    for key, value in metadata.items():
        if key.lower() in SENSITIVE_KEYS:
            result[key] = "[REDACTED]"
        else:
            result[key] = value

    return result
