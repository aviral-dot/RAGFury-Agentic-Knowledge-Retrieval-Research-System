"""Canonical identifiers for RAGFury document chunks."""

from __future__ import annotations

import hashlib


def normalize_source(source: str) -> str:
    """Normalize a document source into a stable representation."""

    return source.strip().replace("\\", "/")


def build_chunk_id(
    *,
    source: str,
    chunk_index: int,
) -> str:
    """Build a deterministic chunk identifier."""

    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")

    normalized_source = normalize_source(source)

    if not normalized_source:
        raise ValueError("source must not be empty")

    source_hash = hashlib.sha1(
        normalized_source.encode("utf-8")
    ).hexdigest()[:12]

    return f"{source_hash}:chunk_{chunk_index:04d}"


def build_document_id(
    *,
    source: str,
    chunk_id: str,
) -> str:
    """Build the canonical identifier used by Qdrant."""

    normalized_source = normalize_source(source)

    if not normalized_source:
        raise ValueError("source must not be empty")

    if not chunk_id:
        raise ValueError("chunk_id must not be empty")

    return f"{normalized_source}:{chunk_id}"
