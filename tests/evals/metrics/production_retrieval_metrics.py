"""Deterministic production retrieval metrics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _source(document: Any) -> str | None:
    """Return the canonical source filename."""

    metadata = getattr(document, "metadata", None) or {}
    value = metadata.get("source")

    return str(value) if value is not None else None


def _page(document: Any) -> int | None:
    """Return the human-visible 1-based PDF page number."""

    metadata = getattr(document, "metadata", None) or {}

    value = metadata.get("page")

    if value is None:
        value = metadata.get("page_label")

    if value is None:
        return None

    try:
        # PyPDFLoader's page metadata is 0-based.
        return int(value) + 1

    except (TypeError, ValueError):
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None


def _is_relevant(
    document: Any,
    expected_source: str,
    expected_pages: set[int] | frozenset[int],
) -> bool:
    """Check whether a retrieved document matches the golden target."""

    return _source(document) == expected_source and _page(document) in expected_pages


def recall_at_k(
    documents: Iterable[Any],
    expected_source: str,
    expected_pages: set[int] | frozenset[int],
    k: int,
) -> float:
    """Calculate page-level Recall@K."""

    if not expected_pages:
        raise ValueError("expected_pages must not be empty")

    if k <= 0:
        raise ValueError("k must be greater than zero")

    top_k = list(documents)[:k]

    retrieved_pages = {
        _page(document)
        for document in top_k
        if _is_relevant(
            document,
            expected_source,
            expected_pages,
        )
    }

    retrieved_pages.discard(None)

    return len(retrieved_pages & set(expected_pages)) / len(expected_pages)


def precision_at_k(
    documents: Iterable[Any],
    expected_source: str,
    expected_pages: set[int] | frozenset[int],
    k: int,
) -> float:
    """Calculate page-level Precision@K."""

    if not expected_pages:
        raise ValueError("expected_pages must not be empty")

    if k <= 0:
        raise ValueError("k must be greater than zero")

    top_k = list(documents)[:k]

    if not top_k:
        return 0.0

    relevant = sum(
        1
        for document in top_k
        if _is_relevant(
            document,
            expected_source,
            expected_pages,
        )
    )

    return relevant / len(top_k)


def hit_rate_at_k(
    documents: Iterable[Any],
    expected_source: str,
    expected_pages: set[int] | frozenset[int],
    k: int,
) -> float:
    """Calculate binary page-level Hit Rate@K."""

    return (
        1.0
        if recall_at_k(
            documents,
            expected_source,
            expected_pages,
            k,
        )
        > 0.0
        else 0.0
    )


def aggregate_metrics(
    results: list[dict[str, float]],
) -> dict[str, float]:
    """Average per-query retrieval metrics."""

    if not results:
        raise ValueError("results must not be empty")

    names = (
        "recall_at_k",
        "precision_at_k",
        "hit_rate_at_k",
    )

    return {
        name: sum(result[name] for result in results) / len(results) for name in names
    }
