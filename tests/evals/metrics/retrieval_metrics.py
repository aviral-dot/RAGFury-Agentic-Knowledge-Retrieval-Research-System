"""Metrics for RAG retrieval evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from deepeval.metrics import ContextualRelevancyMetric

from tests.evals.config import THRESHOLDS
from tests.evals.helpers.eval_models import create_eval_model

eval_model = create_eval_model()


def _source(document: Any) -> str | None:
    """Extract the canonical source identifier from a LangChain Document."""

    metadata = getattr(document, "metadata", None) or {}

    value = metadata.get("source")

    return str(value) if value else None


def recall_at_k(
    retrieved_documents: Iterable[Any],
    expected_sources: set[str],
    k: int,
) -> float:
    """Calculate source-level Recall@K."""

    if not expected_sources:
        raise ValueError("expected_sources must not be empty")

    if k <= 0:
        raise ValueError("k must be greater than zero")

    retrieved_sources = {
        source
        for source in (_source(document) for document in list(retrieved_documents)[:k])
        if source is not None
    }

    return len(retrieved_sources & expected_sources) / len(expected_sources)


def precision_at_k(
    retrieved_documents: Iterable[Any],
    expected_sources: set[str],
    k: int,
) -> float:
    """Calculate source-level Precision@K."""

    if not expected_sources:
        raise ValueError("expected_sources must not be empty")

    if k <= 0:
        raise ValueError("k must be greater than zero")

    top_k = list(retrieved_documents)[:k]

    if not top_k:
        return 0.0

    relevant = sum(1 for document in top_k if _source(document) in expected_sources)

    return relevant / len(top_k)


def hit_rate_at_k(
    retrieved_documents: Iterable[Any],
    expected_sources: set[str],
    k: int,
) -> float:
    """Calculate source-level Hit Rate@K."""

    return (
        1.0
        if recall_at_k(
            retrieved_documents,
            expected_sources,
            k,
        )
        > 0
        else 0.0
    )


def aggregate_retrieval_metrics(
    results: list[dict[str, float]],
) -> dict[str, float]:
    """Average per-query deterministic retrieval metrics."""

    if not results:
        raise ValueError("results must not be empty")

    return {
        key: sum(result[key] for result in results) / len(results)
        for key in (
            "recall_at_k",
            "precision_at_k",
            "hit_rate_at_k",
        )
    }


def get_retrieval_metrics():
    """Return LLM-judged retrieval metrics."""

    return [
        ContextualRelevancyMetric(
            threshold=THRESHOLDS.retrieval_context_relevancy,
            model=eval_model,
            include_reason=True,
        ),
    ]
