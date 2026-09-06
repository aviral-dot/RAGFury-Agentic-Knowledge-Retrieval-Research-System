"""Production-grade deterministic retrieval evaluation."""

from __future__ import annotations

import os

import pytest

from tests.evals.datasets.retrieval_ground_truth import (
    production_retrieval_goldens,
)
from tests.evals.helpers.rag_eval_helpers import build_rag_nodes
from tests.evals.metrics.production_retrieval_metrics import (
    aggregate_metrics,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
)

RETRIEVAL_K = int(os.getenv("EVAL_RETRIEVAL_K", "2"))


@pytest.fixture(scope="module")
def rag_nodes():
    """Build the real production retriever once for the test module."""

    return build_rag_nodes()


@pytest.mark.asyncio
async def test_production_retrieval_metrics(rag_nodes):
    """Evaluate Recall@K, Precision@K and Hit Rate@K deterministically."""

    per_query_results: list[dict[str, float]] = []

    for golden in production_retrieval_goldens:
        result = await rag_nodes.retrieve_docs({"question": golden.input})

        documents = result["retrieved_docs"]

        recall = recall_at_k(
            documents=documents,
            expected_source=golden.expected_source,
            expected_pages=golden.expected_pages,
            k=RETRIEVAL_K,
        )

        precision = precision_at_k(
            documents=documents,
            expected_source=golden.expected_source,
            expected_pages=golden.expected_pages,
            k=RETRIEVAL_K,
        )

        hit_rate = hit_rate_at_k(
            documents=documents,
            expected_source=golden.expected_source,
            expected_pages=golden.expected_pages,
            k=RETRIEVAL_K,
        )

        per_query_results.append(
            {
                "recall_at_k": recall,
                "precision_at_k": precision,
                "hit_rate_at_k": hit_rate,
            }
        )

        retrieved_locations = sorted(
            {
                (
                    document.metadata.get("source"),
                    int(document.metadata["page"]) + 1,
                )
                for document in documents
                if document.metadata.get("source") is not None
                and document.metadata.get("page") is not None
            }
        )

        print(
            "\n"
            f"Query: {golden.input}\n"
            f"Expected: {golden.expected_source}, "
            f"pages={sorted(golden.expected_pages)}\n"
            f"Retrieved: {retrieved_locations}\n"
            f"Recall@{RETRIEVAL_K}: {recall:.4f}\n"
            f"Precision@{RETRIEVAL_K}: {precision:.4f}\n"
            f"Hit Rate@{RETRIEVAL_K}: {hit_rate:.4f}"
        )

    aggregate = aggregate_metrics(per_query_results)

    print("\n" + "=" * 64)
    print("PRODUCTION RETRIEVAL EVALUATION")
    print("=" * 64)
    print(f"Queries: {len(production_retrieval_goldens)}")
    print(f"K: {RETRIEVAL_K}")
    print(f"Recall@K: {aggregate['recall_at_k']:.4f}")
    print(f"Precision@K: {aggregate['precision_at_k']:.4f}")
    print(f"Hit Rate@K: {aggregate['hit_rate_at_k']:.4f}")
    print("=" * 64)

    assert 0.0 <= aggregate["recall_at_k"] <= 1.0
    assert 0.0 <= aggregate["precision_at_k"] <= 1.0
    assert 0.0 <= aggregate["hit_rate_at_k"] <= 1.0
