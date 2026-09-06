"""Component-level evaluation for the RAG retriever."""

import os

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.tracing import observe

from tests.evals.config import THRESHOLDS
from tests.evals.datasets.retrieval_goldens import (
    rag_retrieval_goldens,
)
from tests.evals.datasets.retrieval_ground_truth import (
    expected_sources_for_query,
)
from tests.evals.helpers.rag_eval_helpers import (
    build_rag_nodes,
    documents_to_context,
    update_retrieval_span,
)
from tests.evals.helpers.regression_gate import (
    assert_thresholds,
    print_regression_results,
)
from tests.evals.metrics.retrieval_metrics import (
    aggregate_retrieval_metrics,
    get_retrieval_metrics,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
)

RETRIEVAL_K = int(
    os.getenv(
        "EVAL_RETRIEVAL_K",
        "2",
    )
)


class RetrieverComponent:
    """DeepEval wrapper around the real RAG retrieval node."""

    def __init__(self, rag_nodes):
        self.rag_nodes = rag_nodes

    @observe(
        metrics=get_retrieval_metrics(),
    )
    async def retrieve(
        self,
        query: str,
    ) -> list[str]:
        """Execute the real RAG retrieval node."""

        state = {
            "question": query,
        }

        result = await self.rag_nodes.retrieve_docs(
            state,
        )

        retrieved_context = documents_to_context(
            result["retrieved_docs"],
        )

        update_retrieval_span(
            query=query,
            retrieved_context=retrieved_context,
        )

        return retrieved_context


@pytest.fixture(scope="module")
def retriever_component():
    """Create the real RAG retriever once."""

    rag_nodes = build_rag_nodes()

    return RetrieverComponent(
        rag_nodes=rag_nodes,
    )


@pytest.fixture(scope="module")
def rag_nodes():
    """Create real RAG nodes for deterministic retrieval evaluation."""

    return build_rag_nodes()


@pytest.mark.parametrize(
    "golden",
    rag_retrieval_goldens,
)
@pytest.mark.asyncio
async def test_retriever_component(
    golden: Golden,
    retriever_component: RetrieverComponent,
):
    """
    Run each retrieval golden through the real retriever
    and LLM-based DeepEval judge.

    The Contextual Relevancy regression threshold is defined centrally
    in tests/evals/config.py.
    """

    await retriever_component.retrieve(
        golden.input,
    )

    assert_test(
        golden=golden,
    )


@pytest.mark.asyncio
async def test_deterministic_retrieval_metrics(
    rag_nodes,
):
    """
    Evaluate deterministic retrieval quality.

    This test uses retrieval_ground_truth.py and does NOT use an LLM judge.

    Metrics:
        - Recall@K
        - Precision@K
        - Hit Rate@K
    """

    per_query_results: list[dict[str, float]] = []

    executed_queries = 0

    for golden in rag_retrieval_goldens:
        result = await rag_nodes.retrieve_docs(
            {
                "question": golden.input,
            }
        )

        documents = result["retrieved_docs"]

        expected_sources = expected_sources_for_query(
            golden.input,
        )

        per_query_results.append(
            {
                "recall_at_k": recall_at_k(
                    documents,
                    expected_sources,
                    RETRIEVAL_K,
                ),
                "precision_at_k": precision_at_k(
                    documents,
                    expected_sources,
                    RETRIEVAL_K,
                ),
                "hit_rate_at_k": hit_rate_at_k(
                    documents,
                    expected_sources,
                    RETRIEVAL_K,
                ),
            }
        )

        executed_queries += 1

    if executed_queries == 0:
        raise AssertionError(
            "Deterministic retrieval evaluation executed zero queries."
        )

    aggregate = aggregate_retrieval_metrics(
        per_query_results,
    )

    print(
        "\nDeterministic retrieval metrics "
        f"(K={RETRIEVAL_K}, "
        f"Queries={executed_queries}): "
        f"Recall@K={aggregate['recall_at_k']:.4f}, "
        f"Precision@K={aggregate['precision_at_k']:.4f}, "
        f"HitRate@K={aggregate['hit_rate_at_k']:.4f}"
    )

    regression_results = assert_thresholds(
        metrics=aggregate,
        thresholds={
            "recall_at_k": (THRESHOLDS.retrieval_recall_at_k),
            "precision_at_k": (THRESHOLDS.retrieval_precision_at_k),
            "hit_rate_at_k": (THRESHOLDS.retrieval_hit_rate_at_k),
        },
    )

    print_regression_results(
        regression_results,
    )
