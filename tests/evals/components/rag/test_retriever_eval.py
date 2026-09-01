"""Component-level evaluation for the RAG retriever."""

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.tracing import observe

from tests.evals.datasets.retrieval_goldens import (
    rag_retrieval_goldens,
)
from tests.evals.helpers.rag_eval_helpers import (
    build_rag_nodes,
    documents_to_context,
    update_retrieval_span,
)
from tests.evals.metrics.retrieval_metrics import (
    get_retrieval_metrics,
)


class RetrieverComponent:
    """
    DeepEval wrapper around the real RAG
    retrieval node.
    """

    def __init__(
        self,
        rag_nodes,
    ):
        self.rag_nodes = rag_nodes

    @observe(metrics=get_retrieval_metrics())
    async def retrieve(
        self,
        query: str,
    ) -> list[str]:
        """
        Execute the real RAG retrieval node
        and register its component test case.
        """

        state = {
            "question": query,
        }

        result = await self.rag_nodes.retrieve_docs(state)

        retrieved_context = documents_to_context(result["retrieved_docs"])

        update_retrieval_span(
            query=query,
            retrieved_context=retrieved_context,
        )

        return retrieved_context


@pytest.fixture(scope="module")
def retriever_component():
    """Create the real RAG retriever once."""

    rag_nodes = build_rag_nodes()

    return RetrieverComponent(rag_nodes=rag_nodes)


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
    Run one golden through the real retriever.

    Component metrics are attached to the
    retriever span, so assert_test only needs
    the active golden.
    """

    await retriever_component.retrieve(golden.input)

    assert_test(
        golden=golden,
    )
