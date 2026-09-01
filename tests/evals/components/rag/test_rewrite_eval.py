"""Component-level evaluation for the RAG query-rewrite node."""

import os

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase
from deepeval.tracing import (
    observe,
    update_current_span,
)
from langchain_openai import ChatOpenAI

from src.node.rewrite_nodes import RewriteNodes
from tests.evals.datasets.rewrite_goldens import (
    rag_rewrite_goldens,
)
from tests.evals.helpers.rewrite_eval_helpers import (
    build_rewrite_state,
)
from tests.evals.metrics.rewrite_metrics import (
    get_rewrite_metrics,
)


def build_rewrite_llm() -> ChatOpenAI:
    """
    Build the LangChain chat model required by the real
    RewriteNodes implementation.

    This must be a LangChain model because RewriteNodes
    calls llm.ainvoke().
    """

    return ChatOpenAI(
        model=os.getenv(
            "DEEPEVAL_MODEL",
            "qwen/qwen3.8-27b",
        ),
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        max_tokens=256,
        temperature=0,
        timeout=60,
        max_retries=0,
    )


class RewriteComponent:
    """
    DeepEval wrapper around the real RAG query-rewrite node.
    """

    def __init__(
        self,
        rewrite_nodes: RewriteNodes,
    ):
        self.rewrite_nodes = rewrite_nodes

    @observe(
        metrics=get_rewrite_metrics(),
    )
    async def rewrite(
        self,
        question: str,
        retrieved_context: list[str],
        grade_reason: str,
        expected_output: str,
    ) -> dict:
        """
        Execute the real asynchronous query-rewrite node.
        """

        state = build_rewrite_state(
            question=question,
            retrieved_context=retrieved_context,
            grade_reason=grade_reason,
        )

        result = await self.rewrite_nodes.rewrite_query(
            state,
        )

        rewritten_query = result["question"]

        update_current_span(
            test_case=LLMTestCase(
                input=(
                    f"Original question:\n"
                    f"{question}\n\n"
                    f"Retrieved context:\n"
                    f"{chr(10).join(retrieved_context)}\n\n"
                    f"Grader reason:\n"
                    f"{grade_reason}"
                ),
                actual_output=rewritten_query,
                expected_output=expected_output,
                retrieval_context=retrieved_context,
            )
        )

        return result


@pytest.fixture(scope="module")
def rewrite_component() -> RewriteComponent:
    """
    Create the real production rewrite node once
    for the test module.
    """

    llm = build_rewrite_llm()

    rewrite_nodes = RewriteNodes(
        llm=llm,
    )

    return RewriteComponent(
        rewrite_nodes=rewrite_nodes,
    )


@pytest.mark.parametrize(
    "golden",
    rag_rewrite_goldens,
)
@pytest.mark.asyncio
async def test_rewrite_component(
    golden,
    rewrite_component: RewriteComponent,
):
    """
    Evaluate the real RAG query-rewrite node against fixed
    questions, retrieved contexts, and grader feedback.

    Retrieval itself is intentionally not executed here.
    """

    result = await rewrite_component.rewrite(
        question=golden.input,
        retrieved_context=golden.retrieved_context,
        grade_reason=golden.grade_reason,
        expected_output=golden.expected_output,
    )

    assert result["question"]

    assert_test(
        golden=Golden(
            input=(
                f"Original question:\n"
                f"{golden.input}\n\n"
                f"Retrieved context:\n"
                f"{chr(10).join(golden.retrieved_context)}\n\n"
                f"Grader reason:\n"
                f"{golden.grade_reason}"
            ),
            expected_output=golden.expected_output,
        ),
    )
