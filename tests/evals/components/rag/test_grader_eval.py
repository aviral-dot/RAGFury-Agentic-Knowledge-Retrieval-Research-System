"""Component-level evaluation for the RAG document grader."""

import os

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden as DeepEvalGolden
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span
from langchain_openai import ChatOpenAI

from src.node.grading_nodes import GradingNodes
from tests.evals.datasets.grading_goldens import (
    rag_grading_goldens,
)
from tests.evals.helpers.grading_eval_helpers import (
    build_grader_state,
)
from tests.evals.metrics.grading_metrics import (
    get_grading_metrics,
)


def build_grader_llm() -> ChatOpenAI:
    """
    Build the LangChain chat model required by the real
    GradingNodes implementation.

    This does NOT use GroqEvalModel because GroqEvalModel
    is a DeepEval wrapper and does not expose
    with_structured_output().
    """

    return ChatOpenAI(
        model=os.getenv(
            "DEEPEVAL_MODEL",
            "qwen/qwen3.8-27b",
        ),
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        max_tokens=1024,
        temperature=0,
        timeout=60,
        max_retries=0,
    )


class GraderComponent:
    """
    DeepEval wrapper around the real RAG document grader.
    """

    def __init__(
        self,
        grading_nodes: GradingNodes,
    ):
        self.grading_nodes = grading_nodes

    @observe(
        metrics=get_grading_metrics(),
    )
    async def grade(
        self,
        question: str,
        retrieved_context: list[str],
        expected_relevance: bool,
        expected_reason: str,
    ) -> dict:
        """
        Execute the real asynchronous document grader
        and register its evaluation test case.
        """

        state = build_grader_state(
            question=question,
            retrieved_context=retrieved_context,
        )

        result = await self.grading_nodes.grade_documents(
            state,
        )

        actual_relevance = result["document_relevance"]
        actual_reason = result["grade_reason"]

        update_current_span(
            test_case=LLMTestCase(
                input=(
                    f"Question:\n"
                    f"{question}\n\n"
                    f"Retrieved context:\n"
                    f"{chr(10).join(retrieved_context)}"
                ),
                actual_output=(
                    f"relevant={str(actual_relevance).lower()}\nreason={actual_reason}"
                ),
                expected_output=(
                    f"relevant="
                    f"{str(expected_relevance).lower()}\n"
                    f"reason="
                    f"{expected_reason}"
                ),
                retrieval_context=retrieved_context,
            )
        )

        return result


@pytest.fixture(scope="module")
def grader_component() -> GraderComponent:
    """
    Create the real production grader once for the test module.
    """

    llm = build_grader_llm()

    grading_nodes = GradingNodes(
        llm=llm,
    )

    return GraderComponent(
        grading_nodes=grading_nodes,
    )


@pytest.mark.parametrize(
    "golden",
    rag_grading_goldens,
)
@pytest.mark.asyncio
async def test_grader_component(
    golden,
    grader_component: GraderComponent,
):
    """
    Evaluate the real document grader against fixed contexts.

    Retrieval is intentionally not executed here.
    """

    await grader_component.grade(
        question=golden.input,
        retrieved_context=golden.retrieved_context,
        expected_relevance=golden.expected_relevance,
        expected_reason=golden.expected_reason,
    )

    assert_test(
        golden=DeepEvalGolden(
            input=golden.input,
            expected_output=(f"relevant={str(golden.expected_relevance).lower()}"),
        ),
    )
