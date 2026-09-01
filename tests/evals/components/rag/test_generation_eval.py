"""Component-level evaluation for the RAG answer-generation node."""

import os

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span
from langchain_openai import ChatOpenAI

from src.node.generation_nodes import GenerationNodes
from tests.evals.datasets.generation_goldens import (
    rag_generation_goldens,
)
from tests.evals.helpers.generation_eval_helpers import (
    build_generation_state,
)
from tests.evals.metrics.generation_metrics import (
    get_generation_metrics,
)


def build_generation_llm() -> ChatOpenAI:
    """
    Build the LangChain chat model required by the real
    GenerationNodes implementation.

    This is intentionally separate from GroqEvalModel.

    GroqEvalModel is the DeepEval judge model and exposes
    generate()/a_generate(), while GenerationNodes requires
    the LangChain .ainvoke() interface.
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


class GenerationComponent:
    """
    DeepEval wrapper around the real RAG answer-generation node.
    """

    def __init__(
        self,
        generation_nodes: GenerationNodes,
    ):
        self.generation_nodes = generation_nodes

    @observe(
        metrics=get_generation_metrics(),
    )
    async def generate(
        self,
        question: str,
        retrieved_context: list[str],
        expected_output: str,
    ) -> dict:
        """
        Execute the real asynchronous answer-generation node.

        Retrieval is intentionally not executed here.
        The evaluation supplies fixed retrieved context.
        """

        state = build_generation_state(
            question=question,
            retrieved_context=retrieved_context,
        )

        result = await self.generation_nodes.generate_answer(
            state,
        )

        actual_answer = result["answer"]

        update_current_span(
            test_case=LLMTestCase(
                input=question,
                actual_output=actual_answer,
                expected_output=expected_output,
                retrieval_context=retrieved_context,
            )
        )

        return result


@pytest.fixture(scope="module")
def generation_component() -> GenerationComponent:
    """
    Create the real production generation node once
    for the entire test module.
    """

    llm = build_generation_llm()

    generation_nodes = GenerationNodes(
        llm=llm,
    )

    return GenerationComponent(
        generation_nodes=generation_nodes,
    )


@pytest.mark.parametrize(
    "golden",
    rag_generation_goldens,
)
@pytest.mark.asyncio
async def test_generation_component(
    golden,
    generation_component: GenerationComponent,
):
    """
    Evaluate the real RAG answer-generation node against
    fixed questions and retrieved contexts.

    Retrieval is intentionally not executed here.
    """

    result = await generation_component.generate(
        question=golden.input,
        retrieved_context=golden.retrieved_context,
        expected_output=golden.expected_output,
    )

    assert result["answer"]

    assert_test(
        golden=Golden(
            input=golden.input,
            expected_output=golden.expected_output,
        ),
    )
