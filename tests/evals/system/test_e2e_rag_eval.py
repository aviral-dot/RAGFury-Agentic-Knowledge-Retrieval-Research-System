"""End-to-end evaluation of the complete RAGFury LangGraph workflow."""

from __future__ import annotations

import os
import uuid

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span

from api.main import rag_service
from tests.evals.datasets.e2e_goldens import rag_e2e_goldens
from tests.evals.metrics.generation_metrics import get_generation_metrics


E2E_ENABLED = os.getenv("RUN_E2E_EVAL", "false").lower() == "true"


@pytest.mark.skipif(
    not E2E_ENABLED,
    reason="Set RUN_E2E_EVAL=true to run the live end-to-end evaluation.",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("golden", rag_e2e_goldens)
async def test_e2e_rag_pipeline(golden):
    """Execute a golden question through the real RAGFury graph."""
    if not rag_service.initialized:
        pytest.fail("RAGFury is not initialized. Start the API/infrastructure first.")

    result = await evaluate_e2e_case(golden)

    assert result["answer"].strip(), "E2E pipeline returned an empty answer."
    assert result.get("next_step") == golden.expected_route

    retrieved_docs = result.get("retrieved_docs", [])
    source_names = {
        str(document.metadata.get("source", ""))
        for document in retrieved_docs
        if getattr(document, "metadata", None)
    }

    for expected_source in golden.expected_sources:
        assert expected_source in source_names, (
            f"Expected source '{expected_source}' was not retrieved. "
            f"Retrieved sources: {sorted(source_names)}"
        )

    if golden.should_abstain:
        assert result.get("retrieval_abstained") is True
        assert result.get("abstention_reason")
        return

    assert result.get("retrieval_abstained") is not True
    assert result.get("citations"), "A successful RAG answer must contain citations."

    assert_test(
        golden=Golden(
            input=golden.input,
            expected_output=golden.expected_output,
        )
    )


@observe(metrics=get_generation_metrics())
async def evaluate_e2e_case(golden):
    """Run one real graph execution and expose it to DeepEval."""
    result = await rag_service.query(
        question=golden.input,
        user_id="e2e-evaluator",
        conversation_id=f"e2e-{uuid.uuid4().hex}",
        request_id=uuid.uuid4().hex,
    )

    retrieval_context = [
        document.page_content
        for document in result.get("retrieved_docs", [])
    ]

    update_current_span(
        test_case=LLMTestCase(
            input=golden.input,
            actual_output=result.get("answer", ""),
            expected_output=golden.expected_output,
            retrieval_context=retrieval_context,
        )
    )

    return result
