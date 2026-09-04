from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from src.models.citation import CitationSource
from src.node.generation_nodes import GenerationNodes
from src.node.grading_nodes import DocumentGrade, GradingNodes
from src.node.retrieval_nodes import RAGNodes
from src.node.rewrite_nodes import RewriteNodes


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content=(
                "Full-time employees must work at least 30 hours per week on average."
            ),
            metadata={
                "source": "employee_handbook.pdf",
                "page": 2,
                "chunk_id": "employee_handbook.pdf::chunk_0001",
            },
        ),
        Document(
            page_content=(
                "Employees receive paid annual leave according to company policy."
            ),
            metadata={
                "source": "employee_handbook.pdf",
                "page": 5,
                "chunk_id": "employee_handbook.pdf::chunk_0002",
            },
        ),
    ]


@pytest.fixture
def sample_citations():
    return [
        CitationSource(
            citation_id="1",
            source="employee_handbook.pdf",
            chunk_id="employee_handbook.pdf::chunk_0001",
            page=3,
        ),
        CitationSource(
            citation_id="2",
            source="employee_handbook.pdf",
            chunk_id="employee_handbook.pdf::chunk_0002",
            page=6,
        ),
    ]


@pytest.fixture
def retriever():
    return MagicMock()


@pytest.fixture
def llm():
    return MagicMock()


# ============================================================
# RETRIEVAL → GRADING → GENERATION
# ============================================================


@pytest.mark.asyncio
async def test_rag_pipeline_successful_path(
    retriever,
    llm,
    sample_documents,
):
    """
    Verify the normal RAG path:

        retrieval
            ↓
        relevant documents
            ↓
        generation
    """

    retriever.ainvoke = AsyncMock(return_value=sample_documents)

    retrieval_node = RAGNodes(
        retriever=retriever,
        llm=llm,
    )

    grading_llm = MagicMock()

    grading_llm.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=DocumentGrade(
            relevant=True,
            reason="The retrieved documents contain information relevant to the question.",
        )
    )

    grading_node = GradingNodes(
        llm=grading_llm,
    )

    generation_llm = MagicMock()

    generation_response = MagicMock()
    generation_response.content = (
        "A full-time employee must work at least 30 hours per week on average."
    )

    generation_llm.ainvoke = AsyncMock(return_value=generation_response)

    generation_node = GenerationNodes(
        llm=generation_llm,
    )

    state = {
        "question": "How many hours per week must a full-time employee work?",
        "retrieval_attempts": 0,
    }

    retrieval_result = await retrieval_node.retrieve_docs(state)

    assert retrieval_result["retrieved_docs"] == sample_documents
    assert retrieval_result["retrieval_attempts"] == 1
    assert len(retrieval_result["citations"]) == 2

    grading_state = {
        **state,
        **retrieval_result,
    }

    grading_result = await grading_node.grade_documents(grading_state)

    assert grading_result["document_relevance"] is True
    assert "relevant" in grading_result["grade_reason"].lower()

    generation_state = {
        **grading_state,
        **grading_result,
    }

    generation_result = await generation_node.generate_answer(generation_state)

    assert generation_result["answer"] == (
        "A full-time employee must work at least 30 hours per week on average."
    )

    retriever.ainvoke.assert_awaited_once_with(
        "How many hours per week must a full-time employee work?"
    )

    grading_llm.with_structured_output.return_value.ainvoke.assert_awaited_once()

    generation_llm.ainvoke.assert_awaited_once()


# ============================================================
# EMPTY RETRIEVAL
# ============================================================


@pytest.mark.asyncio
async def test_empty_retrieval_is_handled_without_grader_llm_call(
    retriever,
    llm,
):
    """
    Empty retrieval must not crash the grading stage.

    The grader should return document_relevance=False directly
    without making an LLM call.
    """

    retriever.ainvoke = AsyncMock(return_value=[])

    retrieval_node = RAGNodes(
        retriever=retriever,
        llm=llm,
    )

    grading_llm = MagicMock()

    grading_llm.with_structured_output.return_value.ainvoke = AsyncMock()

    grading_node = GradingNodes(
        llm=grading_llm,
    )

    state = {
        "question": "What is the policy for something not in the documents?",
        "retrieval_attempts": 0,
    }

    retrieval_result = await retrieval_node.retrieve_docs(state)

    assert retrieval_result["retrieved_docs"] == []
    assert retrieval_result["citations"] == []
    assert retrieval_result["retrieval_attempts"] == 1
    assert retrieval_result["retrieval_metadata"]["document_count"] == 0

    grading_result = await grading_node.grade_documents(
        {
            **state,
            **retrieval_result,
        }
    )

    assert grading_result == {
        "document_relevance": False,
        "grade_reason": "No documents were retrieved.",
    }

    grading_llm.with_structured_output.return_value.ainvoke.assert_not_awaited()


# ============================================================
# RETRIEVAL ATTEMPT COUNT
# ============================================================


@pytest.mark.asyncio
async def test_retrieval_attempt_count_increments_across_retries(
    retriever,
    llm,
    sample_documents,
):
    retriever.ainvoke = AsyncMock(
        side_effect=[
            [],
            sample_documents,
        ]
    )

    retrieval_node = RAGNodes(
        retriever=retriever,
        llm=llm,
    )

    first_state = {
        "question": "What is the leave policy?",
        "retrieval_attempts": 0,
    }

    first_result = await retrieval_node.retrieve_docs(first_state)

    assert first_result["retrieval_attempts"] == 1
    assert first_result["retrieved_docs"] == []

    second_state = {
        **first_state,
        **first_result,
        "question": "What are the employee annual leave rules?",
    }

    second_result = await retrieval_node.retrieve_docs(second_state)

    assert second_result["retrieval_attempts"] == 2
    assert second_result["retrieved_docs"] == sample_documents

    assert retriever.ainvoke.await_count == 2

    retriever.ainvoke.assert_any_await("What is the leave policy?")

    retriever.ainvoke.assert_any_await("What are the employee annual leave rules?")


# ============================================================
# RETRIEVAL FAILURE
# ============================================================


@pytest.mark.asyncio
async def test_retrieval_error_propagates(
    retriever,
    llm,
):
    retriever.ainvoke = AsyncMock(side_effect=RuntimeError("retriever unavailable"))

    retrieval_node = RAGNodes(
        retriever=retriever,
        llm=llm,
    )

    state = {
        "question": "What is the leave policy?",
        "retrieval_attempts": 0,
    }

    with pytest.raises(
        RuntimeError,
        match="retriever unavailable",
    ):
        await retrieval_node.retrieve_docs(state)


# ============================================================
# GRADING → REWRITE
# ============================================================


@pytest.mark.asyncio
async def test_irrelevant_documents_trigger_rewrite_path(
    sample_documents,
):
    """
    Verify the failed-relevance path:

        retrieved documents
                ↓
             grading
                ↓
          relevant=False
                ↓
             rewrite
    """

    grading_llm = MagicMock()

    grading_llm.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=DocumentGrade(
            relevant=False,
            reason="The documents do not contain information about the requested topic.",
        )
    )

    grading_node = GradingNodes(
        llm=grading_llm,
    )

    rewrite_response = MagicMock()
    rewrite_response.content = "employee annual leave policy paid leave entitlement"

    rewrite_llm = MagicMock()
    rewrite_llm.ainvoke = AsyncMock(return_value=rewrite_response)

    rewrite_node = RewriteNodes(
        llm=rewrite_llm,
    )

    state = {
        "question": "Tell me about vacation",
        "retrieved_docs": sample_documents,
        "retrieval_attempts": 1,
    }

    grading_result = await grading_node.grade_documents(state)

    assert grading_result["document_relevance"] is False

    rewrite_state = {
        **state,
        **grading_result,
    }

    rewrite_result = await rewrite_node.rewrite_query(rewrite_state)

    assert rewrite_result["question"] == (
        "employee annual leave policy paid leave entitlement"
    )

    grading_llm.with_structured_output.return_value.ainvoke.assert_awaited_once()

    rewrite_llm.ainvoke.assert_awaited_once()


# ============================================================
# REWRITE → RETRIEVAL
# ============================================================


@pytest.mark.asyncio
async def test_rewrite_result_can_be_used_for_next_retrieval(
    retriever,
    llm,
    sample_documents,
):
    retriever.ainvoke = AsyncMock(return_value=sample_documents)

    retrieval_node = RAGNodes(
        retriever=retriever,
        llm=llm,
    )

    rewritten_question = "employee annual leave policy paid leave entitlement"

    state = {
        "question": rewritten_question,
        "retrieved_docs": [],
        "retrieval_attempts": 1,
    }

    result = await retrieval_node.retrieve_docs(state)

    assert result["question"] == rewritten_question
    assert result["retrieved_docs"] == sample_documents
    assert result["retrieval_attempts"] == 2

    retriever.ainvoke.assert_awaited_once_with(rewritten_question)


# ============================================================
# CITATION PROPAGATION
# ============================================================


@pytest.mark.asyncio
async def test_retrieval_builds_citations_for_generation(
    retriever,
    llm,
    sample_documents,
):
    retriever.ainvoke = AsyncMock(return_value=sample_documents)

    retrieval_node = RAGNodes(
        retriever=retriever,
        llm=llm,
    )

    result = await retrieval_node.retrieve_docs(
        {
            "question": "How many hours must a full-time employee work?",
            "retrieval_attempts": 0,
        }
    )

    citations = result["citations"]

    assert len(citations) == 2

    assert citations[0].citation_id == "1"
    assert citations[0].source == "employee_handbook.pdf"
    assert citations[0].chunk_id == ("employee_handbook.pdf::chunk_0001")
    assert citations[0].page == 3

    assert citations[1].citation_id == "2"
    assert citations[1].source == "employee_handbook.pdf"
    assert citations[1].chunk_id == ("employee_handbook.pdf::chunk_0002")
    assert citations[1].page == 6


# ============================================================
# GENERATION USES CITATION CONTEXT
# ============================================================


@pytest.mark.asyncio
async def test_generation_uses_citation_context(
    llm,
    sample_documents,
    sample_citations,
):
    response = MagicMock()
    response.content = "Employees must work at least 30 hours per week [1]."

    llm.ainvoke = AsyncMock(return_value=response)

    generation_node = GenerationNodes(
        llm=llm,
    )

    state = {
        "question": "How many hours per week must a full-time employee work?",
        "retrieved_docs": sample_documents,
        "citations": sample_citations,
    }

    result = await generation_node.generate_answer(state)

    assert result["answer"] == ("Employees must work at least 30 hours per week [1].")

    llm.ainvoke.assert_awaited_once()

    prompt = llm.ainvoke.await_args.args[0]

    assert "employee_handbook.pdf" in prompt
    assert "employee_handbook.pdf::chunk_0001" in prompt
    assert "SOURCE [1]" in prompt
    assert "Page: 3" in prompt
    assert "Full-time employees must work at least 30 hours" in prompt


# ============================================================
# GENERATION FAILURE
# ============================================================


@pytest.mark.asyncio
async def test_generation_error_propagates(
    llm,
    sample_documents,
    sample_citations,
):
    llm.ainvoke = AsyncMock(side_effect=RuntimeError("generation service unavailable"))

    generation_node = GenerationNodes(
        llm=llm,
    )

    state = {
        "question": "What is the leave policy?",
        "retrieved_docs": sample_documents,
        "citations": sample_citations,
    }

    with pytest.raises(
        RuntimeError,
        match="generation service unavailable",
    ):
        await generation_node.generate_answer(state)


# ============================================================
# REWRITE FAILURE
# ============================================================


@pytest.mark.asyncio
async def test_rewrite_error_propagates(
    llm,
    sample_documents,
):
    llm.ainvoke = AsyncMock(side_effect=RuntimeError("rewrite service unavailable"))

    rewrite_node = RewriteNodes(
        llm=llm,
    )

    state = {
        "question": "Tell me about vacation",
        "retrieved_docs": sample_documents,
        "grade_reason": "Retrieved documents were not relevant.",
        "retrieval_attempts": 1,
    }

    with pytest.raises(
        RuntimeError,
        match="rewrite service unavailable",
    ):
        await rewrite_node.rewrite_query(state)


# ============================================================
# INVALID GRADER RESPONSE
# ============================================================


@pytest.mark.asyncio
async def test_invalid_grader_response_is_rejected():
    grading_llm = MagicMock()

    grading_llm.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value={"relevant": True}
    )

    grading_node = GradingNodes(
        llm=grading_llm,
    )

    state = {
        "question": "What is the leave policy?",
        "retrieved_docs": [
            Document(
                page_content="Employees receive paid annual leave.",
                metadata={
                    "source": "employee_handbook.pdf",
                    "chunk_id": "employee_handbook.pdf::chunk_0001",
                },
            )
        ],
    }

    with pytest.raises(
        ValueError,
        match="invalid structured response",
    ):
        await grading_node.grade_documents(state)


# ============================================================
# INVALID GENERATION RESPONSE
# ============================================================


@pytest.mark.asyncio
async def test_generation_rejects_empty_response(
    llm,
    sample_documents,
    sample_citations,
):
    response = MagicMock()
    response.content = "   "

    llm.ainvoke = AsyncMock(return_value=response)

    generation_node = GenerationNodes(
        llm=llm,
    )

    state = {
        "question": "What is the leave policy?",
        "retrieved_docs": sample_documents,
        "citations": sample_citations,
    }

    with pytest.raises(
        ValueError,
        match="empty answer",
    ):
        await generation_node.generate_answer(state)


# ============================================================
# INVALID REWRITE RESPONSE
# ============================================================


@pytest.mark.asyncio
async def test_rewrite_rejects_empty_response(
    llm,
    sample_documents,
):
    response = MagicMock()
    response.content = "   "

    llm.ainvoke = AsyncMock(return_value=response)

    rewrite_node = RewriteNodes(
        llm=llm,
    )

    state = {
        "question": "Tell me about vacation",
        "retrieved_docs": sample_documents,
        "retrieval_attempts": 1,
    }

    with pytest.raises(
        ValueError,
        match="empty query",
    ):
        await rewrite_node.rewrite_query(state)
