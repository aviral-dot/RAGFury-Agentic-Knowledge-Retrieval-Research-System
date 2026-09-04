import pytest
from langchain_core.documents import Document

from src.node.retrieval_nodes import RAGNodes


class FakeRetriever:
    def __init__(self, documents=None, error=None):
        self.documents = documents or []
        self.error = error
        self.calls = []

    async def ainvoke(self, question):
        self.calls.append(question)

        if self.error:
            raise self.error

        return self.documents


@pytest.fixture
def node():
    return lambda retriever: RAGNodes(
        retriever=retriever,
        llm=object(),
    )


@pytest.mark.asyncio
async def test_retrieve_docs_returns_documents_and_metadata(node):
    documents = [
        Document(
            page_content="Annual leave is 20 days.",
            metadata={
                "document_id": "doc-001",
                "chunk_id": "chunk-001",
                "source": "employee_handbook.pdf",
                "page": 4,
                "score": 0.91,
            },
        ),
        Document(
            page_content="Sick leave is available.",
            metadata={
                "document_id": "doc-001",
                "chunk_id": "chunk-002",
                "source": "employee_handbook.pdf",
                "page": 5,
                "score": 0.82,
            },
        ),
    ]

    retriever = FakeRetriever(documents)
    rag_node = node(retriever)

    result = await rag_node.retrieve_docs(
        {
            "question": "What is the leave policy?",
            "retrieval_attempts": 0,
        }
    )

    assert retriever.calls == ["What is the leave policy?"]
    assert result["retrieved_docs"] == documents
    assert result["retrieval_attempts"] == 1

    metadata = result["retrieval_metadata"]

    assert metadata["attempt"] == 1
    assert metadata["document_count"] == 2
    assert metadata["duration_ms"] >= 0
    assert len(metadata["results"]) == 2


@pytest.mark.asyncio
async def test_retrieve_docs_increments_existing_attempt_count(node):
    retriever = FakeRetriever(
        [
            Document(
                page_content="Leave policy.",
                metadata={
                    "document_id": "doc-001",
                    "chunk_id": "chunk-001",
                    "source": "handbook.pdf",
                },
            )
        ]
    )

    rag_node = node(retriever)

    result = await rag_node.retrieve_docs(
        {
            "question": "What is the leave policy?",
            "retrieval_attempts": 2,
        }
    )

    assert result["retrieval_attempts"] == 3
    assert result["retrieval_metadata"]["attempt"] == 3


@pytest.mark.asyncio
async def test_retrieve_docs_handles_empty_results(node):
    retriever = FakeRetriever([])
    rag_node = node(retriever)

    result = await rag_node.retrieve_docs(
        {
            "question": "Something unrelated",
        }
    )

    assert result["retrieved_docs"] == []
    assert result["citations"] == []
    assert result["retrieval_attempts"] == 1

    metadata = result["retrieval_metadata"]

    assert metadata["document_count"] == 0
    assert metadata["results"] == []


@pytest.mark.asyncio
async def test_retrieve_docs_propagates_retriever_error(node):
    retriever = FakeRetriever(error=RuntimeError("Retriever unavailable"))

    rag_node = node(retriever)

    with pytest.raises(
        RuntimeError,
        match="Retriever unavailable",
    ):
        await rag_node.retrieve_docs(
            {
                "question": "What is the leave policy?",
            }
        )


def test_serialize_retrieved_document_returns_safe_metadata():
    document = Document(
        page_content="Annual leave.",
        metadata={
            "document_id": "doc-001",
            "chunk_id": "chunk-001",
            "source": "handbook.pdf",
            "score": 0.95,
            "secret": "should-not-be-exposed",
        },
    )

    result = RAGNodes._serialize_retrieved_document(
        document,
        rank=1,
    )

    assert result == {
        "rank": 1,
        "document_id": "doc-001",
        "chunk_id": "chunk-001",
        "source": "handbook.pdf",
        "score": 0.95,
    }

    assert "secret" not in result


def test_serialize_retrieved_document_handles_missing_metadata():
    document = Document(page_content="No metadata.")

    result = RAGNodes._serialize_retrieved_document(
        document,
        rank=1,
    )

    assert result == {
        "rank": 1,
        "document_id": None,
        "chunk_id": None,
        "source": None,
        "score": None,
    }


def test_build_citations_converts_page_to_one_based():
    documents = [
        Document(
            page_content="Annual leave.",
            metadata={
                "source": "handbook.pdf",
                "chunk_id": "chunk-001",
                "page": 0,
            },
        ),
        Document(
            page_content="Sick leave.",
            metadata={
                "source": "handbook.pdf",
                "chunk_id": "chunk-002",
                "page": 4,
            },
        ),
    ]

    citations = RAGNodes._build_citations(documents)

    assert len(citations) == 2

    assert citations[0].citation_id == "1"
    assert citations[0].source == "handbook.pdf"
    assert citations[0].chunk_id == "chunk-001"
    assert citations[0].page == 1

    assert citations[1].citation_id == "2"
    assert citations[1].page == 5


def test_build_citations_accepts_string_page():
    documents = [
        Document(
            page_content="Annual leave.",
            metadata={
                "source": "handbook.pdf",
                "chunk_id": "chunk-001",
                "page": "7",
            },
        )
    ]

    citations = RAGNodes._build_citations(documents)

    assert len(citations) == 1
    assert citations[0].page == 8


def test_build_citations_sets_invalid_page_to_none():
    documents = [
        Document(
            page_content="Annual leave.",
            metadata={
                "source": "handbook.pdf",
                "chunk_id": "chunk-001",
                "page": "invalid",
            },
        )
    ]

    citations = RAGNodes._build_citations(documents)

    assert len(citations) == 1
    assert citations[0].page is None


def test_build_citations_skips_missing_source():
    documents = [
        Document(
            page_content="Missing source.",
            metadata={
                "chunk_id": "chunk-001",
                "page": 0,
            },
        ),
        Document(
            page_content="Valid document.",
            metadata={
                "source": "handbook.pdf",
                "chunk_id": "chunk-002",
                "page": 1,
            },
        ),
    ]

    citations = RAGNodes._build_citations(documents)

    assert len(citations) == 1
    assert citations[0].citation_id == "2"
    assert citations[0].chunk_id == "chunk-002"


def test_build_citations_skips_missing_chunk_id():
    documents = [
        Document(
            page_content="Missing chunk.",
            metadata={
                "source": "handbook.pdf",
                "page": 0,
            },
        )
    ]

    citations = RAGNodes._build_citations(documents)

    assert citations == []


def test_build_citations_handles_empty_source():
    documents = [
        Document(
            page_content="Empty source.",
            metadata={
                "source": "",
                "chunk_id": "chunk-001",
                "page": 0,
            },
        )
    ]

    citations = RAGNodes._build_citations(documents)

    assert citations == []


def test_build_citations_converts_source_and_chunk_id_to_strings():
    documents = [
        Document(
            page_content="Test.",
            metadata={
                "source": 123,
                "chunk_id": 456,
                "page": 0,
            },
        )
    ]

    citations = RAGNodes._build_citations(documents)

    assert len(citations) == 1
    assert citations[0].source == "123"
    assert citations[0].chunk_id == "456"
