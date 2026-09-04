import uuid
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from langchain_core.documents import Document

from src.vectorstore.vectorstore import RerankingRetriever, VectorStore


@pytest.fixture
def vectorstore(monkeypatch):
    """Create VectorStore without loading real models or Qdrant."""

    fake_qdrant = MagicMock()
    fake_embedding = MagicMock()
    fake_sparse = MagicMock()
    fake_reranker = MagicMock()

    monkeypatch.setattr(
        "src.vectorstore.vectorstore.QdrantClient",
        MagicMock(return_value=fake_qdrant),
    )
    monkeypatch.setattr(
        "src.vectorstore.vectorstore.HuggingFaceEmbeddings",
        MagicMock(return_value=fake_embedding),
    )
    monkeypatch.setattr(
        "src.vectorstore.vectorstore.FastEmbedSparse",
        MagicMock(return_value=fake_sparse),
    )
    monkeypatch.setattr(
        "src.vectorstore.vectorstore.CrossEncoder",
        MagicMock(return_value=fake_reranker),
    )

    store = VectorStore(
        qdrant_url="http://test-qdrant:6333",
        collection_name="test_collection",
        qdrant_api_key="test-key",
        retrieval_k=5,
        rerank_k=2,
        mode="query",
    )

    return store


# ============================================================
# INITIALIZATION
# ============================================================


def test_vectorstore_initializes_with_configuration(
    monkeypatch,
):
    fake_qdrant = MagicMock()
    fake_embedding = MagicMock()
    fake_sparse = MagicMock()
    fake_reranker = MagicMock()

    qdrant_constructor = MagicMock(return_value=fake_qdrant)
    embedding_constructor = MagicMock(return_value=fake_embedding)
    sparse_constructor = MagicMock(return_value=fake_sparse)
    reranker_constructor = MagicMock(return_value=fake_reranker)

    monkeypatch.setattr(
        "src.vectorstore.vectorstore.QdrantClient",
        qdrant_constructor,
    )
    monkeypatch.setattr(
        "src.vectorstore.vectorstore.HuggingFaceEmbeddings",
        embedding_constructor,
    )
    monkeypatch.setattr(
        "src.vectorstore.vectorstore.FastEmbedSparse",
        sparse_constructor,
    )
    monkeypatch.setattr(
        "src.vectorstore.vectorstore.CrossEncoder",
        reranker_constructor,
    )

    store = VectorStore(
        qdrant_url="http://localhost:6333",
        collection_name="my_collection",
        qdrant_api_key="secret",
        retrieval_k=10,
        rerank_k=2,
        mode="query",
    )

    assert store.qdrant_url == "http://localhost:6333"
    assert store.collection_name == "my_collection"
    assert store.qdrant_api_key == "secret"
    assert store.retrieval_k == 10
    assert store.rerank_k == 2
    assert store.mode == "query"

    qdrant_constructor.assert_called_once_with(
        url="http://localhost:6333",
        api_key="secret",
    )

    embedding_constructor.assert_called_once_with(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    sparse_constructor.assert_called_once_with(
        model_name="Qdrant/bm25",
    )

    reranker_constructor.assert_called_once_with(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )


@pytest.mark.parametrize(
    "mode",
    ["invalid", "", "querying", "production"],
)
def test_vectorstore_rejects_invalid_mode(
    monkeypatch,
    mode,
):
    with pytest.raises(
        ValueError,
        match="VectorStore mode must be either",
    ):
        VectorStore(mode=mode)


def test_vectorstore_accepts_ingestion_mode(
    monkeypatch,
):
    monkeypatch.setattr(
        "src.vectorstore.vectorstore.QdrantClient",
        MagicMock(),
    )
    monkeypatch.setattr(
        "src.vectorstore.vectorstore.HuggingFaceEmbeddings",
        MagicMock(),
    )
    monkeypatch.setattr(
        "src.vectorstore.vectorstore.FastEmbedSparse",
        MagicMock(),
    )
    monkeypatch.setattr(
        "src.vectorstore.vectorstore.CrossEncoder",
        MagicMock(),
    )

    store = VectorStore(mode="ingestion")

    assert store.mode == "ingestion"


def test_vectorstore_uses_environment_qdrant_url(
    monkeypatch,
):
    monkeypatch.setenv(
        "QDRANT_URL",
        "http://env-qdrant:6333",
    )

    fake_qdrant = MagicMock()
    monkeypatch.setattr(
        "src.vectorstore.vectorstore.QdrantClient",
        MagicMock(return_value=fake_qdrant),
    )
    monkeypatch.setattr(
        "src.vectorstore.vectorstore.HuggingFaceEmbeddings",
        MagicMock(),
    )
    monkeypatch.setattr(
        "src.vectorstore.vectorstore.FastEmbedSparse",
        MagicMock(),
    )
    monkeypatch.setattr(
        "src.vectorstore.vectorstore.CrossEncoder",
        MagicMock(),
    )

    store = VectorStore()

    assert store.qdrant_url == "http://env-qdrant:6333"


# ============================================================
# QDRANT INITIALIZATION
# ============================================================


def test_initialize_qdrant_connects_existing_collection(
    monkeypatch,
    vectorstore,
):
    vectorstore.qdrant_client.collection_exists.return_value = True

    fake_vectorstore = MagicMock()

    constructor = MagicMock(
        return_value=fake_vectorstore,
    )

    monkeypatch.setattr(
        "src.vectorstore.vectorstore.QdrantVectorStore.from_existing_collection",
        constructor,
    )

    vectorstore._initialize_qdrant()

    assert vectorstore.vectorstore is fake_vectorstore

    constructor.assert_called_once_with(
        embedding=vectorstore.embedding,
        sparse_embedding=vectorstore.sparse_embedding,
        collection_name="test_collection",
        url="http://test-qdrant:6333",
        api_key="test-key",
        retrieval_mode=vectorstore.__class__.__module__
        and __import__(
            "src.vectorstore.vectorstore",
            fromlist=["RetrievalMode"],
        ).RetrievalMode.HYBRID,
    )


def test_initialize_qdrant_rejects_missing_collection(
    vectorstore,
):
    vectorstore.qdrant_client.collection_exists.return_value = False

    with pytest.raises(
        RuntimeError,
        match="does not exist",
    ):
        vectorstore._initialize_qdrant()


def test_initialize_qdrant_allows_missing_collection_when_creating(
    vectorstore,
):
    vectorstore.qdrant_client.collection_exists.return_value = False

    result = vectorstore._initialize_qdrant(
        create_if_missing=True,
    )

    assert result is None
    assert vectorstore.vectorstore is None


# ============================================================
# ADD DOCUMENTS
# ============================================================


def test_add_documents_skips_empty_list(
    vectorstore,
):
    vectorstore.add_documents([])

    vectorstore.qdrant_client.collection_exists.assert_not_called()


def test_add_documents_creates_collection(
    monkeypatch,
    vectorstore,
):
    documents = [
        Document(
            page_content="Employee leave policy",
            metadata={
                "source": "handbook.pdf",
                "chunk_id": "abc:chunk_0000",
            },
        ),
        Document(
            page_content="Working hours policy",
            metadata={
                "source": "handbook.pdf",
                "chunk_id": "abc:chunk_0001",
            },
        ),
    ]

    vectorstore.qdrant_client.collection_exists.return_value = False

    fake_vectorstore = MagicMock()

    constructor = MagicMock(
        return_value=fake_vectorstore,
    )

    monkeypatch.setattr(
        "src.vectorstore.vectorstore.QdrantVectorStore.from_documents",
        constructor,
    )

    monkeypatch.setattr(
        "src.vectorstore.vectorstore.uuid.uuid4",
        lambda: uuid.UUID("00000000-0000-0000-0000-000000000001"),
    )

    vectorstore.add_documents(documents)

    assert vectorstore.vectorstore is fake_vectorstore

    constructor.assert_called_once()

    kwargs = constructor.call_args.kwargs

    assert kwargs["documents"] == documents
    assert kwargs["collection_name"] == "test_collection"
    assert kwargs["url"] == "http://test-qdrant:6333"
    assert kwargs["api_key"] == "test-key"

    assert kwargs["ids"] == [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000001",
    ]


def test_add_documents_preserves_citation_metadata(
    monkeypatch,
    vectorstore,
):
    documents = [
        Document(
            page_content="Leave policy",
            metadata={
                "source": "employee_handbook.pdf",
                "chunk_id": "hash:chunk_0000",
                "page": 7,
            },
        ),
    ]

    vectorstore.qdrant_client.collection_exists.return_value = False

    monkeypatch.setattr(
        "src.vectorstore.vectorstore.QdrantVectorStore.from_documents",
        MagicMock(return_value=MagicMock()),
    )

    vectorstore.add_documents(documents)

    assert documents[0].metadata["source"] == "employee_handbook.pdf"
    assert documents[0].metadata["chunk_id"] == "hash:chunk_0000"
    assert documents[0].metadata["page"] == 7


def test_add_documents_adds_to_existing_collection(
    monkeypatch,
    vectorstore,
):
    documents = [
        Document(
            page_content="Employee handbook",
            metadata={
                "source": "handbook.pdf",
                "chunk_id": "hash:chunk_0000",
            },
        ),
    ]

    vectorstore.qdrant_client.collection_exists.return_value = True

    existing_vectorstore = MagicMock()

    from_existing = MagicMock(
        return_value=existing_vectorstore,
    )

    monkeypatch.setattr(
        "src.vectorstore.vectorstore.QdrantVectorStore.from_existing_collection",
        from_existing,
    )

    monkeypatch.setattr(
        "src.vectorstore.vectorstore.uuid.uuid4",
        lambda: uuid.UUID("00000000-0000-0000-0000-000000000002"),
    )

    vectorstore.add_documents(documents)

    assert vectorstore.vectorstore is existing_vectorstore

    existing_vectorstore.add_documents.assert_called_once()

    kwargs = existing_vectorstore.add_documents.call_args.kwargs

    assert kwargs["documents"] == documents
    assert kwargs["ids"] == [
        "00000000-0000-0000-0000-000000000002",
    ]


def test_add_documents_requires_source_metadata(
    vectorstore,
):
    vectorstore.qdrant_client.collection_exists.return_value = False

    documents = [
        Document(
            page_content="Missing source",
            metadata={
                "chunk_id": "hash:chunk_0000",
            },
        ),
    ]

    with pytest.raises(KeyError, match="source"):
        vectorstore.add_documents(documents)


def test_add_documents_requires_chunk_id_metadata(
    vectorstore,
):
    vectorstore.qdrant_client.collection_exists.return_value = False

    documents = [
        Document(
            page_content="Missing chunk ID",
            metadata={
                "source": "handbook.pdf",
            },
        ),
    ]

    with pytest.raises(KeyError, match="chunk_id"):
        vectorstore.add_documents(documents)


# ============================================================
# INITIALIZE HYBRID RETRIEVER
# ============================================================


def test_initialize_creates_collection_from_new_documents(
    vectorstore,
):
    documents = [
        Document(
            page_content="New document",
            metadata={
                "source": "new.pdf",
                "chunk_id": "hash:chunk_0000",
            },
        ),
    ]

    vectorstore.qdrant_client.collection_exists.return_value = False

    vectorstore.add_documents = MagicMock()

    fake_vectorstore = MagicMock()

    vectorstore.vectorstore = fake_vectorstore

    vectorstore.get_document_count = MagicMock(
        return_value=1,
    )

    vectorstore.initialize(
        new_documents=documents,
    )

    vectorstore.add_documents.assert_called_once_with(
        documents,
    )

    fake_vectorstore.as_retriever.assert_called_once_with(
        search_kwargs={"k": 5},
    )

    assert vectorstore.hybrid_retriever is fake_vectorstore.as_retriever.return_value


def test_initialize_rejects_missing_collection_without_documents(
    vectorstore,
):
    vectorstore.qdrant_client.collection_exists.return_value = False

    with pytest.raises(
        ValueError,
        match="no documents were supplied",
    ):
        vectorstore.initialize()


def test_initialize_adds_documents_to_existing_collection(
    vectorstore,
):
    documents = [
        Document(
            page_content="New document",
            metadata={
                "source": "new.pdf",
                "chunk_id": "hash:chunk_0000",
            },
        ),
    ]

    vectorstore.qdrant_client.collection_exists.return_value = True

    vectorstore.add_documents = MagicMock()

    fake_vectorstore = MagicMock()
    vectorstore.vectorstore = fake_vectorstore

    vectorstore.get_document_count = MagicMock(
        return_value=10,
    )

    vectorstore.initialize(
        new_documents=documents,
    )

    vectorstore.add_documents.assert_called_once_with(
        documents,
    )

    fake_vectorstore.as_retriever.assert_called_once_with(
        search_kwargs={"k": 5},
    )


def test_initialize_connects_existing_collection(
    monkeypatch,
    vectorstore,
):
    vectorstore.qdrant_client.collection_exists.return_value = True
    vectorstore.vectorstore = None

    fake_vectorstore = MagicMock()

    monkeypatch.setattr(
        "src.vectorstore.vectorstore.QdrantVectorStore.from_existing_collection",
        MagicMock(return_value=fake_vectorstore),
    )

    vectorstore.get_document_count = MagicMock(
        return_value=10,
    )

    vectorstore.initialize()

    assert vectorstore.vectorstore is fake_vectorstore
    assert vectorstore.hybrid_retriever is (fake_vectorstore.as_retriever.return_value)

    fake_vectorstore.as_retriever.assert_called_once_with(
        search_kwargs={"k": 5},
    )


# ============================================================
# GET RETRIEVER
# ============================================================


def test_get_retriever_requires_initialization(
    vectorstore,
):
    with pytest.raises(
        ValueError,
        match="Hybrid retriever not initialized",
    ):
        vectorstore.get_retriever()


def test_get_retriever_uses_default_rerank_k(
    vectorstore,
):
    vectorstore.hybrid_retriever = MagicMock()

    retriever = vectorstore.get_retriever()

    assert isinstance(
        retriever,
        RerankingRetriever,
    )

    assert retriever.base_retriever is (vectorstore.hybrid_retriever)

    assert retriever.vector_store is vectorstore
    assert retriever.k == 2


def test_get_retriever_accepts_custom_k(
    vectorstore,
):
    vectorstore.hybrid_retriever = MagicMock()

    retriever = vectorstore.get_retriever(k=4)

    assert retriever.k == 4


# ============================================================
# RERANKING
# ============================================================


def test_rerank_returns_top_k_documents(
    vectorstore,
):
    documents = [
        Document(
            page_content="Document A",
            metadata={"source": "a.pdf"},
        ),
        Document(
            page_content="Document B",
            metadata={"source": "b.pdf"},
        ),
        Document(
            page_content="Document C",
            metadata={"source": "c.pdf"},
        ),
    ]

    vectorstore.reranker.predict.return_value = np.array(
        [0.2, 0.9, 0.5],
    )

    result = vectorstore._rerank(
        query="employee policy",
        documents=documents,
        k=2,
    )

    assert len(result) == 2
    assert result[0].page_content == "Document B"
    assert result[1].page_content == "Document C"


def test_rerank_passes_query_document_pairs(
    vectorstore,
):
    documents = [
        Document(page_content="Document A"),
        Document(page_content="Document B"),
    ]

    vectorstore.reranker.predict.return_value = np.array(
        [0.8, 0.4],
    )

    vectorstore._rerank(
        query="my query",
        documents=documents,
        k=2,
    )

    vectorstore.reranker.predict.assert_called_once_with(
        [
            ("my query", "Document A"),
            ("my query", "Document B"),
        ],
    )


def test_rerank_applies_min_score(
    vectorstore,
):
    documents = [
        Document(page_content="Low"),
        Document(page_content="High"),
    ]

    vectorstore.reranker.predict.return_value = np.array(
        [0.2, 0.8],
    )

    result = vectorstore._rerank(
        query="query",
        documents=documents,
        k=5,
        min_score=0.5,
    )

    assert len(result) == 1
    assert result[0].page_content == "High"


def test_rerank_returns_empty_for_empty_documents(
    vectorstore,
):
    vectorstore.reranker.predict.reset_mock()

    result = vectorstore._rerank(
        query="query",
        documents=[],
        k=2,
    )

    assert result == []

    vectorstore.reranker.predict.assert_not_called()


def test_rerank_raises_when_reranker_unavailable(
    vectorstore,
):
    vectorstore.reranker = None

    with pytest.raises(
        RuntimeError,
        match="Reranking is unavailable",
    ):
        vectorstore._rerank(
            query="query",
            documents=[
                Document(page_content="Document"),
            ],
            k=1,
        )


# ============================================================
# ASYNC RETRIEVAL
# ============================================================


@pytest.mark.asyncio
async def test_retrieve_requires_initialized_retriever(
    vectorstore,
):
    with pytest.raises(
        ValueError,
        match="Hybrid retriever not initialized",
    ):
        await vectorstore.retrieve(
            query="employee leave",
        )


@pytest.mark.asyncio
async def test_retrieve_gets_candidates_and_reranks(
    vectorstore,
):
    documents = [
        Document(
            page_content="Document A",
            metadata={
                "source": "a.pdf",
                "chunk_id": "a:chunk_0000",
            },
        ),
        Document(
            page_content="Document B",
            metadata={
                "source": "b.pdf",
                "chunk_id": "b:chunk_0000",
            },
        ),
    ]

    vectorstore.hybrid_retriever = MagicMock()
    vectorstore.hybrid_retriever.ainvoke = AsyncMock(
        return_value=documents,
    )

    reranked = [
        documents[1],
    ]

    vectorstore._rerank = MagicMock(
        return_value=reranked,
    )

    result = await vectorstore.retrieve(
        query="employee leave",
        k=1,
    )

    assert result == reranked

    vectorstore.hybrid_retriever.ainvoke.assert_awaited_once_with(
        "employee leave",
    )

    vectorstore._rerank.assert_called_once_with(
        query="employee leave",
        documents=documents,
        k=1,
    )


@pytest.mark.asyncio
async def test_retrieve_propagates_retrieval_error(
    vectorstore,
):
    vectorstore.hybrid_retriever = MagicMock()

    vectorstore.hybrid_retriever.ainvoke = AsyncMock(
        side_effect=RuntimeError("Qdrant unavailable"),
    )

    with pytest.raises(
        RuntimeError,
        match="Qdrant unavailable",
    ):
        await vectorstore.retrieve(
            query="employee leave",
        )


# ============================================================
# DOCUMENT COUNT
# ============================================================


def test_get_document_count_returns_qdrant_count(
    vectorstore,
):
    collection_info = MagicMock()
    collection_info.points_count = 42

    vectorstore.qdrant_client.get_collection.return_value = collection_info

    result = vectorstore.get_document_count()

    assert result == 42

    vectorstore.qdrant_client.get_collection.assert_called_once_with(
        "test_collection",
    )


def test_get_document_count_returns_zero_when_qdrant_fails(
    vectorstore,
):
    vectorstore.qdrant_client.get_collection.side_effect = RuntimeError(
        "Qdrant unavailable"
    )

    result = vectorstore.get_document_count()

    assert result == 0


# ============================================================
# RERANKING RETRIEVER
# ============================================================


def test_reranking_retriever_sync_invoke_is_not_supported(
    vectorstore,
):
    retriever = RerankingRetriever(
        base_retriever=MagicMock(),
        vector_store=vectorstore,
        k=2,
    )

    with pytest.raises(
        NotImplementedError,
        match="async-only",
    ):
        retriever._get_relevant_documents(
            query="test",
            run_manager=MagicMock(),
        )


@pytest.mark.asyncio
async def test_reranking_retriever_delegates_to_base_and_reranks(
    vectorstore,
):
    documents = [
        Document(page_content="Document A"),
        Document(page_content="Document B"),
    ]

    base_retriever = MagicMock()

    base_retriever.ainvoke = AsyncMock(
        return_value=documents,
    )

    vectorstore._rerank = MagicMock(
        return_value=[documents[1]],
    )

    retriever = RerankingRetriever(
        base_retriever=base_retriever,
        vector_store=vectorstore,
        k=1,
    )

    result = await retriever.ainvoke(
        "employee policy",
    )

    assert result == [documents[1]]

    base_retriever.ainvoke.assert_awaited_once_with(
        "employee policy",
    )

    vectorstore._rerank.assert_called_once_with(
        "employee policy",
        documents,
        1,
    )
