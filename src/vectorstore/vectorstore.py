"""Qdrant-backed hybrid vector store for dense + sparse retrieval."""

import asyncio
import logging
import os
import time
from typing import List

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import (
    FastEmbedSparse,
    QdrantVectorStore,
    RetrievalMode,
)
from langsmith import traceable
from pydantic import ConfigDict
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from src.utils.loggers import (
    configure_logging,
    get_logger,
    log_event,
)

configure_logging()

logger = get_logger(__name__)


class RerankingRetriever(BaseRetriever):
    """Retrieve hybrid candidates from Qdrant and rerank them."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: object
    vector_store: object
    k: int = 2

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Synchronous BaseRetriever compatibility method.

        This retriever intentionally supports async execution only.
        Use ainvoke() for actual retrieval.
        """
        raise NotImplementedError(
            "RerankingRetriever is async-only. Use ainvoke() instead of invoke()."
        )

    @traceable(
        name="RAGFury Hybrid Retrieval + Reranking",
        run_type="retriever",
    )
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve documents and apply cross-encoder reranking."""

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.DEBUG,
            event="retrieval.reranking_retriever.started",
            k=self.k,
            query_length=len(query),
        )

        try:
            documents = await self.base_retriever.ainvoke(query)

            result = await asyncio.to_thread(
                self.vector_store._rerank,
                query,
                documents,
                self.k,
            )

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="retrieval.reranking_retriever.completed",
                candidate_count=len(documents),
                final_document_count=len(result),
                retrieval_k=self.vector_store.retrieval_k,
                rerank_k=self.k,
                duration_ms=round(elapsed, 2),
            )

            return result

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="retrieval.reranking_retriever.failed",
                error_type=type(exc).__name__,
                k=self.k,
                duration_ms=round(elapsed, 2),
            )

            logger.exception("Reranking retriever execution failed")

            raise


class VectorStore:
    """
    Persistent hybrid retrieval using Qdrant.

    Dense:
        sentence-transformers/all-MiniLM-L6-v2

    Sparse:
        Qdrant/bm25 via FastEmbedSparse

    Hybrid:
        Qdrant native hybrid retrieval

    Reranking:
        cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(
        self,
        qdrant_url: str | None = None,
        collection_name: str = "ragfury_documents",
        qdrant_api_key: str | None = None,
        retrieval_k: int = 5,
        rerank_k: int = 2,
        mode: str = "query",
    ):
        """Initialize the Qdrant-backed vector store."""

        if mode not in {"query", "ingestion"}:
            raise ValueError("VectorStore mode must be either 'query' or 'ingestion'.")

        self.mode = mode

        start_time = time.perf_counter()

        self.qdrant_url = qdrant_url or os.getenv(
            "QDRANT_URL",
            "http://localhost:6333",
        )

        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        self.collection_name = collection_name

        self.retrieval_k = retrieval_k
        self.rerank_k = rerank_k

        log_event(
            logger,
            level=logging.INFO,
            event="vectorstore.initialization.started",
            mode=self.mode,
            qdrant_url=self.qdrant_url,
            collection_name=self.collection_name,
            retrieval_k=self.retrieval_k,
            rerank_k=self.rerank_k,
        )

        # -----------------------------------------------------
        # Qdrant client
        # -----------------------------------------------------

        try:
            client_kwargs = {
                "url": self.qdrant_url,
            }

            if self.qdrant_api_key:
                client_kwargs["api_key"] = self.qdrant_api_key

            self.qdrant_client = QdrantClient(**client_kwargs)

            log_event(
                logger,
                level=logging.DEBUG,
                event="vectorstore.qdrant.client.initialized",
                qdrant_url=self.qdrant_url,
                collection_name=self.collection_name,
            )

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="vectorstore.qdrant.client.initialization.failed",
                error_type=type(exc).__name__,
            )

            logger.exception("Failed to initialize Qdrant client")

            raise

        # -----------------------------------------------------
        # Dense embedding model
        # -----------------------------------------------------

        embedding_start = time.perf_counter()

        try:
            self.embedding = HuggingFaceEmbeddings(
                model_name=("sentence-transformers/all-MiniLM-L6-v2")
            )

            embedding_elapsed = (time.perf_counter() - embedding_start) * 1000

            log_event(
                logger,
                level=logging.DEBUG,
                event="vectorstore.embedding.initialized",
                model_name=("sentence-transformers/all-MiniLM-L6-v2"),
                duration_ms=round(
                    embedding_elapsed,
                    2,
                ),
            )

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="vectorstore.embedding.initialization.failed",
                error_type=type(exc).__name__,
            )

            logger.exception("Failed to initialize dense embedding model")

            raise

        # -----------------------------------------------------
        # Sparse BM25 embedding model
        # -----------------------------------------------------

        sparse_start = time.perf_counter()

        try:
            self.sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")

            sparse_elapsed = (time.perf_counter() - sparse_start) * 1000

            log_event(
                logger,
                level=logging.DEBUG,
                event="vectorstore.sparse_embedding.initialized",
                model_name="Qdrant/bm25",
                duration_ms=round(
                    sparse_elapsed,
                    2,
                ),
            )

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="vectorstore.sparse_embedding.initialization.failed",
                error_type=type(exc).__name__,
            )

            logger.exception("Failed to initialize sparse embedding model")

            raise

        # -----------------------------------------------------
        # Cross-encoder reranker
        # -----------------------------------------------------

        reranker_start = time.perf_counter()

        try:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

            reranker_elapsed = (time.perf_counter() - reranker_start) * 1000

            log_event(
                logger,
                level=logging.DEBUG,
                event="vectorstore.reranker.initialized",
                model_name=("cross-encoder/ms-marco-MiniLM-L-6-v2"),
                duration_ms=round(
                    reranker_elapsed,
                    2,
                ),
            )

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="vectorstore.reranker.initialization.failed",
                error_type=type(exc).__name__,
            )

            logger.exception("Failed to initialize reranker")

            raise

        # -----------------------------------------------------
        # Runtime state
        # -----------------------------------------------------

        self.vectorstore: QdrantVectorStore | None = None
        self.hybrid_retriever = None

        elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="vectorstore.initialization.completed",
            mode=self.mode,
            duration_ms=round(elapsed, 2),
        )

    # =========================================================
    # QDRANT INITIALIZATION
    # =========================================================

    def _initialize_qdrant(
        self,
        create_if_missing: bool = False,
    ) -> None:
        """
        Initialize the Qdrant vector store.

        If the collection already exists, connect to it.

        If create_if_missing is True, create the collection
        using the first ingestion operation.
        """

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.DEBUG,
            event="vectorstore.qdrant.initialization.started",
            collection_name=self.collection_name,
            qdrant_url=self.qdrant_url,
        )

        try:
            collection_exists = self.qdrant_client.collection_exists(
                self.collection_name
            )

            if not collection_exists:
                if not create_if_missing:
                    log_event(
                        logger,
                        level=logging.WARNING,
                        event="vectorstore.qdrant.collection.missing",
                        collection_name=(self.collection_name),
                    )

                    raise RuntimeError(
                        "Qdrant collection "
                        f"'{self.collection_name}' "
                        "does not exist. "
                        "Initialize the vector store with "
                        "new documents first so the collection "
                        "can be created."
                    )

                log_event(
                    logger,
                    level=logging.INFO,
                    event="vectorstore.qdrant.collection.creation.deferred",
                    collection_name=(self.collection_name),
                )

                return

            self.vectorstore = QdrantVectorStore.from_existing_collection(
                embedding=self.embedding,
                sparse_embedding=self.sparse_embedding,
                collection_name=self.collection_name,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                retrieval_mode=RetrievalMode.HYBRID,
            )

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="vectorstore.qdrant.initialization.completed",
                collection_name=self.collection_name,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

        except RuntimeError:
            raise

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="vectorstore.qdrant.initialization.failed",
                collection_name=self.collection_name,
                error_type=type(exc).__name__,
            )

            logger.exception("Failed to initialize Qdrant vector store")

            raise

    # =========================================================
    # ADD DOCUMENTS
    # =========================================================

    def add_documents(
        self,
        documents: List[Document],
    ) -> None:
        """
        Add document chunks to Qdrant.

        Each document is stored with:

            dense vector
            sparse BM25 vector
            document text
            metadata
        """

        if not documents:
            log_event(
                logger,
                level=logging.DEBUG,
                event="vectorstore.documents.add.skipped",
                reason="empty_document_list",
            )

            return

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="vectorstore.documents.add.started",
            document_count=len(documents),
            collection_name=self.collection_name,
        )

        try:
            # -------------------------------------------------
            # Create collection from first document if needed
            # -------------------------------------------------

            collection_exists = self.qdrant_client.collection_exists(
                self.collection_name
            )

            if not collection_exists:
                log_event(
                    logger,
                    level=logging.INFO,
                    event="vectorstore.qdrant.collection.creating",
                    collection_name=(self.collection_name),
                )

                self.vectorstore = QdrantVectorStore.from_documents(
                    documents=documents,
                    embedding=self.embedding,
                    sparse_embedding=(self.sparse_embedding),
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    collection_name=(self.collection_name),
                    retrieval_mode=(RetrievalMode.HYBRID),
                )

                log_event(
                    logger,
                    level=logging.INFO,
                    event="vectorstore.qdrant.collection.created",
                    collection_name=(self.collection_name),
                    document_count=len(documents),
                )

            else:
                # -------------------------------------------------
                # Existing collection
                # -------------------------------------------------

                if self.vectorstore is None:
                    self.vectorstore = QdrantVectorStore.from_existing_collection(
                        embedding=self.embedding,
                        sparse_embedding=(self.sparse_embedding),
                        collection_name=(self.collection_name),
                        url=self.qdrant_url,
                        api_key=self.qdrant_api_key,
                        retrieval_mode=(RetrievalMode.HYBRID),
                    )

                ids = []

                for index, document in enumerate(documents):
                    source = str(
                        document.metadata.get(
                            "source",
                            "unknown",
                        )
                    )

                    chunk_id = str(
                        document.metadata.get(
                            "chunk_id",
                            index,
                        )
                    )

                    document_id = f"{source}:{chunk_id}"

                    ids.append(document_id)

                self.vectorstore.add_documents(
                    documents=documents,
                    ids=ids,
                )

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="vectorstore.documents.add.completed",
                document_count=len(documents),
                collection_name=self.collection_name,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="vectorstore.documents.add.failed",
                document_count=len(documents),
                collection_name=self.collection_name,
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            logger.exception("Failed to add documents to Qdrant")

            raise

    # =========================================================
    # INITIALIZE HYBRID RETRIEVER
    # =========================================================

    def initialize(
        self,
        new_documents: List[Document] | None = None,
    ) -> None:
        """
        Initialize the persistent Qdrant hybrid retriever.

        If new_documents are supplied, they are added to Qdrant.
        """

        start_time = time.perf_counter()

        new_document_count = len(new_documents) if new_documents else 0

        log_event(
            logger,
            level=logging.INFO,
            event="vectorstore.hybrid.initialization.started",
            new_document_count=new_document_count,
            collection_name=self.collection_name,
        )

        try:
            collection_exists = self.qdrant_client.collection_exists(
                self.collection_name
            )

            # -------------------------------------------------
            # First ingestion
            # -------------------------------------------------

            if not collection_exists:
                if not new_documents:
                    raise ValueError(
                        "Qdrant collection does not exist "
                        "and no documents were supplied "
                        "for initial ingestion."
                    )

                self.add_documents(new_documents)

            # -------------------------------------------------
            # Existing collection
            # -------------------------------------------------

            elif new_documents:
                self.add_documents(new_documents)

            # -------------------------------------------------
            # Make sure vector store is connected
            # -------------------------------------------------

            if self.vectorstore is None:
                self.vectorstore = QdrantVectorStore.from_existing_collection(
                    embedding=self.embedding,
                    sparse_embedding=(self.sparse_embedding),
                    collection_name=(self.collection_name),
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    retrieval_mode=(RetrievalMode.HYBRID),
                )

            # -------------------------------------------------
            # Create hybrid retriever
            # -------------------------------------------------

            self.hybrid_retriever = self.vectorstore.as_retriever(
                search_kwargs={
                    "k": self.retrieval_k,
                }
            )

            log_event(
                logger,
                level=logging.INFO,
                event="vectorstore.hybrid_retriever.initialized",
                retrieval_k=self.retrieval_k,
                rerank_k=self.rerank_k,
                collection_name=(self.collection_name),
            )

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="vectorstore.hybrid.initialization.completed",
                document_count=(self.get_document_count()),
                new_document_count=(new_document_count),
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="vectorstore.hybrid.initialization.failed",
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            logger.exception("Failed to initialize Qdrant hybrid vector store")

            raise

    # =========================================================
    # GET RETRIEVER
    # =========================================================

    def get_retriever(
        self,
        k: int | None = None,
    ):
        """
        Return Qdrant hybrid retriever with reranking.

        Qdrant retrieves retrieval_k candidates.
        CrossEncoder returns the final k documents.
        """

        if self.hybrid_retriever is None:
            log_event(
                logger,
                level=logging.ERROR,
                event="vectorstore.retriever.creation.failed",
                reason=("hybrid_retriever_not_initialized"),
            )

            raise ValueError("Hybrid retriever not initialized.")

        final_k = k if k is not None else self.rerank_k

        log_event(
            logger,
            level=logging.DEBUG,
            event="vectorstore.retriever.created",
            retriever_type="qdrant_hybrid_reranking",
            retrieval_k=self.retrieval_k,
            rerank_k=final_k,
        )

        return RerankingRetriever(
            base_retriever=self.hybrid_retriever,
            vector_store=self,
            k=final_k,
        )

    # =========================================================
    # RERANK
    # =========================================================

    def _rerank(
        self,
        query: str,
        documents: List[Document],
        k: int,
        min_score: float = 0.0,
    ) -> List[Document]:
        """Rerank retrieved documents using CrossEncoder."""

        if self.reranker is None:
            raise RuntimeError(
                "Reranking is unavailable because the VectorStore "
                "was initialized in ingestion mode."
            )

        if not documents:
            log_event(
                logger,
                level=logging.DEBUG,
                event="retrieval.reranking.skipped",
                reason="no_documents",
                k=k,
            )

            return []

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.DEBUG,
            event="retrieval.reranking.started",
            input_document_count=len(documents),
            k=k,
            min_score=min_score,
            query_length=len(query),
        )

        try:
            pairs = [
                (
                    query,
                    document.page_content,
                )
                for document in documents
            ]

            scores = self.reranker.predict(pairs)

            ranked_documents = sorted(
                zip(
                    scores,
                    documents,
                ),
                key=lambda item: float(item[0]),
                reverse=True,
            )

            filtered = [
                document
                for score, document in ranked_documents
                if float(score) >= min_score
            ]

            result = filtered[:k]

            elapsed_seconds = time.perf_counter() - start_time
            elapsed_ms = elapsed_seconds * 1000

            top_score = float(ranked_documents[0][0]) if ranked_documents else None

            log_event(
                logger,
                level=logging.INFO,
                event="retrieval.reranking.completed",
                input_document_count=len(documents),
                output_document_count=len(result),
                filtered_document_count=len(filtered),
                k=k,
                min_score=min_score,
                top_score=(round(top_score, 4) if top_score is not None else None),
                duration_ms=round(
                    elapsed_ms,
                    2,
                ),
            )

            log_event(
                logger,
                level=logging.INFO,
                event="retrieval.reranking.results",
                results=[
                    {
                        "rank": rank,
                        "score": round(float(score), 4),
                        "source": document.metadata.get("source"),
                        "chunk_id": document.metadata.get("chunk_id"),
                        "chunk_index": document.metadata.get("chunk_index"),
                    }
                    for rank, (score, document) in enumerate(
                        ranked_documents,
                        start=1,
                    )
                ],
            )
            return result

        except Exception as exc:
            elapsed_seconds = time.perf_counter() - start_time
            elapsed_ms = elapsed_seconds * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="retrieval.reranking.failed",
                input_document_count=len(documents),
                k=k,
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed_ms,
                    2,
                ),
            )

            logger.exception("Document reranking failed")

            raise

    # =========================================================
    # RETRIEVE
    # =========================================================

    async def retrieve(
        self,
        query: str,
        k: int = 2,
    ) -> List[Document]:
        """Retrieve hybrid Qdrant results and rerank them."""

        if self.hybrid_retriever is None:
            log_event(
                logger,
                level=logging.ERROR,
                event="retrieval.failed",
                reason=("hybrid_retriever_not_initialized"),
                k=k,
            )

            raise ValueError("Hybrid retriever not initialized.")

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="retrieval.started",
            query_length=len(query),
            retrieval_k=self.retrieval_k,
            rerank_k=k,
        )

        try:
            # -------------------------------------------------
            # Qdrant hybrid retrieval
            # -------------------------------------------------

            documents = await self.hybrid_retriever.ainvoke(query)

            log_event(
                logger,
                level=logging.INFO,
                event="retrieval.qdrant.candidates",
                candidates=[
                    {
                        "rank": rank,
                        "source": document.metadata.get("source"),
                        "chunk_id": document.metadata.get("chunk_id"),
                        "chunk_index": document.metadata.get("chunk_index"),
                        "content_preview": document.page_content[:500],
                    }
                    for rank, document in enumerate(
                        documents,
                        start=1,
                    )
                ],
            )

            # -------------------------------------------------
            # CrossEncoder reranking
            # -------------------------------------------------

            result = await asyncio.to_thread(
                self._rerank,
                query=query,
                documents=documents,
                k=k,
            )

            elapsed_seconds = time.perf_counter() - start_time
            elapsed_ms = elapsed_seconds * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="retrieval.completed",
                initial_document_count=len(documents),
                final_document_count=len(result),
                retrieval_k=self.retrieval_k,
                rerank_k=k,
                duration_ms=round(
                    elapsed_ms,
                    2,
                ),
            )

            return result

        except Exception as exc:
            elapsed_seconds = time.perf_counter() - start_time
            elapsed_ms = elapsed_seconds * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="retrieval.failed",
                k=k,
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed_ms,
                    2,
                ),
            )

            logger.exception("Qdrant hybrid retrieval failed")

            raise

    # =========================================================
    # DOCUMENT COUNT
    # =========================================================

    def get_document_count(self) -> int:
        """Return the number of points stored in Qdrant."""

        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)

            count = collection_info.points_count or 0

            log_event(
                logger,
                level=logging.DEBUG,
                event="vectorstore.document_count.completed",
                count=count,
                source="qdrant",
                collection_name=(self.collection_name),
            )

            return int(count)

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="vectorstore.document_count.failed",
                error_type=type(exc).__name__,
                source="qdrant",
                collection_name=(self.collection_name),
            )

            logger.exception("Failed to retrieve Qdrant document count")

            return 0
