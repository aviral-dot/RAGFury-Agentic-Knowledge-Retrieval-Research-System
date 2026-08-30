"""Hybrid vector store module for incremental dense + sparse retrieval."""

import json
import logging
import time
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import ConfigDict
from sentence_transformers import CrossEncoder

from src.utils.loggers import (
    configure_logging,
    get_logger,
    log_event,
)

configure_logging()

logger = get_logger(__name__)


class RerankingRetriever(BaseRetriever):
    """Wraps the hybrid retriever so invoke() returns final reranked top-k docs."""

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
            docs = self.base_retriever.invoke(query)

            result = self.vector_store._rerank(
                query=query,
                documents=docs,
                k=self.k,
            )

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.DEBUG,
                event="retrieval.reranking_retriever.completed",
                input_document_count=len(docs),
                output_document_count=len(result),
                k=self.k,
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
    Manages persistent hybrid retrieval using:

    Dense:
        Chroma

    Sparse:
        BM25

    Supports incremental document ingestion.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        bm25_path: str = "./storage/bm25_documents.json",
    ):
        """Initialize vector store."""

        start_time = time.perf_counter()

        self.persist_directory = Path(persist_directory)

        self.bm25_path = Path(bm25_path)

        log_event(
            logger,
            level=logging.INFO,
            event="vectorstore.initialization.started",
            persist_directory=str(self.persist_directory),
            bm25_path=str(self.bm25_path),
        )

        # -----------------------------------------------------
        # Make sure storage directory exists
        # -----------------------------------------------------

        self.bm25_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -----------------------------------------------------
        # Embedding model
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

            logger.exception("Failed to initialize embedding model")

            raise

        # -----------------------------------------------------
        # Reranker
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

        self.vectorstore: Chroma | None = None

        self.dense_retriever = None
        self.sparse_retriever = None
        self.hybrid_retriever = None

        # Documents used by BM25
        self.bm25_documents: List[Document] = []

        elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="vectorstore.initialization.completed",
            duration_ms=round(
                elapsed,
                2,
            ),
        )

    # =========================================================
    # BM25 LOAD
    # =========================================================

    def _load_bm25_documents(self) -> List[Document]:
        """Load previously indexed BM25 documents."""

        start_time = time.perf_counter()

        if not self.bm25_path.exists():
            log_event(
                logger,
                level=logging.DEBUG,
                event="vectorstore.bm25.load.skipped",
                reason="file_not_found",
                path=str(self.bm25_path),
            )

            return []

        try:
            with open(
                self.bm25_path,
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

        except (
            json.JSONDecodeError,
            OSError,
        ) as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="vectorstore.bm25.load.failed",
                path=str(self.bm25_path),
                error_type=type(exc).__name__,
            )

            logger.exception("Failed to load BM25 document store")

            raise RuntimeError(
                f"Could not load BM25 document store: {self.bm25_path}"
            ) from exc

        documents = []

        for item in data:
            documents.append(
                Document(
                    page_content=item["page_content"],
                    metadata=item.get(
                        "metadata",
                        {},
                    ),
                )
            )

        elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="vectorstore.bm25.load.completed",
            document_count=len(documents),
            path=str(self.bm25_path),
            duration_ms=round(
                elapsed,
                2,
            ),
        )

        return documents

    # =========================================================
    # BM25 SAVE
    # =========================================================

    def _save_bm25_documents(
        self,
        documents: List[Document],
    ) -> None:
        """Persist BM25 document corpus to disk."""

        start_time = time.perf_counter()

        data = []

        for document in documents:
            data.append(
                {
                    "page_content": (document.page_content),
                    "metadata": (document.metadata),
                }
            )

        try:
            with open(
                self.bm25_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    data,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.DEBUG,
                event="vectorstore.bm25.save.completed",
                document_count=len(documents),
                path=str(self.bm25_path),
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="vectorstore.bm25.save.failed",
                document_count=len(documents),
                path=str(self.bm25_path),
                error_type=type(exc).__name__,
            )

            logger.exception("Failed to save BM25 document store")

            raise

    # =========================================================
    # CHROMA INITIALIZATION
    # =========================================================

    def _initialize_chroma(self) -> None:
        """Open the existing persistent Chroma store."""

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.DEBUG,
            event="vectorstore.chroma.initialization.started",
            collection_name="ragfury_documents",
            persist_directory=str(self.persist_directory),
        )

        try:
            self.vectorstore = Chroma(
                collection_name="ragfury_documents",
                embedding_function=self.embedding,
                persist_directory=str(self.persist_directory),
            )

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="vectorstore.chroma.initialization.completed",
                collection_name="ragfury_documents",
                persist_directory=str(self.persist_directory),
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="vectorstore.chroma.initialization.failed",
                collection_name="ragfury_documents",
                persist_directory=str(self.persist_directory),
                error_type=type(exc).__name__,
            )

            logger.exception("Failed to initialize Chroma vector store")

            raise

    # =========================================================
    # ADD DOCUMENTS
    # =========================================================

    def add_documents(
        self,
        documents: List[Document],
    ) -> None:
        """
        Add new document chunks incrementally.

        Existing documents already stored in Chroma
        are not re-embedded.
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
        )

        try:
            if self.vectorstore is None:
                log_event(
                    logger,
                    level=logging.DEBUG,
                    event="vectorstore.chroma.lazy_initialization",
                )

                self._initialize_chroma()

            ids = []

            for index, document in enumerate(documents):
                source = document.metadata.get(
                    "source",
                    "unknown",
                )

                chunk_id = document.metadata.get(
                    "chunk_id",
                    index,
                )

                document_id = f"{source}:{chunk_id}"

                ids.append(document_id)

            self.vectorstore.add_documents(
                documents=documents,
                ids=ids,
            )

            log_event(
                logger,
                level=logging.DEBUG,
                event="vectorstore.chroma.documents.added",
                document_count=len(documents),
            )

            self.bm25_documents.extend(documents)

            self._save_bm25_documents(self.bm25_documents)

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="vectorstore.documents.add.completed",
                document_count=len(documents),
                total_bm25_document_count=len(self.bm25_documents),
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
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            logger.exception("Failed to add documents to vector store")

            raise

    # =========================================================
    # INITIALIZE HYBRID RETRIEVER
    # =========================================================

    def initialize(
        self,
        new_documents: List[Document] | None = None,
    ) -> None:
        """
        Initialize the persistent hybrid vector store.

        If new_documents are supplied, only those documents
        are added to the existing store.
        """

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="vectorstore.hybrid.initialization.started",
            new_document_count=(len(new_documents) if new_documents else 0),
        )

        try:
            self.bm25_documents = self._load_bm25_documents()

            self._initialize_chroma()

            if new_documents:
                self.add_documents(new_documents)

            self.dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

            log_event(
                logger,
                level=logging.DEBUG,
                event="vectorstore.dense_retriever.initialized",
                k=5,
            )

            if not self.bm25_documents:
                log_event(
                    logger,
                    level=logging.ERROR,
                    event="vectorstore.bm25.initialization.failed",
                    reason="no_documents_available",
                )

                raise ValueError("No documents available for BM25.")

            self.sparse_retriever = BM25Retriever.from_documents(self.bm25_documents)

            self.sparse_retriever.k = 5

            log_event(
                logger,
                level=logging.DEBUG,
                event="vectorstore.sparse_retriever.initialized",
                k=5,
                document_count=len(self.bm25_documents),
            )

            self.hybrid_retriever = EnsembleRetriever(
                retrievers=[
                    self.dense_retriever,
                    self.sparse_retriever,
                ],
                weights=[
                    0.7,
                    0.3,
                ],
            )

            log_event(
                logger,
                level=logging.INFO,
                event="vectorstore.hybrid_retriever.initialized",
                dense_weight=0.7,
                sparse_weight=0.3,
            )

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="vectorstore.hybrid.initialization.completed",
                document_count=len(self.bm25_documents),
                new_document_count=(len(new_documents) if new_documents else 0),
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

            logger.exception("Failed to initialize hybrid vector store")

            raise

    # =========================================================
    # GET RETRIEVER
    # =========================================================

    def get_retriever(self, k: int = 2):
        """Return hybrid retriever with reranking applied on invoke."""

        if self.hybrid_retriever is None:
            log_event(
                logger,
                level=logging.ERROR,
                event="vectorstore.retriever.creation.failed",
                reason="hybrid_retriever_not_initialized",
            )

            raise ValueError("Hybrid retriever not initialized.")

        log_event(
            logger,
            level=logging.DEBUG,
            event="vectorstore.retriever.created",
            retriever_type="hybrid_reranking",
            k=k,
        )

        return RerankingRetriever(
            base_retriever=self.hybrid_retriever,
            vector_store=self,
            k=k,
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
        """Rerank retrieved documents using a cross-encoder."""

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
                key=lambda item: item[0],
                reverse=True,
            )

            filtered = [
                document for score, document in ranked_documents if score >= min_score
            ]

            result = filtered[:k]

            elapsed = (time.perf_counter() - start_time) * 1000

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
                    elapsed,
                    2,
                ),
            )

            log_event(
                logger,
                level=logging.DEBUG,
                event="retrieval.reranking.scores",
                scores=[
                    round(
                        float(score),
                        4,
                    )
                    for score, _ in ranked_documents
                ],
            )

            return result

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="retrieval.reranking.failed",
                input_document_count=len(documents),
                k=k,
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            logger.exception("Document reranking failed")

            raise

    # =========================================================
    # RETRIEVE
    # =========================================================

    def retrieve(
        self,
        query: str,
        k: int = 2,
    ) -> List[Document]:
        """Retrieve and rerank documents."""

        if self.hybrid_retriever is None:
            log_event(
                logger,
                level=logging.ERROR,
                event="retrieval.failed",
                reason="hybrid_retriever_not_initialized",
                k=k,
            )

            raise ValueError("Hybrid retriever not initialized.")

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="retrieval.started",
            query_length=len(query),
            k=k,
        )

        try:
            documents = self.hybrid_retriever.invoke(query)

            result = self._rerank(
                query=query,
                documents=documents,
                k=k,
            )

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="retrieval.completed",
                initial_document_count=len(documents),
                final_document_count=len(result),
                k=k,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            return result

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="retrieval.failed",
                k=k,
                error_type=type(exc).__name__,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            logger.exception("Hybrid retrieval failed")

            raise

    # =========================================================
    # DOCUMENT COUNT
    # =========================================================

    def get_document_count(self) -> int:
        """Return number of documents stored in Chroma."""

        if self.vectorstore is None:
            log_event(
                logger,
                level=logging.DEBUG,
                event="vectorstore.document_count.unavailable",
                reason="chroma_not_initialized",
            )

            return 0

        try:
            count = self.vectorstore._collection.count()

            log_event(
                logger,
                level=logging.DEBUG,
                event="vectorstore.document_count.completed",
                count=count,
                source="chroma",
            )

            return count

        except Exception as exc:
            fallback_count = len(self.bm25_documents)

            log_event(
                logger,
                level=logging.WARNING,
                event="vectorstore.document_count.fallback",
                error_type=type(exc).__name__,
                fallback_count=fallback_count,
                source="bm25",
            )

            logger.exception(
                "Failed to retrieve Chroma document count; using BM25 document count"
            )

            return fallback_count
