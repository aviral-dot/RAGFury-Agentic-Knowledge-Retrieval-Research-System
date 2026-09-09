"""Qdrant-backed hybrid vector store for production query-time retrieval.

Production architecture:

    User Query
        ↓
    Qdrant Cloud Inference
        ├── Dense embedding
        └── Sparse BM25 embedding
        ↓
    Qdrant native RRF fusion
        ↓
    Top-K documents
        ↓
    RAG pipeline

IMPORTANT:
    This module intentionally does NOT load any local ML models.

    No:
        - HuggingFaceEmbeddings
        - FastEmbedSparse
        - SentenceTransformer
        - CrossEncoder
        - PyTorch
        - Transformers

    Embedding generation is handled by Qdrant Cloud Inference.
"""

import logging
import os
import time
from typing import Any, List

from langchain_core.documents import Document
from langsmith import traceable
from pydantic import ConfigDict
from qdrant_client import QdrantClient, models

from src.config.config import Config
from src.utils.loggers import (
    configure_logging,
    get_logger,
    log_event,
)

configure_logging()

logger = get_logger(__name__)


# ============================================================
# QDRANT CLOUD INFERENCE MODELS
# ============================================================

# These are model identifiers used by Qdrant Cloud.
#
# They DO NOT install or load these models inside the Vercel
# Python runtime.

DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


# ============================================================
# PRODUCTION COLLECTION VECTOR NAMES
# ============================================================

# IMPORTANT:
#
# These names MUST match the existing Qdrant production
# collection configuration.
#
# Current production collection:
#
#   Dense vector name:
#       ""
#
#   Sparse vector name:
#       "langchain-sparse"

DEFAULT_DENSE_VECTOR_NAME = ""
DEFAULT_SPARSE_VECTOR_NAME = "langchain-sparse"


class VectorStore:
    """
    Lightweight production Qdrant Cloud vector store.

    Retrieval strategy:

        Query
          ↓
        Dense Cloud Inference
          +
        Sparse BM25 Cloud Inference
          ↓
        Qdrant native RRF
          ↓
        Final documents

    No local embedding or reranking models are loaded.

    This makes the implementation suitable for serverless
    environments such as Vercel.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        qdrant_url: str | None = None,
        collection_name: str = "ragfury_documents",
        qdrant_api_key: str | None = None,
        retrieval_k: int = 5,
        rerank_k: int = 2,
        mode: str = "query",
    ):
        """
        Initialize the Qdrant Cloud query client.

        Parameters
        ----------
        qdrant_url:
            Qdrant Cloud endpoint.

        collection_name:
            Existing production Qdrant collection.

        qdrant_api_key:
            Qdrant Cloud API key.

        retrieval_k:
            Number of candidates retrieved from each retrieval
            branch before Qdrant RRF fusion.

        rerank_k:
            Number of final documents returned after RRF.

            NOTE:
                There is NO local CrossEncoder reranker anymore.
                This value now represents the final result count.

        mode:
            Production runtime should use "query".

            Ingestion is intentionally handled separately.
        """

        if mode not in {"query", "ingestion"}:
            raise ValueError("VectorStore mode must be either 'query' or 'ingestion'.")

        self.mode = mode

        # --------------------------------------------------------
        # Configuration
        # --------------------------------------------------------

        self.qdrant_url = qdrant_url or os.getenv(
            "QDRANT_URL",
            "http://localhost:6333",
        )

        self.qdrant_api_key = qdrant_api_key or Config.QDRANT_API_KEY

        self.collection_name = collection_name

        self.retrieval_k = max(
            1,
            int(retrieval_k),
        )

        self.rerank_k = max(
            1,
            int(rerank_k),
        )

        # --------------------------------------------------------
        # Qdrant Cloud Inference model identifiers
        # --------------------------------------------------------

        self.dense_model = os.getenv(
            "QDRANT_DENSE_MODEL",
            DEFAULT_DENSE_MODEL,
        )

        self.sparse_model = os.getenv(
            "QDRANT_SPARSE_MODEL",
            DEFAULT_SPARSE_MODEL,
        )

        # --------------------------------------------------------
        # Existing collection vector names
        # --------------------------------------------------------

        self.dense_vector_name = os.getenv(
            "QDRANT_DENSE_VECTOR_NAME",
            DEFAULT_DENSE_VECTOR_NAME,
        )

        self.sparse_vector_name = os.getenv(
            "QDRANT_SPARSE_VECTOR_NAME",
            DEFAULT_SPARSE_VECTOR_NAME,
        )

        # --------------------------------------------------------
        # Runtime state
        # --------------------------------------------------------

        self.qdrant_client: QdrantClient | None = None

        # Compatibility attributes.
        #
        # These are intentionally NOT actual LangChain local
        # vector stores/retrievers/rerankers.

        self.vectorstore = None
        self.hybrid_retriever = None
        self.reranker = None

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="vectorstore.initialization.started",
            mode=self.mode,
            qdrant_url=self.qdrant_url,
            collection_name=self.collection_name,
            retrieval_k=self.retrieval_k,
            final_k=self.rerank_k,
            dense_model=self.dense_model,
            sparse_model=self.sparse_model,
            dense_vector_name=self.dense_vector_name,
            sparse_vector_name=self.sparse_vector_name,
        )

        # ========================================================
        # QDRANT CLIENT
        # ========================================================

        try:
            client_kwargs: dict[str, Any] = {
                "url": self.qdrant_url,
                "cloud_inference": True,
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
                cloud_inference=True,
            )

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="vectorstore.qdrant.client.initialization.failed",
                error_type=type(exc).__name__,
            )

            logger.exception("Failed to initialize Qdrant Cloud client")

            raise

        elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="vectorstore.initialization.completed",
            mode=self.mode,
            duration_ms=round(elapsed, 2),
        )

    # ============================================================
    # COLLECTION INITIALIZATION
    # ============================================================

    def initialize(
        self,
        new_documents: List[Document] | None = None,
    ) -> None:
        """
        Validate and initialize the existing production
        Qdrant collection.

        The deployed API is query-only.

        Document ingestion must be performed by the separate
        ingestion pipeline.
        """

        start_time = time.perf_counter()

        # --------------------------------------------------------
        # Production query runtime only
        # --------------------------------------------------------

        if self.mode == "ingestion":
            raise RuntimeError(
                "Production VectorStore is query-only. "
                "Document ingestion must be performed by "
                "the separate ingestion pipeline."
            )

        if new_documents:
            raise RuntimeError(
                "Document ingestion is not supported by the "
                "production query VectorStore. "
                "Use the dedicated ingestion pipeline."
            )

        log_event(
            logger,
            level=logging.INFO,
            event="vectorstore.hybrid.initialization.started",
            collection_name=self.collection_name,
            retrieval_k=self.retrieval_k,
            final_k=self.rerank_k,
        )

        try:
            if self.qdrant_client is None:
                raise RuntimeError("Qdrant client is not initialized.")

            # ----------------------------------------------------
            # Verify collection exists
            # ----------------------------------------------------

            collection_exists = self.qdrant_client.collection_exists(
                self.collection_name
            )

            if not collection_exists:
                raise RuntimeError(
                    f"Qdrant collection '{self.collection_name}' does not exist."
                )

            # ----------------------------------------------------
            # Verify collection contains documents
            # ----------------------------------------------------

            collection_info = self.qdrant_client.get_collection(self.collection_name)

            points_count = collection_info.points_count or 0

            if points_count <= 0:
                raise RuntimeError(
                    f"Qdrant collection '{self.collection_name}' contains no points."
                )

            # ----------------------------------------------------
            # Mark retrieval as initialized
            # ----------------------------------------------------

            self.hybrid_retriever = True

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="vectorstore.hybrid.initialization.completed",
                collection_name=self.collection_name,
                document_count=points_count,
                retrieval_k=self.retrieval_k,
                final_k=self.rerank_k,
                dense_model=self.dense_model,
                sparse_model=self.sparse_model,
                duration_ms=round(elapsed, 2),
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="vectorstore.hybrid.initialization.failed",
                collection_name=self.collection_name,
                error_type=type(exc).__name__,
                duration_ms=round(elapsed, 2),
            )

            logger.exception("Failed to initialize Qdrant production retriever")

            raise

    # ============================================================
    # GET RETRIEVER
    # ============================================================

    def get_retriever(
        self,
        k: int | None = None,
    ):
        """
        Return a lightweight LangChain-compatible retriever.

        The returned retriever delegates directly to
        VectorStore.retrieve().
        """

        if self.hybrid_retriever is None:
            raise ValueError(
                "Hybrid retriever not initialized. Call initialize() first."
            )

        final_k = max(
            1,
            int(k if k is not None else self.rerank_k),
        )

        log_event(
            logger,
            level=logging.DEBUG,
            event="vectorstore.retriever.created",
            retriever_type="qdrant_cloud_hybrid",
            retrieval_k=self.retrieval_k,
            final_k=final_k,
        )

        return QdrantCloudRetriever(
            vector_store=self,
            k=final_k,
        )

    # ============================================================
    # QUERY / HYBRID RETRIEVAL
    # ============================================================

    @traceable(
        name="RAGFury Qdrant Cloud Hybrid Retrieval",
        run_type="retriever",
    )
    async def retrieve(
        self,
        query: str,
        k: int = 2,
    ) -> List[Document]:
        """
        Retrieve documents using Qdrant Cloud Inference.

        Pipeline:

            query
              ↓
            Dense Cloud Inference
              +
            Sparse BM25 Cloud Inference
              ↓
            Qdrant native RRF
              ↓
            top-k documents
        """

        if self.qdrant_client is None:
            raise RuntimeError("Qdrant client is not initialized.")

        if not query or not query.strip():
            return []

        final_k = max(
            1,
            int(k),
        )

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="retrieval.started",
            query_length=len(query),
            retrieval_k=self.retrieval_k,
            final_k=final_k,
        )

        try:
            # ====================================================
            # CLOUD INFERENCE QUERIES
            # ====================================================

            # Dense embedding generated by Qdrant Cloud.
            dense_query = models.Document(
                text=query,
                model=self.dense_model,
            )

            # Sparse BM25 representation generated by
            # Qdrant Cloud.
            sparse_query = models.Document(
                text=query,
                model=self.sparse_model,
            )

            # ====================================================
            # QDRANT HYBRID RETRIEVAL
            # ====================================================

            response = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    # ----------------------------------------
                    # Dense retrieval
                    # ----------------------------------------
                    models.Prefetch(
                        query=dense_query,
                        using=self.dense_vector_name,
                        limit=self.retrieval_k,
                    ),
                    # ----------------------------------------
                    # Sparse BM25 retrieval
                    # ----------------------------------------
                    models.Prefetch(
                        query=sparse_query,
                        using=self.sparse_vector_name,
                        limit=self.retrieval_k,
                    ),
                ],
                # --------------------------------------------
                # Native Qdrant Reciprocal Rank Fusion
                # --------------------------------------------
                query=models.FusionQuery(
                    fusion=models.Fusion.RRF,
                ),
                # --------------------------------------------
                # Final result count
                # --------------------------------------------
                limit=final_k,
                # --------------------------------------------
                # Return document payload
                # --------------------------------------------
                with_payload=True,
            )

            points = response.points

            # ====================================================
            # CONVERT QDRANT POINTS → LANGCHAIN DOCUMENTS
            # ====================================================

            documents: list[Document] = []

            for rank, point in enumerate(
                points,
                start=1,
            ):
                payload = point.payload or {}

                # -----------------------------------------------
                # Page content
                # -----------------------------------------------

                page_content = payload.get(
                    "page_content",
                    "",
                )

                # -----------------------------------------------
                # Metadata
                # -----------------------------------------------

                metadata = payload.get(
                    "metadata",
                    {},
                )

                if not isinstance(
                    metadata,
                    dict,
                ):
                    metadata = {}

                metadata = dict(metadata)

                # -----------------------------------------------
                # Preserve Qdrant information
                # -----------------------------------------------

                metadata.setdefault(
                    "qdrant_point_id",
                    str(point.id),
                )

                metadata.setdefault(
                    "qdrant_score",
                    float(point.score),
                )

                metadata.setdefault(
                    "retrieval_rank",
                    rank,
                )

                documents.append(
                    Document(
                        page_content=str(page_content),
                        metadata=metadata,
                    )
                )

            # ====================================================
            # LOG RETRIEVAL RESULTS
            # ====================================================

            log_event(
                logger,
                level=logging.INFO,
                event="retrieval.qdrant.completed",
                candidate_count=len(documents),
                retrieval_k=self.retrieval_k,
                final_k=final_k,
            )

            log_event(
                logger,
                level=logging.DEBUG,
                event="retrieval.qdrant.results",
                results=[
                    {
                        "rank": rank,
                        "score": round(
                            float(point.score),
                            6,
                        ),
                        "source": (
                            point.payload.get("metadata", {}).get("source")
                            if isinstance(
                                point.payload,
                                dict,
                            )
                            else None
                        ),
                        "chunk_id": (
                            point.payload.get("metadata", {}).get("chunk_id")
                            if isinstance(
                                point.payload,
                                dict,
                            )
                            else None
                        ),
                    }
                    for rank, point in enumerate(
                        points,
                        start=1,
                    )
                ],
            )

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="retrieval.completed",
                initial_document_count=len(documents),
                final_document_count=len(documents),
                retrieval_k=self.retrieval_k,
                final_k=final_k,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            return documents

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="retrieval.failed",
                error_type=type(exc).__name__,
                retrieval_k=self.retrieval_k,
                final_k=final_k,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            logger.exception("Qdrant Cloud hybrid retrieval failed")

            raise

    # ============================================================
    # DOCUMENT COUNT
    # ============================================================

    def get_document_count(self) -> int:
        """Return the number of points stored in Qdrant."""

        if self.qdrant_client is None:
            raise RuntimeError("Qdrant client is not initialized.")

        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)

            count = collection_info.points_count or 0

            log_event(
                logger,
                level=logging.DEBUG,
                event="vectorstore.document_count.completed",
                count=count,
                source="qdrant",
                collection_name=self.collection_name,
            )

            return int(count)

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="vectorstore.document_count.failed",
                error_type=type(exc).__name__,
                source="qdrant",
                collection_name=self.collection_name,
            )

            logger.exception("Failed to retrieve Qdrant document count")

            raise


# ================================================================
# LANGCHAIN-COMPATIBLE RETRIEVER
# ================================================================


class QdrantCloudRetriever:
    """
    Lightweight LangChain-compatible async retriever.

    This replaces the old:

        BaseRetriever
        +
        local CrossEncoder

    implementation.

    Retrieval is performed entirely through:

        Qdrant Cloud Inference
        +
        Qdrant native RRF
    """

    def __init__(
        self,
        vector_store: VectorStore,
        k: int = 2,
    ):
        self.vector_store = vector_store

        self.k = max(
            1,
            int(k),
        )

    async def ainvoke(
        self,
        query: str,
        **_: Any,
    ) -> List[Document]:
        """LangChain-compatible async retrieval."""

        return await self.vector_store.retrieve(
            query=query,
            k=self.k,
        )
