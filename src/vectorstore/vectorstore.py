"""Hybrid vector store module for incremental dense + sparse retrieval."""

import json
from pathlib import Path
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


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

        self.persist_directory = Path(
            persist_directory
        )

        self.bm25_path = Path(
            bm25_path
        )

        # Make sure storage directory exists
        self.bm25_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Embedding model
        self.embedding = HuggingFaceEmbeddings(
            model_name=(
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            )
        )

        self.vectorstore: Chroma | None = None

        self.dense_retriever = None
        self.sparse_retriever = None
        self.hybrid_retriever = None

        # Documents used by BM25
        self.bm25_documents: List[Document] = []

   

    def _load_bm25_documents(self) -> List[Document]:
        """Load previously indexed BM25 documents."""

        if not self.bm25_path.exists():
            return []

        try:
            with open(
                self.bm25_path,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

        except (json.JSONDecodeError, OSError) as exc:

            raise RuntimeError(
                f"Could not load BM25 document store: "
                f"{self.bm25_path}"
            ) from exc

        documents = []

        for item in data:

            documents.append(
                Document(
                    page_content=item[
                        "page_content"
                    ],
                    metadata=item.get(
                        "metadata",
                        {},
                    ),
                )
            )

        return documents

    

    def _save_bm25_documents(
        self,
        documents: List[Document],
    ) -> None:
        """Persist BM25 document corpus to disk."""

        data = []

        for document in documents:

            data.append(
                {
                    "page_content": (
                        document.page_content
                    ),
                    "metadata": (
                        document.metadata
                    ),
                }
            )

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

    

    def _initialize_chroma(self) -> None:
        """Open the existing persistent Chroma store."""

        self.vectorstore = Chroma(
            collection_name="ragfury_documents",
            embedding_function=self.embedding,
            persist_directory=str(
                self.persist_directory
            ),
        )

   
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
            return

        if self.vectorstore is None:
            self._initialize_chroma()

        print(
            f"➕ Adding {len(documents)} new chunks "
            f"to Chroma..."
        )

        
        ids = []

        for index, document in enumerate(
            documents
        ):

            source = document.metadata.get(
                "source",
                "unknown",
            )

            chunk_id = document.metadata.get(
                "chunk_id",
                index,
            )

            document_id = (
                f"{source}:{chunk_id}"
            )

            ids.append(document_id)

        

        self.vectorstore.add_documents(
            documents=documents,
            ids=ids,
        )

        

        self.bm25_documents.extend(
            documents
        )

        self._save_bm25_documents(
            self.bm25_documents
        )

        print(
            "✅ New documents added successfully."
        )

    

    def initialize(
        self,
        new_documents: List[Document] | None = None,
    ) -> None:
        """
        Initialize the persistent hybrid vector store.

        If new_documents are supplied, only those documents
        are added to the existing store.
        """

        

        self.bm25_documents = (
            self._load_bm25_documents()
        )

        

        self._initialize_chroma()

        

        if new_documents:

            self.add_documents(
                new_documents
            )

       

        self.dense_retriever = (
            self.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )
        )

        

        if not self.bm25_documents:

            raise ValueError(
                "No documents available for BM25."
            )

        self.sparse_retriever = (
            BM25Retriever.from_documents(
                self.bm25_documents
            )
        )

        self.sparse_retriever.k = 4

        

        self.hybrid_retriever = (
            EnsembleRetriever(
                retrievers=[
                    self.dense_retriever,
                    self.sparse_retriever,
                ],
                weights=[
                    0.7,
                    0.3,
                ],
            )
        )

    

    def get_retriever(self):
        """Return hybrid retriever."""

        if self.hybrid_retriever is None:

            raise ValueError(
                "Hybrid retriever not initialized."
            )

        return self.hybrid_retriever

    

    def retrieve(
        self,
        query: str,
        k: int = 4,
    ) -> List[Document]:
        """Retrieve documents using hybrid search."""

        if self.hybrid_retriever is None:

            raise ValueError(
                "Hybrid retriever not initialized."
            )

        documents = (
            self.hybrid_retriever.invoke(
                query
            )
        )

        return documents[:k]

    

    def get_document_count(self) -> int:
        """Return number of documents stored in Chroma."""

        if self.vectorstore is None:
            return 0

        try:
            return self.vectorstore._collection.count()

        except Exception:
            return len(
                self.bm25_documents
            )