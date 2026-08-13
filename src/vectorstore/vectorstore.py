# """Vector store module for document embedding and retrieval"""

# from typing import List

# from langchain_community.vectorstores import FAISS
# from langchain.schema import Document
# from langchain_huggingface import HuggingFaceEmbeddings


# class VectorStore:
#     """Manages vector store operations"""

#     def __init__(self):
#         """Initialize vector store with HuggingFace embeddings"""

#         self.embedding = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2"
#         )

#         self.vectorstore = None
#         self.retriever = None

#     def create_vectorstore(self, documents: List[Document]):
#         """
#         Create vector store from documents.

#         Args:
#             documents: List of documents to embed
#         """

#         self.vectorstore = FAISS.from_documents(
#             documents,
#             self.embedding
#         )

#         self.retriever = self.vectorstore.as_retriever(
#             search_kwargs={"k": 4}
#         )

#     def get_retriever(self):
#         """
#         Get the retriever instance.

#         Returns:
#             Retriever instance
#         """

#         if self.retriever is None:
#             raise ValueError(
#                 "Vector store not initialized. "
#                 "Call create_vectorstore first."
#             )

#         return self.retriever

#     def retrieve(self, query: str, k: int = 4) -> List[Document]:
#         """
#         Retrieve relevant documents for a query.

#         Args:
#             query: Search query
#             k: Number of documents to retrieve

#         Returns:
#             List of relevant documents
#         """

#         if self.vectorstore is None:
#             raise ValueError(
#                 "Vector store not initialized. "
#                 "Call create_vectorstore first."
#             )

#         return self.vectorstore.similarity_search(
#             query,
#             k=k
#         )

########### above is basic retriver code ###############3

"""Hybrid vector store module for dense + sparse document retrieval."""

from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStore:
    """Manages hybrid vector store operations."""

    def __init__(self):
        """Initialize dense and sparse retrievers."""

        # Dense retrieval: semantic similarity
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = None
        self.dense_retriever = None
        self.sparse_retriever = None
        self.hybrid_retriever = None

    def create_vectorstore(self, documents: List[Document]):
        """
        Create dense FAISS and sparse BM25 retrievers,
        then combine them into a hybrid retriever.

        Args:
            documents: List of documents to index.
        """

        if not documents:
            raise ValueError("Documents list cannot be empty.")

        # --------------------------------------------------
        # 1. Dense Retriever - FAISS
        # --------------------------------------------------

        self.vectorstore = FAISS.from_documents(
            documents,
            self.embedding
        )

        self.dense_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        # --------------------------------------------------
        # 2. Sparse Retriever - BM25
        # --------------------------------------------------

        self.sparse_retriever = BM25Retriever.from_documents(
            documents
        )

        self.sparse_retriever.k = 4

        # --------------------------------------------------
        # 3. Hybrid Retriever
        # --------------------------------------------------

        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[
                self.dense_retriever,
                self.sparse_retriever
            ],
            weights=[0.7, 0.3]
        )

    def get_retriever(self):
        """
        Get the hybrid retriever.

        Returns:
            EnsembleRetriever: Hybrid dense + sparse retriever.
        """

        if self.hybrid_retriever is None:
            raise ValueError(
                "Hybrid retriever not initialized. "
                "Call create_vectorstore first."
            )

        return self.hybrid_retriever

    def retrieve(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Retrieve relevant documents using hybrid search.

        Args:
            query: Search query.
            k: Number of documents to retrieve.

        Returns:
            List of relevant documents.
        """

        if self.hybrid_retriever is None:
            raise ValueError(
                "Hybrid retriever not initialized. "
                "Call create_vectorstore first."
            )

        # Retrieve using hybrid search
        documents = self.hybrid_retriever.invoke(query)

        return documents[:k]
