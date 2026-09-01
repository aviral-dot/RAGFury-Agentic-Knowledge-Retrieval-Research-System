"""Semantic threshold-based document chunking."""

import logging
import time
from typing import List, Optional

import numpy as np
from langchain_core.documents import Document
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ThresholdSematicChunker:
    """Split text into semantically coherent chunks using sentence similarity."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.5,
        batch_size: int = 32,
        device: Optional[str] = None,
        max_chunk_chars: int = 1200,
        min_chunk_chars: int = 100,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be greater than 0")

        if min_chunk_chars < 0:
            raise ValueError("min_chunk_chars cannot be negative")

        if min_chunk_chars > max_chunk_chars:
            raise ValueError("min_chunk_chars cannot be greater than max_chunk_chars")

        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self.device = device
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars

        logger.info(
            "Loading semantic chunking model",
            extra={
                "model_name": model_name,
                "device": device or "auto",
            },
        )

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        logger.info(
            "Semantic chunking model loaded",
            extra={
                "model_name": self.model_name,
                "device": str(self.model.device),
            },
        )

    def split(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks."""

        if not text or not text.strip():
            return []

        start_time = time.perf_counter()

        sentences = [
            sentence.strip() for sentence in sent_tokenize(text) if sentence.strip()
        ]

        if not sentences:
            return []

        if len(sentences) == 1:
            return [sentences[0]]

        embeddings = self.model.encode(
            sentences,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Since embeddings are normalized, cosine similarity is
        # equivalent to their dot product.
        similarities = (embeddings[:-1] * embeddings[1:]).sum(axis=1)

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for index, sentence in enumerate(sentences):
            sentence_length = len(sentence)

            if not current_chunk:
                current_chunk = [sentence]
                current_length = sentence_length
                continue

            similarity = float(similarities[index - 1])

            candidate_length = current_length + 1 + sentence_length

            should_split = (
                similarity < self.threshold or candidate_length > self.max_chunk_chars
            )

            if should_split:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length = candidate_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        elapsed = time.perf_counter() - start_time

        average_chunk_size = (
            sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0.0
        )

        logger.info(
            "Semantic chunking completed",
            extra={
                "sentence_count": len(sentences),
                "chunk_count": len(chunks),
                "threshold": self.threshold,
                "batch_size": self.batch_size,
                "max_chunk_chars": self.max_chunk_chars,
                "average_chunk_chars": round(average_chunk_size, 2),
                "processing_time_seconds": round(elapsed, 3),
            },
        )

        return chunks

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Split LangChain documents while preserving and enriching metadata."""

        result: List[Document] = []

        for doc in docs:
            chunks = self.split(doc.page_content)

            for chunk_index, chunk in enumerate(chunks):
                metadata = {
                    **doc.metadata,
                    "chunk_index": chunk_index,
                    "chunking_strategy": "semantic_threshold",
                    "chunking_threshold": self.threshold,
                    "chunking_model": self.model_name,
                }

                result.append(
                    Document(
                        page_content=chunk,
                        metadata=metadata,
                    )
                )

        logger.info(
            "Documents split successfully",
            extra={
                "document_count": len(docs),
                "chunk_count": len(result),
            },
        )

        return result


###### STALE ############
# from langchain_core.documents import Document
# from nltk.tokenize import sent_tokenize
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity


# class ThresholdSematicChunker:
#     def __init__(self, model_name="all-MiniLM-L6-v2", threshold=0.6):
#         self.model = SentenceTransformer(model_name)
#         self.threshold = threshold

#     def split(self, text: str):
#         sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]

#         if not sentences:
#             return []
#         if len(sentences) == 1:
#             return sentences

#         embeddings = self.model.encode(sentences)
#         chunks = []
#         current_chunk = [sentences[0]]

#         for i in range(1, len(sentences)):
#             sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
#             if sim >= self.threshold:
#                 current_chunk.append(sentences[i])
#             else:
#                 chunks.append(" ".join(current_chunk))
#                 current_chunk = [sentences[i]]

#         chunks.append(" ".join(current_chunk))
#         return chunks

#     def split_documents(self, docs):
#         result = []
#         for doc in docs:
#             for chunk in self.split(doc.page_content):
#                 result.append(Document(page_content=chunk, metadata=doc.metadata))

#         return result
