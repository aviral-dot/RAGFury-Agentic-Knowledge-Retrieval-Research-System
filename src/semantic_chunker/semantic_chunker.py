from langchain_core.documents import Document
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class ThresholdSematicChunker:
    def __init__(self, model_name="all-MiniLM-L6-v2", threshold=0.6):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def split(self, text: str):
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]

        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        embeddings = self.model.encode(sentences)
        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            print(f"Similarity: {sim:.4f}")
            if sim >= self.threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]

        chunks.append(" ".join(current_chunk))
        return chunks

    def split_documents(self, docs):
        result = []
        for doc in docs:
            for chunk in self.split(doc.page_content):
                result.append(Document(page_content=chunk, metadata=doc.metadata))

        return result
