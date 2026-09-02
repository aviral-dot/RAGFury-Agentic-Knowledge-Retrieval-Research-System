"""Document ingestion service for RAGFury."""

from pathlib import Path

from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore


class IngestionService:
    """Handles PDF processing and indexing into Qdrant."""

    def __init__(self, data_directory: Path) -> None:
        self.data_directory = data_directory

        self.document_processor = None
        self.vector_store = None

    def initialize(self) -> None:
        """Initialize ingestion components."""

        if not self.data_directory.exists():
            raise FileNotFoundError(
                f"Data directory does not exist: {self.data_directory}"
            )

        self.document_processor = DocumentProcessor(
            model_name="all-MiniLM-L6-v2",
            threshold=0.6,
        )

        self.vector_store = VectorStore(
            mode="ingestion",
        )

    def ingest_pdf(self, pdf_path: Path) -> int:
        """Process one PDF and add its chunks to Qdrant."""

        if self.document_processor is None:
            raise RuntimeError("IngestionService is not initialized.")

        if self.vector_store is None:
            raise RuntimeError("VectorStore is not initialized.")

        chunks = self.document_processor.process_pdf(pdf_path)

        if not chunks:
            return 0

        self.vector_store.add_documents(chunks)

        return len(chunks)

    def ingest_all(self) -> int:
        """Process all PDFs in the data directory."""

        total_chunks = 0

        pdf_files = sorted(self.data_directory.glob("*.pdf"))

        for pdf_file in pdf_files:
            total_chunks += self.ingest_pdf(pdf_file)

        return total_chunks
