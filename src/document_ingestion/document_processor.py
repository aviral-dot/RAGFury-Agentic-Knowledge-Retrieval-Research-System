"""Document processing module for loading and splitting documents."""

from typing import List, Union
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader,
)

from src.semantic_chunker.semantic_chunker import ThresholdSematicChunker


class DocumentProcessor:
    """Handles document loading and processing."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.6,
    ):
        self.model_name = model_name
        self.threshold = threshold

        self.splitter = ThresholdSematicChunker(
            model_name=model_name,
            threshold=threshold,
        )

   

    def load_from_url(
        self,
        url: str,
    ) -> List[Document]:
        """Load document(s) from a URL."""

        loader = WebBaseLoader(url)

        return loader.load()

    def load_from_pdf_dir(
        self,
        directory: Union[str, Path],
    ) -> List[Document]:
        """Load all PDFs from a directory."""

        loader = PyPDFDirectoryLoader(
            str(directory)
        )

        return loader.load()

    def load_from_txt(
        self,
        file_path: Union[str, Path],
    ) -> List[Document]:
        """Load a TXT file."""

        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
        )

        return loader.load()

    def load_from_pdf(
        self,
        file_path: Union[str, Path],
    ) -> List[Document]:
        """Load a single PDF file."""

        loader = PyPDFLoader(
            str(file_path)
        )

        return loader.load()

    

    def load_documents(
        self,
        source: Union[str, Path],
    ) -> List[Document]:
        """
        Load documents from a URL, PDF directory,
        PDF file, or TXT file.
        """

        docs: List[Document] = []

        if isinstance(source, str) and (
            source.startswith("http://")
            or source.startswith("https://")
        ):
            docs.extend(
                self.load_from_url(source)
            )

            return docs

        path = Path(source)

        if path.is_dir():
            docs.extend(
                self.load_from_pdf_dir(path)
            )

            return docs

        if path.suffix.lower() == ".pdf":
            docs.extend(
                self.load_from_pdf(path)
            )

            return docs

        if path.suffix.lower() == ".txt":
            docs.extend(
                self.load_from_txt(path)
            )

            return docs

        raise ValueError(
            f"Unsupported source type: {source}. "
            "Use a URL, PDF file, PDF directory, or TXT file."
        )

    

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """Split documents into semantic chunks."""

        return self.splitter.split_documents(
            documents
        )

    
    def process_pdf(
        self,
        file_path: Union[str, Path],
    ) -> List[Document]:
        """
        Process a single PDF file.

        This method is intentionally used for incremental
        ingestion so already-processed PDFs are not touched.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {file_path}"
            )

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected PDF file, got: {file_path}"
            )

        print(
            f"📄 Processing new PDF: "
            f"{file_path.name}"
        )

        # Load only this PDF
        docs = self.load_from_pdf(
            file_path
        )

        # Semantic chunk only this PDF
        chunks = self.split_documents(
            docs
        )

        print(
            f"📊 Created {len(chunks)} chunks "
            f"from {file_path.name}"
        )

        # Make source metadata consistent
        for chunk in chunks:
            chunk.metadata["source"] = (
                file_path.name
            )

        return chunks

    

    def process_pdfs(
        self,
        directory: Union[str, Path],
    ) -> List[Document]:
        """
        Process all PDFs in a directory.

        NOTE:
        This method is retained for compatibility.

        The FastAPI application should NOT use this method
        during startup anymore. Incremental ingestion should
        use process_pdf() instead.
        """

        docs = self.load_documents(
            directory
        )

        return self.split_documents(
            docs
        )


