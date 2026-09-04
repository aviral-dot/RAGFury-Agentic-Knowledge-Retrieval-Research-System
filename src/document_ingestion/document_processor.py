"""Document processing module for loading and splitting documents."""

import logging
import time
from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document

from src.semantic_chunker.semantic_chunker import ThresholdSematicChunker
from src.utils.loggers import get_logger, log_event
from src.vectorstore.documents_ids import build_chunk_id

logger = get_logger(__name__)


class DocumentProcessor:
    """Handles document loading and semantic chunking."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.6,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold

        log_event(
            logger,
            level=logging.INFO,
            event="document_processor.initializing",
            model_name=model_name,
            threshold=threshold,
        )

        self.splitter = ThresholdSematicChunker(
            model_name=model_name,
            threshold=threshold,
        )

        log_event(
            logger,
            level=logging.INFO,
            event="document_processor.initialized",
            model_name=model_name,
            threshold=threshold,
        )

    # =========================================================
    # URL
    # =========================================================

    def load_from_url(
        self,
        url: str,
    ) -> List[Document]:
        """Load document(s) from a URL."""

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="document.load.url.started",
            url=url,
        )

        try:
            loader = WebBaseLoader(url)

            documents = loader.load()

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="document.load.url.completed",
                url=url,
                document_count=len(documents),
                duration_ms=round(elapsed, 2),
            )

            return documents

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="document.load.url.failed",
                url=url,
                error_type=type(exc).__name__,
            )

            logger.exception(
                "Failed to load documents from URL.",
            )

            raise

    # =========================================================
    # PDF DIRECTORY
    # =========================================================

    def load_from_pdf_dir(
        self,
        directory: Union[str, Path],
    ) -> List[Document]:
        """Load all PDFs from a directory."""

        directory = Path(directory)

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="document.load.pdf_directory.started",
            directory=str(directory),
        )

        if not directory.exists():
            raise FileNotFoundError(f"PDF directory not found: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"Expected directory: {directory}")

        try:
            loader = PyPDFDirectoryLoader(str(directory))

            documents = loader.load()

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="document.load.pdf_directory.completed",
                directory=str(directory),
                document_count=len(documents),
                duration_ms=round(elapsed, 2),
            )

            return documents

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="document.load.pdf_directory.failed",
                directory=str(directory),
                error_type=type(exc).__name__,
            )

            logger.exception(
                "Failed to load PDFs from directory.",
            )

            raise

    # =========================================================
    # TXT
    # =========================================================

    def load_from_txt(
        self,
        file_path: Union[str, Path],
    ) -> List[Document]:
        """Load a TXT file."""

        file_path = Path(file_path)

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="document.load.txt.started",
            file_name=file_path.name,
            file_path=str(file_path),
        )

        if not file_path.exists():
            raise FileNotFoundError(f"TXT file not found: {file_path}")

        try:
            loader = TextLoader(
                str(file_path),
                encoding="utf-8",
            )

            documents = loader.load()

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="document.load.txt.completed",
                file_name=file_path.name,
                document_count=len(documents),
                duration_ms=round(elapsed, 2),
            )

            return documents

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="document.load.txt.failed",
                file_name=file_path.name,
                error_type=type(exc).__name__,
            )

            logger.exception(
                "Failed to load TXT file.",
            )

            raise

    # =========================================================
    # SINGLE PDF
    # =========================================================

    def load_from_pdf(
        self,
        file_path: Union[str, Path],
    ) -> List[Document]:
        """Load a single PDF file."""

        file_path = Path(file_path)

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="document.load.pdf.started",
            file_name=file_path.name,
            file_path=str(file_path),
        )

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected PDF file, got: {file_path}")

        try:
            loader = PyPDFLoader(str(file_path))

            documents = loader.load()

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="document.load.pdf.completed",
                file_name=file_path.name,
                page_count=len(documents),
                duration_ms=round(elapsed, 2),
            )

            return documents

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="document.load.pdf.failed",
                file_name=file_path.name,
                error_type=type(exc).__name__,
            )

            logger.exception(
                "Failed to load PDF file.",
            )

            raise

    # =========================================================
    # GENERIC DOCUMENT LOADER
    # =========================================================

    def load_documents(
        self,
        source: Union[str, Path],
    ) -> List[Document]:
        """
        Load documents from a URL, PDF directory,
        PDF file, or TXT file.
        """

        log_event(
            logger,
            level=logging.DEBUG,
            event="document.load.started",
            source=str(source),
        )

        if isinstance(source, str) and (
            source.startswith("http://") or source.startswith("https://")
        ):
            return self.load_from_url(source)

        path = Path(source)

        if path.is_dir():
            return self.load_from_pdf_dir(path)

        if path.suffix.lower() == ".pdf":
            return self.load_from_pdf(path)

        if path.suffix.lower() == ".txt":
            return self.load_from_txt(path)

        log_event(
            logger,
            level=logging.ERROR,
            event="document.load.unsupported_source",
            source=str(source),
        )

        raise ValueError(
            f"Unsupported source type: {source}. "
            "Use a URL, PDF file, PDF directory, or TXT file."
        )

    # =========================================================
    # SEMANTIC CHUNKING
    # =========================================================

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """Split documents into semantic chunks."""

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="document.chunking.started",
            document_count=len(documents),
            model_name=self.model_name,
            threshold=self.threshold,
        )

        try:
            chunks = self.splitter.split_documents(documents)

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="document.chunking.completed",
                document_count=len(documents),
                chunk_count=len(chunks),
                model_name=self.model_name,
                threshold=self.threshold,
                duration_ms=round(elapsed, 2),
            )

            return chunks

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="document.chunking.failed",
                document_count=len(documents),
                error_type=type(exc).__name__,
            )

            logger.exception(
                "Failed to split documents into semantic chunks.",
            )

            raise

    # =========================================================
    # SINGLE PDF PROCESSING
    # =========================================================

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

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="document.processing.started",
            file_name=file_path.name,
            file_path=str(file_path),
        )

        # -----------------------------------------------------
        # Validate file
        # -----------------------------------------------------

        if not file_path.exists():
            log_event(
                logger,
                level=logging.ERROR,
                event="document.processing.file_not_found",
                file_name=file_path.name,
                file_path=str(file_path),
            )

            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            log_event(
                logger,
                level=logging.ERROR,
                event="document.processing.invalid_file_type",
                file_name=file_path.name,
                suffix=file_path.suffix,
            )

            raise ValueError(f"Expected PDF file, got: {file_path}")

        try:
            # -------------------------------------------------
            # Load only this PDF
            # -------------------------------------------------

            documents = self.load_from_pdf(file_path)

            # -------------------------------------------------
            # Semantic chunk only this PDF
            # -------------------------------------------------

            chunks = self.split_documents(documents)

            # -------------------------------------------------
            # Normalize source metadata
            # -------------------------------------------------

            for index, chunk in enumerate(chunks):
                source = file_path.name

                chunk.metadata["source"] = source
                chunk.metadata["chunk_id"] = build_chunk_id(
                    source=source,
                    chunk_index=index,
                )

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="document.processing.completed",
                file_name=file_path.name,
                page_count=len(documents),
                chunk_count=len(chunks),
                duration_ms=round(elapsed, 2),
            )

            return chunks

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="document.processing.failed",
                file_name=file_path.name,
                error_type=type(exc).__name__,
            )

            logger.exception(
                "Failed to process PDF.",
            )

            raise

    # =========================================================
    # MULTIPLE PDF PROCESSING
    # =========================================================

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

        directory = Path(directory)

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="documents.processing_batch.started",
            directory=str(directory),
        )

        try:
            docs = self.load_documents(directory)

            chunks = self.split_documents(docs)

            # Keep source metadata consistent for compatibility.
            for chunk in chunks:
                source = chunk.metadata.get("source")

                if source:
                    chunk.metadata["source"] = Path(source).name

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="documents.processing_batch.completed",
                directory=str(directory),
                document_count=len(docs),
                chunk_count=len(chunks),
                duration_ms=round(elapsed, 2),
            )

            return chunks

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="documents.processing_batch.failed",
                directory=str(directory),
                error_type=type(exc).__name__,
            )

            logger.exception(
                "Failed to process PDF directory.",
            )

            raise
