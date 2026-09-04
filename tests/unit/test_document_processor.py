from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.documents_ids import build_chunk_id


@pytest.fixture
def processor(monkeypatch):
    """Create DocumentProcessor without loading the real embedding model."""

    fake_splitter = MagicMock()

    monkeypatch.setattr(
        "src.document_ingestion.document_processor.ThresholdSematicChunker",
        MagicMock(return_value=fake_splitter),
    )

    processor = DocumentProcessor(
        model_name="test-model",
        threshold=0.6,
    )

    processor.splitter = fake_splitter

    return processor


# ============================================================
# INITIALIZATION
# ============================================================


def test_processor_initializes_with_configuration(monkeypatch):
    fake_splitter = MagicMock()

    splitter_constructor = MagicMock(return_value=fake_splitter)

    monkeypatch.setattr(
        "src.document_ingestion.document_processor.ThresholdSematicChunker",
        splitter_constructor,
    )

    processor = DocumentProcessor(
        model_name="custom-model",
        threshold=0.7,
    )

    assert processor.model_name == "custom-model"
    assert processor.threshold == 0.7
    assert processor.splitter is fake_splitter

    splitter_constructor.assert_called_once_with(
        model_name="custom-model",
        threshold=0.7,
    )


# ============================================================
# URL LOADING
# ============================================================


def test_load_from_url_returns_documents(monkeypatch, processor):
    documents = [
        Document(page_content="Document from URL"),
    ]

    loader = MagicMock()
    loader.load.return_value = documents

    loader_constructor = MagicMock(return_value=loader)

    monkeypatch.setattr(
        "src.document_ingestion.document_processor.WebBaseLoader",
        loader_constructor,
    )

    result = processor.load_from_url(
        "https://example.com/document",
    )

    assert result == documents

    loader_constructor.assert_called_once_with(
        "https://example.com/document",
    )

    loader.load.assert_called_once_with()


def test_load_from_url_propagates_loader_error(monkeypatch, processor):
    loader = MagicMock()
    loader.load.side_effect = RuntimeError("network failure")

    monkeypatch.setattr(
        "src.document_ingestion.document_processor.WebBaseLoader",
        MagicMock(return_value=loader),
    )

    with pytest.raises(RuntimeError, match="network failure"):
        processor.load_from_url(
            "https://example.com/document",
        )


# ============================================================
# PDF DIRECTORY LOADING
# ============================================================


def test_load_from_pdf_dir_returns_documents(monkeypatch, tmp_path, processor):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    documents = [
        Document(page_content="PDF page 1"),
        Document(page_content="PDF page 2"),
    ]

    loader = MagicMock()
    loader.load.return_value = documents

    loader_constructor = MagicMock(return_value=loader)

    monkeypatch.setattr(
        "src.document_ingestion.document_processor.PyPDFDirectoryLoader",
        loader_constructor,
    )

    result = processor.load_from_pdf_dir(pdf_dir)

    assert result == documents

    loader_constructor.assert_called_once_with(
        str(pdf_dir),
    )

    loader.load.assert_called_once_with()


def test_load_from_pdf_dir_rejects_missing_directory(
    processor,
    tmp_path,
):
    missing_dir = tmp_path / "does_not_exist"

    with pytest.raises(
        FileNotFoundError,
        match="PDF directory not found",
    ):
        processor.load_from_pdf_dir(missing_dir)


def test_load_from_pdf_dir_rejects_file_path(
    processor,
    tmp_path,
):
    file_path = tmp_path / "not_a_directory.txt"
    file_path.write_text("test")

    with pytest.raises(
        NotADirectoryError,
        match="Expected directory",
    ):
        processor.load_from_pdf_dir(file_path)


# ============================================================
# TXT LOADING
# ============================================================


def test_load_from_txt_returns_documents(monkeypatch, tmp_path, processor):
    txt_file = tmp_path / "employee.txt"
    txt_file.write_text("Employee handbook content")

    documents = [
        Document(
            page_content="Employee handbook content",
        )
    ]

    loader = MagicMock()
    loader.load.return_value = documents

    loader_constructor = MagicMock(return_value=loader)

    monkeypatch.setattr(
        "src.document_ingestion.document_processor.TextLoader",
        loader_constructor,
    )

    result = processor.load_from_txt(txt_file)

    assert result == documents

    loader_constructor.assert_called_once_with(
        str(txt_file),
        encoding="utf-8",
    )

    loader.load.assert_called_once_with()


def test_load_from_txt_rejects_missing_file(
    processor,
    tmp_path,
):
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(
        FileNotFoundError,
        match="TXT file not found",
    ):
        processor.load_from_txt(missing_file)


# ============================================================
# SINGLE PDF LOADING
# ============================================================


def test_load_from_pdf_returns_documents(
    monkeypatch,
    tmp_path,
    processor,
):
    pdf_file = tmp_path / "handbook.pdf"
    pdf_file.write_bytes(b"fake pdf")

    documents = [
        Document(page_content="Page 1"),
        Document(page_content="Page 2"),
    ]

    loader = MagicMock()
    loader.load.return_value = documents

    loader_constructor = MagicMock(return_value=loader)

    monkeypatch.setattr(
        "src.document_ingestion.document_processor.PyPDFLoader",
        loader_constructor,
    )

    result = processor.load_from_pdf(pdf_file)

    assert result == documents

    loader_constructor.assert_called_once_with(
        str(pdf_file),
    )

    loader.load.assert_called_once_with()


def test_load_from_pdf_rejects_missing_file(
    processor,
    tmp_path,
):
    missing_file = tmp_path / "missing.pdf"

    with pytest.raises(
        FileNotFoundError,
        match="PDF file not found",
    ):
        processor.load_from_pdf(missing_file)


def test_load_from_pdf_rejects_non_pdf_file(
    processor,
    tmp_path,
):
    txt_file = tmp_path / "document.txt"
    txt_file.write_text("test")

    with pytest.raises(
        ValueError,
        match="Expected PDF file",
    ):
        processor.load_from_pdf(txt_file)


# ============================================================
# GENERIC DOCUMENT ROUTING
# ============================================================


def test_load_documents_routes_url(monkeypatch, processor):
    expected = [
        Document(page_content="URL document"),
    ]

    processor.load_from_url = MagicMock(
        return_value=expected,
    )

    result = processor.load_documents(
        "https://example.com/file",
    )

    assert result == expected

    processor.load_from_url.assert_called_once_with(
        "https://example.com/file",
    )


def test_load_documents_routes_directory(
    processor,
    tmp_path,
):
    directory = tmp_path / "documents"
    directory.mkdir()

    expected = [
        Document(page_content="PDF document"),
    ]

    processor.load_from_pdf_dir = MagicMock(
        return_value=expected,
    )

    result = processor.load_documents(directory)

    assert result == expected

    processor.load_from_pdf_dir.assert_called_once_with(
        directory,
    )


def test_load_documents_routes_pdf(
    processor,
    tmp_path,
):
    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(b"fake")

    expected = [
        Document(page_content="PDF document"),
    ]

    processor.load_from_pdf = MagicMock(
        return_value=expected,
    )

    result = processor.load_documents(pdf_file)

    assert result == expected

    processor.load_from_pdf.assert_called_once_with(
        pdf_file,
    )


def test_load_documents_routes_txt(
    processor,
    tmp_path,
):
    txt_file = tmp_path / "document.txt"
    txt_file.write_text("text")

    expected = [
        Document(page_content="TXT document"),
    ]

    processor.load_from_txt = MagicMock(
        return_value=expected,
    )

    result = processor.load_documents(txt_file)

    assert result == expected

    processor.load_from_txt.assert_called_once_with(
        txt_file,
    )


def test_load_documents_rejects_unsupported_source(
    processor,
    tmp_path,
):
    unsupported = tmp_path / "document.docx"
    unsupported.write_text("test")

    with pytest.raises(
        ValueError,
        match="Unsupported source type",
    ):
        processor.load_documents(unsupported)


# ============================================================
# DOCUMENT SPLITTING
# ============================================================


def test_split_documents_delegates_to_splitter(
    processor,
):
    documents = [
        Document(page_content="Original document"),
    ]

    chunks = [
        Document(page_content="Chunk 1"),
        Document(page_content="Chunk 2"),
    ]

    processor.splitter.split_documents.return_value = chunks

    result = processor.split_documents(documents)

    assert result == chunks

    processor.splitter.split_documents.assert_called_once_with(
        documents,
    )


def test_split_documents_propagates_splitter_error(
    processor,
):
    documents = [
        Document(page_content="Original document"),
    ]

    processor.splitter.split_documents.side_effect = RuntimeError("chunking failed")

    with pytest.raises(
        RuntimeError,
        match="chunking failed",
    ):
        processor.split_documents(documents)


# ============================================================
# SINGLE PDF PROCESSING
# ============================================================


def test_process_pdf_processes_single_pdf(
    processor,
    tmp_path,
):
    pdf_file = tmp_path / "employee_handbook.pdf"
    pdf_file.write_bytes(b"fake pdf")

    loaded_documents = [
        Document(
            page_content="Page 1",
            metadata={"page": 1},
        ),
        Document(
            page_content="Page 2",
            metadata={"page": 2},
        ),
    ]

    chunks = [
        Document(
            page_content="Chunk 1",
            metadata={"page": 1},
        ),
        Document(
            page_content="Chunk 2",
            metadata={"page": 2},
        ),
    ]

    processor.load_from_pdf = MagicMock(
        return_value=loaded_documents,
    )

    processor.split_documents = MagicMock(
        return_value=chunks,
    )

    result = processor.process_pdf(pdf_file)

    assert result == chunks

    processor.load_from_pdf.assert_called_once_with(
        pdf_file,
    )

    processor.split_documents.assert_called_once_with(
        loaded_documents,
    )


def test_process_pdf_sets_source_metadata(
    processor,
    tmp_path,
):
    pdf_file = tmp_path / "employee_handbook.pdf"
    pdf_file.write_bytes(b"fake pdf")

    loaded_documents = [
        Document(page_content="Page 1"),
    ]

    chunks = [
        Document(page_content="Chunk 1"),
        Document(page_content="Chunk 2"),
    ]

    processor.load_from_pdf = MagicMock(
        return_value=loaded_documents,
    )

    processor.split_documents = MagicMock(
        return_value=chunks,
    )

    result = processor.process_pdf(pdf_file)

    assert result[0].metadata["source"] == "employee_handbook.pdf"
    assert result[1].metadata["source"] == "employee_handbook.pdf"


def test_process_pdf_assigns_stable_chunk_ids(
    processor,
    tmp_path,
):
    pdf_file = tmp_path / "employee_handbook.pdf"
    pdf_file.write_bytes(b"fake pdf")

    loaded_documents = [
        Document(page_content="Page 1"),
    ]

    chunks = [
        Document(page_content="Chunk 1"),
        Document(page_content="Chunk 2"),
        Document(page_content="Chunk 3"),
    ]

    processor.load_from_pdf = MagicMock(
        return_value=loaded_documents,
    )

    processor.split_documents = MagicMock(
        return_value=chunks,
    )

    result = processor.process_pdf(pdf_file)

    expected_ids = [
        build_chunk_id(
            source="employee_handbook.pdf",
            chunk_index=0,
        ),
        build_chunk_id(
            source="employee_handbook.pdf",
            chunk_index=1,
        ),
        build_chunk_id(
            source="employee_handbook.pdf",
            chunk_index=2,
        ),
    ]

    actual_ids = [chunk.metadata["chunk_id"] for chunk in result]

    assert actual_ids == expected_ids


def test_process_pdf_overwrites_source_and_chunk_id(
    processor,
    tmp_path,
):
    pdf_file = tmp_path / "employee_handbook.pdf"
    pdf_file.write_bytes(b"fake pdf")

    loaded_documents = [
        Document(page_content="Page 1"),
    ]

    chunks = [
        Document(
            page_content="Chunk 1",
            metadata={
                "source": "old_source.pdf",
                "chunk_id": "old-id",
            },
        ),
    ]

    processor.load_from_pdf = MagicMock(
        return_value=loaded_documents,
    )

    processor.split_documents = MagicMock(
        return_value=chunks,
    )

    result = processor.process_pdf(pdf_file)

    assert result[0].metadata["source"] == "employee_handbook.pdf"

    assert result[0].metadata["chunk_id"] == build_chunk_id(
        source="employee_handbook.pdf",
        chunk_index=0,
    )


def test_process_pdf_preserves_other_metadata(
    processor,
    tmp_path,
):
    pdf_file = tmp_path / "employee_handbook.pdf"
    pdf_file.write_bytes(b"fake pdf")

    loaded_documents = [
        Document(page_content="Page 1"),
    ]

    chunks = [
        Document(
            page_content="Chunk 1",
            metadata={
                "page": 5,
                "section": "Leave Policy",
            },
        ),
    ]

    processor.load_from_pdf = MagicMock(
        return_value=loaded_documents,
    )

    processor.split_documents = MagicMock(
        return_value=chunks,
    )

    result = processor.process_pdf(pdf_file)

    assert result[0].metadata["page"] == 5
    assert result[0].metadata["section"] == "Leave Policy"
    assert result[0].metadata["source"] == "employee_handbook.pdf"


def test_process_pdf_rejects_missing_file(
    processor,
    tmp_path,
):
    missing_file = tmp_path / "missing.pdf"

    with pytest.raises(
        FileNotFoundError,
        match="PDF file not found",
    ):
        processor.process_pdf(missing_file)


def test_process_pdf_rejects_non_pdf(
    processor,
    tmp_path,
):
    txt_file = tmp_path / "document.txt"
    txt_file.write_text("test")

    with pytest.raises(
        ValueError,
        match="Expected PDF file",
    ):
        processor.process_pdf(txt_file)


def test_process_pdf_propagates_loading_error(
    processor,
    tmp_path,
):
    pdf_file = tmp_path / "employee.pdf"
    pdf_file.write_bytes(b"fake")

    processor.load_from_pdf = MagicMock(
        side_effect=RuntimeError("PDF loading failed"),
    )

    with pytest.raises(
        RuntimeError,
        match="PDF loading failed",
    ):
        processor.process_pdf(pdf_file)


def test_process_pdf_propagates_chunking_error(
    processor,
    tmp_path,
):
    pdf_file = tmp_path / "employee.pdf"
    pdf_file.write_bytes(b"fake")

    loaded_documents = [
        Document(page_content="Page 1"),
    ]

    processor.load_from_pdf = MagicMock(
        return_value=loaded_documents,
    )

    processor.split_documents = MagicMock(
        side_effect=RuntimeError("chunking failed"),
    )

    with pytest.raises(
        RuntimeError,
        match="chunking failed",
    ):
        processor.process_pdf(pdf_file)


# ============================================================
# BATCH PDF PROCESSING
# ============================================================


def test_process_pdfs_processes_directory(
    processor,
    tmp_path,
):
    directory = tmp_path / "pdfs"
    directory.mkdir()

    documents = [
        Document(
            page_content="Page 1",
            metadata={"source": "/some/path/employee.pdf"},
        ),
    ]

    chunks = [
        Document(
            page_content="Chunk 1",
            metadata={"source": "/some/path/employee.pdf"},
        ),
    ]

    processor.load_documents = MagicMock(
        return_value=documents,
    )

    processor.split_documents = MagicMock(
        return_value=chunks,
    )

    result = processor.process_pdfs(directory)

    assert result == chunks

    processor.load_documents.assert_called_once_with(
        directory,
    )

    processor.split_documents.assert_called_once_with(
        documents,
    )

    assert result[0].metadata["source"] == "employee.pdf"


def test_process_pdfs_preserves_missing_source(
    processor,
    tmp_path,
):
    directory = tmp_path / "pdfs"
    directory.mkdir()

    documents = [
        Document(page_content="Page 1"),
    ]

    chunks = [
        Document(page_content="Chunk 1"),
    ]

    processor.load_documents = MagicMock(
        return_value=documents,
    )

    processor.split_documents = MagicMock(
        return_value=chunks,
    )

    result = processor.process_pdfs(directory)

    assert "source" not in result[0].metadata


def test_process_pdfs_propagates_error(
    processor,
    tmp_path,
):
    directory = tmp_path / "pdfs"
    directory.mkdir()

    processor.load_documents = MagicMock(
        side_effect=RuntimeError("batch loading failed"),
    )

    with pytest.raises(
        RuntimeError,
        match="batch loading failed",
    ):
        processor.process_pdfs(directory)
