import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_document():
    """A minimal document used by unit tests."""
    return Document(
        page_content=(
            "A full-time employee must work at least 30 hours per week on average."
        ),
        metadata={
            "source": "employee_handbook.pdf",
            "page": 1,
            "chunk_id": "employee_handbook_p1_chunk_001",
        },
    )


@pytest.fixture
def sample_documents():
    """A small deterministic document collection for retrieval tests."""
    return [
        Document(
            page_content=(
                "A full-time employee must work at least 30 hours per week on average."
            ),
            metadata={
                "source": "employee_handbook.pdf",
                "page": 1,
                "chunk_id": "employee_handbook_p1_chunk_001",
            },
        ),
        Document(
            page_content=(
                "Employees are entitled to annual leave according to company policy."
            ),
            metadata={
                "source": "employee_handbook.pdf",
                "page": 2,
                "chunk_id": "employee_handbook_p2_chunk_001",
            },
        ),
        Document(
            page_content=(
                "Employees must notify their manager when they "
                "are unable to attend work."
            ),
            metadata={
                "source": "employee_handbook.pdf",
                "page": 3,
                "chunk_id": "employee_handbook_p3_chunk_001",
            },
        ),
    ]
