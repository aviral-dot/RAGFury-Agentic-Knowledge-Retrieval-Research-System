import numpy as np
import pytest
from langchain_core.documents import Document

from src.semantic_chunker.semantic_chunker import ThresholdSematicChunker


class FakeSentenceTransformer:
    """Deterministic fake embedding model for unit tests."""

    def __init__(self, *args, **kwargs):
        self.device = kwargs.get("device") or "cpu"

    def encode(
        self,
        sentences,
        batch_size,
        convert_to_numpy,
        normalize_embeddings,
        show_progress_bar,
    ):
        # Each sentence receives a deterministic embedding.
        #
        # Sentences at even/odd positions deliberately receive
        # orthogonal vectors so that tests can control similarity.
        embeddings = []

        for index, _ in enumerate(sentences):
            if index % 2 == 0:
                embeddings.append([1.0, 0.0])
            else:
                embeddings.append([0.0, 1.0])

        return np.asarray(embeddings, dtype=np.float32)


@pytest.fixture
def chunker(monkeypatch):
    """Create a chunker without loading the real embedding model."""

    monkeypatch.setattr(
        "src.semantic_chunker.semantic_chunker.SentenceTransformer",
        FakeSentenceTransformer,
    )

    return ThresholdSematicChunker(
        model_name="test-model",
        threshold=0.5,
        batch_size=2,
        device="cpu",
        max_chunk_chars=1200,
        min_chunk_chars=100,
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_chunker_initializes_with_valid_configuration(chunker):
    assert chunker.model_name == "test-model"
    assert chunker.threshold == 0.5
    assert chunker.batch_size == 2
    assert chunker.max_chunk_chars == 1200
    assert chunker.min_chunk_chars == 100


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1, 2.0],
)
def test_chunker_rejects_invalid_threshold(monkeypatch, threshold):
    monkeypatch.setattr(
        "src.semantic_chunker.semantic_chunker.SentenceTransformer",
        FakeSentenceTransformer,
    )

    with pytest.raises(ValueError, match="threshold must be between"):
        ThresholdSematicChunker(
            threshold=threshold,
        )


def test_chunker_rejects_negative_batch_size(monkeypatch):
    monkeypatch.setattr(
        "src.semantic_chunker.semantic_chunker.SentenceTransformer",
        FakeSentenceTransformer,
    )

    with pytest.raises(
        ValueError,
        match="batch_size must be greater than 0",
    ):
        ThresholdSematicChunker(batch_size=0)


def test_chunker_rejects_non_positive_max_chunk_chars(monkeypatch):
    monkeypatch.setattr(
        "src.semantic_chunker.semantic_chunker.SentenceTransformer",
        FakeSentenceTransformer,
    )

    with pytest.raises(
        ValueError,
        match="max_chunk_chars must be greater than 0",
    ):
        ThresholdSematicChunker(max_chunk_chars=0)


def test_chunker_rejects_negative_min_chunk_chars(monkeypatch):
    monkeypatch.setattr(
        "src.semantic_chunker.semantic_chunker.SentenceTransformer",
        FakeSentenceTransformer,
    )

    with pytest.raises(
        ValueError,
        match="min_chunk_chars cannot be negative",
    ):
        ThresholdSematicChunker(min_chunk_chars=-1)


def test_chunker_rejects_min_chunk_larger_than_max_chunk(monkeypatch):
    monkeypatch.setattr(
        "src.semantic_chunker.semantic_chunker.SentenceTransformer",
        FakeSentenceTransformer,
    )

    with pytest.raises(
        ValueError,
        match="min_chunk_chars cannot be greater",
    ):
        ThresholdSematicChunker(
            min_chunk_chars=200,
            max_chunk_chars=100,
        )


# ---------------------------------------------------------------------------
# split()
# ---------------------------------------------------------------------------


def test_split_empty_text_returns_empty_list(chunker):
    assert chunker.split("") == []


def test_split_whitespace_only_text_returns_empty_list(chunker):
    assert chunker.split("   \n\t  ") == []


def test_split_single_sentence_returns_single_chunk(chunker):
    text = "This is a single sentence."

    result = chunker.split(text)

    assert result == ["This is a single sentence."]


def test_split_removes_empty_sentences(chunker):
    text = "First sentence.   Second sentence."

    result = chunker.split(text)

    assert len(result) >= 1

    for chunk in result:
        assert chunk.strip()
        assert not chunk.startswith(" ")
        assert not chunk.endswith(" ")


def test_split_preserves_sentence_content(chunker):
    text = (
        "The employee handbook defines company policies. "
        "Employees should read the handbook carefully."
    )

    result = chunker.split(text)

    combined = " ".join(result)

    assert "The employee handbook defines company policies." in combined
    assert "Employees should read the handbook carefully." in combined


def test_split_splits_when_similarity_is_below_threshold(chunker):
    text = "The company provides health insurance. The moon orbits the Earth."

    result = chunker.split(text)

    # The fake model generates orthogonal embeddings for
    # consecutive sentences, giving similarity 0.0.
    assert len(result) == 2

    assert result[0] == "The company provides health insurance."
    assert result[1] == "The moon orbits the Earth."


def test_split_does_not_split_when_similarity_is_above_threshold(monkeypatch):
    class SimilarSentenceTransformer(FakeSentenceTransformer):
        def encode(
            self,
            sentences,
            batch_size,
            convert_to_numpy,
            normalize_embeddings,
            show_progress_bar,
        ):
            # Identical normalized embeddings produce similarity = 1.0.
            return np.asarray(
                [[1.0, 0.0] for _ in sentences],
                dtype=np.float32,
            )

    monkeypatch.setattr(
        "src.semantic_chunker.semantic_chunker.SentenceTransformer",
        SimilarSentenceTransformer,
    )

    chunker = ThresholdSematicChunker(
        threshold=0.5,
        max_chunk_chars=1200,
    )

    text = (
        "The company provides health insurance. "
        "Employees can enroll during the annual enrollment period."
    )

    result = chunker.split(text)

    assert len(result) == 1
    assert result[0] == (
        "The company provides health insurance. "
        "Employees can enroll during the annual enrollment period."
    )


def test_split_respects_max_chunk_chars(monkeypatch):
    class SimilarSentenceTransformer(FakeSentenceTransformer):
        def encode(
            self,
            sentences,
            batch_size,
            convert_to_numpy,
            normalize_embeddings,
            show_progress_bar,
        ):
            return np.asarray(
                [[1.0, 0.0] for _ in sentences],
                dtype=np.float32,
            )

    monkeypatch.setattr(
        "src.semantic_chunker.semantic_chunker.SentenceTransformer",
        SimilarSentenceTransformer,
    )

    chunker = ThresholdSematicChunker(
        threshold=0.5,
        min_chunk_chars=50,
        max_chunk_chars=60,
    )

    text = (
        "This is the first sentence. "
        "This is the second sentence. "
        "This is the third sentence."
    )

    result = chunker.split(text)

    assert len(result) > 1

    for chunk in result:
        assert len(chunk) <= 60


# ---------------------------------------------------------------------------
# split_documents()
# ---------------------------------------------------------------------------


def test_split_documents_returns_langchain_documents(chunker):
    documents = [
        Document(
            page_content=(
                "The company provides health insurance. "
                "Employees can enroll during annual enrollment."
            ),
            metadata={
                "source": "employee_handbook.pdf",
                "page": 1,
            },
        )
    ]

    result = chunker.split_documents(documents)

    assert result
    assert all(isinstance(document, Document) for document in result)


def test_split_documents_preserves_original_metadata(chunker):
    documents = [
        Document(
            page_content=(
                "The company provides health insurance. "
                "Employees can enroll during annual enrollment."
            ),
            metadata={
                "source": "employee_handbook.pdf",
                "page": 5,
                "custom_field": "important",
            },
        )
    ]

    result = chunker.split_documents(documents)

    assert result

    for chunk in result:
        assert chunk.metadata["source"] == "employee_handbook.pdf"
        assert chunk.metadata["page"] == 5
        assert chunk.metadata["custom_field"] == "important"


def test_split_documents_adds_chunk_metadata(chunker):
    documents = [
        Document(
            page_content=(
                "The company provides health insurance. The moon orbits the Earth."
            ),
            metadata={
                "source": "employee_handbook.pdf",
                "page": 1,
            },
        )
    ]

    result = chunker.split_documents(documents)

    assert len(result) == 2

    assert result[0].metadata["chunk_index"] == 0
    assert result[1].metadata["chunk_index"] == 1

    for chunk in result:
        assert chunk.metadata["chunking_strategy"] == ("semantic_threshold")
        assert chunk.metadata["chunking_threshold"] == 0.5
        assert chunk.metadata["chunking_model"] == "test-model"


def test_split_documents_resets_chunk_index_per_document(chunker):
    documents = [
        Document(
            page_content="First document sentence.",
            metadata={"source": "first.pdf"},
        ),
        Document(
            page_content="Second document sentence.",
            metadata={"source": "second.pdf"},
        ),
    ]

    result = chunker.split_documents(documents)

    assert len(result) == 2

    assert result[0].metadata["chunk_index"] == 0
    assert result[1].metadata["chunk_index"] == 0


def test_split_documents_empty_input_returns_empty_list(chunker):
    assert chunker.split_documents([]) == []
