from __future__ import annotations

from pydantic import BaseModel, Field


class CitationSource(BaseModel):
    """A source that can be referenced by a generated answer."""

    citation_id: str = Field(
        ...,
        description="Stable citation identifier such as '1' or '2'.",
    )

    source: str = Field(
        ...,
        description="Source document name.",
    )

    chunk_id: str = Field(
        ...,
        description="Stable chunk identifier.",
    )

    page: int | None = Field(
        default=None,
        description="1-based source page number when available.",
    )
