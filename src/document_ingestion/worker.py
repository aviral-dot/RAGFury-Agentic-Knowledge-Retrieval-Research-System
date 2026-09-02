"""Standalone document ingestion worker."""

from pathlib import Path

from src.document_ingestion.ingestion_service import IngestionService
from src.utils.loggers import configure_logging


def main() -> None:
    """Run document ingestion."""

    configure_logging()

    service = IngestionService(data_directory=Path("data"))

    service.initialize()

    total_chunks = service.ingest_all()

    print(f"Ingestion completed. Total chunks: {total_chunks}")


if __name__ == "__main__":
    main()
