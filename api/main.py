"""FastAPI backend for RAGFury."""

import json
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.guardrails.exceptions import (
    MaliciousDocumentError,
)

from src.guardrails.guardrail_manager import (
    check_input,
    check_output,
)

from src.config.config import Config
from src.document_ingestion.document_processor import (
    DocumentProcessor,
)
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

from api.schemas import (
    QueryRequest,
    QueryResponse,
    RetrievedDocument,
    HealthResponse,
    SystemInfoResponse,
)


# =============================================================
# RAG SERVICE
# =============================================================


class RAGService:
    """
    Application service responsible for initializing
    and executing the RAGFury pipeline.
    """

    def __init__(
        self,
        data_directory: Path,
    ) -> None:

        self.data_directory = data_directory

        self.llm = None
        self.doc_processor = None
        self.vector_store = None
        self.graph_builder = None
        self.graph = None

        self.num_chunks = 0
        self.initialized = False

        # -----------------------------------------------------
        # Registry of PDFs that have already been processed
        # -----------------------------------------------------

        self.processed_files_path = Path(
            "storage/processed_files.json"
        )

        self.processed_files_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.processed_files = set()

    # =========================================================
    # PROCESSED FILE REGISTRY
    # =========================================================

    def _load_processed_files(self) -> set[str]:
        """Load names of PDFs that have already been indexed."""

        if not self.processed_files_path.exists():
            return set()

        try:

            with open(
                self.processed_files_path,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            return set(data)

        except (
            json.JSONDecodeError,
            OSError,
        ) as exc:

            raise RuntimeError(
                "Could not load processed file registry."
            ) from exc

    def _save_processed_files(self) -> None:
        """Save processed PDF filenames."""

        with open(
            self.processed_files_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                sorted(self.processed_files),
                file,
                indent=2,
            )

    # =========================================================
    # FIND NEW PDF FILES
    # =========================================================

    def _get_new_pdf_files(self) -> List[Path]:
        """
        Return only PDF files that have never been processed.
        """

        pdf_files = sorted(
            self.data_directory.glob("*.pdf")
        )

        new_files = [
            file
            for file in pdf_files
            if file.name not in self.processed_files
        ]

        return new_files

    # =========================================================
    # INCREMENTAL INGESTION
    # =========================================================

    def _sync_documents(self) -> None:
        """
        Detect and process only newly added PDFs.
        """

        print(
            "🔎 Checking for new PDF documents..."
        )

        # -----------------------------------------------------
        # Load registry
        # -----------------------------------------------------

        self.processed_files = (
            self._load_processed_files()
        )

        # -----------------------------------------------------
        # Find new PDFs
        # -----------------------------------------------------

        new_files = self._get_new_pdf_files()

        if not new_files:

            print(
                "✅ No new PDFs found. "
                "Skipping document processing."
            )

            # Load existing persistent vector store.
            self.vector_store.initialize()

            return

        print(
            f"📚 Found {len(new_files)} new PDF(s)."
        )

        all_new_chunks = []

        # -----------------------------------------------------
        # Process ONLY new PDFs
        # -----------------------------------------------------

        for pdf_file in new_files:

            print(
                f"\n📄 New document detected: "
                f"{pdf_file.name}"
            )

            chunks = self.doc_processor.process_pdf(
                pdf_file
            )

            all_new_chunks.extend(chunks)

        # -----------------------------------------------------
        # Add ONLY new chunks to vector store
        # -----------------------------------------------------

        print(
            f"\n🗂️ Adding "
            f"{len(all_new_chunks)} new chunks "
            f"to vector store..."
        )

        self.vector_store.initialize(
            new_documents=all_new_chunks
        )

        # -----------------------------------------------------
        # Mark files as processed
        # -----------------------------------------------------

        for pdf_file in new_files:

            self.processed_files.add(
                pdf_file.name
            )

        self._save_processed_files()

        print(
            "\n✅ Incremental ingestion complete."
        )

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def initialize(self) -> None:
        """Initialize the complete RAGFury pipeline."""

        print(
            "\n🚀 Initializing RAGFury API..."
        )

        # -----------------------------------------------------
        # Validate data directory
        # -----------------------------------------------------

        if not self.data_directory.exists():

            raise FileNotFoundError(
                f"Data directory not found: "
                f"{self.data_directory}"
            )

        # -----------------------------------------------------
        # Load LLM
        # -----------------------------------------------------

        print(
            "🧠 Loading LLM..."
        )

        self.llm = Config.get_llm()

        # -----------------------------------------------------
        # Document processor
        # -----------------------------------------------------

        print(
            "📄 Initializing document processor..."
        )

        self.doc_processor = DocumentProcessor(
            model_name="all-MiniLM-L6-v2",
            threshold=0.6,
        )

        # -----------------------------------------------------
        # Vector store
        # -----------------------------------------------------

        print(
            "🔍 Initializing vector store..."
        )

        self.vector_store = VectorStore()

        # -----------------------------------------------------
        # Incremental document synchronization
        # -----------------------------------------------------

        self._sync_documents()

        # -----------------------------------------------------
        # Number of stored chunks
        # -----------------------------------------------------

        self.num_chunks = (
            self.vector_store.get_document_count()
        )

        print(
            f"📊 Total indexed chunks: "
            f"{self.num_chunks}"
        )

        if self.num_chunks == 0:

            raise ValueError(
                "No indexed documents found."
            )

        # -----------------------------------------------------
        # Build LangGraph
        # -----------------------------------------------------

        print(
            "🔗 Building LangGraph workflow..."
        )

        self.graph_builder = GraphBuilder(
            retriever=(
                self.vector_store.get_retriever()
            ),
            llm=self.llm,
        )

        self.graph = (
            self.graph_builder.build()
        )

        self.initialized = True

        print(
            "✅ RAGFury API initialized successfully!"
        )

    # =========================================================
    # QUERY
    # =========================================================

    def query(
        self,
        question: str,
        user_id: str,
        conversation_id: str,
    ) -> Dict[str, Any]:
        """Execute a question through LangGraph."""

        if not self.initialized:

            raise RuntimeError(
                "RAGFury has not been initialized."
            )

        if self.graph is None:

            raise RuntimeError(
                "LangGraph is not available."
            )

        initial_state = {
            "question": question,
            "user_id": user_id,
            "conversation_id": conversation_id,
        }

        return self.graph.invoke(
            initial_state
        )


# =============================================================
# GLOBAL SERVICE
# =============================================================

DATA_DIRECTORY = Path("data")

rag_service = RAGService(
    data_directory=DATA_DIRECTORY
)


# =============================================================
# FASTAPI LIFESPAN
# =============================================================


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """Initialize RAGFury when FastAPI starts."""

    try:

        rag_service.initialize()

    except Exception as exc:

        print(
            f"❌ RAGFury initialization failed: "
            f"{exc}"
        )

    yield

    print(
        "🛑 RAGFury API shutting down..."
    )


# =============================================================
# FASTAPI APPLICATION
# =============================================================

app = FastAPI(
    title="RAGFury API",
    description=(
        "Agentic Knowledge Retrieval & "
        "Research System"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================
# CORS
# =============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
    ],
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
    ],
    allow_headers=[
        "*",
    ],
)


# =============================================================
# ROOT
# =============================================================


@app.get("/")
async def root():
    """API root endpoint."""

    return {
        "name": "RAGFury API",
        "version": "1.0.0",
        "status": "running",
    }


# =============================================================
# HEALTH
# =============================================================


@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health():
    """Return service health."""

    if rag_service.initialized:

        return HealthResponse(
            status="healthy",
            rag_initialized=True,
        )

    return HealthResponse(
        status="unhealthy",
        rag_initialized=False,
    )


# =============================================================
# SYSTEM INFO
# =============================================================


@app.get(
    "/api/v1/info",
    response_model=SystemInfoResponse,
)
async def system_info():
    """Return information about RAGFury."""

    return SystemInfoResponse(
        name="RAGFury",
        version="1.0.0",
        rag_initialized=(
            rag_service.initialized
        ),
        document_chunks=(
            rag_service.num_chunks
        ),
    )


# =============================================================
# QUERY
# =============================================================


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
)
async def query(
    request: QueryRequest,
):
    """Execute a question through RAGFury."""

    # =========================================================
    # VALIDATE QUESTION
    # =========================================================

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # =========================================================
    # INPUT GUARDRAIL
    # =========================================================

    try:

        input_allowed = await check_input(
            question
        )

    except Exception as exc:

        print(
            f"🚨 Input guardrail failed: {exc}"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Input security validation failed."
            ),
        ) from exc

    if not input_allowed:

        raise HTTPException(
            status_code=400,
            detail=(
                "Your request was blocked by the "
                "RAGFury safety policy."
            ),
        )

    # =========================================================
    # VALIDATE USER ID
    # =========================================================

    user_id = request.user_id.strip()

    if not user_id:

        raise HTTPException(
            status_code=400,
            detail="User ID cannot be empty.",
        )

    # =========================================================
    # CONVERSATION ID
    # =========================================================

    if request.conversation_id:

        conversation_id = (
            request.conversation_id.strip()
        )

        if not conversation_id:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Conversation ID cannot "
                    "be empty."
                ),
            )

    else:

        conversation_id = (
            f"conversation_"
            f"{uuid.uuid4().hex[:12]}"
        )

        print(
            f"🆕 New conversation created: "
            f"{conversation_id}"
        )

    # =========================================================
    # INITIALIZATION CHECK
    # =========================================================

    if not rag_service.initialized:

        raise HTTPException(
            status_code=503,
            detail=(
                "RAGFury is not initialized. "
                "Check the API startup logs."
            ),
        )

    # =========================================================
    # EXECUTE GRAPH
    # =========================================================

    start_time = time.perf_counter()

    try:

        result = rag_service.query(
            question=question,
            user_id=user_id,
            conversation_id=conversation_id,
        )

    except MaliciousDocumentError as exc:

       print(
        f"🚫 MALICIOUS DOCUMENT BLOCKED: {exc}"
       )

       raise HTTPException(
        status_code=400,
        detail=str(exc),
       ) from exc   

    except Exception as exc:

        print(
            "\n" + "=" * 70
        )

        print(
            "❌ QUERY PROCESSING FAILED"
        )

        print(
            "=" * 70
        )

        print(
            f"Error type: "
            f"{type(exc).__name__}"
        )

        print(
            f"Error: {exc}"
        )

        traceback.print_exc()

        print(
            "=" * 70 + "\n"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    # =========================================================
    # OUTPUT GUARDRAIL
    # =========================================================

    answer = result.get(
        "answer",
        "",
    )

    try:

        output_allowed = await check_output(
            answer
        )

    except Exception as exc:

        print(
            f"🚨 Output guardrail failed: {exc}"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Output security validation failed."
            ),
        ) from exc

    if not output_allowed:

        raise HTTPException(
            status_code=403,
            detail=(
                "The generated response was blocked "
                "by the RAGFury safety policy."
            ),
        )

    # =========================================================
    # RESPONSE TIME
    # =========================================================

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # =========================================================
    # RETRIEVED DOCUMENTS
    # =========================================================

    raw_documents = result.get(
        "retrieved_docs",
        [],
    )

    documents: List[
        RetrievedDocument
    ] = []

    for document in raw_documents:

        if hasattr(
            document,
            "page_content",
        ):

            content = (
                document.page_content
            )

            metadata = getattr(
                document,
                "metadata",
                {},
            )

        else:

            content = str(document)

            metadata = {}

        documents.append(
            RetrievedDocument(
                content=content,
                metadata=metadata or {},
            )
        )

    # =========================================================
    # RESPONSE
    # =========================================================

    return QueryResponse(
        question=question,

        answer=answer or "No answer generated.",

        conversation_id=conversation_id,

        next_step=result.get(
            "next_step"
        ),

        documents=documents,

        document_relevance=result.get(
            "document_relevance"
        ),

        grade_reason=result.get(
            "grade_reason"
        ),

        reflection=result.get(
            "reflection"
        ),

        reflection_passed=result.get(
            "reflection_passed"
        ),

        retrieval_attempts=result.get(
            "retrieval_attempts"
        ),

        reflection_attempts=result.get(
            "reflection_attempts"
        ),

        response_time=round(
            elapsed,
            4,
        ),
    )