"""FastAPI backend for RAGFury."""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import (
    AsyncPostgresSaver,
)
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrievedDocument,
    SystemInfoResponse,
)
from src.checkpoint.postgres import (
    get_checkpoint_database_url,
)
from src.config.config import Config
from src.document_ingestion.document_processor import (
    DocumentProcessor,
)
from src.graph_builder.graph_builder import GraphBuilder
from src.guardrails.exceptions import (
    MaliciousDocumentError,
)
from src.guardrails.guardrail_manager import (
    check_input,
    check_output,
)
from src.utils.loggers import (
    configure_logging,
    get_logger,
    log_event,
)
from src.utils.metrics import (
    ABSTENTION_COUNT,
    ERROR_COUNT,
    GRAPH_LATENCY,
    REQUESTS_IN_PROGRESS,
)
from src.utils.observability import (
    clear_observability_context,
    set_request_context,
)
from src.vectorstore.vectorstore import VectorStore

configure_logging()

logger = get_logger(__name__)


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

        # -----------------------------------------------------
        # PostgreSQL LangGraph checkpointer
        # -----------------------------------------------------

        self.checkpointer = None

        self.graph = None

        self.num_chunks = 0
        self.initialized = False

        # -----------------------------------------------------
        # Registry of PDFs that have already been processed
        # -----------------------------------------------------

        self.processed_files_path = Path("storage/processed_files.json")

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
            log_event(
                logger,
                level=logging.ERROR,
                event="ingestion.registry.load.failed",
                registry_path=str(self.processed_files_path),
                error_type=type(exc).__name__,
            )

            logger.exception(
                "Failed to load processed file registry",
            )

            raise RuntimeError("Could not load processed file registry.") from exc

    def _save_processed_files(self) -> None:
        """Save processed PDF filenames."""

        start_time = time.perf_counter()

        try:
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

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.DEBUG,
                event="ingestion.registry.save.completed",
                processed_file_count=len(self.processed_files),
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="ingestion.registry.save.failed",
                error_type=type(exc).__name__,
            )

            logger.exception(
                "Failed to save processed file registry",
            )

            raise

    # =========================================================
    # FIND NEW PDF FILES
    # =========================================================

    def _get_new_pdf_files(self) -> List[Path]:
        """
        Return only PDF files that have never been processed.
        """

        pdf_files = sorted(self.data_directory.glob("*.pdf"))

        new_files = [
            file for file in pdf_files if file.name not in self.processed_files
        ]

        log_event(
            logger,
            level=logging.DEBUG,
            event="ingestion.pdf.discovery.completed",
            total_pdf_count=len(pdf_files),
            new_pdf_count=len(new_files),
        )

        return new_files

    # =========================================================
    # INCREMENTAL INGESTION
    # =========================================================

    def _sync_documents(self) -> None:
        """
        Detect and process only newly added PDFs.
        """

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="ingestion.sync.started",
        )

        # -----------------------------------------------------
        # Load registry
        # -----------------------------------------------------

        self.processed_files = self._load_processed_files()

        log_event(
            logger,
            level=logging.DEBUG,
            event="ingestion.registry.loaded",
            processed_file_count=len(self.processed_files),
        )

        # -----------------------------------------------------
        # Find new PDFs
        # -----------------------------------------------------

        new_files = self._get_new_pdf_files()

        if not new_files:
            log_event(
                logger,
                level=logging.INFO,
                event="ingestion.sync.skipped",
                reason="no_new_documents",
            )

            self.vector_store.initialize()

            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="ingestion.sync.completed",
                new_pdf_count=0,
                chunk_count=0,
                duration_ms=round(
                    elapsed,
                    2,
                ),
            )

            return

        log_event(
            logger,
            level=logging.INFO,
            event="ingestion.pdf.discovery.completed",
            new_pdf_count=len(new_files),
        )

        all_new_chunks = []

        # -----------------------------------------------------
        # Process ONLY new PDFs
        # -----------------------------------------------------

        for pdf_file in new_files:
            pdf_start_time = time.perf_counter()

            log_event(
                logger,
                level=logging.INFO,
                event="ingestion.pdf.processing.started",
                filename=pdf_file.name,
            )

            try:
                chunks = self.doc_processor.process_pdf(pdf_file)

            except Exception as exc:
                log_event(
                    logger,
                    level=logging.ERROR,
                    event="ingestion.pdf.processing.failed",
                    filename=pdf_file.name,
                    error_type=type(exc).__name__,
                )

                logger.exception(
                    "PDF processing failed",
                )

                raise

            all_new_chunks.extend(chunks)

            pdf_elapsed = (time.perf_counter() - pdf_start_time) * 1000

            log_event(
                logger,
                level=logging.INFO,
                event="ingestion.pdf.processing.completed",
                filename=pdf_file.name,
                chunk_count=len(chunks),
                duration_ms=round(
                    pdf_elapsed,
                    2,
                ),
            )

        # -----------------------------------------------------
        # Add ONLY new chunks to vector store
        # -----------------------------------------------------

        log_event(
            logger,
            level=logging.INFO,
            event="vectorstore.indexing.started",
            chunk_count=len(all_new_chunks),
        )

        vector_start_time = time.perf_counter()

        try:
            self.vector_store.initialize(new_documents=all_new_chunks)

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="vectorstore.indexing.failed",
                chunk_count=len(all_new_chunks),
                error_type=type(exc).__name__,
            )

            logger.exception(
                "Vector store indexing failed",
            )

            raise

        vector_elapsed = (time.perf_counter() - vector_start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="vectorstore.indexing.completed",
            chunk_count=len(all_new_chunks),
            duration_ms=round(
                vector_elapsed,
                2,
            ),
        )

        # -----------------------------------------------------
        # Mark files as processed
        # -----------------------------------------------------

        for pdf_file in new_files:
            self.processed_files.add(pdf_file.name)

        self._save_processed_files()

        elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="ingestion.sync.completed",
            new_pdf_count=len(new_files),
            chunk_count=len(all_new_chunks),
            duration_ms=round(
                elapsed,
                2,
            ),
        )

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def initialize(self) -> None:
        """
        Initialize the complete RAGFury pipeline.

        The PostgreSQL checkpointer must already be assigned
        before this method is called.
        """

        start_time = time.perf_counter()

        log_event(
            logger,
            level=logging.INFO,
            event="rag.initialization.started",
        )

        # -----------------------------------------------------
        # Validate data directory
        # -----------------------------------------------------

        if not self.data_directory.exists():
            log_event(
                logger,
                level=logging.ERROR,
                event="rag.initialization.failed",
                reason="data_directory_not_found",
                data_directory=str(self.data_directory),
            )

            raise FileNotFoundError(f"Data directory not found: {self.data_directory}")

        # -----------------------------------------------------
        # Load LLM
        # -----------------------------------------------------

        log_event(
            logger,
            level=logging.INFO,
            event="llm.initialization.started",
        )

        llm_start_time = time.perf_counter()

        try:
            self.llm = Config.get_llm()

        except Exception as exc:
            log_event(
                logger,
                level=logging.ERROR,
                event="llm.initialization.failed",
                error_type=type(exc).__name__,
            )

            logger.exception(
                "LLM initialization failed",
            )

            raise

        log_event(
            logger,
            level=logging.INFO,
            event="llm.initialization.completed",
            duration_ms=round(
                (time.perf_counter() - llm_start_time) * 1000,
                2,
            ),
        )

        # -----------------------------------------------------
        # Document processor
        # -----------------------------------------------------

        log_event(
            logger,
            level=logging.INFO,
            event=("document_processor.initialization.started"),
        )

        self.doc_processor = DocumentProcessor(
            model_name="all-MiniLM-L6-v2",
            threshold=0.6,
        )

        log_event(
            logger,
            level=logging.INFO,
            event=("document_processor.initialization.completed"),
            model_name="all-MiniLM-L6-v2",
            threshold=0.6,
        )

        # -----------------------------------------------------
        # Vector store
        # -----------------------------------------------------

        log_event(
            logger,
            level=logging.INFO,
            event="vectorstore.initialization.started",
        )

        self.vector_store = VectorStore()

        log_event(
            logger,
            level=logging.INFO,
            event="vectorstore.initialization.completed",
        )

        # -----------------------------------------------------
        # Incremental document synchronization
        # -----------------------------------------------------

        self._sync_documents()

        # -----------------------------------------------------
        # Number of stored chunks
        # -----------------------------------------------------

        self.num_chunks = self.vector_store.get_document_count()

        log_event(
            logger,
            level=logging.INFO,
            event="vectorstore.document_count.loaded",
            indexed_chunk_count=self.num_chunks,
        )

        if self.num_chunks == 0:
            log_event(
                logger,
                level=logging.ERROR,
                event="rag.initialization.failed",
                reason="no_indexed_documents",
            )

            raise ValueError("No indexed documents found.")

        # -----------------------------------------------------
        # Build LangGraph
        # -----------------------------------------------------

        log_event(
            logger,
            level=logging.INFO,
            event="graph.build.started",
        )

        self.graph_builder = GraphBuilder(
            retriever=(self.vector_store.get_retriever()),
            llm=self.llm,
            checkpointer=self.checkpointer,
        )

        self.graph = self.graph_builder.build()

        log_event(
            logger,
            level=logging.INFO,
            event="graph.build.completed",
        )

        self.initialized = True

        elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="rag.initialization.completed",
            indexed_chunk_count=self.num_chunks,
            duration_ms=round(
                elapsed,
                2,
            ),
        )

    # =========================================================
    # QUERY
    # =========================================================

    async def query(
        self,
        question: str,
        user_id: str,
        conversation_id: str,
        request_id: str,
    ) -> Dict[str, Any]:
        """
        Execute a question through LangGraph.

        The conversation ID is converted into a unique
        LangGraph thread ID so checkpoints belonging to
        different conversations remain isolated.
        """

        if not self.initialized:
            raise RuntimeError("RAGFury has not been initialized.")

        if self.graph is None:
            raise RuntimeError("LangGraph is not available.")

        start_time = time.perf_counter()

        set_request_context(
            request_id=request_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )

        log_event(
            logger,
            level=logging.INFO,
            event="graph.execution.started",
            request_id=request_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )

        # -----------------------------------------------------
        # LangGraph thread ID
        # -----------------------------------------------------
        #
        # This identifies the persistent execution thread
        # inside PostgreSQL.
        #
        # Example:
        #
        # ragfury:user123:conversation_a81f9e
        #
        # -----------------------------------------------------

        thread_id = f"ragfury:{user_id}:{conversation_id}"

        log_event(
            logger,
            level=logging.DEBUG,
            event="graph.thread.created",
            request_id=request_id,
            conversation_id=conversation_id,
        )

        # -----------------------------------------------------
        # Initial graph state
        # -----------------------------------------------------

        initial_state = {
            "question": question,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "request_id": request_id,
        }

        # -----------------------------------------------------
        # LangGraph checkpoint configuration
        # -----------------------------------------------------
        #
        # IMPORTANT:
        # thread_id does NOT belong inside RAGState.
        #
        # LangGraph uses this configurable value to identify
        # which PostgreSQL checkpoint thread should be used.
        #
        # -----------------------------------------------------

        config = {
            "configurable": {
                "thread_id": thread_id,
            },
            "metadata": {
                "request_id": request_id,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "service": "ragfury",
                "environment": "development",
            },
            "tags": [
                "ragfury",
                "agentic-rag",
            ],
        }

        # -----------------------------------------------------
        # Execute LangGraph asynchronously
        # -----------------------------------------------------

        try:
            result = await self.graph.ainvoke(
                initial_state,
                config=config,
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000

            log_event(
                logger,
                level=logging.ERROR,
                event="graph.execution.failed",
                request_id=request_id,
                user_id=user_id,
                conversation_id=conversation_id,
                duration_ms=round(
                    elapsed,
                    2,
                ),
                error_type=type(exc).__name__,
            )

            logger.exception(
                "LangGraph execution failed",
            )

            raise

        finally:
            clear_observability_context()

        if result.get("next_step") == "abstain":
            ABSTENTION_COUNT.inc()

        elapsed = (time.perf_counter() - start_time) * 1000

        log_event(
            logger,
            level=logging.INFO,
            event="graph.execution.completed",
            request_id=request_id,
            user_id=user_id,
            conversation_id=conversation_id,
            next_step=result.get("next_step"),
            retrieval_attempts=result.get("retrieval_attempts"),
            duration_ms=round(
                elapsed,
                2,
            ),
        )

        return result


# =============================================================
# GLOBAL SERVICE
# =============================================================

DATA_DIRECTORY = Path("data")

rag_service = RAGService(data_directory=DATA_DIRECTORY)


# =============================================================
# FASTAPI LIFESPAN
# =============================================================


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Initialize RAGFury when FastAPI starts.

    PostgreSQL checkpointer lifecycle:

        FastAPI startup
              ↓
        Create AsyncPostgresSaver
              ↓
        Setup checkpoint tables
              ↓
        Inject checkpointer into RAGService
              ↓
        Initialize RAGFury
              ↓
        Application running
              ↓
        FastAPI shutdown
              ↓
        Close checkpointer
    """

    log_event(
        logger,
        level=logging.INFO,
        event="application.startup.started",
    )

    database_url = None
    checkpointer = None

    try:
        # -----------------------------------------------------
        # Get dedicated RAGFury checkpoint database
        # -----------------------------------------------------

        database_url = get_checkpoint_database_url()

        log_event(
            logger,
            level=logging.INFO,
            event=("checkpoint.database.connection.started"),
        )

        # -----------------------------------------------------
        # Create PostgreSQL checkpointer
        # -----------------------------------------------------

        async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
            log_event(
                logger,
                level=logging.INFO,
                event=("checkpoint.database.connection.completed"),
            )

            # -------------------------------------------------
            # Create/check LangGraph checkpoint tables
            # -------------------------------------------------

            log_event(
                logger,
                level=logging.INFO,
                event="checkpoint.setup.started",
            )

            await checkpointer.setup()

            log_event(
                logger,
                level=logging.INFO,
                event="checkpoint.setup.completed",
            )

            # -------------------------------------------------
            # Inject checkpointer into RAG service
            # -------------------------------------------------

            rag_service.checkpointer = checkpointer

            # -------------------------------------------------
            # Initialize complete RAG pipeline
            # -------------------------------------------------

            try:
                await asyncio.to_thread(rag_service.initialize)

            except Exception as exc:
                log_event(
                    logger,
                    level=logging.ERROR,
                    event="rag.initialization.failed",
                    error_type=type(exc).__name__,
                )

                logger.exception(
                    "RAGFury initialization failed",
                )

            # -------------------------------------------------
            # Application runs while checkpointer context
            # remains open.
            # -------------------------------------------------

            yield

    except Exception as exc:
        log_event(
            logger,
            level=logging.ERROR,
            event="application.startup.failed",
            component="postgresql_checkpointer",
            error_type=type(exc).__name__,
        )

        logger.exception(
            "PostgreSQL checkpoint initialization failed",
        )

        # -----------------------------------------------------
        # Preserve the application's previous behavior:
        # FastAPI can still start, but RAGFury remains
        # uninitialized if its infrastructure failed.
        # -----------------------------------------------------

        yield

    finally:
        # -----------------------------------------------------
        # Clear application references
        # -----------------------------------------------------

        rag_service.checkpointer = None
        rag_service.graph = None

        log_event(
            logger,
            level=logging.INFO,
            event="application.shutdown.completed",
        )


# =============================================================
# FASTAPI APPLICATION
# =============================================================

app = FastAPI(
    title="RAGFury API",
    description=("Agentic Knowledge Retrieval & Research System"),
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
)


# =============================================================
# CORS
# =============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.get_cors_origins(),
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
        rag_initialized=(rag_service.initialized),
        document_chunks=(rag_service.num_chunks),
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

    REQUESTS_IN_PROGRESS.inc()

    request_id = uuid.uuid4().hex

    request_start_time = time.perf_counter()

    log_event(
        logger,
        level=logging.INFO,
        event="query.started",
        request_id=request_id,
    )

    # =========================================================
    # VALIDATE QUESTION
    # =========================================================

    question = request.question.strip()

    if not question:
        log_event(
            logger,
            level=logging.WARNING,
            event="query.validation.failed",
            request_id=request_id,
            reason="empty_question",
        )

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # =========================================================
    # INPUT GUARDRAIL
    # =========================================================

    guardrail_start_time = time.perf_counter()

    log_event(
        logger,
        level=logging.INFO,
        event="security.input.started",
        request_id=request_id,
    )

    try:
        input_allowed = await check_input(question)

    except Exception as exc:
        log_event(
            logger,
            level=logging.ERROR,
            event="security.input.failed",
            request_id=request_id,
            error_type=type(exc).__name__,
        )

        logger.exception(
            "Input guardrail failed",
        )

        raise HTTPException(
            status_code=503,
            detail=("Input security validation failed."),
        ) from exc

    input_guardrail_elapsed = (time.perf_counter() - guardrail_start_time) * 1000

    log_event(
        logger,
        level=(logging.INFO if input_allowed else logging.WARNING),
        event=("security.input.allowed" if input_allowed else "security.input.blocked"),
        request_id=request_id,
        duration_ms=round(
            input_guardrail_elapsed,
            2,
        ),
    )

    if not input_allowed:
        raise HTTPException(
            status_code=400,
            detail=("Your request was blocked by the RAGFury safety policy."),
        )

    # =========================================================
    # VALIDATE USER ID
    # =========================================================

    user_id = request.user_id.strip()

    if not user_id:
        log_event(
            logger,
            level=logging.WARNING,
            event="query.validation.failed",
            request_id=request_id,
            reason="empty_user_id",
        )

        raise HTTPException(
            status_code=400,
            detail="User ID cannot be empty.",
        )

    # =========================================================
    # CONVERSATION ID
    # =========================================================

    if request.conversation_id:
        conversation_id = request.conversation_id.strip()

        if not conversation_id:
            log_event(
                logger,
                level=logging.WARNING,
                event="query.validation.failed",
                request_id=request_id,
                reason="empty_conversation_id",
            )

            raise HTTPException(
                status_code=400,
                detail=("Conversation ID cannot be empty."),
            )

    else:
        conversation_id = f"conversation_{uuid.uuid4().hex[:12]}"

        log_event(
            logger,
            level=logging.INFO,
            event="conversation.created",
            request_id=request_id,
        )

    # =========================================================
    # INITIALIZATION CHECK
    # =========================================================

    if not rag_service.initialized:
        log_event(
            logger,
            level=logging.ERROR,
            event="query.rejected",
            request_id=request_id,
            reason="rag_not_initialized",
        )

        raise HTTPException(
            status_code=503,
            detail=("RAGFury is not initialized. Check the API startup logs."),
        )

    # =========================================================
    # EXECUTE GRAPH
    # =========================================================

    try:
        result = await rag_service.query(
            question=question,
            user_id=user_id,
            conversation_id=conversation_id,
            request_id=request_id,
        )

    except MaliciousDocumentError as exc:
        log_event(
            logger,
            level=logging.WARNING,
            event="security.document.blocked",
            request_id=request_id,
            error_type=type(exc).__name__,
        )

        raise HTTPException(
            status_code=400,
            detail={
                "code": "MALICIOUS_DOCUMENT",
                "message": ("The document failed security validation."),
                "request_id": request_id,
            },
        ) from exc

    except Exception as exc:
        elapsed = (time.perf_counter() - request_start_time) * 1000

        ERROR_COUNT.labels(
            component="graph",
            error_type=type(exc).__name__,
        ).inc()

        GRAPH_LATENCY.labels(
            outcome="error",
        ).observe(elapsed / 1000)

        log_event(
            logger,
            level=logging.ERROR,
            event="query.failed",
            request_id=request_id,
            user_id=user_id,
            conversation_id=conversation_id,
            duration_ms=round(
                elapsed,
                2,
            ),
            error_type=type(exc).__name__,
        )

        logger.exception(
            "Query processing failed",
        )

        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred.",
                "request_id": request_id,
            },
        ) from exc

    finally:
        REQUESTS_IN_PROGRESS.dec()

    # =========================================================
    # OUTPUT GUARDRAIL
    # =========================================================

    answer = result.get(
        "answer",
        "",
    )

    output_guardrail_start_time = time.perf_counter()

    log_event(
        logger,
        level=logging.INFO,
        event="security.output.started",
        request_id=request_id,
    )

    try:
        output_allowed = await check_output(answer)

    except Exception as exc:
        log_event(
            logger,
            level=logging.ERROR,
            event="security.output.failed",
            request_id=request_id,
            error_type=type(exc).__name__,
        )

        logger.exception(
            "Output guardrail failed",
        )

        raise HTTPException(
            status_code=503,
            detail=("Output security validation failed."),
        ) from exc

    output_guardrail_elapsed = (
        time.perf_counter() - output_guardrail_start_time
    ) * 1000

    log_event(
        logger,
        level=(logging.INFO if output_allowed else logging.WARNING),
        event=(
            "security.output.allowed" if output_allowed else "security.output.blocked"
        ),
        request_id=request_id,
        answer_length=len(answer),
        duration_ms=round(
            output_guardrail_elapsed,
            2,
        ),
    )

    if not output_allowed:
        raise HTTPException(
            status_code=403,
            detail=("The generated response was blocked by the RAGFury safety policy."),
        )

    # =========================================================
    # RESPONSE TIME
    # =========================================================

    elapsed = time.perf_counter() - request_start_time

    # =========================================================
    # RETRIEVED DOCUMENTS
    # =========================================================

    raw_documents = result.get(
        "retrieved_docs",
        [],
    )

    documents: List[RetrievedDocument] = []

    for document in raw_documents:
        if hasattr(
            document,
            "page_content",
        ):
            content = document.page_content

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
    # QUERY COMPLETION LOG
    # =========================================================

    log_event(
        logger,
        level=logging.INFO,
        event="query.completed",
        request_id=request_id,
        user_id=user_id,
        conversation_id=conversation_id,
        next_step=result.get("next_step"),
        document_count=len(documents),
        answer_length=len(answer or ""),
        retrieval_attempts=result.get("retrieval_attempts"),
        reflection_attempts=result.get("reflection_attempts"),
        duration_ms=round(
            elapsed * 1000,
            2,
        ),
    )

    # =========================================================
    # RESPONSE
    # =========================================================

    return QueryResponse(
        question=question,
        answer=answer or "No answer generated.",
        conversation_id=conversation_id,
        next_step=result.get("next_step"),
        documents=documents,
        document_relevance=result.get("document_relevance"),
        grade_reason=result.get("grade_reason"),
        reflection=result.get("reflection"),
        reflection_passed=result.get("reflection_passed"),
        retrieval_attempts=result.get("retrieval_attempts"),
        reflection_attempts=result.get("reflection_attempts"),
        response_time=round(
            elapsed,
            4,
        ),
    )
