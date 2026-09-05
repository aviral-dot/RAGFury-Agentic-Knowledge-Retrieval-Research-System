"""FastAPI backend for RAGFury."""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import (
    AsyncPostgresSaver,
)
from langsmith import Client, trace

from api.schemas import (
    CitationResponse,
    FeedbackRequest,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrievedDocument,
    SystemInfoResponse,
)
from src.cache.query_cache import QueryCache
from src.checkpoint.postgres import (
    get_checkpoint_database_url,
)
from src.config.config import Config
from src.graph_builder.graph_builder import GraphBuilder
from src.guardrails.exceptions import (
    MaliciousDocumentError,
)
from src.guardrails.guardrail_manager import (
    check_input,
    check_output,
)
from src.rate_limit.rate_limiter import (
    GlobalRateLimiter,
)
from src.utils.langsmith_observability import (
    build_trace_metadata,
    build_trace_tags,
)
from src.utils.loggers import (
    configure_logging,
    get_logger,
    log_event,
)
from src.vectorstore.vectorstore import VectorStore

CACHE_MODEL_VERSION = Config.LLM_MODEL

CACHE_PROMPT_VERSION = os.getenv(
    "RAG_PROMPT_VERSION",
    "v1",
)

CACHE_INDEX_VERSION = os.getenv(
    "RAG_INDEX_VERSION",
    "v1",
)

configure_logging()

logger = get_logger(__name__)

langsmith_client = Client()


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
    ) -> None:

        self.llm = None

        self.vector_store = None
        self.graph_builder = None

        # -----------------------------------------------------
        # PostgreSQL LangGraph checkpointer
        # -----------------------------------------------------

        self.checkpointer = None

        self.graph = None

        self.num_chunks = 0
        self.initialized = False

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
        # Vector store
        # -----------------------------------------------------

        log_event(
            logger,
            level=logging.INFO,
            event="vectorstore.initialization.started",
            mode="query",
        )

        self.vector_store = VectorStore(mode="query")

        log_event(
            logger,
            level=logging.INFO,
            event="vectorstore.initialization.completed",
            mode="query",
        )

        self.vector_store.initialize()
        # -----------------------------------------------------
        # Incremental document synchronization
        # -----------------------------------------------------

        # self._sync_documents()

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

    # ===========================================================
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

        trace_metadata = build_trace_metadata(
            request_id=request_id,
            user_id=user_id,
            conversation_id=conversation_id,
            thread_id=thread_id,
            workflow="rag",
        )

        trace_tags = build_trace_tags(
            workflow="rag",
        )

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
            "metadata": trace_metadata,
            "tags": trace_tags,
        }

        # -----------------------------------------------------
        # Execute LangGraph asynchronously
        # -----------------------------------------------------

        try:
            with trace(
                name="RAGFury Query",
                run_type="chain",
                inputs={
                    "question": question,
                },
                metadata=trace_metadata,
                tags=trace_tags,
            ) as run:
                result = await asyncio.wait_for(
                    self.graph.ainvoke(
                        initial_state,
                        config=config,
                    ),
                    timeout=Config.get_graph_timeout(),
                )

                run_id = str(run.id)

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

        result["run_id"] = run_id

        return result


# =============================================================
# GLOBAL SERVICE
# =============================================================


rag_service = RAGService()

global_rate_limiter = GlobalRateLimiter(
    redis_url=Config.REDIS_URL,
    limit=Config.get_global_rate_limit_per_minute(),
    window_seconds=60,
)

query_cache = QueryCache(
    redis_url=Config.REDIS_URL,
    ttl_seconds=Config.QUERY_CACHE_TTL_SECONDS,
    key_prefix=Config.QUERY_CACHE_KEY_PREFIX,
    lock_ttl_seconds=Config.QUERY_CACHE_LOCK_TTL_SECONDS,
)
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
        try:
            await query_cache.close()

        except Exception:
            logger.exception(
                "Failed to close query cache",
            )

        try:
            await global_rate_limiter.close()

        except Exception:
            logger.exception(
                "Failed to close global rate limiter",
            )

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
    response: Response,
):
    """Execute a question through RAGFury."""

    request_id = uuid.uuid4().hex

    # =========================================================
    # GLOBAL RATE LIMIT
    # =========================================================

    try:
        rate_limit_result = await global_rate_limiter.check(
            request_id=request_id,
        )

    except Exception as exc:
        log_event(
            logger,
            level=logging.ERROR,
            event="rate_limit.redis_error",
            request_id=request_id,
            error_type=type(exc).__name__,
        )

        logger.exception(
            "Global rate limiter failed",
        )

        raise HTTPException(
            status_code=503,
            detail={
                "code": "RATE_LIMIT_SERVICE_UNAVAILABLE",
                "message": (
                    "Request protection service is temporarily "
                    "unavailable. Please try again later."
                ),
                "request_id": request_id,
            },
        ) from exc

    # =========================================================
    # RATE LIMIT HEADERS
    # =========================================================

    response.headers["X-RateLimit-Limit"] = str(rate_limit_result.limit)

    response.headers["X-RateLimit-Remaining"] = str(rate_limit_result.remaining)

    response.headers["X-RateLimit-Reset"] = str(rate_limit_result.retry_after)

    if not rate_limit_result.allowed:
        log_event(
            logger,
            level=logging.WARNING,
            event="rate_limit.exceeded",
            request_id=request_id,
            limit=rate_limit_result.limit,
            remaining=rate_limit_result.remaining,
            retry_after=rate_limit_result.retry_after,
        )

        raise HTTPException(
            status_code=429,
            detail={
                "code": "GLOBAL_RATE_LIMIT_EXCEEDED",
                "message": ("Too many requests. Please try again later."),
                "request_id": request_id,
            },
            headers={
                "Retry-After": str(rate_limit_result.retry_after),
                "X-RateLimit-Limit": str(rate_limit_result.limit),
                "X-RateLimit-Remaining": str(rate_limit_result.remaining),
            },
        )

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

    cache_key = query_cache.build_shared_key(
        question=question,
        model_version=CACHE_MODEL_VERSION,
        prompt_version=CACHE_PROMPT_VERSION,
        index_version=CACHE_INDEX_VERSION,
    )

    result = None
    cache_hit = False
    should_cache_result = False

    lock_key = None
    lock_token = None

    # ---------------------------------------------------------
    # 1. CACHE LOOKUP
    # ---------------------------------------------------------

    try:
        cached_result = await query_cache.get(cache_key)

    except Exception:
        cached_result = None

        logger.warning(
            "Query cache read failed; continuing without cache.",
            extra={
                "request_id": request_id,
            },
        )

    if cached_result is not None:
        result = cached_result
        cache_hit = True

        logger.info(
            "Query cache hit.",
            extra={
                "request_id": request_id,
            },
        )

    # ---------------------------------------------------------
    # 2. CACHE MISS → DISTRIBUTED LOCK
    # ---------------------------------------------------------

    if not cache_hit:
        lock_key = query_cache.build_lock_key(
            computation_key=cache_key,
        )

        try:
            lock_token = await query_cache.acquire_lock(
                lock_key,
            )

        except Exception:
            lock_token = None

            logger.warning(
                "Query cache lock acquisition failed; continuing without cache.",
                extra={
                    "request_id": request_id,
                },
            )

        # -----------------------------------------------------
        # 3. THIS REQUEST WON THE LOCK
        # -----------------------------------------------------

        if lock_token is not None:
            should_cache_result = True

            logger.info(
                "Query cache lock acquired; executing RAG.",
                extra={
                    "request_id": request_id,
                },
            )

        # -----------------------------------------------------
        # 4. THIS REQUEST LOST THE LOCK
        # -----------------------------------------------------

        else:
            try:
                result = await query_cache.wait_for_result(
                    cache_key,
                    timeout_seconds=(Config.QUERY_CACHE_WAIT_TIMEOUT_SECONDS),
                )

            except Exception:
                result = None

                logger.warning(
                    "Query cache wait failed; continuing with RAG.",
                    extra={
                        "request_id": request_id,
                    },
                )

            if result is not None:
                logger.info(
                    "Query cache result received after waiting.",
                    extra={
                        "request_id": request_id,
                    },
                )

            else:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "The requested result is currently "
                        "being generated. Please retry."
                    ),
                )

    # ---------------------------------------------------------
    # 5. EXECUTE RAG + OUTPUT GUARDRAIL + CACHE
    # ---------------------------------------------------------

    try:
        # -----------------------------------------------------
        # EXECUTE RAG ONLY WHEN REQUIRED
        # -----------------------------------------------------

        try:
            if result is None:
                result = await rag_service.query(
                    question=question,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                )

        except MaliciousDocumentError as exc:
            logger.warning(
                "Malicious document detected during RAG execution.",
                extra={
                    "request_id": request_id,
                    "error": str(exc),
                },
            )

            raise HTTPException(
                status_code=400,
                detail={
                    "error": "malicious_document",
                    "message": str(exc),
                },
            ) from exc

        except Exception as exc:
            logger.exception(
                "RAG execution failed.",
                extra={
                    "request_id": request_id,
                },
            )

            raise HTTPException(
                status_code=500,
                detail="Internal server error.",
            ) from exc

        # =====================================================
        # OUTPUT GUARDRAIL
        # =====================================================

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
                "security.output.allowed"
                if output_allowed
                else "security.output.blocked"
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
                detail=(
                    "The generated response was blocked by the RAGFury safety policy."
                ),
            )

        # =====================================================
        # CACHE GENERATED RESULT
        # =====================================================

        if should_cache_result:
            try:
                # -------------------------------------------------
                # BUILD JSON-SERIALIZABLE CACHE PAYLOAD
                # -------------------------------------------------
                cacheable_result = {
                    "answer": result.get("answer", ""),
                    "documents": [
                        {
                            "page_content": document.page_content,
                            "metadata": document.metadata,
                        }
                        for document in result.get("documents", [])
                    ],
                }

                await query_cache.set(
                    cache_key,
                    cacheable_result,
                )

                logger.info(
                    "Query result cached successfully.",
                    extra={
                        "request_id": request_id,
                        "document_count": len(cacheable_result["documents"]),
                    },
                )

            except Exception:
                logger.exception(
                    "Query cache write failed; returning response without cache.",
                    extra={
                        "request_id": request_id,
                    },
                )

    finally:
        # =====================================================
        # ALWAYS RELEASE DISTRIBUTED LOCK
        # =====================================================

        if lock_key is not None and lock_token is not None:
            try:
                await query_cache.release_lock(
                    lock_key,
                    lock_token,
                )

                logger.info(
                    "Query cache lock released.",
                    extra={
                        "request_id": request_id,
                    },
                )

            except Exception:
                logger.warning(
                    "Query cache lock release failed.",
                    extra={
                        "request_id": request_id,
                    },
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

    # =========================================================
    # CITATIONS
    # =========================================================

    raw_citations = result.get(
        "citations",
        [],
    )

    citations: List[CitationResponse] = []

    for citation in raw_citations:
        if hasattr(
            citation,
            "model_dump",
        ):
            citation_data = citation.model_dump()

        else:
            citation_data = citation

        citations.append(
            CitationResponse(
                citation_id=str(
                    citation_data.get("citation_id"),
                ),
                source=str(
                    citation_data.get("source"),
                ),
                chunk_id=str(
                    citation_data.get("chunk_id"),
                ),
                page=citation_data.get("page"),
            )
        )

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
        request_id=request_id,
        citations=citations,
        run_id=result.get("run_id"),
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


@app.post("/api/v1/feedback")
async def submit_feedback(request: FeedbackRequest):
    try:
        langsmith_client.create_feedback(
            run_id=request.run_id,
            key="user-feedback",
            score=request.score,
            comment=request.comment,
        )

        return {
            "status": "success",
            "run_id": request.run_id,
        }

    except Exception as exc:
        logger.exception("Failed to submit LangSmith feedback")

        raise HTTPException(
            status_code=500,
            detail="Failed to submit feedback.",
        ) from exc
