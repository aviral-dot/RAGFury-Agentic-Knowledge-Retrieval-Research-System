"""LangGraph entrypoint for RAGFury."""

# import asyncio
import logging
from contextlib import asynccontextmanager

from src.checkpoint.postgres import (
    create_checkpointer,
    get_checkpoint_database_url,
)
from src.config.config import Config
from src.graph_builder.graph_builder import GraphBuilder
from src.utils.loggers import (
    get_logger,
    log_event,
)
from src.vectorstore.vectorstore import VectorStore

logger = get_logger(__name__)


@asynccontextmanager
async def create_graph():
    """
    Create the RAGFury LangGraph application.

    The PostgreSQL checkpointer remains alive for
    the lifetime of the graph context.
    """

    log_event(
        logger,
        level=logging.INFO,
        event="langgraph.initialization.started",
    )

    # =========================================================
    # LLM
    # =========================================================

    llm = Config.get_llm()

    # =========================================================
    # VECTOR STORE
    # =========================================================

    # vector_store = await asyncio.to_thread(VectorStore)

    # await asyncio.to_thread(vector_store.initialize)

    # retriever = await asyncio.to_thread(vector_store.get_retriever)

    vector_store = VectorStore()

    vector_store.initialize()

    retriever = vector_store.get_retriever()

    # =========================================================
    # CHECKPOINTER
    # =========================================================

    database_url = get_checkpoint_database_url()

    async with create_checkpointer(database_url) as checkpointer:
        # -----------------------------------------------------
        # GRAPH BUILDER
        # -----------------------------------------------------

        # graph_builder = await asyncio.to_thread(
        #     GraphBuilder,
        #     retriever=retriever,
        #     llm=llm,
        #     checkpointer=checkpointer,
        # )

        graph_builder = GraphBuilder(
            retriever=retriever,
            llm=llm,
            checkpointer=checkpointer,
        )

        graph = graph_builder.build()

        log_event(
            logger,
            level=logging.INFO,
            event="langgraph.initialization.completed",
            checkpointer_enabled=True,
        )

        yield graph
