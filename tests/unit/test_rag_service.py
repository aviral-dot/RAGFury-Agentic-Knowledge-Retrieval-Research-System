import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.main import RAGService

# ============================================================
# Helpers
# ============================================================


def make_initialized_service():
    service = RAGService()

    service.llm = MagicMock()
    service.vector_store = MagicMock()
    service.graph_builder = MagicMock()
    service.graph = MagicMock()

    service.num_chunks = 10
    service.initialized = True

    return service


# ============================================================
# Initialization
# ============================================================


class TestRAGServiceInitialization:
    def test_initial_state(self):
        service = RAGService()

        assert service.llm is None
        assert service.vector_store is None
        assert service.graph_builder is None
        assert service.checkpointer is None
        assert service.graph is None
        assert service.num_chunks == 0
        assert service.initialized is False

    def test_initialize_success(self):
        fake_llm = MagicMock()
        fake_vector_store = MagicMock()
        fake_retriever = MagicMock()
        fake_graph_builder = MagicMock()
        fake_graph = MagicMock()

        fake_vector_store.get_document_count.return_value = 25
        fake_vector_store.get_retriever.return_value = fake_retriever

        fake_graph_builder.build.return_value = fake_graph

        with (
            patch(
                "api.main.Config.get_llm",
                return_value=fake_llm,
            ),
            patch(
                "api.main.VectorStore",
                return_value=fake_vector_store,
            ),
            patch(
                "api.main.GraphBuilder",
                return_value=fake_graph_builder,
            ),
        ):
            service = RAGService()

            service.initialize()

        assert service.llm is fake_llm
        assert service.vector_store is fake_vector_store
        assert service.graph_builder is fake_graph_builder
        assert service.graph is fake_graph
        assert service.num_chunks == 25
        assert service.initialized is True

        fake_vector_store.initialize.assert_called_once()
        fake_vector_store.get_document_count.assert_called_once()
        fake_vector_store.get_retriever.assert_called_once()

        fake_graph_builder.build.assert_called_once()

    def test_initialize_uses_query_vector_store_mode(self):
        fake_llm = MagicMock()
        fake_vector_store = MagicMock()
        fake_vector_store.get_document_count.return_value = 5
        fake_vector_store.get_retriever.return_value = MagicMock()

        fake_graph_builder = MagicMock()
        fake_graph_builder.build.return_value = MagicMock()

        with (
            patch(
                "api.main.Config.get_llm",
                return_value=fake_llm,
            ),
            patch(
                "api.main.VectorStore",
                return_value=fake_vector_store,
            ) as vector_store_cls,
            patch(
                "api.main.GraphBuilder",
                return_value=fake_graph_builder,
            ),
        ):
            service = RAGService()
            service.initialize()

        vector_store_cls.assert_called_once_with(
            mode="query",
        )

    def test_initialize_fails_when_llm_initialization_fails(self):
        service = RAGService()

        with patch(
            "api.main.Config.get_llm",
            side_effect=RuntimeError("LLM initialization failed"),
        ):
            with pytest.raises(
                RuntimeError,
                match="LLM initialization failed",
            ):
                service.initialize()

        assert service.initialized is False

    def test_initialize_fails_when_no_documents_are_indexed(self):
        fake_llm = MagicMock()
        fake_vector_store = MagicMock()

        fake_vector_store.get_document_count.return_value = 0

        with (
            patch(
                "api.main.Config.get_llm",
                return_value=fake_llm,
            ),
            patch(
                "api.main.VectorStore",
                return_value=fake_vector_store,
            ),
        ):
            service = RAGService()

            with pytest.raises(
                ValueError,
                match="No indexed documents found",
            ):
                service.initialize()

        assert service.initialized is False
        assert service.num_chunks == 0

    def test_initialize_propagates_vector_store_failure(self):
        fake_llm = MagicMock()

        with (
            patch(
                "api.main.Config.get_llm",
                return_value=fake_llm,
            ),
            patch(
                "api.main.VectorStore",
                side_effect=RuntimeError("vector store failed"),
            ),
        ):
            service = RAGService()

            with pytest.raises(
                RuntimeError,
                match="vector store failed",
            ):
                service.initialize()

        assert service.initialized is False

    def test_initialize_propagates_graph_build_failure(self):
        fake_llm = MagicMock()
        fake_vector_store = MagicMock()

        fake_vector_store.get_document_count.return_value = 10
        fake_vector_store.get_retriever.return_value = MagicMock()

        with (
            patch(
                "api.main.Config.get_llm",
                return_value=fake_llm,
            ),
            patch(
                "api.main.VectorStore",
                return_value=fake_vector_store,
            ),
            patch(
                "api.main.GraphBuilder",
                side_effect=RuntimeError("graph build failed"),
            ),
        ):
            service = RAGService()

            with pytest.raises(
                RuntimeError,
                match="graph build failed",
            ):
                service.initialize()

        assert service.initialized is False


# ============================================================
# Query validation
# ============================================================


class TestRAGServiceQueryValidation:
    @pytest.mark.asyncio
    async def test_query_rejects_uninitialized_service(self):
        service = RAGService()

        with pytest.raises(
            RuntimeError,
            match="RAGFury has not been initialized",
        ):
            await service.query(
                question="What is the leave policy?",
                user_id="user-1",
                conversation_id="conversation-1",
                request_id="request-1",
            )

    @pytest.mark.asyncio
    async def test_query_rejects_missing_graph(self):
        service = RAGService()
        service.initialized = True
        service.graph = None

        with pytest.raises(
            RuntimeError,
            match="LangGraph is not available",
        ):
            await service.query(
                question="What is the leave policy?",
                user_id="user-1",
                conversation_id="conversation-1",
                request_id="request-1",
            )


# ============================================================
# Successful query
# ============================================================


class TestRAGServiceQuery:
    @pytest.mark.asyncio
    async def test_query_success(self):
        service = make_initialized_service()

        service.graph.ainvoke = AsyncMock(
            return_value={
                "next_step": "rag",
                "answer": "Employees receive paid leave.",
                "retrieved_docs": [],
                "retrieval_attempts": 1,
            }
        )

        fake_trace = MagicMock()
        fake_trace.id = "trace-123"

        fake_trace_context = MagicMock()
        fake_trace_context.__enter__.return_value = fake_trace
        fake_trace_context.__exit__.return_value = False

        with (
            patch(
                "api.main.trace",
                return_value=fake_trace_context,
            ),
            patch(
                "api.main.build_trace_metadata",
                return_value={
                    "request_id": "request-1",
                },
            ),
            patch(
                "api.main.build_trace_tags",
                return_value=["rag"],
            ),
            patch(
                "api.main.Config.get_graph_timeout",
                return_value=30,
            ),
        ):
            result = await service.query(
                question="What is the leave policy?",
                user_id="user-1",
                conversation_id="conversation-1",
                request_id="request-1",
            )

        assert result["answer"] == "Employees receive paid leave."
        assert result["next_step"] == "rag"
        assert result["run_id"] == "trace-123"

        service.graph.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_query_builds_correct_thread_id(self):
        service = make_initialized_service()

        service.graph.ainvoke = AsyncMock(
            return_value={
                "answer": "test answer",
            }
        )

        fake_trace = MagicMock()
        fake_trace.id = "trace-123"

        fake_trace_context = MagicMock()
        fake_trace_context.__enter__.return_value = fake_trace
        fake_trace_context.__exit__.return_value = False

        with (
            patch(
                "api.main.trace",
                return_value=fake_trace_context,
            ),
            patch(
                "api.main.build_trace_metadata",
                return_value={},
            ),
            patch(
                "api.main.build_trace_tags",
                return_value=[],
            ),
            patch(
                "api.main.Config.get_graph_timeout",
                return_value=30,
            ),
        ):
            await service.query(
                question="What is the leave policy?",
                user_id="user-42",
                conversation_id="conversation-99",
                request_id="request-123",
            )

        call_args = service.graph.ainvoke.await_args

        graph_state = call_args.args[0]
        graph_config = call_args.kwargs["config"]

        assert graph_state == {
            "question": "What is the leave policy?",
            "user_id": "user-42",
            "conversation_id": "conversation-99",
            "request_id": "request-123",
        }

        assert (
            graph_config["configurable"]["thread_id"]
            == "ragfury:user-42:conversation-99"
        )

    @pytest.mark.asyncio
    async def test_query_passes_trace_metadata_and_tags(self):
        service = make_initialized_service()

        service.graph.ainvoke = AsyncMock(return_value={"answer": "test"})

        fake_trace = MagicMock()
        fake_trace.id = "trace-456"

        fake_trace_context = MagicMock()
        fake_trace_context.__enter__.return_value = fake_trace
        fake_trace_context.__exit__.return_value = False

        metadata = {
            "request_id": "req-1",
            "user_id": "user-1",
            "conversation_id": "conv-1",
        }

        tags = [
            "rag",
            "production",
        ]

        with (
            patch(
                "api.main.trace",
                return_value=fake_trace_context,
            ),
            patch(
                "api.main.build_trace_metadata",
                return_value=metadata,
            ),
            patch(
                "api.main.build_trace_tags",
                return_value=tags,
            ),
            patch(
                "api.main.Config.get_graph_timeout",
                return_value=30,
            ),
        ):
            await service.query(
                question="test question",
                user_id="user-1",
                conversation_id="conv-1",
                request_id="req-1",
            )

        call_args = service.graph.ainvoke.await_args

        config = call_args.kwargs["config"]

        assert config["metadata"] == metadata
        assert config["tags"] == tags

    @pytest.mark.asyncio
    async def test_query_propagates_graph_error(self):
        service = make_initialized_service()

        service.graph.ainvoke = AsyncMock(
            side_effect=RuntimeError("graph execution failed")
        )

        fake_trace = MagicMock()
        fake_trace.id = "trace-error"

        fake_trace_context = MagicMock()
        fake_trace_context.__enter__.return_value = fake_trace
        fake_trace_context.__exit__.return_value = False

        with (
            patch(
                "api.main.trace",
                return_value=fake_trace_context,
            ),
            patch(
                "api.main.build_trace_metadata",
                return_value={},
            ),
            patch(
                "api.main.build_trace_tags",
                return_value=[],
            ),
            patch(
                "api.main.Config.get_graph_timeout",
                return_value=30,
            ),
        ):
            with pytest.raises(
                RuntimeError,
                match="graph execution failed",
            ):
                await service.query(
                    question="test question",
                    user_id="user-1",
                    conversation_id="conv-1",
                    request_id="req-1",
                )

    @pytest.mark.asyncio
    async def test_query_times_out(self):
        service = make_initialized_service()

        async def slow_graph(*args, **kwargs):
            await asyncio.sleep(0.1)

            return {
                "answer": "should not complete",
            }

        service.graph.ainvoke = slow_graph

        fake_trace = MagicMock()
        fake_trace.id = "trace-timeout"

        fake_trace_context = MagicMock()
        fake_trace_context.__enter__.return_value = fake_trace
        fake_trace_context.__exit__.return_value = False

        with (
            patch(
                "api.main.trace",
                return_value=fake_trace_context,
            ),
            patch(
                "api.main.build_trace_metadata",
                return_value={},
            ),
            patch(
                "api.main.build_trace_tags",
                return_value=[],
            ),
            patch(
                "api.main.Config.get_graph_timeout",
                return_value=0.01,
            ),
        ):
            with pytest.raises(
                asyncio.TimeoutError,
            ):
                await service.query(
                    question="slow question",
                    user_id="user-1",
                    conversation_id="conv-1",
                    request_id="req-1",
                )

    @pytest.mark.asyncio
    async def test_query_adds_run_id_to_existing_graph_result(self):
        service = make_initialized_service()

        original_result = {
            "answer": "Final answer",
            "retrieval_attempts": 2,
        }

        service.graph.ainvoke = AsyncMock(
            return_value=original_result,
        )

        fake_trace = MagicMock()
        fake_trace.id = "trace-final"

        fake_trace_context = MagicMock()
        fake_trace_context.__enter__.return_value = fake_trace
        fake_trace_context.__exit__.return_value = False

        with (
            patch(
                "api.main.trace",
                return_value=fake_trace_context,
            ),
            patch(
                "api.main.build_trace_metadata",
                return_value={},
            ),
            patch(
                "api.main.build_trace_tags",
                return_value=[],
            ),
            patch(
                "api.main.Config.get_graph_timeout",
                return_value=30,
            ),
        ):
            result = await service.query(
                question="test",
                user_id="user",
                conversation_id="conversation",
                request_id="request",
            )

        assert result is original_result
        assert result["run_id"] == "trace-final"
        assert result["answer"] == "Final answer"
        assert result["retrieval_attempts"] == 2
