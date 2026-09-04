from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.node.chat_nodes import ChatNode
from src.node.generation_nodes import GenerationNodes
from src.node.grading_nodes import DocumentGrade, GradingNodes
from src.node.retrieval_nodes import RAGNodes
from src.node.rewrite_nodes import RewriteNodes

# ============================================================
# Shared helpers
# ============================================================


def make_document(
    content="Full-time employees must work at least 30 hours per week.",
    source="employee_handbook.pdf",
    page=5,
    chunk_id="chunk-001",
):
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "page": page,
            "chunk_id": chunk_id,
        },
    )


def make_rag_state(**overrides):
    state = {
        "question": "How many hours must a full-time employee work?",
        "user_id": "user-123",
        "conversation_id": "conversation-123",
        "request_id": "request-123",
    }

    state.update(overrides)

    return state


# ============================================================
# RAGNodes
# ============================================================


class TestRAGNodes:
    def test_initialization_stores_dependencies(self):
        retriever = MagicMock()
        llm = MagicMock()

        nodes = RAGNodes(
            retriever=retriever,
            llm=llm,
        )

        assert nodes.retriever is retriever
        assert nodes.llm is llm

    def test_serialize_retrieved_document(self):
        document = Document(
            page_content="Employee content",
            metadata={
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "source": "handbook.pdf",
                "score": 0.91,
            },
        )

        result = RAGNodes._serialize_retrieved_document(
            document,
            rank=1,
        )

        assert result == {
            "rank": 1,
            "document_id": "doc-1",
            "chunk_id": "chunk-1",
            "source": "handbook.pdf",
            "score": 0.91,
        }

    def test_serialize_retrieved_document_handles_missing_metadata(self):
        document = Document(
            page_content="Employee content",
            metadata={},
        )

        result = RAGNodes._serialize_retrieved_document(
            document,
            rank=2,
        )

        assert result == {
            "rank": 2,
            "document_id": None,
            "chunk_id": None,
            "source": None,
            "score": None,
        }

    def test_build_citations_from_documents(self):
        documents = [
            make_document(
                page=0,
                chunk_id="chunk-001",
            ),
            make_document(
                page=4,
                chunk_id="chunk-002",
            ),
        ]

        citations = RAGNodes._build_citations(documents)

        assert len(citations) == 2

        assert citations[0].citation_id == "1"
        assert citations[0].source == "employee_handbook.pdf"
        assert citations[0].chunk_id == "chunk-001"
        assert citations[0].page == 1

        assert citations[1].citation_id == "2"
        assert citations[1].chunk_id == "chunk-002"
        assert citations[1].page == 5

    def test_build_citations_skips_missing_source(self):
        documents = [
            Document(
                page_content="Content",
                metadata={
                    "page": 1,
                    "chunk_id": "chunk-001",
                },
            )
        ]

        citations = RAGNodes._build_citations(documents)

        assert citations == []

    def test_build_citations_skips_missing_chunk_id(self):
        documents = [
            Document(
                page_content="Content",
                metadata={
                    "source": "handbook.pdf",
                    "page": 1,
                },
            )
        ]

        citations = RAGNodes._build_citations(documents)

        assert citations == []

    def test_build_citations_handles_missing_page(self):
        documents = [
            Document(
                page_content="Content",
                metadata={
                    "source": "handbook.pdf",
                    "chunk_id": "chunk-001",
                },
            )
        ]

        citations = RAGNodes._build_citations(documents)

        assert len(citations) == 1
        assert citations[0].page is None

    def test_build_citations_converts_zero_based_page_to_one_based(self):
        documents = [
            make_document(
                page=0,
                chunk_id="chunk-001",
            )
        ]

        citations = RAGNodes._build_citations(documents)

        assert citations[0].page == 1

    def test_build_citations_handles_invalid_page(self):
        documents = [
            Document(
                page_content="Content",
                metadata={
                    "source": "handbook.pdf",
                    "page": "not-a-number",
                    "chunk_id": "chunk-001",
                },
            )
        ]

        citations = RAGNodes._build_citations(documents)

        assert len(citations) == 1
        assert citations[0].page is None

    @pytest.mark.asyncio
    async def test_retrieve_docs_returns_documents_and_citations(self):
        retriever = MagicMock()
        retriever.ainvoke = AsyncMock(
            return_value=[
                make_document(
                    page=4,
                    chunk_id="chunk-001",
                ),
                make_document(
                    page=7,
                    chunk_id="chunk-002",
                ),
            ]
        )

        nodes = RAGNodes(
            retriever=retriever,
            llm=MagicMock(),
        )

        state = make_rag_state(
            retrieval_attempts=0,
        )

        result = await nodes.retrieve_docs(state)

        retriever.ainvoke.assert_awaited_once_with(state["question"])

        assert result["question"] == state["question"]
        assert len(result["retrieved_docs"]) == 2
        assert len(result["citations"]) == 2
        assert result["retrieval_attempts"] == 1

        assert result["retrieval_metadata"]["attempt"] == 1
        assert result["retrieval_metadata"]["document_count"] == 2

    @pytest.mark.asyncio
    async def test_retrieve_docs_increments_existing_attempt_count(self):
        retriever = MagicMock()
        retriever.ainvoke = AsyncMock(return_value=[make_document()])

        nodes = RAGNodes(
            retriever=retriever,
            llm=MagicMock(),
        )

        state = make_rag_state(
            retrieval_attempts=2,
        )

        result = await nodes.retrieve_docs(state)

        assert result["retrieval_attempts"] == 3

    @pytest.mark.asyncio
    async def test_retrieve_docs_handles_empty_retrieval(self):
        retriever = MagicMock()
        retriever.ainvoke = AsyncMock(return_value=[])

        nodes = RAGNodes(
            retriever=retriever,
            llm=MagicMock(),
        )

        result = await nodes.retrieve_docs(make_rag_state())

        assert result["retrieved_docs"] == []
        assert result["citations"] == []
        assert result["retrieval_attempts"] == 1
        assert result["retrieval_metadata"]["document_count"] == 0

    @pytest.mark.asyncio
    async def test_retrieve_docs_propagates_retriever_error(self):
        retriever = MagicMock()
        retriever.ainvoke = AsyncMock(side_effect=RuntimeError("retrieval failed"))

        nodes = RAGNodes(
            retriever=retriever,
            llm=MagicMock(),
        )

        with pytest.raises(RuntimeError, match="retrieval failed"):
            await nodes.retrieve_docs(make_rag_state())


# ============================================================
# GradingNodes
# ============================================================


class TestGradingNodes:
    def test_initialization_creates_structured_grader(self):
        llm = MagicMock()

        structured_grader = MagicMock()

        llm.with_structured_output.return_value = structured_grader

        nodes = GradingNodes(llm=llm)

        assert nodes.llm is llm
        assert nodes.grader is structured_grader

        llm.with_structured_output.assert_called_once_with(
            DocumentGrade,
            method="json_schema",
        )

    @pytest.mark.asyncio
    async def test_grade_documents_returns_false_when_no_documents(self):
        llm = MagicMock()

        nodes = GradingNodes(llm=llm)

        state = make_rag_state(retrieved_docs=[])

        result = await nodes.grade_documents(state)

        assert result == {
            "document_relevance": False,
            "grade_reason": "No documents were retrieved.",
        }

        llm.with_structured_output.return_value.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_grade_documents_returns_relevant_result(self):
        llm = MagicMock()

        grader = MagicMock()
        grader.ainvoke = AsyncMock(
            return_value=DocumentGrade(
                relevant=True,
                reason="The document directly answers the question.",
            )
        )

        llm.with_structured_output.return_value = grader

        nodes = GradingNodes(llm=llm)

        document = make_document()

        state = make_rag_state(retrieved_docs=[document])

        result = await nodes.grade_documents(state)

        assert result["document_relevance"] is True
        assert result["grade_reason"] == "The document directly answers the question."

        grader.ainvoke.assert_awaited_once()

        prompt = grader.ainvoke.await_args.args[0]

        assert state["question"] in prompt
        assert document.page_content in prompt

    @pytest.mark.asyncio
    async def test_grade_documents_returns_irrelevant_result(self):
        llm = MagicMock()

        grader = MagicMock()
        grader.ainvoke = AsyncMock(
            return_value=DocumentGrade(
                relevant=False,
                reason="The retrieved document is unrelated.",
            )
        )

        llm.with_structured_output.return_value = grader

        nodes = GradingNodes(llm=llm)

        result = await nodes.grade_documents(
            make_rag_state(retrieved_docs=[make_document()])
        )

        assert result["document_relevance"] is False
        assert result["grade_reason"] == ("The retrieved document is unrelated.")

    @pytest.mark.asyncio
    async def test_grade_documents_rejects_invalid_structured_response(self):
        llm = MagicMock()

        grader = MagicMock()
        grader.ainvoke = AsyncMock(
            return_value={
                "relevant": True,
                "reason": "Looks relevant",
            }
        )

        llm.with_structured_output.return_value = grader

        nodes = GradingNodes(llm=llm)

        with pytest.raises(
            ValueError,
            match="invalid structured response",
        ):
            await nodes.grade_documents(
                make_rag_state(retrieved_docs=[make_document()])
            )

    @pytest.mark.asyncio
    async def test_grade_documents_propagates_llm_error(self):
        llm = MagicMock()

        grader = MagicMock()
        grader.ainvoke = AsyncMock(side_effect=RuntimeError("grading failed"))

        llm.with_structured_output.return_value = grader

        nodes = GradingNodes(llm=llm)

        with pytest.raises(RuntimeError, match="grading failed"):
            await nodes.grade_documents(
                make_rag_state(retrieved_docs=[make_document()])
            )


# ============================================================
# GenerationNodes
# ============================================================


class TestGenerationNodes:
    def test_initialization_stores_llm(self):
        llm = MagicMock()

        nodes = GenerationNodes(llm=llm)

        assert nodes.llm is llm

    def test_build_citation_context(self):
        document = make_document(
            content="Employees must work at least 30 hours per week.",
            page=4,
            chunk_id="chunk-001",
        )

        citation = MagicMock()
        citation.citation_id = "1"
        citation.source = "employee_handbook.pdf"
        citation.chunk_id = "chunk-001"
        citation.page = 5

        result = GenerationNodes._build_citation_context(
            documents=[document],
            citations=[citation],
        )

        assert "SOURCE [1]" in result
        assert "Document: employee_handbook.pdf" in result
        assert "Page: 5" in result
        assert "Chunk ID: chunk-001" in result
        assert "Employees must work at least 30 hours per week." in result

    def test_build_citation_context_returns_empty_without_documents(self):
        result = GenerationNodes._build_citation_context(
            documents=[],
            citations=[],
        )

        assert result == ""

    def test_build_citation_context_returns_empty_without_citations(self):
        result = GenerationNodes._build_citation_context(
            documents=[make_document()],
            citations=[],
        )

        assert result == ""

    def test_build_citation_context_skips_document_without_chunk_id(self):
        document = Document(
            page_content="Some content",
            metadata={
                "source": "handbook.pdf",
                "page": 1,
            },
        )

        citation = MagicMock()
        citation.citation_id = "1"
        citation.chunk_id = "chunk-001"

        result = GenerationNodes._build_citation_context(
            documents=[document],
            citations=[citation],
        )

        assert result == ""

    def test_build_citation_context_skips_document_without_matching_citation(
        self,
    ):
        document = make_document(chunk_id="chunk-001")

        citation = MagicMock()
        citation.citation_id = "1"
        citation.chunk_id = "different-chunk"

        result = GenerationNodes._build_citation_context(
            documents=[document],
            citations=[citation],
        )

        assert result == ""

    def test_build_citation_context_handles_unavailable_page(self):
        document = make_document()

        citation = MagicMock()
        citation.citation_id = "1"
        citation.source = "handbook.pdf"
        citation.chunk_id = "chunk-001"
        citation.page = None

        result = GenerationNodes._build_citation_context(
            documents=[document],
            citations=[citation],
        )

        assert "Page: unavailable" in result

    @pytest.mark.asyncio
    async def test_generate_answer_returns_answer(self):
        llm = MagicMock()

        llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="A full-time employee must work at least 30 hours per week. [1]"
            )
        )

        nodes = GenerationNodes(llm=llm)

        document = make_document(
            content="A full-time employee must work at least 30 hours per week."
        )

        citation = MagicMock()
        citation.citation_id = "1"
        citation.source = "employee_handbook.pdf"
        citation.chunk_id = "chunk-001"
        citation.page = 6

        result = await nodes.generate_answer(
            make_rag_state(
                retrieved_docs=[document],
                citations=[citation],
            )
        )

        assert result["answer"] == (
            "A full-time employee must work at least 30 hours per week. [1]"
        )

        llm.ainvoke.assert_awaited_once()

        prompt = llm.ainvoke.await_args.args[0]

        assert "SOURCE [1]" in prompt
        assert "chunk-001" in prompt
        assert "Citation rules" in prompt

    @pytest.mark.asyncio
    async def test_generate_answer_strips_response_whitespace(self):
        llm = MagicMock()

        llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="   Answer with whitespace.   ")
        )

        nodes = GenerationNodes(llm=llm)

        result = await nodes.generate_answer(
            make_rag_state(
                retrieved_docs=[make_document()],
                citations=[
                    MagicMock(
                        citation_id="1",
                        source="handbook.pdf",
                        chunk_id="chunk-001",
                        page=1,
                    )
                ],
            )
        )

        assert result["answer"] == "Answer with whitespace."

    @pytest.mark.asyncio
    async def test_generate_answer_rejects_response_without_content(self):
        llm = MagicMock()

        response = MagicMock()
        response.content = None

        llm.ainvoke = AsyncMock(return_value=response)

        nodes = GenerationNodes(llm=llm)

        with pytest.raises(
            ValueError,
            match="without content",
        ):
            await nodes.generate_answer(make_rag_state())

    @pytest.mark.asyncio
    async def test_generate_answer_rejects_empty_response(self):
        llm = MagicMock()

        llm.ainvoke = AsyncMock(return_value=AIMessage(content="   "))

        nodes = GenerationNodes(llm=llm)

        with pytest.raises(
            ValueError,
            match="empty answer",
        ):
            await nodes.generate_answer(make_rag_state())

    @pytest.mark.asyncio
    async def test_generate_answer_propagates_llm_error(self):
        llm = MagicMock()

        llm.ainvoke = AsyncMock(side_effect=RuntimeError("generation failed"))

        nodes = GenerationNodes(llm=llm)

        with pytest.raises(
            RuntimeError,
            match="generation failed",
        ):
            await nodes.generate_answer(make_rag_state())


# ============================================================
# RewriteNodes
# ============================================================


class TestRewriteNodes:
    def test_initialization_stores_llm(self):
        llm = MagicMock()

        nodes = RewriteNodes(llm=llm)

        assert nodes.llm is llm

    @pytest.mark.asyncio
    async def test_rewrite_query_returns_rewritten_question(self):
        llm = MagicMock()

        llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content=" minimum weekly working hours for full-time employees "
            )
        )

        nodes = RewriteNodes(llm=llm)

        state = make_rag_state(
            retrieved_docs=[make_document()],
            grade_reason="The retrieved document was not sufficiently relevant.",
            retrieval_attempts=1,
            reflection_attempts=0,
        )

        result = await nodes.rewrite_query(state)

        assert result["question"] == (
            "minimum weekly working hours for full-time employees"
        )

        llm.ainvoke.assert_awaited_once()

        prompt = llm.ainvoke.await_args.args[0]

        assert state["question"] in prompt
        assert "Grader reason:" in prompt
        assert "Improved search query:" in prompt

    @pytest.mark.asyncio
    async def test_rewrite_query_uses_no_documents_context(self):
        llm = MagicMock()

        llm.ainvoke = AsyncMock(return_value=AIMessage(content="better search query"))

        nodes = RewriteNodes(llm=llm)

        result = await nodes.rewrite_query(make_rag_state(retrieved_docs=[]))

        assert result["question"] == "better search query"

        prompt = llm.ainvoke.await_args.args[0]

        assert "No documents were retrieved." in prompt

    @pytest.mark.asyncio
    async def test_rewrite_query_reads_reflection_attempts(self):
        llm = MagicMock()

        llm.ainvoke = AsyncMock(return_value=AIMessage(content="rewritten query"))

        nodes = RewriteNodes(llm=llm)

        await nodes.rewrite_query(
            make_rag_state(
                retrieved_docs=[],
                retrieval_attempts=2,
                reflection_attempts=3,
            )
        )

        # The node should have prepared rewrite attempt 4.
        prompt = llm.ainvoke.await_args.args[0]

        assert "rewritten query" not in prompt
        assert "Previous search query:" in prompt

    @pytest.mark.asyncio
    async def test_rewrite_query_strips_whitespace(self):
        llm = MagicMock()

        llm.ainvoke = AsyncMock(return_value=AIMessage(content="   rewritten query   "))

        nodes = RewriteNodes(llm=llm)

        result = await nodes.rewrite_query(make_rag_state())

        assert result["question"] == "rewritten query"

    @pytest.mark.asyncio
    async def test_rewrite_query_rejects_response_without_content(self):
        llm = MagicMock()

        response = MagicMock()
        response.content = None

        llm.ainvoke = AsyncMock(return_value=response)

        nodes = RewriteNodes(llm=llm)

        with pytest.raises(
            ValueError,
            match="without content",
        ):
            await nodes.rewrite_query(make_rag_state())

    @pytest.mark.asyncio
    async def test_rewrite_query_rejects_empty_response(self):
        llm = MagicMock()

        llm.ainvoke = AsyncMock(return_value=AIMessage(content="   "))

        nodes = RewriteNodes(llm=llm)

        with pytest.raises(
            ValueError,
            match="empty query",
        ):
            await nodes.rewrite_query(make_rag_state())

    @pytest.mark.asyncio
    async def test_rewrite_query_propagates_llm_error(self):
        llm = MagicMock()

        llm.ainvoke = AsyncMock(side_effect=RuntimeError("rewrite failed"))

        nodes = RewriteNodes(llm=llm)

        with pytest.raises(
            RuntimeError,
            match="rewrite failed",
        ):
            await nodes.rewrite_query(make_rag_state())


# ============================================================
# ChatNode helper methods
# ============================================================


class TestChatNodeHelpers:
    def test_format_history_converts_user_messages(self):
        history = [
            {
                "role": "user",
                "content": "Hello",
            }
        ]

        result = ChatNode._format_history(history)

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hello"

    def test_format_history_converts_assistant_messages(self):
        history = [
            {
                "role": "assistant",
                "content": "Hi there",
            }
        ]

        result = ChatNode._format_history(history)

        assert len(result) == 1
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "Hi there"

    def test_format_history_skips_unknown_roles(self):
        history = [
            {
                "role": "system",
                "content": "Internal message",
            },
            {
                "role": "user",
                "content": "Hello",
            },
        ]

        result = ChatNode._format_history(history)

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hello"

    def test_format_history_skips_empty_content(self):
        history = [
            {
                "role": "user",
                "content": "",
            },
            {
                "role": "assistant",
                "content": "Valid answer",
            },
        ]

        result = ChatNode._format_history(history)

        assert len(result) == 1
        assert result[0].content == "Valid answer"

    def test_format_history_handles_empty_history(self):
        result = ChatNode._format_history([])

        assert result == []

    def test_format_memories_formats_memory_key(self):
        memories = [{"memory": "User prefers concise answers."}]

        result = ChatNode._format_memories(memories)

        assert result == "- User prefers concise answers."

    def test_format_memories_formats_text_key(self):
        memories = [{"text": "User works with Python."}]

        result = ChatNode._format_memories(memories)

        assert result == "- User works with Python."

    def test_format_memories_formats_string_memory(self):
        memories = ["User likes Python."]

        result = ChatNode._format_memories(memories)

        assert result == "- User likes Python."

    def test_format_memories_handles_empty_memories(self):
        result = ChatNode._format_memories([])

        assert result == "No relevant long-term memories found."

    def test_format_memories_skips_empty_memory_values(self):
        memories = [
            {"memory": ""},
            {"memory": "Valid memory"},
        ]

        result = ChatNode._format_memories(memories)

        assert result == "- Valid memory"

    def test_format_memories_handles_only_invalid_memories(self):
        memories = [
            {"memory": ""},
            {"text": ""},
        ]

        result = ChatNode._format_memories(memories)

        assert result == "No relevant long-term memories found."


# ============================================================
# ChatNode execution
# ============================================================


class TestChatNodeRun:
    @pytest.mark.asyncio
    async def test_run_generates_answer_and_persists_memory(self):
        llm = MagicMock()

        llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Hello, how can I help?")
        )

        fake_memory_manager = MagicMock()

        fake_memory_manager.get_context.return_value = {
            "recent_history": [
                {
                    "role": "user",
                    "content": "Previous question",
                }
            ],
            "long_term_memories": [{"memory": "User prefers concise answers."}],
        }

        fake_redis = MagicMock()
        fake_memory_manager.redis = fake_redis

        with patch(
            "src.node.chat_nodes.MemoryManager",
            return_value=fake_memory_manager,
        ):
            with patch("src.node.chat_nodes.memory_queue") as mock_queue:
                node = ChatNode(llm=llm)

                result = await node.run(
                    make_rag_state(question="What can you help me with?")
                )

        assert result["answer"] == "Hello, how can I help?"

        assert len(result["chat_history"]) == 1
        assert result["chat_history"][0]["content"] == "Previous question"

        assert len(result["relevant_memories"]) == 1

        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], HumanMessage)
        assert isinstance(result["messages"][1], AIMessage)

        fake_memory_manager.get_context.assert_called_once()

        fake_redis.add_turn.assert_called_once()

        mock_queue.enqueue.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_rejects_llm_response_without_content(self):
        llm = MagicMock()

        response = MagicMock()
        response.content = None

        llm.ainvoke = AsyncMock(return_value=response)

        fake_memory_manager = MagicMock()

        fake_memory_manager.get_context.return_value = {
            "recent_history": [],
            "long_term_memories": [],
        }

        with patch(
            "src.node.chat_nodes.MemoryManager",
            return_value=fake_memory_manager,
        ):
            node = ChatNode(llm=llm)

            with pytest.raises(
                ValueError,
                match="without content",
            ):
                await node.run(make_rag_state())

    @pytest.mark.asyncio
    async def test_run_rejects_empty_llm_response(self):
        llm = MagicMock()

        llm.ainvoke = AsyncMock(return_value=AIMessage(content="   "))

        fake_memory_manager = MagicMock()

        fake_memory_manager.get_context.return_value = {
            "recent_history": [],
            "long_term_memories": [],
        }

        with patch(
            "src.node.chat_nodes.MemoryManager",
            return_value=fake_memory_manager,
        ):
            node = ChatNode(llm=llm)

            with pytest.raises(
                ValueError,
                match="empty response",
            ):
                await node.run(make_rag_state())

    @pytest.mark.asyncio
    async def test_run_propagates_memory_context_error(self):
        llm = MagicMock()

        fake_memory_manager = MagicMock()

        fake_memory_manager.get_context.side_effect = RuntimeError("memory unavailable")

        with patch(
            "src.node.chat_nodes.MemoryManager",
            return_value=fake_memory_manager,
        ):
            node = ChatNode(llm=llm)

            with pytest.raises(
                RuntimeError,
                match="memory unavailable",
            ):
                await node.run(make_rag_state())

    @pytest.mark.asyncio
    async def test_run_propagates_llm_error(self):
        llm = MagicMock()

        llm.ainvoke = AsyncMock(side_effect=RuntimeError("chat llm failed"))

        fake_memory_manager = MagicMock()

        fake_memory_manager.get_context.return_value = {
            "recent_history": [],
            "long_term_memories": [],
        }

        with patch(
            "src.node.chat_nodes.MemoryManager",
            return_value=fake_memory_manager,
        ):
            node = ChatNode(llm=llm)

            with pytest.raises(
                RuntimeError,
                match="chat llm failed",
            ):
                await node.run(make_rag_state())
