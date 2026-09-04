import pytest

from src.node.chat_nodes import ChatNode


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeMemoryManager:
    def __init__(self):
        self.redis = FakeRedis()

    def get_context(self, **kwargs):
        return {
            "recent_history": [],
            "long_term_memories": [],
        }


class FakeRedis:
    def __init__(self):
        self.calls = []

    def add_turn(self, **kwargs):
        self.calls.append(kwargs)


class FakeQueue:
    def __init__(self):
        self.calls = []

    def enqueue(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class FakeLLM:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)

        if self.error:
            raise self.error

        return self.response


@pytest.fixture
def state():
    return {
        "question": "What is the leave policy?",
        "user_id": "user-123",
        "conversation_id": "conversation-123",
    }


@pytest.fixture
def patch_memory(monkeypatch):
    memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        "src.node.chat_nodes.MemoryManager",
        lambda: memory_manager,
    )

    return memory_manager


@pytest.fixture
def patch_queue(monkeypatch):
    queue = FakeQueue()

    monkeypatch.setattr(
        "src.node.chat_nodes.memory_queue",
        queue,
    )

    return queue


@pytest.mark.asyncio
async def test_chat_node_propagates_memory_retrieval_error(
    monkeypatch,
    state,
    patch_queue,
):
    class FailingMemoryManager:
        def __init__(self):
            self.redis = FakeRedis()

        def get_context(self, **kwargs):
            raise RuntimeError("Memory service unavailable")

    monkeypatch.setattr(
        "src.node.chat_nodes.MemoryManager",
        FailingMemoryManager,
    )

    llm = FakeLLM(response=FakeResponse("This should not be called."))

    node = ChatNode(llm)

    with pytest.raises(RuntimeError, match="Memory service unavailable"):
        await node.run(state)

    assert llm.calls == []


@pytest.mark.asyncio
async def test_chat_node_propagates_llm_error(
    state,
    patch_memory,
    patch_queue,
):
    llm = FakeLLM(error=RuntimeError("LLM unavailable"))

    node = ChatNode(llm)

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        await node.run(state)

    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_chat_node_rejects_response_without_content(
    state,
    patch_memory,
    patch_queue,
):
    class InvalidResponse:
        pass

    llm = FakeLLM(response=InvalidResponse())

    node = ChatNode(llm)

    with pytest.raises(
        ValueError,
        match="response without content",
    ):
        await node.run(state)

    assert patch_queue.calls == []
    assert patch_memory.redis.calls == []


@pytest.mark.asyncio
async def test_chat_node_rejects_empty_llm_response(
    state,
    patch_memory,
    patch_queue,
):
    llm = FakeLLM(response=FakeResponse("   "))

    node = ChatNode(llm)

    with pytest.raises(
        ValueError,
        match="empty response",
    ):
        await node.run(state)

    assert patch_queue.calls == []
    assert patch_memory.redis.calls == []


@pytest.mark.asyncio
async def test_chat_node_propagates_redis_write_error(
    monkeypatch,
    state,
    patch_queue,
):
    class FailingRedis:
        def add_turn(self, **kwargs):
            raise RuntimeError("Redis unavailable")

    class MemoryManagerWithFailingRedis:
        def __init__(self):
            self.redis = FailingRedis()

        def get_context(self, **kwargs):
            return {
                "recent_history": [],
                "long_term_memories": [],
            }

    monkeypatch.setattr(
        "src.node.chat_nodes.MemoryManager",
        MemoryManagerWithFailingRedis,
    )

    llm = FakeLLM(response=FakeResponse("The employee gets annual leave."))

    node = ChatNode(llm)

    with pytest.raises(
        RuntimeError,
        match="Redis unavailable",
    ):
        await node.run(state)

    assert patch_queue.calls == []


@pytest.mark.asyncio
async def test_chat_node_propagates_memory_queue_error(
    monkeypatch,
    state,
):
    memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        "src.node.chat_nodes.MemoryManager",
        lambda: memory_manager,
    )

    class FailingQueue:
        def enqueue(self, *args, **kwargs):
            raise RuntimeError("Memory queue unavailable")

    monkeypatch.setattr(
        "src.node.chat_nodes.memory_queue",
        FailingQueue(),
    )

    llm = FakeLLM(response=FakeResponse("The employee gets annual leave."))

    node = ChatNode(llm)

    with pytest.raises(
        RuntimeError,
        match="Memory queue unavailable",
    ):
        await node.run(state)

    # Redis persistence happens before the background
    # memory queue operation.
    assert len(memory_manager.redis.calls) == 1


@pytest.mark.asyncio
async def test_chat_node_success_persists_turn_and_enqueues_memory(
    state,
    patch_memory,
    patch_queue,
):
    llm = FakeLLM(response=FakeResponse("The employee gets annual leave."))

    node = ChatNode(llm)

    result = await node.run(state)

    assert result["answer"] == ("The employee gets annual leave.")

    assert len(patch_memory.redis.calls) == 1

    redis_call = patch_memory.redis.calls[0]

    assert redis_call["user_id"] == "user-123"
    assert redis_call["conversation_id"] == "conversation-123"
    assert redis_call["user_message"] == ("What is the leave policy?")
    assert redis_call["assistant_message"] == ("The employee gets annual leave.")

    assert len(patch_queue.calls) == 1

    _, queue_kwargs = patch_queue.calls[0]

    assert queue_kwargs["user_id"] == "user-123"
    assert queue_kwargs["conversation_id"] == "conversation-123"
    assert queue_kwargs["user_message"] == ("What is the leave policy?")
    assert queue_kwargs["assistant_message"] == ("The employee gets annual leave.")
