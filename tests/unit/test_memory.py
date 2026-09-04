import json

import pytest

from src.memory.memo_memory import Mem0Memory
from src.memory.memory_manager import MemoryManager
from src.memory.redis_memory import RedisMemory

# ============================================================
# REDIS MEMORY
# ============================================================


class FakeRedisClient:
    def __init__(self):
        self.storage = {}
        self.expirations = {}

    def lrange(self, key, start, end):
        messages = self.storage.get(key, [])

        if end == -1:
            return messages[start:]

        return messages[start : end + 1]

    def rpush(self, key, value):
        self.storage.setdefault(key, []).append(value)

    def ltrim(self, key, start, end):
        messages = self.storage.get(key, [])

        if end == -1:
            self.storage[key] = messages[start:]
        else:
            self.storage[key] = messages[start : end + 1]

    def expire(self, key, ttl):
        self.expirations[key] = ttl

    def delete(self, key):
        self.storage.pop(key, None)

    def ping(self):
        return True


@pytest.fixture
def fake_redis(monkeypatch):
    client = FakeRedisClient()

    class FakeRedis:
        @classmethod
        def from_url(cls, *args, **kwargs):
            return client

    monkeypatch.setattr(
        "src.memory.redis_memory.redis.Redis",
        FakeRedis,
    )

    return client


def test_redis_memory_initializes_with_defaults(fake_redis):
    memory = RedisMemory()

    assert memory.redis_url
    assert memory.max_messages == 20
    assert memory.ttl_seconds == 1800
    assert memory.client is fake_redis


def test_redis_memory_custom_configuration(fake_redis):
    memory = RedisMemory(
        redis_url="redis://test:6379",
        max_messages=10,
        ttl_seconds=300,
    )

    assert memory.redis_url == "redis://test:6379"
    assert memory.max_messages == 10
    assert memory.ttl_seconds == 300


def test_redis_memory_generates_deterministic_key(fake_redis):
    memory = RedisMemory()

    assert (
        memory._key(
            "user-123",
            "conversation-456",
        )
        == "chat:user-123:conversation-456"
    )


def test_redis_memory_add_message(fake_redis):
    memory = RedisMemory(
        max_messages=20,
        ttl_seconds=1800,
    )

    memory.add_message(
        user_id="user-123",
        conversation_id="conversation-456",
        role="user",
        content="Hello",
    )

    key = "chat:user-123:conversation-456"

    stored = fake_redis.storage[key]

    assert len(stored) == 1

    assert json.loads(stored[0]) == {
        "role": "user",
        "content": "Hello",
    }

    assert fake_redis.expirations[key] == 1800


def test_redis_memory_get_history(fake_redis):
    memory = RedisMemory()

    memory.add_message(
        "user-123",
        "conversation-456",
        "user",
        "Hello",
    )

    memory.add_message(
        "user-123",
        "conversation-456",
        "assistant",
        "Hi there",
    )

    history = memory.get_history(
        "user-123",
        "conversation-456",
    )

    assert history == [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hi there",
        },
    ]


def test_redis_memory_respects_max_messages(fake_redis):
    memory = RedisMemory(max_messages=2)

    memory.add_message(
        "user-123",
        "conversation-456",
        "user",
        "message-1",
    )

    memory.add_message(
        "user-123",
        "conversation-456",
        "assistant",
        "message-2",
    )

    memory.add_message(
        "user-123",
        "conversation-456",
        "user",
        "message-3",
    )

    history = memory.get_history(
        "user-123",
        "conversation-456",
    )

    assert history == [
        {
            "role": "assistant",
            "content": "message-2",
        },
        {
            "role": "user",
            "content": "message-3",
        },
    ]


def test_redis_memory_add_turn(fake_redis):
    memory = RedisMemory()

    memory.add_turn(
        user_id="user-123",
        conversation_id="conversation-456",
        user_message="What is RAG?",
        assistant_message="RAG means Retrieval-Augmented Generation.",
    )

    history = memory.get_history(
        "user-123",
        "conversation-456",
    )

    assert history == [
        {
            "role": "user",
            "content": "What is RAG?",
        },
        {
            "role": "assistant",
            "content": "RAG means Retrieval-Augmented Generation.",
        },
    ]


def test_redis_memory_clear_history(fake_redis):
    memory = RedisMemory()

    memory.add_message(
        "user-123",
        "conversation-456",
        "user",
        "Hello",
    )

    key = "chat:user-123:conversation-456"

    assert key in fake_redis.storage

    memory.clear_history(
        "user-123",
        "conversation-456",
    )

    assert key not in fake_redis.storage


def test_redis_memory_health_check(fake_redis):
    memory = RedisMemory()

    assert memory.health_check() is True


# ============================================================
# MEM0 MEMORY
# ============================================================


class FakeMem0Client:
    def __init__(self):
        self.search_calls = []
        self.add_calls = []
        self.get_all_calls = []

        self.search_result = {
            "results": [
                {"memory": "User prefers Python."},
                {"memory": "User is building a RAG system."},
                {"memory": ""},
                {"other": "ignored"},
            ]
        }

        self.add_result = {
            "results": [
                {"id": "memory-1"},
            ]
        }

        self.get_all_result = [
            {"memory": "User prefers Python."},
            {"memory": "User uses FastAPI."},
        ]

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.search_result

    def add(self, *args, **kwargs):
        self.add_calls.append(
            {
                "args": args,
                "kwargs": kwargs,
            }
        )
        return self.add_result

    def get_all(self, **kwargs):
        self.get_all_calls.append(kwargs)
        return self.get_all_result


@pytest.fixture
def fake_mem0(monkeypatch):
    client = FakeMem0Client()

    class FakeMemory:
        @classmethod
        def from_config(cls, config):
            return client

    monkeypatch.setattr(
        "src.memory.memo_memory.Memory",
        FakeMemory,
    )

    return client


def test_mem0_memory_initializes_without_real_mem0(fake_mem0):
    memory = Mem0Memory()

    assert memory.memory is fake_mem0


def test_mem0_memory_search(fake_mem0):
    memory = Mem0Memory()

    result = memory.search(
        user_id="user-123",
        query="What programming language do I prefer?",
        limit=5,
    )

    assert result == [
        "User prefers Python.",
        "User is building a RAG system.",
    ]

    assert fake_mem0.search_calls == [
        {
            "query": "What programming language do I prefer?",
            "filters": {
                "user_id": "user-123",
            },
            "limit": 5,
        }
    ]


def test_mem0_memory_search_uses_requested_limit(fake_mem0):
    memory = Mem0Memory()

    memory.search(
        user_id="user-123",
        query="RAG",
        limit=3,
    )

    assert fake_mem0.search_calls[0]["limit"] == 3


def test_mem0_memory_add(fake_mem0):
    memory = Mem0Memory()

    result = memory.add(
        user_id="user-123",
        user_message="I use Python.",
        assistant_message="Noted.",
    )

    assert result == {
        "results": [
            {"id": "memory-1"},
        ]
    }

    call = fake_mem0.add_calls[0]

    assert call["kwargs"] == {
        "user_id": "user-123",
    }

    assert call["args"][0] == [
        {
            "role": "user",
            "content": "I use Python.",
        },
        {
            "role": "assistant",
            "content": "Noted.",
        },
    ]


def test_mem0_memory_get_all(fake_mem0):
    memory = Mem0Memory()

    result = memory.get_all(
        user_id="user-123",
    )

    assert result == [
        {"memory": "User prefers Python."},
        {"memory": "User uses FastAPI."},
    ]

    assert fake_mem0.get_all_calls == [
        {
            "user_id": "user-123",
        }
    ]


def test_mem0_memory_search_propagates_backend_error(
    fake_mem0,
):
    memory = Mem0Memory()

    def failing_search(**kwargs):
        raise RuntimeError("Mem0 unavailable")

    fake_mem0.search = failing_search

    with pytest.raises(RuntimeError, match="Mem0 unavailable"):
        memory.search(
            user_id="user-123",
            query="RAG",
        )


def test_mem0_memory_add_propagates_backend_error(
    fake_mem0,
):
    memory = Mem0Memory()

    def failing_add(*args, **kwargs):
        raise RuntimeError("Mem0 unavailable")

    fake_mem0.add = failing_add

    with pytest.raises(RuntimeError, match="Mem0 unavailable"):
        memory.add(
            user_id="user-123",
            user_message="Hello",
            assistant_message="Hi",
        )


# ============================================================
# MEMORY MANAGER
# ============================================================


class FakeRedisMemory:
    def __init__(self):
        self.history_calls = []
        self.add_turn_calls = []

    def get_history(self, **kwargs):
        self.history_calls.append(kwargs)

        return [
            {
                "role": "user",
                "content": "Previous question",
            },
            {
                "role": "assistant",
                "content": "Previous answer",
            },
        ]

    def add_turn(self, **kwargs):
        self.add_turn_calls.append(kwargs)


class FakeMem0Memory:
    def __init__(self):
        self.search_calls = []
        self.add_calls = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)

        return [
            "User prefers concise answers.",
            "User is building RAGFury.",
        ]

    def add(self, **kwargs):
        self.add_calls.append(kwargs)

        return {
            "results": [
                {"id": "memory-1"},
            ]
        }


@pytest.fixture
def memory_manager(monkeypatch):
    redis_memory = FakeRedisMemory()
    mem0_memory = FakeMem0Memory()

    monkeypatch.setattr(
        "src.memory.memory_manager.RedisMemory",
        lambda: redis_memory,
    )

    monkeypatch.setattr(
        "src.memory.memory_manager.Mem0Memory",
        lambda: mem0_memory,
    )

    manager = MemoryManager()

    return manager, redis_memory, mem0_memory


@pytest.mark.asyncio
async def test_memory_manager_get_context(
    memory_manager,
):
    manager, redis_memory, mem0_memory = memory_manager

    result = await manager.get_context(
        user_id="user-123",
        conversation_id="conversation-456",
        query="What do you remember about me?",
    )

    assert result == {
        "recent_history": [
            {
                "role": "user",
                "content": "Previous question",
            },
            {
                "role": "assistant",
                "content": "Previous answer",
            },
        ],
        "long_term_memories": [
            "User prefers concise answers.",
            "User is building RAGFury.",
        ],
    }

    assert redis_memory.history_calls == [
        {
            "user_id": "user-123",
            "conversation_id": "conversation-456",
        }
    ]

    assert mem0_memory.search_calls == [
        {
            "user_id": "user-123",
            "query": "What do you remember about me?",
            "limit": 5,
        }
    ]


@pytest.mark.asyncio
async def test_memory_manager_get_context_propagates_redis_error(
    memory_manager,
):
    manager, redis_memory, _ = memory_manager

    def failing_history(**kwargs):
        raise RuntimeError("Redis unavailable")

    redis_memory.get_history = failing_history

    with pytest.raises(RuntimeError, match="Redis unavailable"):
        await manager.get_context(
            user_id="user-123",
            conversation_id="conversation-456",
            query="RAG",
        )


@pytest.mark.asyncio
async def test_memory_manager_get_context_propagates_mem0_error(
    memory_manager,
):
    manager, _, mem0_memory = memory_manager

    def failing_search(**kwargs):
        raise RuntimeError("Mem0 unavailable")

    mem0_memory.search = failing_search

    with pytest.raises(RuntimeError, match="Mem0 unavailable"):
        await manager.get_context(
            user_id="user-123",
            conversation_id="conversation-456",
            query="RAG",
        )


@pytest.mark.asyncio
async def test_memory_manager_save_turn(
    memory_manager,
):
    manager, redis_memory, mem0_memory = memory_manager

    result = await manager.save_turn(
        user_id="user-123",
        conversation_id="conversation-456",
        user_message="What is RAG?",
        assistant_message="RAG retrieves relevant documents.",
    )

    assert result == {
        "results": [
            {"id": "memory-1"},
        ]
    }

    assert redis_memory.add_turn_calls == [
        {
            "user_id": "user-123",
            "conversation_id": "conversation-456",
            "user_message": "What is RAG?",
            "assistant_message": "RAG retrieves relevant documents.",
        }
    ]

    assert mem0_memory.add_calls == [
        {
            "user_id": "user-123",
            "user_message": "What is RAG?",
            "assistant_message": "RAG retrieves relevant documents.",
        }
    ]


@pytest.mark.asyncio
async def test_memory_manager_save_turn_propagates_redis_error(
    memory_manager,
):
    manager, redis_memory, _ = memory_manager

    def failing_add_turn(**kwargs):
        raise RuntimeError("Redis unavailable")

    redis_memory.add_turn = failing_add_turn

    with pytest.raises(RuntimeError, match="Redis unavailable"):
        await manager.save_turn(
            user_id="user-123",
            conversation_id="conversation-456",
            user_message="Hello",
            assistant_message="Hi",
        )


@pytest.mark.asyncio
async def test_memory_manager_save_turn_propagates_mem0_error(
    memory_manager,
):
    manager, _, mem0_memory = memory_manager

    def failing_add(**kwargs):
        raise RuntimeError("Mem0 unavailable")

    mem0_memory.add = failing_add

    with pytest.raises(RuntimeError, match="Mem0 unavailable"):
        await manager.save_turn(
            user_id="user-123",
            conversation_id="conversation-456",
            user_message="Hello",
            assistant_message="Hi",
        )
