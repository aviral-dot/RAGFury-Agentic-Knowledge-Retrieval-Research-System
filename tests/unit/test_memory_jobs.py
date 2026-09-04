import pytest

import src.memory.memory_jobs as memory_jobs


class FakeMemoryManager:
    def __init__(self):
        self.calls = []

    async def save_turn(
        self,
        *,
        user_id,
        conversation_id,
        user_message,
        assistant_message,
    ):
        self.calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "user_message": user_message,
                "assistant_message": assistant_message,
            }
        )


@pytest.mark.asyncio
async def test_save_memory_turn_delegates_to_memory_manager(monkeypatch):
    manager = FakeMemoryManager()

    monkeypatch.setattr(
        memory_jobs,
        "MemoryManager",
        lambda: manager,
    )

    await memory_jobs.save_memory_turn(
        user_id="user-123",
        conversation_id="conversation-123",
        user_message="What is the leave policy?",
        assistant_message="The employee receives annual leave.",
    )

    assert len(manager.calls) == 1
    assert manager.calls[0] == {
        "user_id": "user-123",
        "conversation_id": "conversation-123",
        "user_message": "What is the leave policy?",
        "assistant_message": "The employee receives annual leave.",
    }


@pytest.mark.asyncio
async def test_save_memory_turn_creates_memory_manager(monkeypatch):
    created = []

    class TrackingMemoryManager:
        def __init__(self):
            created.append(self)

        async def save_turn(self, **kwargs):
            pass

    monkeypatch.setattr(
        memory_jobs,
        "MemoryManager",
        TrackingMemoryManager,
    )

    await memory_jobs.save_memory_turn(
        user_id="user-123",
        conversation_id="conversation-123",
        user_message="Hello",
        assistant_message="Hi there.",
    )

    assert len(created) == 1


@pytest.mark.asyncio
async def test_save_memory_turn_propagates_memory_error(monkeypatch):
    class FailingMemoryManager:
        async def save_turn(self, **kwargs):
            raise RuntimeError("Memory persistence failed")

    monkeypatch.setattr(
        memory_jobs,
        "MemoryManager",
        FailingMemoryManager,
    )

    with pytest.raises(
        RuntimeError,
        match="Memory persistence failed",
    ):
        await memory_jobs.save_memory_turn(
            user_id="user-123",
            conversation_id="conversation-123",
            user_message="Hello",
            assistant_message="Hi there.",
        )


@pytest.mark.asyncio
async def test_save_memory_turn_preserves_empty_messages(monkeypatch):
    manager = FakeMemoryManager()

    monkeypatch.setattr(
        memory_jobs,
        "MemoryManager",
        lambda: manager,
    )

    await memory_jobs.save_memory_turn(
        user_id="user-123",
        conversation_id="conversation-123",
        user_message="",
        assistant_message="",
    )

    assert manager.calls[0]["user_message"] == ""
    assert manager.calls[0]["assistant_message"] == ""
