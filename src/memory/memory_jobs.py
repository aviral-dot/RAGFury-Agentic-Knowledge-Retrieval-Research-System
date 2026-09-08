import asyncio


async def _save_memory_turn_async(
    user_id: str,
    conversation_id: str,
    user_message: str,
    assistant_message: str,
):
    from src.memory.memory_manager import MemoryManager

    memory_manager = MemoryManager()

    await memory_manager.save_turn(
        user_id=user_id,
        conversation_id=conversation_id,
        user_message=user_message,
        assistant_message=assistant_message,
    )


def save_memory_turn(
    user_id: str,
    conversation_id: str,
    user_message: str,
    assistant_message: str,
):
    """RQ-compatible synchronous entrypoint for the async memory job."""
    return asyncio.run(
        _save_memory_turn_async(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )
    )
