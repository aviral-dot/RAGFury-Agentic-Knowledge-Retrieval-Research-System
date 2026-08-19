from src.memory.memory_manager import MemoryManager


def save_memory_turn(
    user_id: str,
    conversation_id: str,
    user_message: str,
    assistant_message: str,
):
    memory_manager = MemoryManager()

    memory_manager.save_turn(
        user_id=user_id,
        conversation_id=conversation_id,
        user_message=user_message,
        assistant_message=assistant_message,
    )