from src.memory.redis_memory import RedisMemory
from src.memory.memo_memory import Mem0Memory


class MemoryManager:
    """Coordinates short-term Redis and long-term Mem0 memory."""

    def __init__(self):
        self.redis = RedisMemory()
        self.mem0 = Mem0Memory()

    def get_context(
        self,
        user_id: str,
        conversation_id: str,
        query: str,
    ):
        recent_history = self.redis.get_history(
            user_id=user_id,
            conversation_id=conversation_id,
        )

        long_term_memories = self.mem0.search(
            user_id=user_id,
            query=query,
            limit=5,
        )

        print("short term:", recent_history)
        print("long  term:", long_term_memories)

        return {
            "recent_history": recent_history,
            "long_term_memories": long_term_memories,
        }

    def save_turn(
        self,
        user_id: str,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
    ):


        self.mem0.add(
            user_id=user_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )