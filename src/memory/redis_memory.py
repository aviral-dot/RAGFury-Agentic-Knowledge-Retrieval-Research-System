import json
import os

import redis


class RedisMemory:
    """Short-term conversational memory backed by Redis."""

    def __init__(self, redis_url: str | None = None, max_messages: int = 20):
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL",
            "redis://localhost:6379",
        )

        self.max_messages = max_messages

        self.client = redis.Redis.from_url(
            self.redis_url,
            decode_responses=True,
        )

    def _key(self, user_id: str, conversation_id: str) -> str:
        return f"chat:{user_id}:{conversation_id}"

    def get_history(
        self,
        user_id: str,
        conversation_id: str,
    ) -> list[dict]:
        key = self._key(user_id, conversation_id)

        messages = self.client.lrange(
            key,
            -self.max_messages,
            -1,
        )

        return [json.loads(message) for message in messages]

    def add_message(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:

        key = self._key(user_id, conversation_id)

        message = {
            "role": role,
            "content": content,
        }

        self.client.rpush(
            key,
            json.dumps(message),
        )

        self.client.ltrim(
            key,
            -self.max_messages,
            -1,
        )

    def add_turn(
        self,
        user_id: str,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:

        self.add_message(
            user_id,
            conversation_id,
            "user",
            user_message,
        )

        self.add_message(
            user_id,
            conversation_id,
            "assistant",
            assistant_message,
        )

    def clear_history(
        self,
        user_id: str,
        conversation_id: str,
    ) -> None:

        self.client.delete(self._key(user_id, conversation_id))

    def health_check(self) -> bool:
        return bool(self.client.ping())
