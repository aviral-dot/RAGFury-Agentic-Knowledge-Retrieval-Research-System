import os

from mem0 import Memory
from dotenv import load_dotenv
load_dotenv()
 
class Mem0Memory:
    """Long-term semantic memory backed by Mem0 + Qdrant."""

    def __init__(self):
        config = {
            "llm": {
        "provider": "openai",
        "config": {
            "model": "openrouter/free",
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "openai_base_url": "https://openrouter.ai/api/v1",
            "temperature": 0.1,
        },
    },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "ragfury_memories",
                    "url": os.getenv(
                        "QDRANT_URL",
                        "http://localhost:6333",
                    ),
                    "embedding_model_dims": 768,
                },
            },
            "embedder": {
                "provider": "gemini",
                "config": {
                    "model": "models/gemini-embedding-001",
                    "embedding_dims": 768,
                    "api_key": os.getenv("GEMINI_API_KEY"),
                },
            },
        }

        self.memory = Memory.from_config(config)

    def search(
    self,
    user_id: str,
    query: str,
    limit: int = 5,
):
        result =  self.memory.search(
        query=query,
        filters={
            "user_id": user_id
        },
        limit=limit,
    )
        print("MEM0 RESULT:", result)
        print("==============================\n")

        memories = [
        item["memory"]
        for item in result.get("results", [])
        if item.get("memory")
    ]

        print("EXTRACTED MEMORIES:", memories)
        print("==============================\n")

        return memories

    def add(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
    ):
        result= self.memory.add(
            [
                {
                    "role": "user",
                    "content": user_message,
                },
                {
                    "role": "assistant",
                    "content": assistant_message,
                },
            ],
            user_id=user_id
        )

        print("MEM0 RESULT:", result)
        print("==============================\n")

        return result

    def get_all(self, user_id: str):
        return self.memory.get_all(
            user_id=user_id
        )