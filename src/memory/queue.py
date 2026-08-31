from redis import Redis
from rq import Queue

from src.config.config import Config

redis_connection = Redis.from_url(
    Config.REDIS_URL,
    decode_responses=True,
)

memory_queue = Queue(
    "memory",
    connection=redis_connection,
)
