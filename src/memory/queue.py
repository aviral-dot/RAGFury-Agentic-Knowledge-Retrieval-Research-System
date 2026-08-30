from redis import Redis
from rq import Queue

redis_connection = Redis(
    host="localhost",
    port=6379,
    decode_responses=True,
)

memory_queue = Queue(
    "memory",
    connection=redis_connection,
)
