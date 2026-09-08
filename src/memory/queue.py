"""RQ queue configuration for long-term memory jobs."""

import os

from redis import Redis
from rq import Queue


def get_redis_url() -> str:
    """Return the Redis URL from the runtime environment."""

    redis_url = os.getenv("REDIS_URL")

    if not redis_url:
        raise RuntimeError("REDIS_URL is required to use the memory worker queue.")

    return redis_url


def get_memory_queue() -> Queue:
    """
    Create and return the RQ memory queue.

    Redis is initialized lazily so importing the application does not
    require Redis connectivity during serverless cold start.
    """

    redis_connection = Redis.from_url(
        get_redis_url(),
    )

    return Queue(
        "memory",
        connection=redis_connection,
    )
