from __future__ import annotations

import asyncio
import json

import pytest

from src.cache.query_cache import QueryCache


@pytest.fixture
def cache() -> QueryCache:
    return QueryCache(
        redis_url="redis://localhost:6379",
        ttl_seconds=300,
        key_prefix="test:query_cache",
        lock_prefix="test:query_lock",
        lock_ttl_seconds=10,
    )


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------


def test_shared_key_is_deterministic(cache: QueryCache) -> None:
    key_1 = cache.build_shared_key(
        question="What is the leave policy?",
        model_version="model-v1",
        prompt_version="prompt-v1",
        index_version="index-v1",
    )

    key_2 = cache.build_shared_key(
        question="What is the leave policy?",
        model_version="model-v1",
        prompt_version="prompt-v1",
        index_version="index-v1",
    )

    assert key_1 == key_2


def test_shared_key_normalizes_case_and_whitespace(
    cache: QueryCache,
) -> None:
    key_1 = cache.build_shared_key(
        question="  What   is the Leave Policy? ",
        model_version="model-v1",
        prompt_version="prompt-v1",
        index_version="index-v1",
    )

    key_2 = cache.build_shared_key(
        question="what is the leave policy?",
        model_version="model-v1",
        prompt_version="prompt-v1",
        index_version="index-v1",
    )

    assert key_1 == key_2


def test_different_questions_produce_different_keys(
    cache: QueryCache,
) -> None:
    key_1 = cache.build_shared_key(
        question="What is the leave policy?",
        model_version="model-v1",
        prompt_version="prompt-v1",
        index_version="index-v1",
    )

    key_2 = cache.build_shared_key(
        question="What is the remote work policy?",
        model_version="model-v1",
        prompt_version="prompt-v1",
        index_version="index-v1",
    )

    assert key_1 != key_2


def test_model_version_is_part_of_cache_key(
    cache: QueryCache,
) -> None:
    key_1 = cache.build_shared_key(
        question="What is the leave policy?",
        model_version="model-v1",
        prompt_version="prompt-v1",
        index_version="index-v1",
    )

    key_2 = cache.build_shared_key(
        question="What is the leave policy?",
        model_version="model-v2",
        prompt_version="prompt-v1",
        index_version="index-v1",
    )

    assert key_1 != key_2


def test_prompt_version_is_part_of_cache_key(
    cache: QueryCache,
) -> None:
    key_1 = cache.build_shared_key(
        question="What is the leave policy?",
        model_version="model-v1",
        prompt_version="prompt-v1",
        index_version="index-v1",
    )

    key_2 = cache.build_shared_key(
        question="What is the leave policy?",
        model_version="model-v1",
        prompt_version="prompt-v2",
        index_version="index-v1",
    )

    assert key_1 != key_2


def test_index_version_is_part_of_cache_key(
    cache: QueryCache,
) -> None:
    key_1 = cache.build_shared_key(
        question="What is the leave policy?",
        model_version="model-v1",
        prompt_version="prompt-v1",
        index_version="index-v1",
    )

    key_2 = cache.build_shared_key(
        question="What is the leave policy?",
        model_version="model-v1",
        prompt_version="prompt-v1",
        index_version="index-v2",
    )

    assert key_1 != key_2


# ---------------------------------------------------------------------------
# User-scoped cache keys
# ---------------------------------------------------------------------------


def test_user_scoped_key_is_different_for_different_users(
    cache: QueryCache,
) -> None:
    key_1 = cache.build_user_key(
        question="What is the leave policy?",
        user_id="user-1",
        conversation_id="conversation-1",
        model_version="model-v1",
        prompt_version="prompt-v1",
        index_version="index-v1",
    )

    key_2 = cache.build_user_key(
        question="What is the leave policy?",
        user_id="user-2",
        conversation_id="conversation-1",
        model_version="model-v1",
        prompt_version="prompt-v1",
        index_version="index-v1",
    )

    assert key_1 != key_2


def test_user_scoped_key_is_different_for_different_conversations(
    cache: QueryCache,
) -> None:
    key_1 = cache.build_user_key(
        question="What is the leave policy?",
        user_id="user-1",
        conversation_id="conversation-1",
        model_version="model-v1",
        prompt_version="prompt-v1",
        index_version="index-v1",
    )

    key_2 = cache.build_user_key(
        question="What is the leave policy?",
        user_id="user-1",
        conversation_id="conversation-2",
        model_version="model-v1",
        prompt_version="prompt-v1",
        index_version="index-v1",
    )

    assert key_1 != key_2


# ---------------------------------------------------------------------------
# Redis integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_and_get(cache: QueryCache) -> None:
    key = "test:query_cache:set_get"

    response = {
        "answer": "Employees receive 20 days of annual leave.",
        "documents": [],
    }

    await cache.delete(key)

    await cache.set(key, response)

    result = await cache.get(key)

    assert result == response

    await cache.delete(key)


@pytest.mark.asyncio
async def test_delete(cache: QueryCache) -> None:
    key = "test:query_cache:delete"

    response = {
        "answer": "test response",
    }

    await cache.delete(key)

    await cache.set(key, response)

    assert await cache.get(key) == response

    deleted = await cache.delete(key)

    assert deleted is True
    assert await cache.get(key) is None


@pytest.mark.asyncio
async def test_get_missing_key_returns_none(
    cache: QueryCache,
) -> None:
    key = "test:query_cache:missing"

    await cache.delete(key)

    result = await cache.get(key)

    assert result is None


@pytest.mark.asyncio
async def test_invalid_cached_json_raises_value_error(
    cache: QueryCache,
) -> None:
    key = "test:query_cache:invalid_json"

    await cache.redis.set(
        key,
        "{invalid-json",
        ex=30,
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        await cache.get(key)

    await cache.delete(key)


@pytest.mark.asyncio
async def test_non_object_cached_value_raises_value_error(
    cache: QueryCache,
) -> None:
    key = "test:query_cache:non_object"

    await cache.redis.set(
        key,
        json.dumps(["not", "an", "object"]),
        ex=30,
    )

    with pytest.raises(
        ValueError,
        match="must be a JSON object",
    ):
        await cache.get(key)

    await cache.delete(key)


# ---------------------------------------------------------------------------
# Locking / single-flight
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_request_acquires_lock(
    cache: QueryCache,
) -> None:
    key = cache.build_lock_key(
        computation_key="test-computation",
    )

    await cache.redis.delete(key)

    token = await cache.acquire_lock(key)

    assert token is not None

    await cache.release_lock(key, token)


@pytest.mark.asyncio
async def test_second_request_cannot_acquire_existing_lock(
    cache: QueryCache,
) -> None:
    key = cache.build_lock_key(
        computation_key="same-computation",
    )

    await cache.redis.delete(key)

    first_token = await cache.acquire_lock(key)
    second_token = await cache.acquire_lock(key)

    assert first_token is not None
    assert second_token is None

    await cache.release_lock(key, first_token)


@pytest.mark.asyncio
async def test_correct_token_releases_lock(
    cache: QueryCache,
) -> None:
    key = cache.build_lock_key(
        computation_key="release-test",
    )

    await cache.redis.delete(key)

    token = await cache.acquire_lock(key)

    assert token is not None

    released = await cache.release_lock(key, token)

    assert released is True

    assert await cache.redis.get(key) is None


@pytest.mark.asyncio
async def test_wrong_token_cannot_release_lock(
    cache: QueryCache,
) -> None:
    key = cache.build_lock_key(
        computation_key="wrong-token-test",
    )

    await cache.redis.delete(key)

    token = await cache.acquire_lock(key)

    assert token is not None

    released = await cache.release_lock(
        key,
        "wrong-token",
    )

    assert released is False

    assert await cache.redis.get(key) == token

    await cache.release_lock(key, token)


@pytest.mark.asyncio
async def test_concurrent_requests_only_one_acquires_lock(
    cache: QueryCache,
) -> None:
    key = cache.build_lock_key(
        computation_key="concurrent-test",
    )

    await cache.redis.delete(key)

    results = await asyncio.gather(
        cache.acquire_lock(key),
        cache.acquire_lock(key),
        cache.acquire_lock(key),
        cache.acquire_lock(key),
        cache.acquire_lock(key),
    )

    successful_tokens = [token for token in results if token is not None]

    assert len(successful_tokens) == 1

    await cache.release_lock(
        key,
        successful_tokens[0],
    )


# ---------------------------------------------------------------------------
# Wait-for-result / single-flight
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_for_result_returns_cached_response(
    cache: QueryCache,
) -> None:
    key = "test:query_cache:wait"

    await cache.delete(key)

    response = {
        "answer": "cached answer",
    }

    async def populate_cache() -> None:
        await asyncio.sleep(0.1)
        await cache.set(key, response)

    asyncio.create_task(populate_cache())

    result = await cache.wait_for_result(
        key,
        timeout_seconds=2,
        poll_interval_seconds=0.05,
    )

    assert result == response

    await cache.delete(key)


@pytest.mark.asyncio
async def test_wait_for_result_times_out(
    cache: QueryCache,
) -> None:
    key = "test:query_cache:timeout"

    await cache.delete(key)

    result = await cache.wait_for_result(
        key,
        timeout_seconds=0.2,
        poll_interval_seconds=0.05,
    )

    assert result is None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ping(cache: QueryCache) -> None:
    assert await cache.ping() is True
