import pytest

from src.rate_limit.rate_limiter import (
    GlobalRateLimiter,
    RateLimitResult,
)


class FakeScript:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)

        if self.error:
            raise self.error

        return self.result


class FakeRedis:
    def __init__(self, script_result=None, script_error=None):
        self.script = FakeScript(
            result=script_result,
            error=script_error,
        )
        self.closed = False

    def register_script(self, script):
        self.registered_script = script
        return self.script

    async def aclose(self):
        self.closed = True


@pytest.fixture
def limiter(monkeypatch):
    fake_redis = FakeRedis(
        script_result=[1, 30, 29, 60],
    )

    monkeypatch.setattr(
        "src.rate_limit.rate_limiter.redis.from_url",
        lambda *args, **kwargs: fake_redis,
    )

    limiter = GlobalRateLimiter(
        redis_url="redis://localhost:6379",
        limit=30,
        window_seconds=60,
    )

    return limiter, fake_redis


# ============================================================
# CONSTRUCTOR VALIDATION
# ============================================================


def test_rate_limiter_rejects_non_positive_limit():
    with pytest.raises(
        ValueError,
        match="Rate limit must be greater than zero",
    ):
        GlobalRateLimiter(
            redis_url="redis://localhost:6379",
            limit=0,
        )


def test_rate_limiter_rejects_negative_limit():
    with pytest.raises(
        ValueError,
        match="Rate limit must be greater than zero",
    ):
        GlobalRateLimiter(
            redis_url="redis://localhost:6379",
            limit=-1,
        )


def test_rate_limiter_rejects_non_positive_window():
    with pytest.raises(
        ValueError,
        match="Rate-limit window must be greater than zero",
    ):
        GlobalRateLimiter(
            redis_url="redis://localhost:6379",
            limit=30,
            window_seconds=0,
        )


def test_rate_limiter_rejects_negative_window():
    with pytest.raises(
        ValueError,
        match="Rate-limit window must be greater than zero",
    ):
        GlobalRateLimiter(
            redis_url="redis://localhost:6379",
            limit=30,
            window_seconds=-1,
        )


# ============================================================
# ALLOWED REQUEST
# ============================================================


@pytest.mark.asyncio
async def test_check_returns_allowed_result(limiter):
    rate_limiter, fake_redis = limiter

    result = await rate_limiter.check(
        request_id="request-123",
    )

    assert isinstance(result, RateLimitResult)

    assert result.allowed is True
    assert result.limit == 30
    assert result.remaining == 29
    assert result.retry_after == 60
    assert result.reset_after == 60


# ============================================================
# DENIED REQUEST
# ============================================================


@pytest.mark.asyncio
async def test_check_returns_denied_result(monkeypatch):
    fake_redis = FakeRedis(
        script_result=[0, 30, 0, 17],
    )

    monkeypatch.setattr(
        "src.rate_limit.rate_limiter.redis.from_url",
        lambda *args, **kwargs: fake_redis,
    )

    rate_limiter = GlobalRateLimiter(
        redis_url="redis://localhost:6379",
        limit=30,
        window_seconds=60,
    )

    result = await rate_limiter.check(
        request_id="request-456",
    )

    assert isinstance(result, RateLimitResult)

    assert result.allowed is False
    assert result.limit == 30
    assert result.remaining == 0
    assert result.retry_after == 17
    assert result.reset_after == 17


# ============================================================
# REDIS SCRIPT ARGUMENTS
# ============================================================


@pytest.mark.asyncio
async def test_check_passes_correct_request_id_to_redis_script(
    limiter,
):
    rate_limiter, fake_redis = limiter

    await rate_limiter.check(
        request_id="request-789",
    )

    assert len(fake_redis.script.calls) == 1

    call = fake_redis.script.calls[0]

    assert call["keys"] == [
        "ragfury:rate_limit:global",
    ]

    assert call["args"][2] == "30"
    assert call["args"][3] == "60"
    assert call["args"][4] == "request-789"


@pytest.mark.asyncio
async def test_check_uses_custom_rate_limit_configuration(
    monkeypatch,
):
    fake_redis = FakeRedis(
        script_result=[1, 10, 9, 120],
    )

    monkeypatch.setattr(
        "src.rate_limit.rate_limiter.redis.from_url",
        lambda *args, **kwargs: fake_redis,
    )

    rate_limiter = GlobalRateLimiter(
        redis_url="redis://localhost:6379",
        limit=10,
        window_seconds=120,
        key="custom:rate:key",
    )

    result = await rate_limiter.check(
        request_id="request-custom",
    )

    assert result.allowed is True
    assert result.limit == 10
    assert result.remaining == 9

    call = fake_redis.script.calls[0]

    assert call["keys"] == [
        "custom:rate:key",
    ]

    assert call["args"][2] == "10"
    assert call["args"][3] == "120"
    assert call["args"][4] == "request-custom"


# ============================================================
# REDIS FAILURE
# ============================================================


@pytest.mark.asyncio
async def test_check_propagates_redis_errors(monkeypatch):
    fake_redis = FakeRedis(
        script_error=RuntimeError("Redis unavailable"),
    )

    monkeypatch.setattr(
        "src.rate_limit.rate_limiter.redis.from_url",
        lambda *args, **kwargs: fake_redis,
    )

    rate_limiter = GlobalRateLimiter(
        redis_url="redis://localhost:6379",
        limit=30,
    )

    with pytest.raises(
        RuntimeError,
        match="Redis unavailable",
    ):
        await rate_limiter.check(
            request_id="request-error",
        )


# ============================================================
# CLOSE
# ============================================================


@pytest.mark.asyncio
async def test_close_closes_redis_connection(limiter):
    rate_limiter, fake_redis = limiter

    assert fake_redis.closed is False

    await rate_limiter.close()

    assert fake_redis.closed is True
