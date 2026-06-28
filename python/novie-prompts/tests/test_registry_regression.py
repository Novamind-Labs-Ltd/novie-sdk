import pytest

from novie_prompts import testing
from novie_prompts.registry import get_managed_prompt


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setenv("NOVIE_OBSERVABILITY_LANGFUSE_ENABLED", "true")
    yield
    testing.reset()


def test_regression_max_retries_is_one_and_no_fallback_kwarg():
    fake, _ = testing.install_fake(text="LIVE")
    get_managed_prompt("planner", fallback="CONST")
    assert fake.last_call["max_retries"] == 1, "max_retries=0 means retry-forever (ADR-040)"
    assert "fetch_timeout_seconds" in fake.last_call
    assert fake.last_call["fetch_timeout_seconds"] == 2
    assert "fallback" not in fake.last_call, "passing fallback= makes the SDK swallow errors"
    # cache_ttl_seconds==60 is the only package-side guard for the spec's
    # "TTL-expired → cached without blocking" P0 row (the rest is SDK behavior).
    assert fake.last_call["cache_ttl_seconds"] == 60
