import httpx
import pytest
from langfuse.api import NotFoundError
from novie_prompts import registry, telemetry, client


class _Prompt:
    def __init__(self, text):
        self.prompt = text


@pytest.fixture
def recorder():
    seen = []
    telemetry.set_recorder(seen.append)
    yield seen
    telemetry.set_recorder(None)


def _patch_client(monkeypatch, fake):
    monkeypatch.setattr(client, "get_client", lambda: fake)


def test_disabled_returns_fallback(monkeypatch, recorder):
    _patch_client(monkeypatch, None)
    assert registry.get_managed_prompt("p", fallback="FB") == "FB"
    assert recorder == ["prompt_fallback_total__p__disabled"]


def test_live_returns_body(monkeypatch, recorder):
    class C:
        def get_prompt(self, name, **kw):
            assert kw["max_retries"] == 1          # never 0 (would hang)
            assert "fallback" not in kw            # must NOT pass fallback= to SDK
            return _Prompt("LIVE")
    _patch_client(monkeypatch, C())
    assert registry.get_managed_prompt("p", fallback="FB") == "LIVE"
    assert recorder == ["prompt_served_live_total__p"]


def test_timeout_returns_fallback_with_reason(monkeypatch, recorder):
    class C:
        def get_prompt(self, name, **kw):
            raise httpx.ReadTimeout("slow")
    _patch_client(monkeypatch, C())
    assert registry.get_managed_prompt("p", fallback="FB") == "FB"
    assert recorder == ["prompt_fallback_total__p__timeout"]


def test_missing_returns_fallback_with_reason(monkeypatch, recorder):
    class C:
        def get_prompt(self, name, **kw):
            raise NotFoundError("nope")
    _patch_client(monkeypatch, C())
    assert registry.get_managed_prompt("p", fallback="FB") == "FB"
    assert recorder == ["prompt_fallback_total__p__missing"]


def test_chat_type_returns_fallback(monkeypatch, recorder):
    class C:
        def get_prompt(self, name, **kw):
            return _Prompt([{"role": "system", "content": "x"}])  # list, not str
    _patch_client(monkeypatch, C())
    assert registry.get_managed_prompt("p", fallback="FB") == "FB"
    assert recorder == ["prompt_fallback_total__p__chat_type"]


def test_cache_ttl_and_timeout_passed_through(monkeypatch, recorder):
    """Finding #6: cache_ttl_seconds and fetch_timeout_seconds arrive in the SDK call,
    and a configured ttl=0 arrives floored to 1."""
    from novie_prompts import config
    config.set_config(cache_ttl_seconds=0, fetch_timeout_seconds=0,
                      enabled=True, host="http://lf:3000",
                      public_key="pk", secret_key="sk")

    captured = {}

    class C:
        def get_prompt(self, name, **kw):
            captured.update(kw)
            return _Prompt("LIVE")

    _patch_client(monkeypatch, C())
    registry.get_managed_prompt("p", fallback="FB")
    assert "cache_ttl_seconds" in captured
    assert "fetch_timeout_seconds" in captured
    assert captured["cache_ttl_seconds"] == 1       # 0 floored to 1
    assert captured["fetch_timeout_seconds"] == 1   # 0 floored to 1
    config.set_config()  # restore defaults
