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
