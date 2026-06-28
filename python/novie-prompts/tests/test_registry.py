import httpx
import pytest

from novie_prompts import testing
from novie_prompts.registry import get_managed_prompt


def teardown_function():
    testing.reset()


def test_disabled_returns_fallback_and_records(monkeypatch):
    monkeypatch.setenv("NOVIE_OBSERVABILITY_LANGFUSE_ENABLED", "false")
    _, rec = testing.install_fake(text="LIVE")  # even with a fake client, disabled wins
    out = get_managed_prompt("reception/supervisor", fallback="CONST")
    assert out == "CONST"
    assert rec.fallbacks == [("reception/supervisor", "disabled")]
    assert rec.lives == []


def test_no_client_returns_fallback_disabled(monkeypatch):
    monkeypatch.setenv("NOVIE_OBSERVABILITY_LANGFUSE_ENABLED", "true")
    from novie_prompts import client, telemetry

    client.set_client_for_test(None)  # enabled but no client (construction failed)
    rec = testing.RecordingRecorder()
    telemetry.set_recorder(rec)
    out = get_managed_prompt("pm/system", fallback="CONST")
    assert out == "CONST"
    assert rec.fallbacks == [("pm/system", "disabled")]


def test_enabled_live_returns_langfuse_text_and_records(monkeypatch):
    monkeypatch.setenv("NOVIE_OBSERVABILITY_LANGFUSE_ENABLED", "true")
    fake, rec = testing.install_fake(text="LIVE BODY")
    out = get_managed_prompt("analyst/system", fallback="CONST", label="production")
    assert out == "LIVE BODY"
    assert rec.lives == ["analyst/system"]
    assert rec.fallbacks == []
    assert fake.last_call["label"] == "production"


@pytest.fixture
def _enabled(monkeypatch):
    monkeypatch.setenv("NOVIE_OBSERVABILITY_LANGFUSE_ENABLED", "true")


def test_timeout_classified_as_timeout(_enabled):
    _, rec = testing.install_fake(raises=httpx.ReadTimeout("slow"))
    assert get_managed_prompt("planner", fallback="CONST") == "CONST"
    assert rec.fallbacks == [("planner", "timeout")]


def test_not_found_classified_as_missing(_enabled):
    from langfuse.api import NotFoundError

    # NotFoundError construction varies by SDK version; build the cheapest valid instance.
    try:
        err = NotFoundError(body="missing")
    except TypeError:
        err = NotFoundError("missing")
    _, rec = testing.install_fake(raises=err)
    assert get_managed_prompt("planner", fallback="CONST") == "CONST"
    assert rec.fallbacks == [("planner", "missing")]


def test_chat_type_classified_as_chat_type(_enabled):
    _, rec = testing.install_fake(text=[{"role": "system", "content": "x"}])  # list, not str
    assert get_managed_prompt("planner", fallback="CONST") == "CONST"
    assert rec.fallbacks == [("planner", "chat_type")]


def test_generic_exception_classified_as_exception(_enabled):
    _, rec = testing.install_fake(raises=ValueError("weird"))
    assert get_managed_prompt("planner", fallback="CONST") == "CONST"
    assert rec.fallbacks == [("planner", "exception")]


class _RaisingRecorder:
    def record_fallback(self, name, reason):
        raise RuntimeError("buggy recorder")

    def record_live(self, name):
        raise RuntimeError("buggy recorder")


def test_raising_recorder_does_not_break_never_raises(_enabled):
    # NEVER-raises must hold even when the injected recorder is buggy (ADR-075 D6).
    from novie_prompts import client, telemetry

    client.set_client_for_test(testing.FakeClient(text="LIVE BODY"))
    telemetry.set_recorder(_RaisingRecorder())
    # record_live raises → would cascade; the telemetry guard must absorb it.
    assert get_managed_prompt("planner", fallback="CONST") == "LIVE BODY"

    # And on the fallback path, a raising record_fallback must not escape either.
    client.set_client_for_test(None)
    assert get_managed_prompt("planner", fallback="CONST") == "CONST"
