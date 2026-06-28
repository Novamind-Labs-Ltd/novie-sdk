import pytest

from novie_prompts import testing


def teardown_function():
    testing.reset()


def test_recording_recorder_collects():
    rec = testing.RecordingRecorder()
    rec.record_fallback("p", "timeout")
    rec.record_live("q")
    assert rec.fallbacks == [("p", "timeout")]
    assert rec.lives == ["q"]


def test_fake_client_returns_text_prompt_and_captures_kwargs():
    fake = testing.FakeClient(text="hello")
    prompt = fake.get_prompt("planner", label="production", max_retries=1)
    assert prompt.prompt == "hello"
    assert fake.last_call == {"name": "planner", "label": "production", "max_retries": 1}


def test_fake_client_raises_when_configured():
    boom = ValueError("nope")
    fake = testing.FakeClient(raises=boom)
    with pytest.raises(ValueError):
        fake.get_prompt("planner")


def test_install_fake_wires_client_and_recorder():
    from novie_prompts import client, telemetry

    fake, rec = testing.install_fake(text="hi")
    assert client.get_client() is fake
    assert telemetry.has_recorder() is True
    assert isinstance(rec, testing.RecordingRecorder)
