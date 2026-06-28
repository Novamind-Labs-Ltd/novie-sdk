from novie_prompts import telemetry, registry, client


def test_noop_when_unset():
    telemetry.set_recorder(None)
    telemetry.record_fallback("planner", "timeout")  # must not raise
    telemetry.record_live("planner")


def test_emits_composite_keys():
    seen = []
    telemetry.set_recorder(seen.append)
    telemetry.record_fallback("planner", "timeout")
    telemetry.record_live("reception/supervisor")
    assert seen == [
        "prompt_fallback_total__planner__timeout",
        "prompt_served_live_total__reception/supervisor",
    ]


def test_throwing_recorder_does_not_propagate(monkeypatch):
    """Finding #1: a recorder that raises must NOT surface from get_managed_prompt."""
    def boom(key):
        raise RuntimeError("metrics exploded")

    telemetry.set_recorder(boom)
    monkeypatch.setattr(client, "get_client", lambda: None)
    try:
        result = registry.get_managed_prompt("p", fallback="FB")
        assert result == "FB"   # fail-soft still returns fallback
    finally:
        telemetry.set_recorder(None)
