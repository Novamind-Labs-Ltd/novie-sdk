from novie_prompts import telemetry


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
