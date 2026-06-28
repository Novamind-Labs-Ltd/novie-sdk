from novie_prompts import telemetry


class _Rec:
    def __init__(self):
        self.fallbacks = []
        self.lives = []

    def record_fallback(self, name, reason):
        self.fallbacks.append((name, reason))

    def record_live(self, name):
        self.lives.append(name)


def setup_function():
    telemetry.set_recorder(None)


def test_no_recorder_is_noop_and_does_not_raise():
    assert telemetry.has_recorder() is False
    telemetry.record_fallback("p", "disabled")  # must not raise
    telemetry.record_live("p")  # must not raise


def test_injected_recorder_receives_calls():
    rec = _Rec()
    telemetry.set_recorder(rec)
    assert telemetry.has_recorder() is True
    telemetry.record_fallback("planner", "timeout")
    telemetry.record_live("supervisor")
    assert rec.fallbacks == [("planner", "timeout")]
    assert rec.lives == ["supervisor"]
