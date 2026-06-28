from novie_prompts import registry


def _fake_fetch(monkeypatch, value="FETCHED"):
    monkeypatch.setattr(registry, "get_managed_prompt",
                        lambda name, *, fallback, label="production": value)


def test_content_always_fetches(monkeypatch):
    _fake_fetch(monkeypatch)
    out = registry.resolve_prompt("p", fallback="C", tier="content",
                                  is_prod=True, control_plane_fetch_enabled=False)
    assert out == "FETCHED"


def test_control_plane_prod_uses_constant(monkeypatch):
    _fake_fetch(monkeypatch)  # would return FETCHED if called — must NOT be called
    out = registry.resolve_prompt("p", fallback="C", tier="control_plane",
                                  is_prod=True, control_plane_fetch_enabled=True)
    assert out == "C"  # prod control-plane is ALWAYS the constant


def test_control_plane_nonprod_t2_on_fetches(monkeypatch):
    _fake_fetch(monkeypatch)
    out = registry.resolve_prompt("p", fallback="C", tier="control_plane",
                                  is_prod=False, control_plane_fetch_enabled=True)
    assert out == "FETCHED"


def test_control_plane_nonprod_t2_off_uses_constant(monkeypatch):
    _fake_fetch(monkeypatch)
    out = registry.resolve_prompt("p", fallback="C", tier="control_plane",
                                  is_prod=False, control_plane_fetch_enabled=False)
    assert out == "C"


def test_content_ignores_env_flags(monkeypatch):
    """Finding #6: content tier always fetches, regardless of is_prod / control_plane_fetch_enabled."""
    _fake_fetch(monkeypatch)
    out = registry.resolve_prompt("p", fallback="C", tier="content",
                                  is_prod=False, control_plane_fetch_enabled=True)
    assert out == "FETCHED"  # proves content never gates on env


def test_unknown_tier_fails_safe_to_constant(monkeypatch):
    """ADR-075 D4: unknown/typo'd tier returns constant, never fetched."""
    fetch_was_called = []
    def fake_fetch(name, *, fallback, label="production"):
        fetch_was_called.append(True)
        return "FETCHED"
    monkeypatch.setattr(registry, "get_managed_prompt", fake_fetch)
    out = registry.resolve_prompt("p", fallback="C", tier="UNKNOWN_TYPO",
                                  is_prod=False, control_plane_fetch_enabled=True)
    assert out == "C"  # must use constant
    assert not fetch_was_called  # must NOT call get_managed_prompt
