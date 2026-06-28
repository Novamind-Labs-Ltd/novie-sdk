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
