from novie_prompts import client, config


def test_disabled_returns_none():
    config.set_config(enabled=False)
    client.reset_client()
    assert client.get_client() is None


def test_enabled_constructs_non_network(monkeypatch):
    # Construction must be non-network: a fake Langfuse that records it was built.
    built = {}

    class FakeLangfuse:
        def __init__(self, **kw):
            built.update(kw)

    monkeypatch.setattr(client, "_Langfuse", FakeLangfuse)
    config.set_config(enabled=True, host="http://lf:3000",
                      public_key="pk", secret_key="sk")
    client.reset_client()
    c = client.get_client()
    assert c is not None
    assert built["host"] == "http://lf:3000"


def test_construction_failure_returns_none(monkeypatch):
    def boom(**kw):
        raise RuntimeError("bad init")

    monkeypatch.setattr(client, "_Langfuse", boom)
    config.set_config(enabled=True, host="http://lf:3000",
                      public_key="pk", secret_key="sk")
    client.reset_client()
    assert client.get_client() is None  # init failure = disabled mode = instant fallback


def test_enabled_but_no_host_returns_none():
    """Finding #6: enabled=True but host=None → the `or` guard → None."""
    config.set_config(enabled=True, host=None)
    client.reset_client()
    assert client.get_client() is None


def test_set_config_invalidates_singleton(monkeypatch):
    """Finding #3: set_config must invalidate cached singleton."""
    config.set_config(enabled=False)
    assert client.get_client() is None          # disabled → None

    class FakeLangfuse:
        def __init__(self, **kw):
            pass

    monkeypatch.setattr(client, "_Langfuse", FakeLangfuse)
    config.set_config(enabled=True, host="http://lf:3000",
                      public_key="pk", secret_key="sk")
    # No explicit reset_client() — set_config must have done it
    assert client.get_client() is not None
