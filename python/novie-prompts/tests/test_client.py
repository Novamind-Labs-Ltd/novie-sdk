from novie_prompts import client, config


def setup_function():
    config.reset()
    client.reset_client()


def test_no_connection_returns_none():
    assert client.get_client() is None


def test_test_override_takes_precedence():
    sentinel = object()
    client.set_client_for_test(sentinel)
    assert client.get_client() is sentinel


def test_override_none_is_respected():
    client.set_client_for_test(None)
    # even with a connection configured, an explicit None override wins
    config.configure(host="http://lf", public_key="pk", secret_key="sk")
    assert client.get_client() is None


def test_construction_failure_yields_none(monkeypatch):
    # Point at a connection but force the langfuse import/construction to blow up.
    config.configure(host="http://lf", public_key="pk", secret_key="sk")

    def _boom(*a, **k):
        raise RuntimeError("no langfuse")

    monkeypatch.setattr(client, "_build_client", _boom)
    assert client.get_client() is None  # construction failure = disabled mode = instant fallback
