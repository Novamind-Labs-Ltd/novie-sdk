from novie_prompts import config


def setup_function():
    config.reset()


def test_no_connection_by_default():
    assert config.get_connection() is None


def test_configure_sets_connection():
    config.configure(host="http://lf:3000", public_key="pk", secret_key="sk")
    conn = config.get_connection()
    assert conn is not None
    assert conn.host == "http://lf:3000"
    assert conn.public_key == "pk"
    assert conn.secret_key == "sk"


def test_is_enabled_defaults_false(monkeypatch):
    monkeypatch.delenv("NOVIE_OBSERVABILITY_LANGFUSE_ENABLED", raising=False)
    assert config.is_enabled() is False


def test_is_enabled_reads_env_per_call(monkeypatch):
    monkeypatch.setenv("NOVIE_OBSERVABILITY_LANGFUSE_ENABLED", "true")
    assert config.is_enabled() is True
    monkeypatch.setenv("NOVIE_OBSERVABILITY_LANGFUSE_ENABLED", "false")
    assert config.is_enabled() is False  # re-read per call = kill switch


def test_timeout_constants():
    assert config.FETCH_TIMEOUT_SECONDS == 2
    assert config.CACHE_TTL_SECONDS == 60
