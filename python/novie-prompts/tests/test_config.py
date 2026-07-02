import pytest

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


def test_connection_repr_redacts_credentials():
    config.configure(host="https://lf", public_key="pk-lf-abc", secret_key="sk-lf-SECRET")
    text = repr(config.get_connection())
    assert "sk-lf-SECRET" not in text
    assert "pk-lf-abc" not in text
    assert "<redacted>" in text
    assert "https://lf" in text  # host is fine to show


def test_is_enabled_defaults_false(monkeypatch):
    monkeypatch.delenv("NOVIE_OBSERVABILITY_LANGFUSE_ENABLED", raising=False)
    assert config.is_enabled() is False


def test_is_enabled_reads_env_per_call(monkeypatch):
    monkeypatch.setenv("NOVIE_OBSERVABILITY_LANGFUSE_ENABLED", "true")
    assert config.is_enabled() is True
    monkeypatch.setenv("NOVIE_OBSERVABILITY_LANGFUSE_ENABLED", "false")
    assert config.is_enabled() is False  # re-read per call = kill switch


def test_is_enabled_tolerates_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv("NOVIE_OBSERVABILITY_LANGFUSE_ENABLED", " true ")
    assert config.is_enabled() is True  # a stray space in the env file must not silently disable


def test_timeout_constants():
    assert config.FETCH_TIMEOUT_SECONDS == 2
    assert config.CACHE_TTL_SECONDS == 60


def test_resolve_label_defaults_to_development(monkeypatch):
    monkeypatch.delenv("NOVIE_RUNTIME_MODE", raising=False)
    monkeypatch.delenv("NOVIE_ENV", raising=False)
    assert config.resolve_label() == "development"


@pytest.mark.parametrize(
    "runtime_mode,expected",
    [("production", "production"), ("uat", "uat"), ("dev", "development"),
     ("PRODUCTION", "production"), (" uat ", "uat")],
)
def test_resolve_label_reads_runtime_mode(monkeypatch, runtime_mode, expected):
    monkeypatch.setenv("NOVIE_RUNTIME_MODE", runtime_mode)
    monkeypatch.setenv("NOVIE_ENV", "production")  # must not win over an explicit RUNTIME_MODE
    assert config.resolve_label() == expected


def test_resolve_label_dev_never_escalates_via_legacy_env(monkeypatch):
    # Mirrors is_production_mode()'s escalation guard: explicit dev wins even
    # when a stray legacy NOVIE_ENV=production is also set.
    monkeypatch.setenv("NOVIE_RUNTIME_MODE", "dev")
    monkeypatch.setenv("NOVIE_ENV", "production")
    assert config.resolve_label() == "development"


@pytest.mark.parametrize("legacy,expected", [("production", "production"), ("uat", "uat"), ("staging", "development")])
def test_resolve_label_falls_back_to_legacy_env(monkeypatch, legacy, expected):
    monkeypatch.delenv("NOVIE_RUNTIME_MODE", raising=False)
    monkeypatch.setenv("NOVIE_ENV", legacy)
    assert config.resolve_label() == expected


def test_resolve_label_reads_env_per_call(monkeypatch):
    monkeypatch.setenv("NOVIE_RUNTIME_MODE", "production")
    assert config.resolve_label() == "production"
    monkeypatch.setenv("NOVIE_RUNTIME_MODE", "dev")
    assert config.resolve_label() == "development"  # re-read per call, same as is_enabled()
