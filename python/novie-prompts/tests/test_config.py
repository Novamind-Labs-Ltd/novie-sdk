from novie_prompts import config


def test_defaults():
    config.set_config()  # reset to defaults
    assert config.is_enabled() is False
    assert config.cache_ttl_seconds() == 60
    assert config.fetch_timeout_seconds() == 2
    assert config.host() is None


def test_ttl_floored_at_1():
    config.set_config(cache_ttl_seconds=0)
    assert config.cache_ttl_seconds() == 1  # 0 would disable caching; floor it


def test_fetch_timeout_floored_at_1():
    """Finding #5: fetch_timeout_seconds=0 means 'always fall back' silently; floor it."""
    config.set_config(fetch_timeout_seconds=0)
    assert config.fetch_timeout_seconds() == 1


def test_overrides_round_trip():
    config.set_config(enabled=True, host="http://lf:3000",
                      cache_ttl_seconds=30, fetch_timeout_seconds=1,
                      public_key="pk", secret_key="sk")
    assert config.is_enabled() is True
    assert config.host() == "http://lf:3000"
    cur = config.current()
    assert cur.public_key == "pk" and cur.secret_key == "sk"
