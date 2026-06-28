import socket
import time
import pytest
from novie_prompts import config, client, registry


@pytest.fixture
def black_hole_host():
    # A listening socket that accepts then never responds → read timeout.
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    srv.close()


def test_cold_outage_returns_within_budget(black_hole_host):
    config.set_config(enabled=True, host=black_hole_host,
                      public_key="pk", secret_key="sk",
                      fetch_timeout_seconds=2, cache_ttl_seconds=60)
    client.reset_client()
    start = time.monotonic()
    out = registry.get_managed_prompt("p", fallback="FB")
    elapsed = time.monotonic() - start
    assert out == "FB"
    # max_retries=1 → ~1 attempt; per-phase timeout means a small multiple, not ∞.
    # Generous ceiling that still FAILS loudly if someone sets max_retries=0 (hangs).
    assert elapsed < 15, f"took {elapsed:.1f}s — check max_retries is 1, not 0"
