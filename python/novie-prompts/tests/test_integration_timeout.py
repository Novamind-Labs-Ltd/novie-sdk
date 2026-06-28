import socket
import threading
import time

import pytest

from novie_prompts import client, config, telemetry
from novie_prompts.registry import get_managed_prompt
from novie_prompts.testing import RecordingRecorder


@pytest.fixture
def black_hole_host():
    """A socket that accepts connections and never sends a byte → forces a read timeout."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]
    accepted: list[socket.socket] = []
    stop = threading.Event()

    def _accept_loop():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
                accepted.append(conn)  # hold it open, send nothing
            except OSError:
                continue

    t = threading.Thread(target=_accept_loop, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    stop.set()
    t.join(timeout=1)
    for c in accepted:
        c.close()
    srv.close()


def test_hanging_socket_falls_back_within_budget(black_hole_host, monkeypatch):
    monkeypatch.setenv("NOVIE_OBSERVABILITY_LANGFUSE_ENABLED", "true")
    monkeypatch.setattr(config, "FETCH_TIMEOUT_SECONDS", 1)  # keep the test fast
    config.reset()
    client.reset_client()
    config.configure(host=black_hole_host, public_key="pk-lf-test", secret_key="sk-lf-test")
    rec = RecordingRecorder()
    telemetry.set_recorder(rec)

    start = time.monotonic()
    out = get_managed_prompt("planner", fallback="CONST")
    elapsed = time.monotonic() - start

    assert out == "CONST"
    # max_retries=1 + per-phase timeout=1s → worst case a small multiple; 10s is a safe ceiling.
    assert elapsed < 10, f"fetch was not latency-bounded: {elapsed:.1f}s"
    assert rec.fallbacks and rec.fallbacks[0][0] == "planner"
    assert rec.fallbacks[0][1] in {"timeout", "exception"}  # timeout preferred; never hangs
