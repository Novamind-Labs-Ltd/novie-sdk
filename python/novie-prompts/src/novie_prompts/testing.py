"""CI seam for all 6 consumer repos (no Langfuse in CI). The hard-to-fake case
is a real socket timeout; disabled/force-value are just config/monkeypatch."""
from __future__ import annotations
import contextlib
import httpx
from . import client


class _TimeoutClient:
    def get_prompt(self, name, **kw):
        raise httpx.ReadTimeout("forced by fake_registry")


@contextlib.contextmanager
def fake_registry(*, mode: str = "timeout"):
    original = client.get_client
    if mode == "timeout":
        client.get_client = lambda: _TimeoutClient()  # type: ignore[assignment]
    else:
        raise ValueError(f"unknown fake_registry mode: {mode}")
    try:
        yield
    finally:
        client.get_client = original  # type: ignore[assignment]
