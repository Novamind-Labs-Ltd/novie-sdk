"""``CapabilityClient`` (``platform_services.py``) — non-streaming
``POST /invocations`` request/response shape.

``CapabilityClient`` predates the SDK's httpx-based ``_CapabilityCaller``
(``platform_namespace.py``) and talks to the platform via stdlib
``urllib`` instead, so ``request.urlopen`` is monkeypatched rather than
using ``httpx.MockTransport``.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from novie_agent_sdk.platform_services import CapabilityClient


class _FakeUrlopenResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeUrlopenResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _client(**overrides: Any) -> CapabilityClient:
    headers = {"x-novie-org-id": "tenant-1", "x-novie-project-id": "project-1"}
    headers.update(overrides.pop("headers", {}))
    return CapabilityClient(
        "http://platform.test",
        headers,
        agent_id=overrides.pop("agent_id", "analyst"),
        **overrides,
    )


@pytest.mark.asyncio
async def test_invoke_with_diagnostics_posts_invocations_envelope(monkeypatch) -> None:
    monkeypatch.setenv("NOVIE_AGENT_PLATFORM_SHARED_SECRET", "secret")
    seen: dict[str, Any] = {}

    def fake_urlopen(req, timeout=None):  # noqa: ANN001, ARG001
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeUrlopenResponse(
            json.dumps({"status": "ok", "output": {"count": 1}}).encode("utf-8")
        )

    monkeypatch.setattr("novie_agent_sdk.platform_services.request.urlopen", fake_urlopen)

    client = _client()
    diagnostics = await client.invoke_with_diagnostics(
        "platform.knowledge.search", {"query": "widgets"}
    )

    assert diagnostics.ok is True
    assert diagnostics.result == {"count": 1}
    assert seen["url"] == "http://platform.test/invocations"
    assert seen["method"] == "POST"
    assert seen["body"] == {
        "capability_id": "platform.knowledge.search",
        "provider_id": "platform.knowledge",
        "mode": "execute",
        "inputs": {"query": "widgets"},
    }


@pytest.mark.asyncio
async def test_invoke_with_diagnostics_reads_error_message_with_explanation_fallback(
    monkeypatch,
) -> None:
    """Canonical failure detail key is ``error_message``; tolerate a
    platform build that still sends ``explanation`` (Task 2 recipe)."""

    def fake_urlopen(req, timeout=None):  # noqa: ANN001, ARG001
        return _FakeUrlopenResponse(
            json.dumps(
                {
                    "status": "denied",
                    "error_code": "denied_by_binding",
                    "explanation": "no grant (legacy)",
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr("novie_agent_sdk.platform_services.request.urlopen", fake_urlopen)

    client = _client()
    diagnostics = await client.invoke_with_diagnostics("platform.knowledge.search", {})

    assert diagnostics.ok is False
    assert diagnostics.detail == "no grant (legacy)"


@pytest.mark.asyncio
async def test_invoke_with_diagnostics_prefers_error_message_over_explanation(
    monkeypatch,
) -> None:
    def fake_urlopen(req, timeout=None):  # noqa: ANN001, ARG001
        return _FakeUrlopenResponse(
            json.dumps(
                {
                    "status": "denied",
                    "error_code": "denied_by_binding",
                    "error_message": "no grant",
                    "explanation": "should not win",
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr("novie_agent_sdk.platform_services.request.urlopen", fake_urlopen)

    client = _client()
    diagnostics = await client.invoke_with_diagnostics("platform.knowledge.search", {})

    assert diagnostics.ok is False
    assert diagnostics.detail == "no grant"
