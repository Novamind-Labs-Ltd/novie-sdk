"""``CapabilityClient`` (``platform_services.py``) — non-streaming
``POST /invocations`` request/response shape.

``CapabilityClient`` predates the SDK's httpx-based ``_CapabilityCaller``
(``platform_namespace.py``) and talks to the platform via stdlib
``urllib`` instead, so ``request.urlopen`` is monkeypatched rather than
using ``httpx.MockTransport``.
"""
from __future__ import annotations

import json
from io import BytesIO
from typing import Any
from urllib import error

import pytest

from novie_agent_sdk.platform_services import CapabilityClient, HttpExternalAgentCheckpointService
from novie_agent_sdk import ExternalAgentCheckpointPutError
from novie_protocol.contracts import ExecutionContext, IdentityContext, TenantScope


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


def _checkpoint_ctx() -> ExecutionContext:
    return ExecutionContext(
        request_id="req-1",
        session_id="sess-1",
        thread_id="thread-1",
        tenant=TenantScope(tenant_id="tenant-1", workspace_id="workspace-1"),
        identity=IdentityContext(principal_id="agent:demo", principal_type="service"),
    )


@pytest.mark.asyncio
async def test_checkpoint_service_put_raises_on_binding_denied(monkeypatch) -> None:
    def fake_urlopen(req, timeout=None):  # noqa: ANN001, ARG001
        return _FakeUrlopenResponse(
            json.dumps(
                {"status": "denied", "error_code": "denied_by_binding", "error_message": "no grant"}
            ).encode("utf-8")
        )

    monkeypatch.setattr("novie_agent_sdk.platform_services.request.urlopen", fake_urlopen)

    service = HttpExternalAgentCheckpointService(_client())

    with pytest.raises(ExternalAgentCheckpointPutError) as excinfo:
        await service.put(
            _checkpoint_ctx(),
            owner_agent_id="demo",
            thread_id="thread-1",
            payload={"phase": "x"},
        )

    assert excinfo.value.kind == "binding_denied"
    assert excinfo.value.error_code == "denied_by_binding"
    assert excinfo.value.detail == "no grant"


@pytest.mark.asyncio
async def test_checkpoint_service_put_returns_record_on_success(monkeypatch) -> None:
    def fake_urlopen(req, timeout=None):  # noqa: ANN001, ARG001
        return _FakeUrlopenResponse(
            json.dumps(
                {
                    "status": "ok",
                    "output": {"checkpoint": {"checkpoint_id": "ck-1", "thread_id": "thread-1"}},
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr("novie_agent_sdk.platform_services.request.urlopen", fake_urlopen)

    service = HttpExternalAgentCheckpointService(_client())
    record = await service.put(
        _checkpoint_ctx(),
        owner_agent_id="demo",
        thread_id="thread-1",
        payload={"phase": "x"},
    )

    assert record.checkpoint_id == "ck-1"


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
async def test_invoke_with_diagnostics_classifies_http_403_binding_denial(monkeypatch) -> None:
    def fake_urlopen(req, timeout=None):  # noqa: ANN001, ARG001
        raise error.HTTPError(
            req.full_url,
            403,
            "Forbidden",
            hdrs=None,
            fp=BytesIO(
                json.dumps({"error_code": "denied_by_binding"}).encode("utf-8")
            ),
        )

    monkeypatch.setattr("novie_agent_sdk.platform_services.request.urlopen", fake_urlopen)

    diagnostics = await _client().invoke_with_diagnostics("platform.knowledge.search", {})

    assert diagnostics.ok is False
    assert diagnostics.kind == "binding_denied"
    assert diagnostics.error_code == "denied_by_binding"
    assert "denied_by_binding" in diagnostics.detail


@pytest.mark.asyncio
async def test_invoke_with_diagnostics_puts_signed_headers_on_request(monkeypatch) -> None:
    monkeypatch.setenv("NOVIE_AGENT_PLATFORM_SHARED_SECRET", "secret")
    captured: dict[str, str] = {}

    def fake_urlopen(req, timeout=None):  # noqa: ANN001, ARG001
        captured.update({key.lower(): value for key, value in req.header_items()})
        return _FakeUrlopenResponse(
            json.dumps({"status": "ok", "output": {}}).encode("utf-8")
        )

    monkeypatch.setattr("novie_agent_sdk.platform_services.request.urlopen", fake_urlopen)

    diagnostics = await _client(
        headers={"x-novie-service-principal": "agent:analyst"}
    ).invoke_with_diagnostics("platform.knowledge.search", {})

    assert diagnostics.ok is True
    assert captured["x-novie-service-principal"] == "agent:analyst"
    assert captured["x-novie-sig"]
    assert captured["x-novie-timestamp"]


@pytest.mark.asyncio
async def test_invoke_with_diagnostics_passes_configured_timeout_to_urlopen(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        seen["timeout"] = timeout
        return _FakeUrlopenResponse(
            json.dumps({"status": "ok", "output": {}}).encode("utf-8")
        )

    monkeypatch.setattr("novie_agent_sdk.platform_services.request.urlopen", fake_urlopen)

    client = _client(timeout_seconds=12.5)
    diagnostics = await client.invoke_with_diagnostics("platform.knowledge.search", {})

    assert diagnostics.ok is True
    assert seen["timeout"] == 12.5


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
