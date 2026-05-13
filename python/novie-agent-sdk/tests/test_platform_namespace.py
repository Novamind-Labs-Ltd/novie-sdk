"""EXPERT_AGENT_SDK W4 — platform-callback namespace tests.

Locks the W4 surface:

- ``build_platform_namespace`` returns ``_UnavailablePlatformNamespace``
  when ``base_url`` is missing or required tenant/project headers are
  absent, and a live ``PlatformNamespace`` otherwise.
- Acceptance bullet: "Tests cover missing platform URL, binding
  denial, and transport failure."
- ``KnowledgeNamespace.search`` and ``CheckpointsNamespace.{put, get,
  list}`` route through ``_CapabilityCaller`` and surface non-OK
  outcomes via ``last_diagnostics()``.
- ``classify_envelope_error`` mirrors analyst classification so
  agents that already read the analyst's flags interpret SDK-emitted
  ones the same way.

All HTTP traffic is captured by ``httpx.MockTransport`` so tests run
without a real server.
"""
# ruff: noqa: I001
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from novie_agent_sdk import (
    CapabilityCallDiagnostics,
    PlatformNamespace,
    build_platform_namespace,
    classify_envelope_error,
)
from novie_agent_sdk.platform_namespace import (
    _UnavailablePlatformNamespace,
)


def _incoming_headers(**overrides: str) -> dict[str, str]:
    base = {
        "x-novie-tenant-id": "tenant-1",
        "x-novie-workspace-id": "workspace-1",
        "x-novie-project-id": "project-1",
        "x-novie-user-id": "user-1",
        "x-novie-session-id": "session-1",
        "x-novie-request-id": "req-1",
    }
    base.update(overrides)
    return base


def _build_with_responder(
    responder: Callable[[httpx.Request], httpx.Response],
    *,
    base_url: str = "http://platform.test",
    incoming: dict[str, str] | None = None,
    agent_id: str = "demo",
) -> PlatformNamespace:
    transport = httpx.MockTransport(responder)
    client = httpx.AsyncClient(transport=transport, base_url=base_url)
    ns = build_platform_namespace(
        incoming or _incoming_headers(),
        agent_id=agent_id,
        base_url=base_url,
        client=client,
    )
    assert isinstance(ns, PlatformNamespace), "expected live namespace"
    return ns


def _ok_envelope(result: dict[str, Any]) -> dict[str, Any]:
    return {"status": "ok", "result": result}


# ── classify_envelope_error ─────────────────────────────────────────────────


def test_classify_envelope_error_binding_denied_via_code() -> None:
    assert classify_envelope_error("denied_by_binding", None) == "binding_denied"


def test_classify_envelope_error_binding_denied_via_403() -> None:
    assert classify_envelope_error(None, 403) == "binding_denied"


def test_classify_envelope_error_other_falls_back_to_platform_unavailable() -> None:
    assert classify_envelope_error(None, 500) == "platform_unavailable"
    assert classify_envelope_error("internal_error", 500) == "platform_unavailable"


# ── Factory: missing config → unavailable namespace ─────────────────────────


def test_factory_returns_unavailable_when_base_url_missing(monkeypatch) -> None:
    """Acceptance bullet: 'Tests cover missing platform URL'."""
    monkeypatch.delenv("NOVIE_PLATFORM_BASE_URL", raising=False)
    ns = build_platform_namespace(
        _incoming_headers(), agent_id="demo", base_url=None,
    )
    assert isinstance(ns, _UnavailablePlatformNamespace)
    assert ns.is_available is False


def test_factory_returns_unavailable_when_tenant_missing(monkeypatch) -> None:
    """No tenant/org header → platform would 400 — degrade up front."""
    monkeypatch.delenv("NOVIE_ORG_ID", raising=False)
    monkeypatch.delenv("NOVIE_PROJECT_ID", raising=False)
    headers = _incoming_headers()
    headers.pop("x-novie-tenant-id")
    headers.pop("x-novie-workspace-id")
    headers.pop("x-novie-project-id")
    ns = build_platform_namespace(
        headers, agent_id="demo", base_url="http://platform.test",
    )
    assert isinstance(ns, _UnavailablePlatformNamespace)


def test_factory_uses_env_base_url_when_kwarg_omitted(monkeypatch) -> None:
    monkeypatch.setenv("NOVIE_PLATFORM_BASE_URL", "http://from-env.test")
    ns = build_platform_namespace(
        _incoming_headers(), agent_id="demo", base_url=None,
    )
    assert isinstance(ns, PlatformNamespace)


# ── Unavailable namespace surfaces diagnostics, doesn't raise ───────────────


@pytest.mark.asyncio
async def test_unavailable_namespace_returns_diagnostics_on_invoke() -> None:
    ns = _UnavailablePlatformNamespace(reason="missing base url")
    diagnostics = await ns.invoke_capability("platform.knowledge.search", {"q": "x"})
    assert diagnostics.ok is False
    assert diagnostics.kind == "unconfigured"
    assert diagnostics.error_code == "platform_unavailable"
    assert ns.last_diagnostics() == (diagnostics,)


@pytest.mark.asyncio
async def test_unavailable_knowledge_search_returns_empty_list() -> None:
    ns = _UnavailablePlatformNamespace(reason="missing base url")
    out = await ns.knowledge.search("widgets")
    assert out == []
    assert any(d.kind == "unconfigured" for d in ns.last_diagnostics())


@pytest.mark.asyncio
async def test_unavailable_checkpoints_get_returns_none() -> None:
    ns = _UnavailablePlatformNamespace(reason="missing base url")
    out = await ns.checkpoints.get(owner_agent_id="demo", thread_id="t-1")
    assert out is None


# ── Live namespace: knowledge.search ────────────────────────────────────────


@pytest.mark.asyncio
async def test_knowledge_search_returns_hits_on_ok() -> None:
    captured: dict[str, Any] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json=_ok_envelope({"results": [{"id": "doc-1"}, {"id": "doc-2"}]}),
        )

    ns = _build_with_responder(responder)
    hits = await ns.knowledge.search("widgets", top_k=3)

    assert hits == [{"id": "doc-1"}, {"id": "doc-2"}]
    assert "/capabilities/platform.knowledge.search/invoke" in captured["url"]
    assert captured["body"]["arguments"]["query"] == "widgets"
    assert captured["body"]["arguments"]["top_k"] == 3
    # default_project_id from forward headers populated automatically.
    assert captured["body"]["arguments"]["project_id"] == "project-1"
    assert ns.last_diagnostics() == ()


@pytest.mark.asyncio
async def test_knowledge_search_records_no_results_diagnostic() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_envelope({"results": []}))

    ns = _build_with_responder(responder)
    hits = await ns.knowledge.search("nothing")
    assert hits == []
    diagnostics = ns.last_diagnostics()
    assert any(d.kind == "no_results" for d in diagnostics)


@pytest.mark.asyncio
async def test_knowledge_search_records_schema_violation_when_results_not_list() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_envelope({"results": "not-a-list"}))

    ns = _build_with_responder(responder)
    hits = await ns.knowledge.search("x")
    assert hits == []
    diagnostics = ns.last_diagnostics()
    assert any(d.kind == "schema_violation" for d in diagnostics)


@pytest.mark.asyncio
async def test_knowledge_search_overrides_project_id_argument() -> None:
    captured: dict[str, Any] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_ok_envelope({"results": []}))

    ns = _build_with_responder(responder)
    await ns.knowledge.search("x", project_id="custom-project")
    assert captured["body"]["arguments"]["project_id"] == "custom-project"


# ── Live namespace: binding denial / transport failure ──────────────────────


@pytest.mark.asyncio
async def test_invoke_capability_returns_binding_denied_on_403() -> None:
    """Acceptance bullet: 'Tests cover binding denial'."""
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"error_code": "denied_by_binding", "explanation": "no grant"},
        )

    ns = _build_with_responder(responder)
    diagnostics = await ns.invoke_capability(
        "platform.knowledge.search", {"query": "x"},
    )
    assert diagnostics.ok is False
    assert diagnostics.kind == "binding_denied"
    assert diagnostics.error_code == "denied_by_binding"
    # Recorded on the namespace for handler retrieval.
    assert ns.last_diagnostics() == (diagnostics,)


@pytest.mark.asyncio
async def test_invoke_capability_returns_transport_error() -> None:
    """Acceptance bullet: 'Tests cover transport failure'."""
    def responder(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    ns = _build_with_responder(responder)
    diagnostics = await ns.invoke_capability(
        "platform.knowledge.search", {"query": "x"},
    )
    assert diagnostics.ok is False
    assert diagnostics.kind == "transport_error"
    assert "connection refused" in diagnostics.detail


@pytest.mark.asyncio
async def test_invoke_capability_returns_platform_unavailable_on_500() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error_code": "internal_error"})

    ns = _build_with_responder(responder)
    diagnostics = await ns.invoke_capability(
        "platform.knowledge.search", {"query": "x"},
    )
    assert diagnostics.kind == "platform_unavailable"


@pytest.mark.asyncio
async def test_invoke_capability_classifies_non_ok_envelope() -> None:
    """An HTTP 200 with ``status=error`` is still a degradation —
    surface the envelope's ``error_code`` through the same kind
    classification used for HTTP status codes."""
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "error",
                "error_code": "denied_by_binding",
                "explanation": "no grant",
            },
        )

    ns = _build_with_responder(responder)
    diagnostics = await ns.invoke_capability(
        "platform.knowledge.search", {"query": "x"},
    )
    assert diagnostics.ok is False
    assert diagnostics.kind == "binding_denied"


@pytest.mark.asyncio
async def test_invoke_capability_returns_schema_violation_on_non_dict_envelope() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "an", "object"])

    ns = _build_with_responder(responder)
    diagnostics = await ns.invoke_capability("platform.knowledge.search", {})
    assert diagnostics.kind == "schema_violation"
    assert diagnostics.error_code == "non_object_envelope"


@pytest.mark.asyncio
async def test_invoke_capability_returns_schema_violation_on_non_json() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>",
                              headers={"content-type": "text/html"})

    ns = _build_with_responder(responder)
    diagnostics = await ns.invoke_capability("platform.knowledge.search", {})
    assert diagnostics.kind == "schema_violation"


# ── Checkpoints namespace ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_checkpoints_put_returns_dict_on_success() -> None:
    captured: dict[str, Any] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json=_ok_envelope({"checkpoint": {"checkpoint_id": "ck-1"}}),
        )

    ns = _build_with_responder(responder)
    record = await ns.checkpoints.put(
        owner_agent_id="demo",
        thread_id="thread-1",
        payload={"phase": "synthesis"},
        summary="phase complete",
    )
    assert record == {"checkpoint_id": "ck-1"}
    assert "/capabilities/platform.external_agent_checkpoint.put/invoke" in captured["url"]
    assert captured["body"]["arguments"]["thread_id"] == "thread-1"
    assert captured["body"]["arguments"]["summary"] == "phase complete"
    assert captured["body"]["arguments"]["payload"] == {"phase": "synthesis"}


@pytest.mark.asyncio
async def test_checkpoints_put_returns_none_on_binding_denied() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error_code": "denied_by_binding"})

    ns = _build_with_responder(responder)
    record = await ns.checkpoints.put(
        owner_agent_id="demo",
        thread_id="thread-1",
        payload={"phase": "x"},
    )
    assert record is None
    assert any(
        d.kind == "binding_denied" for d in ns.last_diagnostics()
    )


@pytest.mark.asyncio
async def test_checkpoints_get_returns_record_or_none() -> None:
    state = {"first": True}

    def responder(request: httpx.Request) -> httpx.Response:
        if state["first"]:
            state["first"] = False
            return httpx.Response(
                200,
                json=_ok_envelope({"checkpoint": {"checkpoint_id": "ck-2"}}),
            )
        return httpx.Response(200, json=_ok_envelope({}))

    ns = _build_with_responder(responder)
    found = await ns.checkpoints.get(owner_agent_id="demo", thread_id="t-1")
    assert found == {"checkpoint_id": "ck-2"}
    missing = await ns.checkpoints.get(owner_agent_id="demo", thread_id="t-1")
    assert missing is None


@pytest.mark.asyncio
async def test_checkpoints_list_returns_records() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_ok_envelope({
                "checkpoints": [
                    {"checkpoint_id": "ck-1"},
                    {"checkpoint_id": "ck-2"},
                ],
            }),
        )

    ns = _build_with_responder(responder)
    records = await ns.checkpoints.list(
        owner_agent_id="demo", thread_id="t-1", limit=10,
    )
    assert [r["checkpoint_id"] for r in records] == ["ck-1", "ck-2"]


@pytest.mark.asyncio
async def test_checkpoints_list_falls_back_to_items_key() -> None:
    """Some platform responses key the list under ``items`` instead of
    ``checkpoints``; both shapes are accepted."""
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_ok_envelope({"items": [{"checkpoint_id": "ck-9"}]}),
        )

    ns = _build_with_responder(responder)
    records = await ns.checkpoints.list(
        owner_agent_id="demo", thread_id="t-1",
    )
    assert records == [{"checkpoint_id": "ck-9"}]


@pytest.mark.asyncio
async def test_checkpoints_list_returns_empty_on_failure() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error_code": "internal_error"})

    ns = _build_with_responder(responder)
    records = await ns.checkpoints.list(
        owner_agent_id="demo", thread_id="t-1",
    )
    assert records == []
    assert any(
        d.kind == "platform_unavailable" for d in ns.last_diagnostics()
    )


# ── Diagnostics → metadata projection ────────────────────────────────────────


def test_capability_call_diagnostics_to_metadata_entry_shape() -> None:
    """Acceptance bullet: 'Callback failures degrade predictably and
    can be reported in final metadata.' Handlers stuff this into
    ``ArtifactResult.metadata`` so consumers see degradation
    symbolically."""
    diagnostics = CapabilityCallDiagnostics(
        ok=False,
        capability_id="platform.knowledge.search",
        kind="binding_denied",
        error_code="denied_by_binding",
        detail="...",
    )
    entry = diagnostics.to_metadata_entry()
    assert entry == {
        "capability_id": "platform.knowledge.search",
        "ok": False,
        "kind": "binding_denied",
        "error_code": "denied_by_binding",
    }


# ── Forwarded headers carry tenant/project/auth ──────────────────────────────


@pytest.mark.asyncio
async def test_forwarded_headers_include_tenant_and_project() -> None:
    """Acceptance bullet: 'Dev and production header requirements are
    explicit.' The platform receives the agent's tenant + project so
    binding checks succeed; agent identity rides as
    ``service_principal``."""
    captured: dict[str, str] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        for key, value in request.headers.items():
            if key.lower().startswith("x-novie-"):
                captured[key.lower()] = value
        return httpx.Response(200, json=_ok_envelope({"results": []}))

    ns = _build_with_responder(responder, agent_id="my-agent")
    await ns.knowledge.search("x")

    assert captured["x-novie-org-id"] == "tenant-1"
    assert captured["x-novie-project-id"] == "project-1"
    assert captured["x-novie-user-id"] == "user-1"
