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

import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from novie_agent_sdk import (
    CapabilityCallDiagnostics,
    ExternalAgentCheckpointPutError,
    PlatformLlmCallError,
    PlatformLlmTimeoutError,
    PlatformLlmTransportError,
    PlatformNamespace,
    QuotaExceededError,
    ToolCallAccumulator,
    build_platform_namespace,
    classify_envelope_error,
)
from novie_agent_sdk.platform_namespace import (
    _DEFAULT_ARTIFACT_TIMEOUT_SECONDS,
    _DEFAULT_LLM_TIMEOUT_SECONDS,
    _DEFAULT_STATE_TIMEOUT_SECONDS,
    _DEFAULT_TIMEOUT_SECONDS,
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


def _sse_stream(*payloads: Any, done: bool = True) -> bytes:
    """Encode objects as an OpenAI-compatible ``data:`` SSE stream."""
    lines = [f"data: {json.dumps(payload)}\n\n" for payload in payloads]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def _openai_chunk(
    *,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "novie/default",
        "choices": [
            {"index": 0, "delta": delta or {}, "finish_reason": finish_reason}
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


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


def test_platform_namespace_exposes_openai_proxy_headers(monkeypatch) -> None:
    monkeypatch.setenv("NOVIE_AGENT_PLATFORM_SHARED_SECRET", "secret")
    ns = _build_with_responder(lambda request: httpx.Response(200, json={"ok": True}))

    assert ns.openai_base_url == "http://platform.test/v1"
    headers = ns.openai_headers(path="/v1/chat/completions")
    assert headers["x-novie-org-id"] == "tenant-1"
    assert headers["x-novie-service-principal"] == "agent:demo"
    assert headers["x-novie-on-behalf-of-user-id"] == "user-1"
    assert "x-novie-user-id" not in headers
    assert headers["x-novie-sig"].startswith("sha256=")

    platform_headers = ns.platform_headers(method="POST", path="/invocations")
    assert platform_headers["x-novie-user-id"] == "user-1"
    assert "x-novie-service-principal" not in platform_headers


def test_openai_proxy_preserves_explicit_billing_user(monkeypatch) -> None:
    monkeypatch.setenv("NOVIE_AGENT_PLATFORM_SHARED_SECRET", "secret")
    ns = _build_with_responder(
        lambda request: httpx.Response(200, json={"ok": True}),
        incoming=_incoming_headers(
            **{"x-novie-on-behalf-of-user-id": "billing-user"}
        ),
        agent_id="analyst",
    )

    headers = ns.openai_headers(path="/v1/chat/completions")
    assert headers["x-novie-service-principal"] == "agent:analyst"
    assert headers["x-novie-on-behalf-of-user-id"] == "billing-user"


def _ok_envelope(result: dict[str, Any]) -> dict[str, Any]:
    return {"status": "ok", "output": result}


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
    assert captured["url"].endswith("/invocations")
    assert captured["body"]["capability_id"] == "platform.knowledge.search"
    assert captured["body"]["provider_id"] == "platform.knowledge"
    assert captured["body"]["mode"] == "execute"
    assert captured["body"]["inputs"]["query"] == "widgets"
    assert captured["body"]["inputs"]["top_k"] == 3
    # default_project_id from forward headers populated automatically.
    assert captured["body"]["inputs"]["project_id"] == "project-1"
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
async def test_mid_run_ask_blocks_inline_write_capability() -> None:
    called = False

    def responder(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json=_ok_envelope({"ok": True}))

    ns = _build_with_responder(responder)
    ns.set_mid_run_ask_active(True)

    diagnostics = await ns.invoke_capability(
        "platform.pms.issue.update_status",
        {"issue_id": "issue-1", "state": "Todo"},
    )

    assert diagnostics.ok is False
    assert diagnostics.kind == "binding_denied"
    assert diagnostics.error_code == "mid_run_ask_inline_write_denied"
    assert called is False


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
    assert captured["body"]["inputs"]["project_id"] == "custom-project"


# ── Live namespace: web.search ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_web_search_returns_platform_result_on_ok() -> None:
    captured: dict[str, Any] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json=_ok_envelope(
                {
                    "available": True,
                    "provider": "tavily",
                    "answer": "Use the platform capability.",
                    "results": [{"title": "Result", "url": "https://example.test"}],
                    "count": 1,
                }
            ),
        )

    ns = _build_with_responder(responder)
    out = await ns.web.search("widgets", max_results=3, search_depth="basic")

    assert out["provider"] == "tavily"
    assert out["results"] == [{"title": "Result", "url": "https://example.test"}]
    assert captured["url"].endswith("/invocations")
    assert captured["body"]["capability_id"] == "platform.web.search"
    assert captured["body"]["inputs"]["query"] == "widgets"
    assert captured["body"]["inputs"]["max_results"] == 3
    assert captured["body"]["inputs"]["search_depth"] == "basic"
    assert ns.last_diagnostics() == ()


@pytest.mark.asyncio
async def test_web_search_records_no_results_diagnostic() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_ok_envelope(
                {
                    "available": False,
                    "provider": "tavily",
                    "error": "web_search_unconfigured",
                    "results": [],
                    "count": 0,
                }
            ),
        )

    ns = _build_with_responder(responder)
    out = await ns.web.search("widgets")
    assert out["results"] == []
    assert any(
        d.capability_id == "platform.web.search" and d.kind == "no_results"
        for d in ns.last_diagnostics()
    )


# ── Live namespace: artifacts.read ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_artifacts_read_uses_budgeted_platform_capability() -> None:
    captured: dict[str, Any] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json=_ok_envelope(
                {
                    "available": True,
                    "artifact_id": "artifact-1",
                    "mode": "search",
                    "content": "1. offset=12\nbounded excerpt",
                    "metadata": {"count": 1},
                }
            ),
        )

    ns = _build_with_responder(responder)
    out = await ns.artifacts.search(
        "artifact-1",
        "pricing",
        purpose="compare pricing evidence",
        max_bytes=4096,
    )

    assert out["content"] == "1. offset=12\nbounded excerpt"
    assert captured["url"].endswith("/invocations")
    assert captured["body"]["capability_id"] == "platform.artifacts.read"
    assert captured["body"]["inputs"] == {
        "artifact_id": "artifact-1",
        "mode": "search",
        "purpose": "compare pricing evidence",
        "offset": 0,
        "max_bytes": 4096,
        "allow_full": False,
        "query": "pricing",
    }
    assert ns.last_diagnostics() == ()


@pytest.mark.asyncio
async def test_artifacts_read_does_not_expose_full_read_allow_flag() -> None:
    captured: dict[str, Any] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json=_ok_envelope(
                {
                    "available": False,
                    "artifact_id": "artifact-1",
                    "error": "artifact_full_read_requires_explicit_allow",
                }
            ),
        )

    ns = _build_with_responder(responder)
    out = await ns.artifacts.read("artifact-1", mode="full", max_bytes=4096)

    assert out["available"] is False
    assert captured["body"]["inputs"]["allow_full"] is False
    assert any(
        d.capability_id == "platform.artifacts.read" and d.kind == "no_results"
        for d in ns.last_diagnostics()
    )


@pytest.mark.asyncio
async def test_artifacts_read_text_formats_platform_artifact_payload() -> None:
    calls: list[dict[str, Any]] = []
    encoded = base64.b64encode(
        json.dumps(
            {
                "final_payload": {
                    "final_markdown": "# Final Report\n\nReadable body."
                }
            }
        ).encode("utf-8")
    ).decode("ascii")

    def responder(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json=_ok_envelope(
                {
                    "available": True,
                    "artifact_id": "artifact-1",
                    "mode": "chunks",
                    "metadata": {
                        "encoding": "base64",
                        "content_type": "application/json",
                    },
                    "content": {"data": encoded, "next_offset": 4096},
                    "excerpts": [{"offset": 32, "excerpt": "bounded excerpt"}],
                }
            ),
        )

    ns = _build_with_responder(responder)
    out = await ns.artifacts.read_text(
        "artifact://artifact-1",
        mode="chunks",
        offset=128,
        max_bytes=4096,
    )

    assert "# Final Report" in out
    assert "Readable body." in out
    assert "bounded excerpt" in out
    assert "Next offset: 4096" in out
    assert encoded not in out
    assert calls[0]["inputs"]["artifact_id"] == "artifact-1"
    assert calls[0]["inputs"]["purpose"] == "agent evidence retrieval"


@pytest.mark.asyncio
async def test_artifacts_read_text_caches_exact_repeated_reads() -> None:
    call_count = 0

    def responder(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            200,
            json=_ok_envelope(
                {
                    "available": True,
                    "artifact_id": "artifact-1",
                    "mode": "summary",
                    "summary": "Cached artifact summary",
                }
            ),
        )

    ns = _build_with_responder(responder)
    first = await ns.artifacts.read_text("artifact-1", mode="summary")
    second = await ns.artifacts.read_text("artifact://artifact-1", mode="summary")

    assert first == second
    assert "Cached artifact summary" in second
    assert call_count == 1


@pytest.mark.asyncio
async def test_artifact_writes_use_dedicated_long_timeout() -> None:
    ns = _build_with_responder(
        lambda request: httpx.Response(200, json=_ok_envelope({"available": True}))
    )
    captured: list[tuple[str, float | None]] = []

    async def fake_invoke(capability_id, arguments, *, timeout_seconds=None):  # type: ignore[no-untyped-def]
        captured.append((capability_id, timeout_seconds))
        if capability_id == "platform.artifacts.create":
            return CapabilityCallDiagnostics(
                ok=True,
                capability_id=capability_id,
                result={"artifact_ref": "artifact://art-1"},
            )
        return CapabilityCallDiagnostics(
            ok=True,
            capability_id=capability_id,
            result={"available": True},
        )

    ns.invoke_capability = fake_invoke  # type: ignore[method-assign]

    await ns.artifacts.create(
        artifact_type="implementation_plan_document",
        content="# Plan",
    )
    await ns.artifacts.read("artifact://art-1")
    await ns.artifacts.search_index(workflow_id="wf-1")
    await ns.workpads.snapshot(workflow_id="wf-1")
    await ns.workpads.record_entry(kind="progress")
    await ns.workpads.set_final_deliverable("artifact://art-1")
    await ns.checkpoints.put(
        owner_agent_id="agent",
        thread_id="thread-1",
        payload={"state": "ok"},
    )
    await ns.knowledge.search("still short")

    assert captured == [
        ("platform.artifacts.create", _DEFAULT_ARTIFACT_TIMEOUT_SECONDS),
        ("platform.artifacts.read", _DEFAULT_ARTIFACT_TIMEOUT_SECONDS),
        ("platform.artifacts.search", _DEFAULT_ARTIFACT_TIMEOUT_SECONDS),
        ("platform.workpads.snapshot", _DEFAULT_STATE_TIMEOUT_SECONDS),
        ("platform.workpads.record_entry", _DEFAULT_STATE_TIMEOUT_SECONDS),
        ("platform.workpads.set_final_deliverable", _DEFAULT_ARTIFACT_TIMEOUT_SECONDS),
        ("platform.external_agent_checkpoint.put", _DEFAULT_STATE_TIMEOUT_SECONDS),
        ("platform.knowledge.search", None),
    ]


@pytest.mark.asyncio
async def test_artifacts_search_index_requires_items_list() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_envelope({"items": "bad"}))

    ns = _build_with_responder(responder)
    out = await ns.artifacts.search_index(workflow_id="wf-1")

    assert out == []
    assert any(
        d.capability_id == "platform.artifacts.search"
        and d.kind == "schema_violation"
        for d in ns.last_diagnostics()
    )


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
    classification used for HTTP status codes. Uses the canonical
    ``/invocations`` failure-detail key, ``error_message``."""
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "error",
                "error_code": "denied_by_binding",
                "error_message": "no grant",
            },
        )

    ns = _build_with_responder(responder)
    diagnostics = await ns.invoke_capability(
        "platform.knowledge.search", {"query": "x"},
    )
    assert diagnostics.ok is False
    assert diagnostics.kind == "binding_denied"
    assert diagnostics.detail == "no grant"


@pytest.mark.asyncio
async def test_invoke_capability_tolerates_legacy_explanation_key() -> None:
    """Some platform builds haven't fully cut over to ``error_message``
    yet — ``explanation`` must still populate ``.detail`` as a fallback."""
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "error",
                "error_code": "denied_by_binding",
                "explanation": "no grant (legacy)",
            },
        )

    ns = _build_with_responder(responder)
    diagnostics = await ns.invoke_capability(
        "platform.knowledge.search", {"query": "x"},
    )
    assert diagnostics.ok is False
    assert diagnostics.detail == "no grant (legacy)"


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


def test_external_agent_checkpoint_put_error_carries_diagnostics_fields() -> None:
    exc = ExternalAgentCheckpointPutError(
        capability_id="platform.external_agent_checkpoint.put",
        kind="binding_denied",
        error_code="denied_by_binding",
        detail="no grant",
    )
    assert isinstance(exc, RuntimeError)
    assert exc.capability_id == "platform.external_agent_checkpoint.put"
    assert exc.kind == "binding_denied"
    assert exc.error_code == "denied_by_binding"
    assert exc.detail == "no grant"
    assert "binding_denied" in str(exc)


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
    assert captured["url"].endswith("/invocations")
    assert captured["body"]["capability_id"] == "platform.external_agent_checkpoint.put"
    assert captured["body"]["inputs"]["thread_id"] == "thread-1"
    assert captured["body"]["inputs"]["summary"] == "phase complete"
    assert captured["body"]["inputs"]["payload"] == {"phase": "synthesis"}


@pytest.mark.asyncio
async def test_checkpoints_put_raises_on_binding_denied() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error_code": "denied_by_binding"})

    ns = _build_with_responder(responder)

    with pytest.raises(ExternalAgentCheckpointPutError) as excinfo:
        await ns.checkpoints.put(
            owner_agent_id="demo",
            thread_id="thread-1",
            payload={"phase": "x"},
        )

    assert excinfo.value.kind == "binding_denied"


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


# ── LLM namespace: surface failures instead of silently returning {} ────────


@pytest.mark.asyncio
async def test_llm_structured_raises_transport_error_instead_of_returning_empty() -> None:
    """Pre-0.3.3 SDKs returned ``{}`` from ``LlmNamespace.structured`` when
    the underlying capability call timed out; that fed an empty dict into
    callers' Pydantic schemas and bubbled up as ``summary Field required``
    on the analyst's ``ProductBriefArtifact``.  The 0.3.3 contract is to
    raise ``PlatformLlmTransportError`` so callers can distinguish a real
    LLM-returned empty object from a transport failure."""
    def responder(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timeout")

    ns = _build_with_responder(responder)
    with pytest.raises(PlatformLlmTransportError) as excinfo:
        await ns.llm.structured(
            [{"role": "user", "content": "hi"}],
            output_schema={"type": "object", "required": ["summary"]},
        )
    assert excinfo.value.capability_id == "platform.llm.structured"
    assert excinfo.value.kind == "timeout"
    assert excinfo.value.is_transient is True


@pytest.mark.asyncio
async def test_llm_structured_preserves_platform_timeout_envelope() -> None:
    envelope = {
        "kind": "timeout",
        "capability_id": "platform.llm.structured",
        "model": "openai/gpt-5.4",
        "phase": "structured_output",
        "timeout_seconds": 120,
        "retryable": True,
        "reason_code": "platform_llm_structured_timeout",
        "raw_detail": "ReadTimeout",
    }

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "denied",
                "error_code": "platform_llm_structured_timeout",
                "explanation": "structured timed out",
                "metadata": {"error_envelope": envelope},
            },
        )

    ns = _build_with_responder(responder)
    with pytest.raises(PlatformLlmTimeoutError) as excinfo:
        await ns.llm.structured(
            [{"role": "user", "content": "hi"}],
            output_schema={"type": "object"},
        )
    assert excinfo.value.reason_code == "platform_llm_structured_timeout"
    assert excinfo.value.timeout_seconds == 120
    assert excinfo.value.error_envelope["model"] == "openai/gpt-5.4"
    assert excinfo.value.is_transient is True


@pytest.mark.asyncio
async def test_llm_structured_forwards_per_call_timeout() -> None:
    captured: dict[str, Any] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.update(body["inputs"])
        return httpx.Response(200, json=_ok_envelope({"structured": {"ok": True}}))

    ns = _build_with_responder(responder)
    await ns.llm.structured(
        [{"role": "user", "content": "hi"}],
        output_schema={"type": "object"},
        timeout_seconds=240,
    )
    assert captured["timeout_seconds"] == 240.0


@pytest.mark.asyncio
async def test_llm_chat_raises_call_error_on_envelope_5xx() -> None:
    """Non-quota envelope failures (binding denied, schema violation,
    unavailable) must also raise rather than silently degrade so callers
    don't feed an empty ``content`` into downstream prompts."""
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error_code": "internal_error"})

    ns = _build_with_responder(responder)
    with pytest.raises(PlatformLlmCallError) as excinfo:
        await ns.llm.chat([{"role": "user", "content": "hi"}])
    assert excinfo.value.capability_id == "platform.llm.chat"
    assert excinfo.value.kind == "platform_unavailable"


@pytest.mark.asyncio
async def test_llm_chat_streams_over_openai_chat_completions() -> None:
    captured_paths: list[str] = []
    captured_bodies: list[dict[str, Any]] = []
    captured_sig: list[str | None] = []

    def responder(request: httpx.Request) -> httpx.Response:
        captured_paths.append(request.url.path)
        captured_bodies.append(json.loads(request.content.decode("utf-8")))
        captured_sig.append(request.headers.get("x-novie-sig"))
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            content=_sse_stream(
                _openai_chunk(delta={"role": "assistant"}),
                _openai_chunk(delta={"content": "hello"}),
                _openai_chunk(
                    delta={},
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": 3,
                        "completion_tokens": 4,
                        "total_tokens": 7,
                    },
                ),
            ),
            headers={"content-type": "text/event-stream"},
        )

    ns = _build_with_responder(responder)

    result = await ns.llm.chat(
        [{"role": "user", "content": "hi"}],
        model="anthropic/claude-opus-4.6",
        temperature=0.2,
        max_output_tokens=256,
    )

    assert result["content"] == "hello"
    assert result["usage_metadata"]["total_tokens"] == 7
    # ``usage`` renames OpenAI's prompt/completion token fields onto
    # LangChain's ``usage_metadata`` vocabulary.
    assert result["usage_metadata"]["input_tokens"] == 3
    assert result["usage_metadata"]["output_tokens"] == 4
    # The terminal chunk's ``finish_reason`` carries through to
    # ``response_metadata`` on the synthesized ``completed`` result.
    assert result["response_metadata"]["finish_reason"] == "stop"
    assert captured_paths == ["/v1/chat/completions"]
    # Re-signed for the new path (a stale signed path would 401 on the platform).
    assert captured_sig[0]
    # ``arguments`` mapped onto the OpenAI body: ``stream`` on,
    # ``max_output_tokens`` renamed to ``max_tokens``.
    body = captured_bodies[0]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["model"] == "anthropic/claude-opus-4.6"
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 256
    assert "arguments" not in body
    assert "caller_type" not in body


@pytest.mark.asyncio
async def test_llm_chat_accumulates_streamed_content_deltas() -> None:
    """``llm.chat`` consumes the SSE stream but returns one final result.

    The endpoint emits token-delta chunks before the terminal chunk; the SDK
    accumulates them into the synthesized ``completed`` result rather than
    surfacing them to the non-streaming ``llm.chat`` caller.
    """
    def responder(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            content=_sse_stream(
                _openai_chunk(delta={"content": "hel"}),
                _openai_chunk(delta={"content": "lo"}),
                _openai_chunk(
                    delta={},
                    finish_reason="stop",
                    usage={"total_tokens": 7},
                ),
            ),
            headers={"content-type": "text/event-stream"},
        )

    ns = _build_with_responder(responder)

    result = await ns.llm.chat([{"role": "user", "content": "hi"}])

    assert result["content"] == "hello"
    assert result["usage_metadata"]["total_tokens"] == 7


@pytest.mark.asyncio
async def test_llm_stream_chat_normalises_tool_chunks_for_non_langchain_agents() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            content=_sse_stream(
                _openai_chunk(delta={"tool_calls": [
                    {
                        "index": 0,
                        "id": "toolu_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"quer'},
                    }
                ]}),
                _openai_chunk(delta={"tool_calls": [
                    {
                        "index": 0,
                        "type": "function",
                        "function": {"arguments": 'y":"hello"}'},
                    }
                ]}),
                _openai_chunk(
                    delta={},
                    finish_reason="tool_calls",
                    usage={"total_tokens": 7},
                ),
            ),
            headers={"content-type": "text/event-stream"},
        )

    ns = _build_with_responder(responder)
    accumulator = ToolCallAccumulator()
    chunks: list[dict[str, Any]] = []

    async for event in ns.llm.stream_chat([{"role": "user", "content": "hi"}]):
        chunks.extend(accumulator.add_event(event))

    assert [chunk["id"] for chunk in chunks] == ["toolu_1", "toolu_1"]
    assert accumulator.tool_calls() == [
        {"id": "toolu_1", "name": "lookup", "args": {"query": "hello"}}
    ]


@pytest.mark.asyncio
async def test_llm_chat_diagnostics_path_accumulates_tool_calls() -> None:
    """``llm.chat`` (the diagnostics path, not ``stream_chat``'s raw event
    stream) must also accumulate tool-call deltas into the synthesized
    ``completed`` result rather than dropping them."""
    def responder(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            content=_sse_stream(
                _openai_chunk(delta={"tool_calls": [
                    {
                        "index": 0,
                        "id": "toolu_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"quer'},
                    }
                ]}),
                _openai_chunk(delta={"tool_calls": [
                    {
                        "index": 0,
                        "type": "function",
                        "function": {"arguments": 'y":"hello"}'},
                    }
                ]}),
                _openai_chunk(
                    delta={},
                    finish_reason="tool_calls",
                    usage={"total_tokens": 7},
                ),
            ),
            headers={"content-type": "text/event-stream"},
        )

    ns = _build_with_responder(responder)

    diagnostics = await ns._llm_caller.invoke_stream_with_diagnostics(  # noqa: SLF001
        "platform.llm.chat", {"messages": [{"role": "user", "content": "hi"}]}
    )

    assert diagnostics.ok is True
    assert diagnostics.result is not None
    assert diagnostics.result["tool_calls"] == [
        {"id": "toolu_1", "name": "lookup", "args": {"query": "hello"}}
    ]


@pytest.mark.asyncio
async def test_llm_stream_diagnostics_classifies_read_timeout_as_heartbeat_timeout() -> None:
    """A real ``httpx.ReadTimeout`` (e.g. the platform stops sending
    heartbeat bytes for longer than the read timeout) must be classified as
    ``stream_heartbeat_timeout``, not fall through to the generic
    transport-error branch (which drops the error_code entirely) -- the
    handler previously caught the builtin ``TimeoutError``, which
    ``httpx.ReadTimeout`` never subclasses, so it was silently unreachable.
    """
    def responder(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated heartbeat timeout", request=request)

    ns = _build_with_responder(responder)

    diagnostics = await ns._llm_caller.invoke_stream_with_diagnostics(  # noqa: SLF001
        "platform.llm.chat", {"messages": [{"role": "user", "content": "hi"}]}
    )

    assert diagnostics.ok is False
    assert diagnostics.error_code == "stream_heartbeat_timeout"


@pytest.mark.asyncio
async def test_llm_event_stream_classifies_read_timeout_as_heartbeat_timeout() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated heartbeat timeout", request=request)

    ns = _build_with_responder(responder)

    events = [
        event
        async for event in ns._llm_caller.invoke_event_stream(  # noqa: SLF001
            "platform.llm.chat", {"messages": [{"role": "user", "content": "hi"}]}
        )
    ]

    assert events[-1]["type"] == "error"
    assert events[-1]["error_code"] == "stream_heartbeat_timeout"


@pytest.mark.asyncio
async def test_translate_openai_sse_errors_when_stream_closes_without_done() -> None:
    """A clean mid-generation connection close (no ``[DONE]``, no httpx
    exception) must surface as an ``error`` event, not a ``completed`` one
    — otherwise a truncated response silently looks like success."""
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse_stream(
                _openai_chunk(delta={"content": "partial"}),
                done=False,
            ),
            headers={"content-type": "text/event-stream"},
        )

    ns = _build_with_responder(responder)

    events = [
        event
        async for event in ns._llm_caller.invoke_event_stream(  # noqa: SLF001
            "platform.llm.chat", {"messages": [{"role": "user", "content": "hi"}]}
        )
    ]

    assert events[-1]["type"] == "error"
    assert events[-1]["error_code"] == "stream_closed_without_completion"
    assert not any(event["type"] == "completed" for event in events)


@pytest.mark.asyncio
async def test_llm_stream_chat_raises_on_stream_error_event() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        # Mid-stream failures arrive as an OpenAI-shaped ``error`` payload.
        return httpx.Response(
            200,
            content=_sse_stream({
                "error": {
                    "message": "provider failed",
                    "type": "server_error",
                    "code": "internal_error",
                    "param": None,
                }
            }),
            headers={"content-type": "text/event-stream"},
        )

    ns = _build_with_responder(responder)

    with pytest.raises(PlatformLlmCallError) as excinfo:
        async for _event in ns.llm.stream_chat([{"role": "user", "content": "hi"}]):
            pass

    assert excinfo.value.capability_id == "platform.llm.chat"
    assert excinfo.value.error_code == "internal_error"
    assert excinfo.value.detail == "provider failed"


@pytest.mark.asyncio
async def test_llm_chat_raises_when_chat_completions_endpoint_missing() -> None:
    captured_paths: list[str] = []

    def responder(request: httpx.Request) -> httpx.Response:
        captured_paths.append(request.url.path)
        return httpx.Response(404, json={"error_code": "not_found"})

    ns = _build_with_responder(responder)

    with pytest.raises(PlatformLlmCallError) as excinfo:
        await ns.llm.chat([{"role": "user", "content": "hi"}])

    assert excinfo.value.capability_id == "platform.llm.chat"
    assert excinfo.value.error_code == "not_found"
    assert captured_paths == ["/v1/chat/completions"]


@pytest.mark.asyncio
async def test_llm_structured_still_raises_quota_exceeded_separately() -> None:
    """``QuotaExceededError`` is the one exception that should NOT be
    folded into ``PlatformLlmCallError`` — handlers branch on it to surface
    a user-visible 'org pool exhausted' message instead of retrying."""
    def responder(request: httpx.Request) -> httpx.Response:
        # envelope=ok with quota signalled inside ``result.error`` is the
        # contract LLMUsageRecord uses today; covers the "successful HTTP
        # but capability says no" pathway.
        return httpx.Response(
            200,
            json=_ok_envelope({
                "error": "quota_exceeded",
                "org_id": "org-1",
                "remaining_tokens": 0,
                "reason": "out of tokens",
            }),
        )

    ns = _build_with_responder(responder)
    with pytest.raises(QuotaExceededError) as excinfo:
        await ns.llm.structured(
            [{"role": "user", "content": "hi"}],
            output_schema={"type": "object"},
        )
    assert excinfo.value.org_id == "org-1"
    # Sanity-check that we didn't accidentally subclass ``PlatformLlmCallError``
    # — quota signalling is a known business state, not a transport problem.
    assert not isinstance(excinfo.value, PlatformLlmCallError)


@pytest.mark.asyncio
async def test_llm_namespace_uses_dedicated_long_timeout_caller() -> None:
    """LLM calls share the platform endpoint with knowledge / checkpoints
    but legitimately take 10–30 s; they MUST NOT share the 8 s default
    timeout that's appropriate for short capabilities or every slow LLM
    structured call would surface as a transport error."""
    captured_caps: list[str] = []

    def responder(request: httpx.Request) -> httpx.Response:
        # Path is ``/invocations`` for both callers; the capability id
        # rides in the body now, so pull it back out from there to
        # confirm both callers route through the same endpoint shape —
        # only the timeout differs.
        body = json.loads(request.content.decode("utf-8"))
        captured_caps.append(body["capability_id"])
        return httpx.Response(200, json=_ok_envelope({"structured": {"k": 1}}))

    transport = httpx.MockTransport(responder)
    client = httpx.AsyncClient(transport=transport, base_url="http://platform.test")
    ns = build_platform_namespace(
        _incoming_headers(),
        agent_id="demo",
        base_url="http://platform.test",
        client=client,
    )
    assert isinstance(ns, PlatformNamespace)

    # Default caller (knowledge / checkpoints) keeps the short timeout.
    assert ns._caller._timeout == _DEFAULT_TIMEOUT_SECONDS  # noqa: SLF001
    # LLM caller gets the long-form timeout.
    assert ns._llm_caller._timeout == _DEFAULT_LLM_TIMEOUT_SECONDS  # noqa: SLF001
    assert ns._llm_caller is not ns._caller  # noqa: SLF001

    await ns.llm.structured(
        [{"role": "user", "content": "x"}],
        output_schema={"type": "object"},
    )
    assert captured_caps == ["platform.llm.structured"]


@pytest.mark.asyncio
async def test_invoke_llm_capability_routes_through_llm_caller() -> None:
    """``invoke_llm_capability`` must use the long-timeout caller so the
    ``LlmNamespace`` methods inherit the right latency tier even when
    they're rebound or wrapped (e.g. by ``LlmFacade``)."""
    counts = {"default": 0, "llm": 0}

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_envelope({"hits": []}))

    transport = httpx.MockTransport(responder)
    client = httpx.AsyncClient(transport=transport, base_url="http://platform.test")
    ns = build_platform_namespace(
        _incoming_headers(),
        agent_id="demo",
        base_url="http://platform.test",
        client=client,
    )
    assert isinstance(ns, PlatformNamespace)

    # Patch each caller's ``invoke_with_diagnostics`` so we can witness
    # which one ``invoke_llm_capability`` reaches for.
    original_default = ns._caller.invoke_with_diagnostics  # noqa: SLF001
    original_llm = ns._llm_caller.invoke_with_diagnostics  # noqa: SLF001

    async def default_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        counts["default"] += 1
        return await original_default(*args, **kwargs)

    async def llm_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        counts["llm"] += 1
        return await original_llm(*args, **kwargs)

    ns._caller.invoke_with_diagnostics = default_wrapper  # type: ignore[method-assign]  # noqa: SLF001
    ns._llm_caller.invoke_with_diagnostics = llm_wrapper  # type: ignore[method-assign]  # noqa: SLF001

    await ns.invoke_capability("platform.knowledge.search", {"query": "x"})
    await ns.invoke_llm_capability("platform.llm.structured", {"messages": []})

    assert counts == {"default": 1, "llm": 1}


@pytest.mark.asyncio
async def test_llm_chat_stream_404_raises_without_legacy_invoke() -> None:
    counts = {"stream": 0, "invoke": 0}

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_envelope({"content": "ok"}))

    transport = httpx.MockTransport(responder)
    client = httpx.AsyncClient(transport=transport, base_url="http://platform.test")
    ns = build_platform_namespace(
        _incoming_headers(),
        agent_id="demo",
        base_url="http://platform.test",
        client=client,
    )
    assert isinstance(ns, PlatformNamespace)

    async def stream_404(*args, **kwargs):  # type: ignore[no-untyped-def]
        counts["stream"] += 1
        return CapabilityCallDiagnostics(
            ok=False,
            capability_id="platform.llm.chat",
            kind="platform_unavailable",
            error_code="stream_endpoint_not_found",
        )

    original_invoke = ns._llm_caller.invoke_with_diagnostics  # noqa: SLF001

    async def invoke_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        counts["invoke"] += 1
        return await original_invoke(*args, **kwargs)

    ns._llm_caller.invoke_stream_with_diagnostics = stream_404  # type: ignore[method-assign]  # noqa: SLF001
    ns._llm_caller.invoke_with_diagnostics = invoke_wrapper  # type: ignore[method-assign]  # noqa: SLF001

    with pytest.raises(PlatformLlmCallError) as excinfo:
        await ns.llm.chat([{"role": "user", "content": "one"}])

    assert excinfo.value.capability_id == "platform.llm.chat"
    assert excinfo.value.error_code == "stream_endpoint_not_found"
    assert counts == {"stream": 1, "invoke": 0}
