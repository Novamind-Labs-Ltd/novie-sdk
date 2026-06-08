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
    PlatformLlmCallError,
    PlatformLlmTransportError,
    PlatformNamespace,
    QuotaExceededError,
    ToolCallAccumulator,
    build_platform_namespace,
    classify_envelope_error,
)
from novie_agent_sdk.platform_namespace import (
    _DEFAULT_LLM_TIMEOUT_SECONDS,
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
    assert captured["body"]["arguments"]["project_id"] == "custom-project"


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
    assert "/capabilities/platform.web.search/invoke" in captured["url"]
    assert captured["body"]["arguments"]["query"] == "widgets"
    assert captured["body"]["arguments"]["max_results"] == 3
    assert captured["body"]["arguments"]["search_depth"] == "basic"
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
    assert "/capabilities/platform.artifacts.read/invoke" in captured["url"]
    assert captured["body"]["arguments"] == {
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
    assert captured["body"]["arguments"]["allow_full"] is False
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
    assert calls[0]["arguments"]["artifact_id"] == "artifact-1"
    assert calls[0]["arguments"]["purpose"] == "agent evidence retrieval"


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
    assert excinfo.value.kind == "transport_error"
    assert excinfo.value.is_transient is True


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
async def test_llm_chat_uses_streaming_capability_endpoint() -> None:
    captured_paths: list[str] = []

    def responder(request: httpx.Request) -> httpx.Response:
        captured_paths.append(request.url.path)
        assert request.url.path == "/capabilities/platform.llm.chat/invoke-stream"
        return httpx.Response(
            200,
            content=(
                b'{"type":"accepted","invocation_id":"i1"}\n'
                b'{"type":"heartbeat","invocation_id":"i1","seq":1}\n'
                b'{"type":"completed","invocation_id":"i1",'
                b'"result":{"content":"hello","usage_metadata":{"total_tokens":7}}}\n'
            ),
            headers={"content-type": "application/x-ndjson"},
        )

    ns = _build_with_responder(responder)

    result = await ns.llm.chat([{"role": "user", "content": "hi"}])

    assert result["content"] == "hello"
    assert result["usage_metadata"]["total_tokens"] == 7
    assert captured_paths == ["/capabilities/platform.llm.chat/invoke-stream"]


@pytest.mark.asyncio
async def test_llm_chat_ignores_intermediate_stream_chunks() -> None:
    """``llm.chat`` consumes the stream endpoint but returns the final result.

    The platform emits ``chunk`` events before ``completed`` for live token
    streaming. Those chunks are for ``llm.stream_chat`` callers and must not
    make non-streaming ``llm.chat`` fail with ``unknown_stream_event``.
    """
    captured_paths: list[str] = []

    def responder(request: httpx.Request) -> httpx.Response:
        captured_paths.append(request.url.path)
        assert request.url.path == "/capabilities/platform.llm.chat/invoke-stream"
        return httpx.Response(
            200,
            content=(
                b'{"type":"accepted","invocation_id":"i1"}\n'
                b'{"type":"chunk","invocation_id":"i1","delta":{"content":"hel"}}\n'
                b'{"type":"chunk","invocation_id":"i1","delta":{"content":"lo"}}\n'
                b'{"type":"completed","invocation_id":"i1",'
                b'"result":{"content":"hello","usage_metadata":{"total_tokens":7}}}\n'
            ),
            headers={"content-type": "application/x-ndjson"},
        )

    ns = _build_with_responder(responder)

    result = await ns.llm.chat([{"role": "user", "content": "hi"}])

    assert result["content"] == "hello"
    assert result["usage_metadata"]["total_tokens"] == 7
    assert captured_paths == ["/capabilities/platform.llm.chat/invoke-stream"]


@pytest.mark.asyncio
async def test_llm_stream_chat_normalises_tool_chunks_for_non_langchain_agents() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/capabilities/platform.llm.chat/invoke-stream"
        return httpx.Response(
            200,
            content=(
                b'{"type":"chunk","delta":{"content":"","tool_call_chunks":['
                b'{"type":"function","id":"toolu_1","function":'
                b'{"name":"lookup","arguments":"{\\"quer"}}]}}\n'
                b'{"type":"chunk","delta":{"content":"","tool_call_chunks":['
                b'{"type":"function","id":null,"function":'
                b'{"name":null,"arguments":"y\\":\\"hello\\"}"}}]}}\n'
                b'{"type":"completed","result":{"content":"","usage_metadata":{"total_tokens":7}}}\n'
            ),
            headers={"content-type": "application/x-ndjson"},
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
async def test_llm_chat_falls_back_to_legacy_invoke_when_stream_endpoint_missing() -> None:
    captured_paths: list[str] = []

    def responder(request: httpx.Request) -> httpx.Response:
        captured_paths.append(request.url.path)
        if request.url.path.endswith("/invoke-stream"):
            return httpx.Response(404, json={"error_code": "not_found"})
        return httpx.Response(200, json=_ok_envelope({"content": "legacy"}))

    ns = _build_with_responder(responder)

    result = await ns.llm.chat([{"role": "user", "content": "hi"}])

    assert result["content"] == "legacy"
    assert captured_paths == [
        "/capabilities/platform.llm.chat/invoke-stream",
        "/capabilities/platform.llm.chat/invoke",
    ]


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
        # Path is ``/capabilities/{id}/invoke``; pull the id back out so
        # we can confirm both callers route through the same endpoint
        # shape — only the timeout differs.
        path = request.url.path
        captured_caps.append(path.split("/")[-2])
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
