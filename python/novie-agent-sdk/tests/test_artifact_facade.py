"""EXPERT_AGENT_SDK W3 — ``artifact_agent`` SDK facade tests.

Locks the W3 authoring surface:

- ``artifact_agent(manifest=...)`` returns an ``ArtifactAgentApp``
  with ``.handle`` / ``.serve`` / ``.build_app``.
- Context projection: ``input_text`` / ``inputs`` / ``project`` /
  ``member`` / ``runtime_context`` / ``attachments`` /
  ``capability_id`` are filled from the platform's request shape.
- ``ctx.artifact(...)`` produces a frozen ``ArtifactResult`` that the
  SDK projects onto invoke + stream output shapes.
- Handler is async-only (sync handlers raise at registration).
- Invoke wire shape: ``response.output.kind == "artifact"`` with all
  five fields populated.
- Stream wire shape: progress events appear before the final
  artifact event, in handler-emit order; SDK auto-appends ``done``.
- Default ``ctx.platform`` raises ``NotImplementedError`` (W4 fills);
  ``artifact_agent(platform=...)`` injection works.
- Acceptance bullet: a minimal artifact agent can be implemented
  in under 30 lines of Python (counted by an actual line-count
  assertion on a string fixture).
"""
# ruff: noqa: I001
from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from novie_agent_sdk import (
    ArtifactAgentApp,
    ArtifactAgentContext,
    ArtifactResult,
    NeedsConfirmationResult,
    artifact_agent,
)
from novie_agent_sdk.artifact_facade import _build_context, _coerce_outcome
from novie_agent_sdk.runtime import RequestHeaders


def _manifest_dict(agent_id: str = "demo", protocol_mode: str = "stream") -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "name": "Demo",
        "version": "0.1.0",
        "kind": "expert_basic",
        "runtime": "external_a2a",
        "capabilities": [],
        "declared_gates": [],
        "protocol_mode": protocol_mode,
        "endpoint": "http://localhost:8888",
        "supports_streaming": protocol_mode == "stream",
    }


def _request_headers(**overrides: str) -> RequestHeaders:
    base = {
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "project_id": "project-1",
        "user_id": "user-1",
        "service_principal": "agent:demo",
        "session_id": "session-1",
        "step_id": "step-1",
        "request_id": "req-1",
    }
    base.update(overrides)
    return RequestHeaders(**base)


# ── Construction surface ─────────────────────────────────────────────────────


def test_artifact_agent_returns_app_from_manifest_dict() -> None:
    app = artifact_agent(manifest=_manifest_dict())
    assert isinstance(app, ArtifactAgentApp)
    # Underlying Agent escape hatch is preserved.
    assert app.agent is not None


def test_artifact_agent_loads_manifest_from_path(tmp_path: Path) -> None:
    manifest_path = tmp_path / "agent.json"
    manifest_path.write_text(json.dumps(_manifest_dict()), encoding="utf-8")
    app = artifact_agent(manifest=manifest_path)
    assert isinstance(app, ArtifactAgentApp)


def test_handle_rejects_sync_handler() -> None:
    app = artifact_agent(manifest=_manifest_dict())

    def sync_handler(ctx: Any) -> Any:  # pragma: no cover — never called
        return None

    with pytest.raises(TypeError):
        app.handle(sync_handler)


def test_handle_returns_decorated_function() -> None:
    """``@app.handle`` should preserve the original function reference
    (so authors can still call it from tests directly)."""
    app = artifact_agent(manifest=_manifest_dict())

    async def my_handler(ctx: ArtifactAgentContext) -> ArtifactResult:
        return ctx.artifact(artifact_type="x", summary="x")

    result = app.handle(my_handler)
    assert result is my_handler


# ── Context projection ──────────────────────────────────────────────────────


def test_build_context_projects_inputs_and_runtime_fields() -> None:
    headers = _request_headers()
    ctx = _build_context(
        input_payload={
            "context": {
                "session_id": "s-1",
                "thread_id": "t-1",
                "request_id": "r-1",
                "tenant": {"tenant_id": "ten-1", "workspace_id": "ws-1"},
                "identity": {
                    "principal_id": "p-1",
                    "principal_type": "user",
                    "roles": ["editor"],
                },
            },
            "inputs": {
                "text": "hello world",
                "capability_id": "agent.demo.go",
            },
            "brief": {"user_goal": "G"},
        },
        headers=headers,
        platform=None,
        progress_emitter=None,
    )
    assert ctx.input_text == "hello world"
    assert ctx.inputs["text"] == "hello world"
    assert ctx.capability_id == "agent.demo.go"
    assert ctx.project["tenant_id"] == "ten-1"
    assert ctx.project["workspace_id"] == "ws-1"
    assert ctx.project["session_id"] == "s-1"
    assert ctx.project["thread_id"] == "t-1"
    assert ctx.project["project_id"] == "project-1"  # from headers
    assert ctx.member["principal_id"] == "p-1"
    assert ctx.member["roles"] == ("editor",)
    assert ctx.member["service_principal"] == "agent:demo"
    assert ctx.runtime_context["session_id"] == "s-1"


def test_build_context_falls_back_to_brief_for_input_text() -> None:
    headers = _request_headers()
    ctx = _build_context(
        input_payload={
            "inputs": {},
            "brief": {"user_goal": "G", "summary": "S"},
        },
        headers=headers,
        platform=None,
        progress_emitter=None,
    )
    assert ctx.input_text == "G"


def test_build_context_resolves_capability_from_grants() -> None:
    headers = _request_headers()
    ctx = _build_context(
        input_payload={
            "inputs": {
                "capability_grants": [
                    {"capability_id": "agent.demo.fallback"},
                ],
            },
        },
        headers=headers,
        platform=None,
        progress_emitter=None,
    )
    assert ctx.capability_id == "agent.demo.fallback"


def test_build_context_attachments_from_inputs_then_brief() -> None:
    headers = _request_headers()
    ctx_from_inputs = _build_context(
        input_payload={
            "inputs": {"attachments": [{"id": "a-1"}, "skip-non-mapping"]},
            "brief": {"attachments": [{"id": "b-1"}]},
        },
        headers=headers,
        platform=None,
        progress_emitter=None,
    )
    # inputs source wins; non-mapping entries are filtered out.
    assert [a["id"] for a in ctx_from_inputs.attachments] == ["a-1"]

    ctx_from_brief = _build_context(
        input_payload={
            "inputs": {},
            "brief": {"attachments": [{"id": "b-1"}]},
        },
        headers=headers,
        platform=None,
        progress_emitter=None,
    )
    assert [a["id"] for a in ctx_from_brief.attachments] == ["b-1"]


def test_build_context_handles_malformed_runtime_context() -> None:
    """Non-mapping ``context``/``inputs``/``brief`` should not crash —
    project to empty defaults instead."""
    headers = _request_headers()
    ctx = _build_context(
        input_payload={
            "context": "not-a-mapping",
            "inputs": ["not", "a", "mapping"],
            "brief": 42,
        },
        headers=headers,
        platform=None,
        progress_emitter=None,
    )
    assert ctx.input_text == ""
    assert ctx.inputs == {}
    assert ctx.attachments == []


# ── ctx.artifact() and result coercion ──────────────────────────────────────


def test_ctx_artifact_returns_frozen_dataclass() -> None:
    headers = _request_headers()
    ctx = _build_context(
        input_payload={"inputs": {}}, headers=headers,
        platform=None, progress_emitter=None,
    )
    result = ctx.artifact(
        artifact_type="report",
        summary="done",
        content={"body": "x"},
        metadata={"confidence": "high"},
        provenance={"model": "claude"},
    )
    assert isinstance(result, ArtifactResult)
    # Frozen — direct mutation raises.
    with pytest.raises((AttributeError, Exception)):
        result.summary = "tampered"  # type: ignore[misc]


def test_ctx_artifact_to_invoke_output_shape() -> None:
    result = ArtifactResult(
        artifact_type="market_report",
        summary="done",
        content={"body": "x"},
        metadata={"k": "v"},
        provenance={"src": "y"},
    )
    output = result.to_invoke_output()
    assert output == {
        "kind": "artifact",
        "artifact_type": "market_report",
        "summary": "done",
        "content": {"body": "x"},
        "metadata": {"k": "v"},
        "provenance": {"src": "y"},
    }


def test_ctx_needs_confirmation_to_invoke_response_shape() -> None:
    headers = _request_headers()
    ctx = _build_context(
        input_payload={"inputs": {}}, headers=headers,
        platform=None, progress_emitter=None,
    )
    result = ctx.needs_confirmation(
        prompt="Approve sending the report?",
        confirmation_id="confirm-1",
        resume_reference={"invocation_id": "invoke-1"},
        timeout_policy={"after": "PT1H", "on_timeout": "cancel"},
        metadata={"risk": "external_write"},
        reason="external side effect",
    )

    assert isinstance(result, NeedsConfirmationResult)
    response = result.to_invoke_response()
    assert response["status"] == "needs_confirmation"
    assert response["output"]["kind"] == "needs_confirmation"
    assert response["confirmation"] == {
        "confirmation_id": "confirm-1",
        "prompt": "Approve sending the report?",
        "allowed_actions": ["approve", "request_changes", "reject"],
        "resume_reference": {"invocation_id": "invoke-1"},
        "timeout_policy": {"after": "PT1H", "on_timeout": "cancel"},
        "metadata": {"risk": "external_write"},
        "reason": "external side effect",
    }


def test_coerce_outcome_accepts_dict_passthrough() -> None:
    out = _coerce_outcome(
        {"artifact_type": "x", "summary": "y", "content": 1}
    )
    assert isinstance(out, ArtifactResult)
    assert out.artifact_type == "x"
    assert out.content == 1


def test_coerce_outcome_accepts_needs_confirmation() -> None:
    result = NeedsConfirmationResult(prompt="Approve?")
    assert _coerce_outcome(result) is result


def test_coerce_outcome_rejects_none() -> None:
    with pytest.raises(RuntimeError, match="returned None"):
        _coerce_outcome(None)


def test_coerce_outcome_rejects_dict_without_artifact_type() -> None:
    with pytest.raises(RuntimeError, match="artifact_type"):
        _coerce_outcome({"summary": "x"})


def test_coerce_outcome_rejects_other_types() -> None:
    with pytest.raises(RuntimeError, match="must return"):
        _coerce_outcome("just a string")  # type: ignore[arg-type]


# ── Platform namespace contract (W4: live or unavailable namespace) ─────────


def test_default_platform_unavailable_when_no_base_url(monkeypatch) -> None:
    """W4: with no ``NOVIE_PLATFORM_BASE_URL`` set, the resolved
    ``ctx.platform`` is an unavailable namespace that returns
    ``platform_unavailable`` diagnostics rather than raising. Handlers
    branch on ``platform.is_available``."""
    from novie_agent_sdk.platform_namespace import _UnavailablePlatformNamespace

    monkeypatch.delenv("NOVIE_PLATFORM_BASE_URL", raising=False)
    app = artifact_agent(manifest=_manifest_dict())
    resolved = app._resolve_platform(_request_headers())  # noqa: SLF001
    assert isinstance(resolved, _UnavailablePlatformNamespace)
    assert resolved.is_available is False


def test_platform_namespace_can_be_injected() -> None:
    sentinel = object()
    app = artifact_agent(manifest=_manifest_dict(), platform=sentinel)
    # Injection always wins regardless of headers / base url.
    assert app._resolve_platform(_request_headers()) is sentinel  # noqa: SLF001


def test_platform_namespace_built_when_base_url_and_headers_present(
    monkeypatch,
) -> None:
    """When base URL + tenant/project headers are available, the
    resolver returns a live ``PlatformNamespace`` (not the
    unavailable stand-in)."""
    from novie_agent_sdk.platform_namespace import PlatformNamespace

    monkeypatch.delenv("NOVIE_PLATFORM_BASE_URL", raising=False)
    app = artifact_agent(
        manifest=_manifest_dict(),
        platform_base_url="http://platform.test",
    )
    resolved = app._resolve_platform(_request_headers())  # noqa: SLF001
    assert isinstance(resolved, PlatformNamespace)
    assert resolved.is_available is True


# ── Invoke wire contract ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invoke_endpoint_returns_artifact_output_shape() -> None:
    from fastapi.testclient import TestClient

    app = artifact_agent(manifest=_manifest_dict(protocol_mode="simple"))

    @app.handle
    async def handle(ctx: ArtifactAgentContext) -> ArtifactResult:
        await ctx.progress("Reading context")
        return ctx.artifact(
            artifact_type="market_report",
            summary="ready",
            content={"text": ctx.input_text or "default"},
            metadata={"confidence": "medium"},
        )

    fastapi_app = app.build_app()
    client = TestClient(fastapi_app)

    resp = client.post(
        "/invoke",
        json={"input": {"inputs": {"text": "Tell me about widgets"}}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "completed"
    output = body["output"]
    assert output["kind"] == "artifact"
    assert output["artifact_type"] == "market_report"
    assert output["summary"] == "ready"
    assert output["content"] == {"text": "Tell me about widgets"}
    assert output["metadata"] == {"confidence": "medium"}
    assert output["provenance"] == {}


@pytest.mark.asyncio
async def test_invoke_handler_progress_log_buffers_events() -> None:
    """Invoke mode buffers progress events into ``ctx.progress_log``
    so authors can write a single mode-agnostic handler."""
    from fastapi.testclient import TestClient

    captured: dict[str, Any] = {}

    app = artifact_agent(manifest=_manifest_dict(protocol_mode="simple"))

    @app.handle
    async def handle(ctx: ArtifactAgentContext) -> ArtifactResult:
        await ctx.progress("step-1")
        await ctx.progress("step-2", metadata={"phase": "deep"})
        captured["log"] = list(ctx.progress_log)
        return ctx.artifact(artifact_type="x", summary="ok")

    client = TestClient(app.build_app())
    resp = client.post("/invoke", json={"input": {"inputs": {}}})
    assert resp.status_code == 200, resp.text
    assert [e["text"] for e in captured["log"]] == ["step-1", "step-2"]
    assert captured["log"][1]["metadata"] == {"phase": "deep"}


@pytest.mark.asyncio
async def test_invoke_endpoint_can_return_needs_confirmation() -> None:
    from fastapi.testclient import TestClient

    app = artifact_agent(manifest=_manifest_dict(protocol_mode="simple"))

    @app.handle
    async def handle(ctx: ArtifactAgentContext) -> NeedsConfirmationResult:
        return ctx.needs_confirmation(
            prompt="Approve publishing this artifact?",
            confirmation_id="confirm-1",
            resume_reference={"capability_id": ctx.capability_id},
        )

    client = TestClient(app.build_app())
    resp = client.post(
        "/invoke",
        json={"input": {"inputs": {"capability_id": "agent.demo.publish"}}},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "needs_confirmation"
    assert body["output"]["kind"] == "needs_confirmation"
    assert body["confirmation"]["prompt"] == "Approve publishing this artifact?"
    assert body["confirmation"]["resume_reference"] == {
        "capability_id": "agent.demo.publish",
    }


# ── Stream wire contract ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_endpoint_emits_progress_then_artifact_then_done() -> None:
    """Acceptance bullet: progress events appear in platform session
    timeline. The wire stream is: progress* → artifact → SDK-appended
    done event."""
    from fastapi.testclient import TestClient

    app = artifact_agent(manifest=_manifest_dict(protocol_mode="stream"))

    @app.handle
    async def handle(ctx: ArtifactAgentContext) -> ArtifactResult:
        await ctx.progress("Reading context")
        await asyncio.sleep(0)  # let queue drain so order is observable
        await ctx.progress("Calling model", metadata={"phase": "llm"})
        await asyncio.sleep(0)
        return ctx.artifact(
            artifact_type="market_report",
            summary="Market report complete",
            content="hello",
            metadata={"confidence": "medium"},
        )

    client = TestClient(app.build_app())
    events: list[dict[str, Any]] = []
    with client.stream(
        "POST", "/stream", json={"input": {"inputs": {}}},
    ) as resp:
        assert resp.status_code == 200, resp.text
        for line in resp.iter_lines():
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))

    kinds = [e["kind"] for e in events]
    # Two progress events + one artifact + one SDK-appended done.
    assert kinds.count("progress") == 2
    assert kinds.count("artifact") == 1
    assert kinds[-1] == "done"
    # Order: progress events come first, artifact lands before done.
    progress_idxs = [i for i, k in enumerate(kinds) if k == "progress"]
    artifact_idx = kinds.index("artifact")
    done_idx = kinds.index("done")
    assert all(i < artifact_idx for i in progress_idxs)
    assert artifact_idx < done_idx
    artifact_event = events[artifact_idx]
    assert artifact_event["artifact_type"] == "market_report"
    assert artifact_event["summary"] == "Market report complete"
    assert artifact_event["metadata"] == {"confidence": "medium"}


@pytest.mark.asyncio
async def test_stream_handler_exception_propagates_as_500() -> None:
    """A handler that raises should fail the stream with a terminal_error."""
    from fastapi.testclient import TestClient

    app = artifact_agent(manifest=_manifest_dict(protocol_mode="stream"))

    @app.handle
    async def handle(ctx: ArtifactAgentContext) -> ArtifactResult:
        raise RuntimeError("boom")

    client = TestClient(app.build_app())
    events: list[dict[str, Any]] = []
    with client.stream(
        "POST", "/stream", json={"input": {"inputs": {}}},
    ) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if not line:
                continue
            events.append(json.loads(line))

    assert events[-1]["kind"] == "terminal_error"
    assert events[-1]["error"] == "boom"
    assert events[-1]["metadata"]["terminal_source"] == "sdk_exception_guard"


@pytest.mark.asyncio
async def test_stream_endpoint_can_emit_needs_confirmation_then_done() -> None:
    from fastapi.testclient import TestClient

    app = artifact_agent(manifest=_manifest_dict(protocol_mode="stream"))

    @app.handle
    async def handle(ctx: ArtifactAgentContext) -> NeedsConfirmationResult:
        await ctx.progress("Checking side effects")
        return ctx.needs_confirmation(
            prompt="Approve external write?",
            allowed_actions=("approve", "reject"),
        )

    client = TestClient(app.build_app())
    events: list[dict[str, Any]] = []
    with client.stream(
        "POST", "/stream", json={"input": {"inputs": {}}},
    ) as resp:
        assert resp.status_code == 200, resp.text
        for line in resp.iter_lines():
            line = line.strip()
            if line:
                events.append(json.loads(line))

    kinds = [event["kind"] for event in events]
    assert kinds == ["progress", "needs_confirmation", "done"]
    confirmation = events[1]["confirmation"]
    assert confirmation["prompt"] == "Approve external write?"
    assert confirmation["allowed_actions"] == ["approve", "reject"]


# ── Acceptance bullet: < 30 lines for a minimal agent ───────────────────────


def test_minimal_artifact_agent_under_30_lines() -> None:
    """Acceptance bullet: 'A minimal artifact agent can be implemented
    in under 30 lines of Python.' Counted on the canonical example
    fixture below — re-evaluated whenever the surface changes."""
    minimal = textwrap.dedent(
        '''
        from novie_agent_sdk import artifact_agent, ArtifactAgentContext

        app = artifact_agent(manifest=".well-known/agent.json")

        @app.handle
        async def handle(ctx: ArtifactAgentContext):
            await ctx.progress("Reading context")
            return ctx.artifact(
                artifact_type="market_report",
                summary="Market report complete",
                content={"text": ctx.input_text},
                metadata={"confidence": "medium"},
            )

        fastapi_app = app.build_app()
        '''
    ).strip()
    non_blank = [
        line for line in minimal.splitlines() if line.strip()
    ]
    assert len(non_blank) < 30, (
        f"minimal artifact agent example exceeded 30 lines: "
        f"{len(non_blank)} non-blank lines"
    )


# ── Proxies to the underlying Agent ─────────────────────────────────────────


def test_app_build_app_proxies_to_agent() -> None:
    app = artifact_agent(manifest=_manifest_dict())

    @app.handle
    async def handle(ctx: ArtifactAgentContext) -> ArtifactResult:
        return ctx.artifact(artifact_type="x", summary="y")

    fastapi_app = app.build_app()
    routes = [getattr(r, "path", "") for r in fastapi_app.routes]
    # Both invoke + stream wired (handler registers both).
    assert "/invoke" in routes
    assert "/stream" in routes
    assert "/.well-known/agent.json" in routes
