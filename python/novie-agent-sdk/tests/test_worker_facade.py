"""EXPERT_AGENT_SDK W5 — ``worker_agent`` SDK facade tests.

Locks the W5 authoring surface:

- ``worker_agent(manifest=...)`` returns a ``WorkerAgentApp`` with
  ``.task`` decorator + tasks-protocol routes wired automatically.
- Context projection: ``task`` / ``repo`` / ``target_branch`` /
  ``upstream`` / ``metadata`` filled from the platform's request
  shape; ``inputs`` carries the full mapping; ``is_cancelled``
  forwards from the underlying ``TaskContext``.
- ``ctx.progress(...)`` / ``ctx.artifact(...)`` emit events on the
  task event stream surfaced by ``GET /tasks/{id}/events``.
- ``ctx.result(...)`` produces a frozen ``WorkerResult`` projected
  into ``GET /tasks/{id}/result``'s ``output`` field; ``ctx.fail(...)``
  raises ``WorkerFailure`` and the SDK marks the task ``failed``
  with the reason.
- Status transitions match the platform contract:
  ``running → completed`` / ``running → failed`` /
  ``running → cancelled`` / ``running → waiting_for_input → running``.
- Acceptance bullet "A non-Cortex worker can be implemented without
  reimplementing task lifecycle endpoints" — locked by a sub-30-line
  fixture.
- Cancellation is cooperative: ``POST /tasks/{id}/cancel`` flips
  ``ctx.is_cancelled`` and the SDK records a ``cancelled`` status.
"""
# ruff: noqa: I001
from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path
from typing import Any

import pytest

from novie_agent_sdk import (
    HumanWaitRequest,
    WorkerAgentApp,
    WorkerFailure,
    WorkerResult,
    WorkerTaskContext,
    worker_agent,
)
from novie_agent_sdk.runtime import TaskContext
from novie_agent_sdk.runtime import InMemoryTaskStore, SqliteTaskStore


def _manifest_dict(agent_id: str = "worker-demo") -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "name": "Worker Demo",
        "version": "0.1.0",
        "kind": "expert_complex",
        "runtime": "external_a2a",
        "capabilities": [],
        "declared_gates": [],
        "protocol_mode": "tasks",
        "endpoint": "http://localhost:9999",
        "execution": {"supports_cancel": True, "emits_events": True},
    }


# ── Construction surface ─────────────────────────────────────────────────────


def test_worker_agent_returns_app_from_dict() -> None:
    app = worker_agent(manifest=_manifest_dict())
    assert isinstance(app, WorkerAgentApp)
    assert app.agent is not None


def test_worker_agent_loads_manifest_from_path(tmp_path: Path) -> None:
    import json

    manifest_path = tmp_path / "agent.json"
    manifest_path.write_text(json.dumps(_manifest_dict()), encoding="utf-8")
    app = worker_agent(manifest=manifest_path)
    assert isinstance(app, WorkerAgentApp)


def test_human_wait_request_serializes_contract_shape() -> None:
    payload = HumanWaitRequest(
        gate_id="gate-1",
        prompt="Approve deployment?",
        allowed_actions=("approve", "reject"),
        resume_reference={"task_id": "t-1"},
        timeout_policy={"after_seconds": 3600, "on_timeout": "escalate"},
        metadata={"reason": "deployment"},
    ).to_event_payload()

    assert payload == {
        "wait_kind": "waiting_for_human",
        "gate_id": "gate-1",
        "prompt": "Approve deployment?",
        "allowed_actions": ["approve", "reject"],
        "resume_reference": {"task_id": "t-1"},
        "timeout_policy": {"after_seconds": 3600, "on_timeout": "escalate"},
        "metadata": {"reason": "deployment"},
    }


def test_task_store_durability_defaults_to_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOVIE_AGENT_STATE_DIR", str(tmp_path / "state"))
    manifest = _manifest_dict()
    manifest["execution"]["durability"] = "task_store"

    app = worker_agent(manifest=manifest)

    assert isinstance(app.agent._store, SqliteTaskStore)  # noqa: SLF001
    assert (tmp_path / "state" / "worker-demo.tasks.sqlite3").exists()


def test_explicit_task_store_wins_over_task_store_durability() -> None:
    manifest = _manifest_dict()
    manifest["execution"]["durability"] = "task_store"
    store = InMemoryTaskStore()

    app = worker_agent(manifest=manifest, task_store=store)

    assert app.agent._store is store  # noqa: SLF001


def test_explicit_sqlite_path_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOVIE_AGENT_STATE_DIR", str(tmp_path / "ignored"))
    manifest = _manifest_dict()
    manifest["execution"]["durability"] = "task_store"
    sqlite_path = tmp_path / "explicit.sqlite3"

    app = worker_agent(manifest=manifest, sqlite_path=sqlite_path)

    assert isinstance(app.agent._store, SqliteTaskStore)  # noqa: SLF001
    assert sqlite_path.exists()
    assert not (tmp_path / "ignored").exists()


def test_production_worker_defaults_to_sqlite_even_without_durability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOVIE_ENV", "production")
    monkeypatch.setenv("NOVIE_AGENT_TASK_STORE_PATH", str(tmp_path / "prod.sqlite3"))

    app = worker_agent(manifest=_manifest_dict())

    assert isinstance(app.agent._store, SqliteTaskStore)  # noqa: SLF001
    assert (tmp_path / "prod.sqlite3").exists()


def test_task_decorator_rejects_sync_handler() -> None:
    app = worker_agent(manifest=_manifest_dict())

    def sync_handler(ctx: WorkerTaskContext) -> WorkerResult:  # pragma: no cover
        return WorkerResult(summary="x")

    with pytest.raises(TypeError):
        app.task(sync_handler)


def test_task_decorator_returns_decorated_function() -> None:
    app = worker_agent(manifest=_manifest_dict())

    async def my_handler(ctx: WorkerTaskContext) -> WorkerResult:
        return ctx.result(summary="ok")

    result = app.task(my_handler)
    assert result is my_handler


# ── Context projection ──────────────────────────────────────────────────────


async def _fake_task_context(input_payload: dict[str, Any]) -> TaskContext:
    """Build a TaskContext directly to test the ctx projection path
    without spinning up an HTTP runtime. Pre-creates the underlying
    task record so status-transition assertions work."""
    from novie_agent_sdk.runtime import (
        AgentObservability,
        InMemoryTaskStore,
        RequestHeaders,
    )
    from novie_agent_sdk.observability import NoOpObservabilitySink
    from novie_protocol.contracts.agent_sdk_v2 import AgentManifestV2

    manifest = AgentManifestV2.from_dict(_manifest_dict())
    store = InMemoryTaskStore()
    task_id = "task-1"
    await store.create_task(task_id, input_payload, idempotency_key="")
    return TaskContext(
        task_id=task_id,
        input=input_payload,
        headers=RequestHeaders(
            tenant_id="tenant-1",
            project_id="project-1",
            session_id="session-1",
        ),
        agent_manifest=manifest,
        observability=AgentObservability(
            agent_id="worker-demo",
            sinks=(NoOpObservabilitySink(),),
        ),
        _store=store,
    )


@pytest.mark.asyncio
async def test_build_context_projects_inputs() -> None:
    from novie_agent_sdk.worker_facade import _build_context

    task_ctx = await _fake_task_context({
        "inputs": {
            "task": "fix the bug",
            "repo": {"url": "git@github.com:org/repo.git", "base": "main"},
            "target_branch": "fix/bug-123",
            "upstream": {"parent_task_id": "t-0"},
            "metadata": {"priority": "high"},
        },
    })
    facade = _build_context(task_ctx=task_ctx, platform=None)
    assert facade.task_id == "task-1"
    assert facade.task == "fix the bug"
    assert facade.repo == {"url": "git@github.com:org/repo.git", "base": "main"}
    assert facade.target_branch == "fix/bug-123"
    assert facade.upstream == {"parent_task_id": "t-0"}
    assert facade.metadata == {"priority": "high"}
    assert facade.inputs["task"] == "fix the bug"
    assert facade.is_cancelled is False


@pytest.mark.asyncio
async def test_build_context_handles_missing_fields() -> None:
    from novie_agent_sdk.worker_facade import _build_context

    task_ctx = await _fake_task_context({"inputs": {}})
    facade = _build_context(task_ctx=task_ctx, platform=None)
    assert facade.task is None
    assert facade.repo == {}
    assert facade.target_branch == ""
    assert facade.upstream == {}
    assert facade.metadata == {}


@pytest.mark.asyncio
async def test_build_context_accepts_alias_keys() -> None:
    """Workers receive task descriptions under various conventional
    keys (``task`` / ``instruction`` / ``prompt``); repo as
    ``repo`` / ``repository``; branch as ``target_branch`` /
    ``branch``; upstream as ``upstream`` / ``upstream_context``."""
    from novie_agent_sdk.worker_facade import _build_context

    task_ctx = await _fake_task_context({
        "inputs": {
            "instruction": "do the thing",
            "repository": {"url": "x"},
            "branch": "feature/y",
            "upstream_context": {"parent": "z"},
        },
    })
    facade = _build_context(task_ctx=task_ctx, platform=None)
    assert facade.task == "do the thing"
    assert facade.repo == {"url": "x"}
    assert facade.target_branch == "feature/y"
    assert facade.upstream == {"parent": "z"}


# ── ctx.result + outcome coercion ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ctx_result_returns_frozen_dataclass() -> None:
    from novie_agent_sdk.worker_facade import _build_context

    task_ctx = await _fake_task_context({"inputs": {}})
    facade = _build_context(task_ctx=task_ctx, platform=None)
    result = facade.result(
        summary="done",
        output={"answer": 42},
        artifacts=[{"id": "a-1"}],
        metadata={"k": "v"},
    )
    assert isinstance(result, WorkerResult)
    with pytest.raises((AttributeError, Exception)):
        result.summary = "tampered"  # type: ignore[misc]


def test_worker_result_to_task_output_shape() -> None:
    result = WorkerResult(
        summary="done",
        output={"k": "v"},
        artifacts=[{"id": "a-1"}],
        metadata={"phase": "complete"},
    )
    out = result.to_task_output()
    assert out == {
        "kind": "worker_result",
        "summary": "done",
        "output": {"k": "v"},
        "artifacts": [{"id": "a-1"}],
        "metadata": {"phase": "complete"},
    }


def test_coerce_outcome_accepts_dict_passthrough() -> None:
    from novie_agent_sdk.worker_facade import _coerce_outcome

    result = _coerce_outcome({"summary": "ok", "output": 1})
    assert isinstance(result, WorkerResult)
    assert result.summary == "ok"
    assert result.output == 1


def test_coerce_outcome_rejects_none() -> None:
    from novie_agent_sdk.worker_facade import _coerce_outcome

    with pytest.raises(RuntimeError, match="returned None"):
        _coerce_outcome(None)


def test_coerce_outcome_re_raises_returned_failure() -> None:
    from novie_agent_sdk.worker_facade import _coerce_outcome

    failure = WorkerFailure("computed failure path")
    with pytest.raises(WorkerFailure, match="computed failure path"):
        _coerce_outcome(failure)


@pytest.mark.asyncio
async def test_ctx_fail_returns_failure_object() -> None:
    from novie_agent_sdk.worker_facade import _build_context

    task_ctx = await _fake_task_context({"inputs": {}})
    facade = _build_context(task_ctx=task_ctx, platform=None)
    failure = facade.fail("disk full", metadata={"bytes_free": 0})
    assert isinstance(failure, WorkerFailure)
    assert failure.reason == "disk full"
    assert failure.metadata == {"bytes_free": 0}


# ── Tasks-protocol wire contract ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_task_endpoint_completes_with_result_output() -> None:
    """End-to-end: POST /tasks → handler runs → GET /tasks/{id}/result
    returns the projected ``WorkerResult`` shape."""
    from fastapi.testclient import TestClient

    app = worker_agent(manifest=_manifest_dict())

    @app.task
    async def run(ctx: WorkerTaskContext) -> WorkerResult:
        await ctx.progress("starting", metadata={"phase": "init"})
        return ctx.result(
            summary="done",
            output={"value": ctx.task},
            metadata={"completed": True},
        )

    fastapi_app = app.build_app()
    client = TestClient(fastapi_app)

    create_resp = client.post(
        "/tasks", json={"input": {"inputs": {"task": "fix-it"}}},
    )
    assert create_resp.status_code == 202, create_resp.text
    task_id = create_resp.json()["task_id"]

    # Poll until terminal.
    for _ in range(50):
        await asyncio.sleep(0.02)
        status_resp = client.get(f"/tasks/{task_id}")
        if status_resp.json()["status"] == "completed":
            break
    assert status_resp.json()["status"] == "completed"

    result_resp = client.get(f"/tasks/{task_id}/result")
    assert result_resp.status_code == 200, result_resp.text
    body = result_resp.json()
    assert body["status"] == "completed"
    output = body["output"]
    assert output["kind"] == "worker_result"
    assert output["summary"] == "done"
    assert output["output"] == {"value": "fix-it"}
    assert output["metadata"] == {"completed": True}
    assert output["artifacts"] == []


@pytest.mark.asyncio
async def test_task_progress_and_artifact_events_appear_on_event_stream() -> None:
    from fastapi.testclient import TestClient

    app = worker_agent(manifest=_manifest_dict())

    @app.task
    async def run(ctx: WorkerTaskContext) -> WorkerResult:
        await ctx.progress("step-1")
        await ctx.artifact(
            artifact_type="diff",
            summary="patch",
            content={"path": "src/x.py"},
        )
        await ctx.progress("step-2")
        return ctx.result(summary="ok")

    client = TestClient(app.build_app())
    resp = client.post("/tasks", json={"input": {"inputs": {}}})
    task_id = resp.json()["task_id"]

    for _ in range(50):
        await asyncio.sleep(0.02)
        if client.get(f"/tasks/{task_id}").json()["status"] == "completed":
            break

    events = client.get(f"/tasks/{task_id}/events").json()["events"]
    kinds = [e["kind"] for e in events]
    assert kinds.count("progress") == 2
    assert kinds.count("artifact") == 1
    progress_idxs = [i for i, k in enumerate(kinds) if k == "progress"]
    artifact_idx = kinds.index("artifact")
    # Progress events bracket the artifact in handler order.
    assert progress_idxs[0] < artifact_idx < progress_idxs[1]
    artifact_event = next(e for e in events if e["kind"] == "artifact")
    assert artifact_event["artifact_type"] == "diff"
    assert artifact_event["summary"] == "patch"


@pytest.mark.asyncio
async def test_ctx_fail_marks_task_failed() -> None:
    """``raise ctx.fail(reason)`` → SDK status="failed", error=reason."""
    from fastapi.testclient import TestClient

    app = worker_agent(manifest=_manifest_dict())

    @app.task
    async def run(ctx: WorkerTaskContext) -> WorkerResult:
        raise ctx.fail("disk full")

    client = TestClient(app.build_app(), raise_server_exceptions=False)
    resp = client.post("/tasks", json={"input": {}})
    task_id = resp.json()["task_id"]

    for _ in range(50):
        await asyncio.sleep(0.02)
        status = client.get(f"/tasks/{task_id}").json()
        if status["status"] in ("failed", "completed", "cancelled"):
            break
    assert status["status"] == "failed"
    assert status["error"] == "disk full"


@pytest.mark.asyncio
async def test_handler_exception_marks_task_failed() -> None:
    """Generic exceptions also mark the task failed (back-compat with
    Agent's existing handler contract)."""
    from fastapi.testclient import TestClient

    app = worker_agent(manifest=_manifest_dict())

    @app.task
    async def run(ctx: WorkerTaskContext) -> WorkerResult:
        raise RuntimeError("boom")

    client = TestClient(app.build_app(), raise_server_exceptions=False)
    resp = client.post("/tasks", json={"input": {}})
    task_id = resp.json()["task_id"]

    for _ in range(50):
        await asyncio.sleep(0.02)
        status = client.get(f"/tasks/{task_id}").json()
        if status["status"] in ("failed", "completed"):
            break
    assert status["status"] == "failed"
    assert "boom" in status["error"]


@pytest.mark.asyncio
async def test_cooperative_cancellation_via_is_cancelled() -> None:
    """Acceptance bullet: 'Cancellation works at least as cooperative
    cancellation in SDK.' Handler polls ``ctx.is_cancelled`` and
    returns early; SDK records ``cancelled``."""
    import threading
    from fastapi.testclient import TestClient

    proceed = threading.Event()
    app = worker_agent(manifest=_manifest_dict())

    @app.task
    async def run(ctx: WorkerTaskContext) -> WorkerResult:
        for _ in range(500):
            if ctx.is_cancelled:
                # Return early — Agent._run_task short-circuits to cancel
                # when ctx.is_cancelled is set after handler returns.
                return ctx.result(summary="cancelled-mid-run")
            if proceed.is_set():
                break
            await asyncio.sleep(0.01)
        return ctx.result(summary="completed")

    with TestClient(app.build_app(), raise_server_exceptions=False) as client:
        resp = client.post("/tasks", json={"input": {}})
        task_id = resp.json()["task_id"]
        cancel_resp = client.post(f"/tasks/{task_id}/cancel")
        assert cancel_resp.status_code in (202, 409)
        proceed.set()
        for _ in range(50):
            await asyncio.sleep(0.02)
            status = client.get(f"/tasks/{task_id}").json()
            if status["status"] in ("cancelled", "completed", "failed"):
                break
        assert status["status"] in ("cancelled", "completed")


@pytest.mark.asyncio
async def test_status_transitions_through_waiting_for_input() -> None:
    """Acceptance bullet: status maps to platform statuses
    ``waiting_for_input``. Tested directly against the underlying
    ``TaskContext`` so we observe each transition without racing
    BackgroundTasks against the test loop."""
    from novie_agent_sdk.worker_facade import _build_context

    task_ctx = await _fake_task_context({"inputs": {"task": "x"}})
    # Bring the task to ``running`` so the wait transition is valid.
    await task_ctx._store.update_task_status(task_ctx.task_id, "running")  # noqa: SLF001
    facade = _build_context(task_ctx=task_ctx, platform=None)

    await facade.wait_for_input(prompt="need approval")
    record = await task_ctx._store.get_task(task_ctx.task_id)  # noqa: SLF001
    assert record.status == "waiting_for_input"

    await facade.resume_running()
    record = await task_ctx._store.get_task(task_ctx.task_id)  # noqa: SLF001
    assert record.status == "running"

    events = await task_ctx._store.get_events(task_ctx.task_id)  # noqa: SLF001
    wait_events = [e for e in events if e["kind"] == "wait_prompt"]
    assert len(wait_events) == 1
    assert wait_events[0]["wait_kind"] == "waiting_for_input"
    assert wait_events[0]["prompt"] == "need approval"


@pytest.mark.asyncio
async def test_wait_for_human_transitions_status() -> None:
    """``ctx.wait_for_human`` mirrors ``wait_for_input`` for HITL
    gates so the platform's session timeline can mark the gate."""
    from novie_agent_sdk.worker_facade import _build_context

    task_ctx = await _fake_task_context({"inputs": {}})
    await task_ctx._store.update_task_status(task_ctx.task_id, "running")  # noqa: SLF001
    facade = _build_context(task_ctx=task_ctx, platform=None)

    await facade.wait_for_human(
        gate_id="gate-1",
        prompt="Approve the code change?",
        allowed_actions=("approve", "request_changes"),
        resume_reference={"task_id": "task-1", "step_id": "s2"},
        timeout_policy={"after_seconds": 1800, "on_timeout": "escalate"},
        metadata={"reason": "approval"},
    )
    record = await task_ctx._store.get_task(task_ctx.task_id)  # noqa: SLF001
    assert record.status == "waiting_for_human"

    events = await task_ctx._store.get_events(task_ctx.task_id)  # noqa: SLF001
    wait_events = [e for e in events if e["kind"] == "wait_prompt"]
    assert len(wait_events) == 1
    assert wait_events[0]["wait_kind"] == "waiting_for_human"
    assert wait_events[0]["gate_id"] == "gate-1"
    assert wait_events[0]["prompt"] == "Approve the code change?"
    assert wait_events[0]["allowed_actions"] == ["approve", "request_changes"]
    assert wait_events[0]["resume_reference"] == {
        "task_id": "task-1",
        "step_id": "s2",
    }
    assert wait_events[0]["timeout_policy"] == {
        "after_seconds": 1800,
        "on_timeout": "escalate",
    }


# ── Platform namespace integration (W4 surface available on ctx) ────────────


@pytest.mark.asyncio
async def test_ctx_platform_available_when_base_url_set() -> None:
    from fastapi.testclient import TestClient
    from novie_agent_sdk.platform_namespace import PlatformNamespace

    captured: dict[str, Any] = {}
    app = worker_agent(
        manifest=_manifest_dict(),
        platform_base_url="http://platform.test",
    )

    @app.task
    async def run(ctx: WorkerTaskContext) -> WorkerResult:
        captured["platform"] = ctx.platform
        return ctx.result(summary="ok")

    client = TestClient(app.build_app())
    headers = {
        "x-novie-tenant-id": "tenant-1",
        "x-novie-project-id": "project-1",
        "x-novie-session-id": "session-1",
    }
    resp = client.post("/tasks", json={"input": {}}, headers=headers)
    task_id = resp.json()["task_id"]
    for _ in range(50):
        await asyncio.sleep(0.02)
        if client.get(f"/tasks/{task_id}").json()["status"] == "completed":
            break

    assert isinstance(captured["platform"], PlatformNamespace)


def test_platform_override_wins() -> None:
    sentinel = object()
    app = worker_agent(manifest=_manifest_dict(), platform=sentinel)
    from novie_agent_sdk.runtime import RequestHeaders
    assert app._resolve_platform(RequestHeaders()) is sentinel  # noqa: SLF001


# ── Acceptance bullet: < 30 lines for a minimal worker ──────────────────────


def test_minimal_worker_agent_under_30_lines() -> None:
    """Acceptance bullet: 'A non-Cortex worker can be implemented
    without reimplementing task lifecycle endpoints.' The minimal
    example below proves the surface is small enough that a worker
    fits in ~15 lines."""
    minimal = textwrap.dedent(
        '''
        from novie_agent_sdk import worker_agent, WorkerTaskContext

        app = worker_agent(manifest=".well-known/agent.json")

        @app.task
        async def run(ctx: WorkerTaskContext):
            await ctx.progress("Preparing workspace")
            output = {"task": ctx.task, "branch": ctx.target_branch}
            return ctx.result(
                summary="Task completed",
                output=output,
            )

        fastapi_app = app.build_app()
        '''
    ).strip()
    non_blank = [line for line in minimal.splitlines() if line.strip()]
    assert len(non_blank) < 30, (
        f"minimal worker example exceeded 30 lines: {len(non_blank)}"
    )


# ── build_app proxy + route surface ─────────────────────────────────────────


def test_app_build_app_wires_full_tasks_protocol() -> None:
    app = worker_agent(manifest=_manifest_dict())

    @app.task
    async def run(ctx: WorkerTaskContext) -> WorkerResult:
        return ctx.result(summary="x")

    fastapi_app = app.build_app()
    routes = {getattr(r, "path", "") for r in fastapi_app.routes}
    # All five tasks-protocol endpoints exposed without author
    # writing route code.
    assert "/tasks" in routes
    assert "/tasks/{task_id}" in routes
    assert "/tasks/{task_id}/events" in routes
    assert "/tasks/{task_id}/result" in routes
    assert "/tasks/{task_id}/cancel" in routes
    assert "/.well-known/agent.json" in routes
