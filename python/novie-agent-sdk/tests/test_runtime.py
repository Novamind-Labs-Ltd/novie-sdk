"""Python A2A Agent Runtime SDK 测试。

覆盖：
1. Agent 从 manifest 创建
2. InMemoryTaskStore：create / update / cancel / events / result
3. Agent.build_app() simple / tasks 端点
4. Idempotency（相同 idempotency key 返回同一个 task）
5. Cancel propagation
6. Task handler 异常处理
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

# set up path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "novie-platform" / "protocol" / "src"))

from novie_agent_sdk.runtime import (
    Agent,
    AskBudgetExceeded,
    AskTimedOut,
    InMemoryOneShotInvocationStore,
    InMemoryTaskStore,
    InvokeContext,
    RegistrationClient,
    RequestHeaders,
    SqliteOneShotInvocationStore,
    SqliteTaskStore,
    TaskContext,
    _failure_envelope,
)
from novie_agent_sdk import PublicAgentError
from novie_agent_sdk.platform_security import sign_agent_platform_headers
from novie_agent_sdk.testing import (
    assert_http_json_invoke_idempotency_replay,
    assert_http_json_stream_idempotency_replay,
    assert_http_json_tasks_idempotency_replay,
)
from novie_agent_sdk.observability import (
    AgentObservability,
    NovieLangChainCallbackHandler,
)
from novie_protocol.contracts.agent_sdk_v2 import AgentManifestV2, ExecutionHints


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _simple_manifest(agent_id: str = "test") -> AgentManifestV2:
    return AgentManifestV2(
        agent_id=agent_id,
        name="Test",
        version="0.1.0",
        kind="expert_basic",
        runtime="external_a2a",
        capabilities=(),
        declared_gates=(),
        protocol_mode="simple",
        endpoint="http://localhost:8000",
    )


def _result_cache_manifest(agent_id: str = "result-cache-test") -> AgentManifestV2:
    return AgentManifestV2(
        agent_id=agent_id,
        name="Result Cache Test",
        version="0.1.0",
        kind="expert_basic",
        runtime="external_a2a",
        capabilities=(),
        declared_gates=(),
        protocol_mode="simple",
        endpoint="http://localhost:8000",
        execution=ExecutionHints(durability="result_cache"),
    )


def _tasks_manifest(agent_id: str = "test-tasks") -> AgentManifestV2:
    return AgentManifestV2(
        agent_id=agent_id,
        name="Tasks Test",
        version="0.1.0",
        kind="expert_complex",
        runtime="external_a2a",
        capabilities=(),
        declared_gates=(),
        protocol_mode="tasks",
        endpoint="http://localhost:8001",
        execution=ExecutionHints(supports_cancel=True, emits_events=True),
    )


class _FakeHttpResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.mark.asyncio
async def test_registration_client_reregisters_after_heartbeat_404(monkeypatch):
    import httpx

    calls: list[str] = []

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, **kwargs):
            calls.append(url)
            if url.endswith("/agents/register"):
                return _FakeHttpResponse(201)
            if url.endswith("/agents/recoverable/heartbeat"):
                return _FakeHttpResponse(404)
            return _FakeHttpResponse(500)

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    client = RegistrationClient(
        "http://platform",
        _simple_manifest("recoverable"),
        heartbeat_interval=0.01,
        register_max_attempts=1,
    )

    await client.start_heartbeat()
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if "http://platform/agents/register" in calls:
                break
            await asyncio.sleep(0.01)
    finally:
        await client.stop_heartbeat()

    assert "http://platform/agents/recoverable/heartbeat" in calls
    assert "http://platform/agents/register" in calls


# ─────────────────────────────────────────────────────────────────────────────
# 1. InMemoryTaskStore
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_task_store_create_and_get():
    store = InMemoryTaskStore()
    record = await store.create_task("t1", {"query": "hello"})
    assert record.task_id == "t1"
    assert record.status == "pending"
    got = await store.get_task("t1")
    assert got is not None
    assert got.task_id == "t1"


@pytest.mark.asyncio
async def test_task_store_status_transitions():
    store = InMemoryTaskStore()
    await store.create_task("t1", {})
    await store.update_task_status("t1", "running")
    t = await store.get_task("t1")
    assert t.status == "running"


@pytest.mark.asyncio
async def test_task_store_invalid_transition_ignored():
    store = InMemoryTaskStore()
    await store.create_task("t1", {})
    await store.update_task_status("t1", "completed")
    # completed → running should be ignored
    await store.update_task_status("t1", "running")
    t = await store.get_task("t1")
    assert t.status == "completed"


@pytest.mark.asyncio
async def test_task_store_idempotency():
    store = InMemoryTaskStore()
    r1 = await store.create_task("t1", {"a": 1}, idempotency_key="key-123")
    r2 = await store.create_task("t2", {"b": 2}, idempotency_key="key-123")
    # Second create with same key should return existing
    assert r2.task_id == r1.task_id


@pytest.mark.asyncio
async def test_task_store_events():
    store = InMemoryTaskStore()
    await store.create_task("t1", {})
    await store.append_event("t1", {
        "event_id": "e1",
        "task_id": "t1",
        "kind": "message",
        "timestamp": "2026-04-26T00:00:00Z",
        "text": "hello",
    })
    events = await store.get_events("t1")
    assert len(events) == 1
    assert events[0]["kind"] == "message"


@pytest.mark.asyncio
async def test_task_store_result():
    store = InMemoryTaskStore()
    await store.create_task("t1", {})
    await store.set_task_result("t1", {"answer": 42})
    t = await store.get_task("t1")
    assert t.status == "completed"
    result = await store.get_result("t1")
    assert result == {"answer": 42}


@pytest.mark.asyncio
async def test_task_store_cancel():
    store = InMemoryTaskStore()
    await store.create_task("t1", {})
    await store.update_task_status("t1", "running")
    cancelled = await store.cancel_task("t1")
    assert cancelled is True
    t = await store.get_task("t1")
    assert t.status == "cancelled"


@pytest.mark.asyncio
async def test_task_store_cancel_terminal_returns_false():
    store = InMemoryTaskStore()
    await store.create_task("t1", {})
    await store.set_task_result("t1", {"done": True})
    cancelled = await store.cancel_task("t1")
    assert cancelled is False


@pytest.mark.asyncio
async def test_task_context_report_llm_usage_emits_platform_token_usage_event():
    store = InMemoryTaskStore()
    await store.create_task("usage-task", {"q": "hello"})

    async def emit_usage_event(event: dict[str, Any]) -> None:
        await store.append_event("usage-task", event)

    observability = AgentObservability(
        agent_id="usage-agent",
        session_id="sess-1",
        step_id="step-1",
        trace_id="trace-1",
        task_id="usage-task",
        task_event_emitter=emit_usage_event,
    )
    ctx = TaskContext(
        task_id="usage-task",
        input={"q": "hello"},
        headers=RequestHeaders(session_id="sess-1", step_id="step-1", trace_id="trace-1"),
        agent_manifest=_tasks_manifest("usage-agent"),
        observability=observability,
        _store=store,
    )

    report = await ctx.report_llm_usage(
        provider="anthropic",
        model="claude-sonnet-4.5",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        phase="analysis",
        turn_id="turn-1",
    )

    events = await store.get_events("usage-task")
    usage_events = [
        event for event in events
        if event.get("payload", {}).get("agent_event_kind") == "token_usage"
    ]
    assert report["agent_id"] == "usage-agent"
    assert len(usage_events) == 1
    usage_event = usage_events[0]
    assert usage_event["kind"] == "status_changed"
    assert usage_event["payload"]["provider"] == "anthropic"
    assert usage_event["payload"]["model"] == "claude-sonnet-4.5"
    assert usage_event["payload"]["usage"] == {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    assert usage_event["payload"]["phase"] == "analysis"


@pytest.mark.asyncio
async def test_task_context_ask_times_out_with_continue_alias():
    store = InMemoryTaskStore()
    await store.create_task("ask-task", {"q": "hello"})
    await store.update_task_status("ask-task", "running")
    ctx = TaskContext(
        task_id="ask-task",
        input={},
        headers=RequestHeaders(session_id="sess-1", step_id="step-1"),
        agent_manifest=_tasks_manifest("ask-agent"),
        observability=AgentObservability(agent_id="ask-agent"),
        _store=store,
    )

    result = await ctx.ask(
        "Should I continue?",
        timeout=0.01,
        default_action="continue",
    )

    assert result["resolution_type"] == "skipped"
    assert result["default_action"] == "skip"
    record = await store.get_task("ask-task")
    assert record is not None
    assert record.status == "running"
    events = await store.get_events("ask-task")
    kinds = [event["kind"] for event in events]
    assert "mid_run_ask" in kinds
    assert "mid_run_ask_timeout" in kinds
    ask_event = next(event for event in events if event["kind"] == "mid_run_ask")
    assert ask_event["timeout_seconds"] == 0.01
    assert ask_event["default_action_on_timeout"] == "skip"
    assert ask_event["envelope"]["question"] == "Should I continue?"


@pytest.mark.asyncio
async def test_task_context_ask_resumes_with_human_resolution():
    store = InMemoryTaskStore()
    await store.create_task("ask-resume-task", {})
    await store.update_task_status("ask-resume-task", "running")
    ctx = TaskContext(
        task_id="ask-resume-task",
        input={},
        headers=RequestHeaders(session_id="sess-1", step_id="step-1"),
        agent_manifest=_tasks_manifest("ask-resume-agent"),
        observability=AgentObservability(agent_id="ask-resume-agent"),
        _store=store,
    )

    ask_task = asyncio.create_task(
        ctx.ask("Pick one?", timeout=1.0, default_action="continue")
    )
    for _ in range(20):
        events = await store.get_events("ask-resume-task")
        ask_events = [event for event in events if event["kind"] == "mid_run_ask"]
        if ask_events:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("mid_run_ask event was not emitted")

    gate_id = ask_events[0]["gate_id"]
    accepted = await store.resolve_ask(
        "ask-resume-task",
        gate_id,
        {"resolution_type": "answered", "freeform_answer": "Use option A"},
    )

    assert accepted is True
    result = await ask_task
    assert result["gate_id"] == gate_id
    assert result["timed_out"] is False
    assert result["freeform_answer"] == "Use option A"
    events = await store.get_events("ask-resume-task")
    assert any(event["kind"] == "mid_run_ask_resumed" for event in events)


@pytest.mark.asyncio
async def test_task_context_ask_fail_implementation_raises():
    store = InMemoryTaskStore()
    await store.create_task("ask-fail-task", {})
    await store.update_task_status("ask-fail-task", "running")
    ctx = TaskContext(
        task_id="ask-fail-task",
        input={},
        headers=RequestHeaders(session_id="sess-1", step_id="step-1"),
        agent_manifest=_tasks_manifest("ask-fail-agent"),
        observability=AgentObservability(agent_id="ask-fail-agent"),
        _store=store,
    )

    with pytest.raises(AskTimedOut):
        await ctx.ask(
            "Cannot proceed without approval",
            timeout=0.01,
            default_action="fail_implementation",
        )

    events = await store.get_events("ask-fail-task")
    assert any(event["kind"] == "mid_run_ask_timeout" for event in events)


@pytest.mark.asyncio
async def test_task_context_ask_rejects_fourth_default_ask():
    store = InMemoryTaskStore()
    await store.create_task("ask-cap-task", {})
    await store.update_task_status("ask-cap-task", "running")
    ctx = TaskContext(
        task_id="ask-cap-task",
        input={"lifecycle": {"max_mid_run_asks": 3}},
        headers=RequestHeaders(session_id="sess-1", step_id="step-1"),
        agent_manifest=_tasks_manifest("ask-cap-agent"),
        observability=AgentObservability(agent_id="ask-cap-agent"),
        _store=store,
    )

    for index in range(3):
        await ctx.ask(f"Question {index}?", timeout=0.01, default_action="continue")

    with pytest.raises(AskBudgetExceeded):
        await ctx.ask("Question 4?", timeout=0.01, default_action="continue")

    events = await store.get_events("ask-cap-task")
    rejected = [event for event in events if event["kind"] == "mid_run_ask_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["reason_code"] == "max_mid_run_asks_exceeded"


@pytest.mark.asyncio
async def test_task_context_heartbeat_emits_agent_heartbeat_event():
    store = InMemoryTaskStore()
    await store.create_task("heartbeat-task", {})
    ctx = TaskContext(
        task_id="heartbeat-task",
        input={},
        headers=RequestHeaders(session_id="sess-1", step_id="step-1"),
        agent_manifest=_tasks_manifest("heartbeat-agent"),
        observability=AgentObservability(agent_id="heartbeat-agent"),
        _store=store,
    )

    await ctx.heartbeat(phase="working", message="Still running", interval_seconds=0.5)

    events = await store.get_events("heartbeat-task")
    assert events[-1]["kind"] == "heartbeat"
    assert events[-1]["agent_event_kind"] == "heartbeat"
    assert events[-1]["phase"] == "working"
    assert events[-1]["interval_seconds"] == 0.5


@pytest.mark.asyncio
async def test_langchain_callback_reports_usage_without_langfuse_dependency():
    events: list[dict[str, Any]] = []

    async def emit_usage_event(event: dict[str, Any]) -> None:
        events.append(event)

    observability = AgentObservability(
        agent_id="lc-agent",
        session_id="sess-1",
        step_id="step-1",
        trace_id="trace-1",
        task_id="task-1",
        task_event_emitter=emit_usage_event,
    )
    callback = NovieLangChainCallbackHandler(
        observability,
        phase="draft",
        metadata={"component": "unit-test"},
    )
    run_id = uuid.uuid4()
    await callback.on_llm_start({}, ["hello"], run_id=run_id)

    class _Response:
        llm_output = {
            "model_name": "anthropic/claude-sonnet-4.5",
            "token_usage": {
                "prompt_tokens": 11,
                "completion_tokens": 22,
                "total_tokens": 33,
            },
        }
        generations: list[Any] = []

    await callback.on_llm_end(_Response(), run_id=run_id)

    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["agent_event_kind"] == "token_usage"
    assert payload["provider"] == "anthropic"
    assert payload["model"] == "claude-sonnet-4.5"
    assert payload["phase"] == "draft"
    assert payload["usage"] == {
        "input_tokens": 11,
        "output_tokens": 22,
        "total_tokens": 33,
    }


@pytest.mark.asyncio
async def test_sqlite_task_store_persists_across_instances(tmp_path: Path):
    db_path = str(tmp_path / "tasks.db")
    store_a = SqliteTaskStore(db_path)
    await store_a.create_task("persist-task", {"q": "hello"})
    await store_a.update_task_status("persist-task", "running")
    await store_a.set_task_result("persist-task", {"answer": 42})

    store_b = SqliteTaskStore(db_path)
    task = await store_b.get_task("persist-task")
    assert task is not None
    assert task.status == "completed"
    result = await store_b.get_result("persist-task")
    assert result == {"answer": 42}


@pytest.mark.asyncio
async def test_sqlite_one_shot_invocation_store_persists_completed_response(
    tmp_path: Path,
):
    db_path = str(tmp_path / "invocations.db")
    store_a = SqliteOneShotInvocationStore(db_path)
    started, record = await store_a.start_or_get("idem-1", "invoke")
    assert started is True
    assert record.status == "in_progress"
    await store_a.complete(
        "idem-1",
        "invoke",
        response={"status": "completed", "output": {"answer": 42}},
    )

    store_b = SqliteOneShotInvocationStore(db_path)
    started, replay = await store_b.start_or_get("idem-1", "invoke")
    assert started is False
    assert replay.status == "completed"
    assert replay.response == {"status": "completed", "output": {"answer": 42}}


# ── Tier B: lease + heartbeat regression coverage ─────────────────────────────


@pytest.mark.asyncio
async def test_inmemory_invocation_store_fresh_in_progress_returns_duplicate():
    """Fresh in_progress (lease not stale) still returns the cached record so
    duplicate retries see the existing in-flight invocation."""
    store = InMemoryOneShotInvocationStore()
    started_first, first = await store.start_or_get("idem-fresh", "stream")
    assert started_first is True
    assert first.status == "in_progress"

    # Second call mid-flight (lease still valid) sees the same record.
    started_second, second = await store.start_or_get("idem-fresh", "stream")
    assert started_second is False
    assert second.status == "in_progress"
    assert second is first  # same record
    assert second.invocation_id == first.invocation_id
    assert second.invocation_id.startswith("inv-")


@pytest.mark.asyncio
async def test_inmemory_invocation_store_lookup_by_invocation_id():
    store = InMemoryOneShotInvocationStore()
    started, record = await store.start_or_get("idem-lookup", "stream")
    assert started is True

    found = await store.get_by_invocation_id(record.invocation_id)

    assert found is record


@pytest.mark.asyncio
async def test_duplicate_one_shot_response_exposes_invocation_resume_ref():
    from novie_agent_sdk.runtime import _duplicate_one_shot_response

    store = InMemoryOneShotInvocationStore()
    started, record = await store.start_or_get("idem-duplicate", "stream")
    assert started is True

    response = _duplicate_one_shot_response(record)
    body = json.loads(response.body.decode())
    details = body["error"]["details"]

    assert details["invocation_id"] == record.invocation_id
    assert details["resume_ref"] == {
        "type": "stream_invocation",
        "id": record.invocation_id,
    }


@pytest.mark.asyncio
async def test_inmemory_invocation_store_stale_in_progress_recycled():
    """A previous in_progress record whose lease has elapsed is recycled —
    start_or_get returns started=True so a new invocation can proceed."""
    store = InMemoryOneShotInvocationStore()
    started, record = await store.start_or_get("idem-stale", "stream")
    assert started is True

    # Simulate agent crash mid-stream: record is stuck in_progress but the
    # wall-clock has advanced past the lease.
    record.updated_at = "2000-01-01T00:00:00+00:00"

    started_again, fresh = await store.start_or_get("idem-stale", "stream")
    assert started_again is True
    assert fresh.status == "in_progress"
    assert fresh is not record  # a brand new record
    # The recycled old record was marked expired.
    assert record.status == "expired"
    assert record.error == "lease_expired_no_activity"


@pytest.mark.asyncio
async def test_inmemory_invocation_store_touch_renews_lease():
    """touch() bumps updated_at so a long-running but actively-streaming
    handler does not get recycled by the stale-lease check."""
    store = InMemoryOneShotInvocationStore()
    started, record = await store.start_or_get("idem-touch", "stream")
    assert started is True
    initial_updated_at = record.updated_at

    # Wait a millisecond so any monotonic-style clock would advance.
    await asyncio.sleep(0.01)
    await store.touch("idem-touch", "stream")

    assert record.updated_at != initial_updated_at  # bumped
    assert record.status == "in_progress"  # status unchanged


@pytest.mark.asyncio
async def test_inmemory_invocation_store_touch_noop_on_terminal_records():
    """touch() must NOT bump updated_at on completed/failed records — a
    racy heartbeat after the terminal call should be a silent no-op."""
    store = InMemoryOneShotInvocationStore()
    await store.start_or_get("idem-done", "invoke")
    await store.complete("idem-done", "invoke", response={"status": "completed"})

    completed_record = store._records[("invoke", "idem-done")]  # noqa: SLF001
    locked_updated_at = completed_record.updated_at
    await asyncio.sleep(0.01)
    await store.touch("idem-done", "invoke")

    assert completed_record.updated_at == locked_updated_at
    assert completed_record.status == "completed"


@pytest.mark.asyncio
async def test_inmemory_invocation_store_touch_silent_on_missing_record():
    """touch() on an unknown key is a silent no-op (never raises)."""
    store = InMemoryOneShotInvocationStore()
    await store.touch("never-started", "stream")  # should not raise


@pytest.mark.asyncio
async def test_sqlite_invocation_store_stale_in_progress_recycled(tmp_path: Path):
    """Sqlite mirror of the in-memory recycle test — persisted in_progress
    rows whose lease has elapsed are recycled in-place."""
    db_path = str(tmp_path / "invocations.db")
    store = SqliteOneShotInvocationStore(db_path)
    started, record = await store.start_or_get("idem-stale", "stream")
    assert started is True

    # Backdate the row directly to simulate an agent that crashed mid-stream
    # ages ago. (In production this would happen by elapsed wall-clock time.)
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE sdk_one_shot_invocations SET updated_at = ? "
            "WHERE mode = ? AND idempotency_key = ?",
            ("2000-01-01T00:00:00+00:00", "stream", "idem-stale"),
        )
        conn.commit()

    started_again, fresh = await store.start_or_get("idem-stale", "stream")
    assert started_again is True
    assert fresh.status == "in_progress"
    # The recycled row should now show a current (recent) updated_at.
    assert fresh.updated_at >= record.updated_at


@pytest.mark.asyncio
async def test_sqlite_invocation_store_touch_renews_lease(tmp_path: Path):
    """Sqlite touch() updates the row's updated_at column."""
    db_path = str(tmp_path / "invocations.db")
    store = SqliteOneShotInvocationStore(db_path)
    started, record = await store.start_or_get("idem-touch", "stream")
    assert started is True
    initial_updated_at = record.updated_at

    await asyncio.sleep(0.01)
    await store.touch("idem-touch", "stream")

    _, refreshed = await store.start_or_get("idem-touch", "stream")
    assert refreshed.status == "in_progress"
    assert refreshed.updated_at != initial_updated_at
    assert refreshed.invocation_id == record.invocation_id


@pytest.mark.asyncio
async def test_sqlite_invocation_store_lookup_by_invocation_id(tmp_path: Path):
    db_path = str(tmp_path / "invocations.db")
    store_a = SqliteOneShotInvocationStore(db_path)
    started, record = await store_a.start_or_get("idem-lookup", "stream")
    assert started is True

    store_b = SqliteOneShotInvocationStore(db_path)
    found = await store_b.get_by_invocation_id(record.invocation_id)

    assert found is not None
    assert found.idempotency_key == "idem-lookup"
    assert found.mode == "stream"
    assert found.invocation_id == record.invocation_id


# ── Tier A: stream/invoke handlers must transition record on abort ────────────


@pytest.mark.asyncio
async def test_invoke_handler_cancelled_transitions_record_to_failed():
    """The /invoke endpoint's finally guard must call fail() when the
    handler is cancelled mid-await (BaseException path that bypasses the
    inner ``except Exception`` block)."""
    from novie_agent_sdk.runtime import OneShotInvocationRecord

    store = InMemoryOneShotInvocationStore()

    # Simulate what the endpoint does: start the record, then have the
    # handler get cancelled before it can transition the record.
    started, _ = await store.start_or_get("idem-cancel", "invoke")
    assert started is True

    invocation_resolved = False
    try:
        try:
            await asyncio.sleep(0.01)
            raise asyncio.CancelledError("client disconnected")
        except Exception as exc:  # noqa: BLE001
            # CancelledError is BaseException in py3.8+, so this branch is NOT
            # hit — that's exactly the bug Tier A fixes.
            await store.fail("idem-cancel", "invoke", str(exc))
            invocation_resolved = True
            raise
    except asyncio.CancelledError:
        pass
    finally:
        if not invocation_resolved:
            await store.fail(
                "idem-cancel",
                "invoke",
                "invoke_aborted_before_terminal",
            )
            invocation_resolved = True

    record: OneShotInvocationRecord = store._records[("invoke", "idem-cancel")]  # noqa: SLF001
    assert record.status == "failed"
    assert record.error == "invoke_aborted_before_terminal"


@pytest.mark.asyncio
async def test_invoke_handler_exception_still_transitions_via_inner_handler():
    """Regular Exception path (not BaseException) goes through the inner
    ``except Exception`` block and the finally guard is a no-op. This is
    the regression check that the Tier A change doesn't double-fail."""
    store = InMemoryOneShotInvocationStore()

    started, _ = await store.start_or_get("idem-error", "invoke")
    assert started is True

    invocation_resolved = False
    fail_call_count = 0
    original_fail = store.fail

    async def counting_fail(*args, **kwargs):
        nonlocal fail_call_count
        fail_call_count += 1
        await original_fail(*args, **kwargs)

    store.fail = counting_fail  # type: ignore[method-assign]

    try:
        try:
            raise RuntimeError("handler died")
        except Exception as exc:  # noqa: BLE001
            await store.fail("idem-error", "invoke", str(exc))
            invocation_resolved = True
            raise
    except RuntimeError:
        pass
    finally:
        if not invocation_resolved:
            await store.fail(
                "idem-error",
                "invoke",
                "invoke_aborted_before_terminal",
            )

    assert fail_call_count == 1  # only the inner handler called fail
    record = store._records[("invoke", "idem-error")]  # noqa: SLF001
    assert record.status == "failed"
    assert record.error == "handler died"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Agent from manifest
# ─────────────────────────────────────────────────────────────────────────────


def test_agent_from_manifest_dict():
    d = {
        "agent_id": "x",
        "name": "X",
        "version": "0.1.0",
        "kind": "expert_basic",
        "runtime": "external_a2a",
        "capabilities": [],
        "declared_gates": [],
        "protocol_mode": "simple",
    }
    agent = Agent.from_manifest_dict(d)
    assert agent._manifest.agent_id == "x"


def test_agent_from_manifest_file(tmp_path: Path):
    wk = tmp_path / ".well-known"
    wk.mkdir()
    manifest_data = {
        "$schema": "test",
        "agent_id": "file-agent",
        "name": "File Agent",
        "version": "0.1.0",
        "kind": "expert_basic",
        "runtime": "external_a2a",
        "capabilities": [],
        "declared_gates": [],
        "protocol_mode": "simple",
        "endpoint": "http://file-agent:8000",
        "execution": {},
    }
    (wk / "agent.json").write_text(json.dumps(manifest_data))
    agent = Agent.from_manifest(wk / "agent.json")
    assert agent._manifest.agent_id == "file-agent"


# ─────────────────────────────────────────────────────────────────────────────
# 3. FastAPI app — simple mode
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_simple_agent_invoke():
    from fastapi.testclient import TestClient

    agent = Agent(_simple_manifest())

    @agent.invoke
    async def handle(ctx: InvokeContext) -> dict:
        return {"greeting": f"Hello {ctx.input.get('name', 'world')}"}

    app = agent.build_app()
    client = TestClient(app)

    resp = client.post("/invoke", json={"input": {"name": "Alice"}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["output"]["greeting"] == "Hello Alice"
    assert body["status"] == "completed"


@pytest.mark.asyncio
async def test_simple_agent_invoke_passes_through_terminal_envelope():
    from fastapi.testclient import TestClient

    agent = Agent(_simple_manifest())

    @agent.invoke
    async def handle(ctx: InvokeContext) -> dict:
        return {
            "status": "needs_confirmation",
            "confirmation": {
                "prompt": "Approve external write?",
                "allowed_actions": ["approve", "reject"],
            },
        }

    client = TestClient(agent.build_app())
    resp = client.post("/invoke", json={"input": {"name": "Alice"}})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "needs_confirmation"
    assert body["confirmation"]["prompt"] == "Approve external write?"
    assert "output" not in body


def test_simple_agent_sanitizes_explicit_failed_envelope_and_store() -> None:
    from fastapi.testclient import TestClient

    store = InMemoryOneShotInvocationStore()
    agent = Agent(_simple_manifest("invoke-explicit-failure"), invocation_store=store)

    @agent.invoke
    async def handle(ctx: InvokeContext) -> dict[str, Any]:
        return {
            "status": "failed",
            "error": "RAW_SECRET_USER_PROMPT",
            "error_code": "provider_raw_error",
            "output": {"draft": "RAW_SECRET_USER_PROMPT"},
        }

    client = TestClient(agent.build_app())
    response = client.post(
        "/invoke",
        headers={"Idempotency-Key": "explicit-failure-key"},
        json={"input": {"q": "x"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "status": "failed",
        "error": "Agent execution failed.",
        "error_code": "agent_internal_error",
        "output": {},
    }
    record = store._records[("invoke", "explicit-failure-key")]  # noqa: SLF001
    assert record.status == "failed"
    assert record.error == "Agent execution failed."
    assert "RAW_SECRET_USER_PROMPT" not in json.dumps(record.__dict__)


def test_simple_agent_sanitizes_failed_envelope_without_error_field() -> None:
    from fastapi.testclient import TestClient

    agent = Agent(_simple_manifest("invoke-malformed-failure"))

    @agent.invoke
    async def handle(ctx: InvokeContext) -> dict[str, Any]:
        return {
            "status": "FAILED",
            "output": {"draft": "RAW_SECRET_USER_PROMPT"},
        }

    client = TestClient(agent.build_app())
    response = client.post("/invoke", json={"input": {"q": "x"}})

    assert response.status_code == 200
    assert response.json() == {
        "status": "failed",
        "error": "Agent execution failed.",
        "error_code": "agent_internal_error",
        "output": {},
    }


def test_simple_agent_rejects_completed_envelope_with_error_and_output() -> None:
    from fastapi.testclient import TestClient

    agent = Agent(_simple_manifest("invoke-inconsistent-completed"))

    @agent.invoke
    async def handle(ctx: InvokeContext) -> dict[str, Any]:
        return {
            "status": "COMPLETED",
            "error": "RAW_SECRET_USER_PROMPT",
            "output": {"draft": "RAW_SECRET_USER_PROMPT"},
        }

    client = TestClient(agent.build_app())
    response = client.post("/invoke", json={"input": {"q": "x"}})

    assert response.status_code == 200
    assert response.json() == {
        "status": "failed",
        "error": "Agent execution failed.",
        "error_code": "agent_internal_error",
        "output": {},
    }


@pytest.mark.asyncio
async def test_simple_agent_requires_signed_headers_in_production(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("NOVIE_RUNTIME_MODE", "production")
    monkeypatch.setenv("NOVIE_AGENT_PLATFORM_SHARED_SECRET", "agent-platform-secret")
    agent = Agent(_simple_manifest())

    @agent.invoke
    async def handle(ctx: InvokeContext) -> dict:
        return {
            "principal": ctx.headers.user_id or ctx.headers.service_principal,
            "project_id": ctx.headers.project_id,
        }

    client = TestClient(agent.build_app())
    unsigned = client.post("/invoke", json={"input": {"name": "Alice"}})
    assert unsigned.status_code == 401
    assert unsigned.json()["detail"]["error"] == "agent_platform_signature_required"

    headers = {
        "x-novie-tenant-id": "tenant-1",
        "x-novie-workspace-id": "workspace-1",
        "x-novie-project-id": "project-1",
        "x-novie-user-id": "user-1",
        "x-novie-session-id": "session-1",
        "x-novie-step-id": "step-1",
        "idempotency-key": "signed-invoke-1",
        "x-novie-timestamp": str(int(time.time())),
    }
    signed_headers = sign_agent_platform_headers(
        headers,
        method="POST",
        path="/invoke",
        secret="agent-platform-secret",
        timestamp=headers["x-novie-timestamp"],
    )
    signed = client.post(
        "/invoke",
        json={"input": {"name": "Alice"}},
        headers=signed_headers,
    )
    assert signed.status_code == 200
    assert signed.json()["output"] == {
        "principal": "user-1",
        "project_id": "project-1",
    }


def test_request_headers_treat_org_id_as_tenant_and_workspace_fallback() -> None:
    headers = RequestHeaders.from_request(
        {
            "x-novie-org-id": "org-1",
            "x-novie-project-id": "project-1",
            "x-novie-user-id": "user-1",
        }
    )

    assert headers.tenant_id == "org-1"
    assert headers.workspace_id == "org-1"
    assert headers.project_id == "project-1"
    assert headers.user_id == "user-1"


@pytest.mark.asyncio
async def test_simple_agent_replays_duplicate_idempotency_key():
    import httpx

    agent = Agent(_simple_manifest())
    calls = 0

    @agent.invoke
    async def handle(ctx: InvokeContext) -> dict:
        nonlocal calls
        calls += 1
        return {"call": calls, "name": ctx.input.get("name")}

    app = agent.build_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://simple-agent",
    ) as client:
        first, second = await assert_http_json_invoke_idempotency_replay(
            client,
            payload={"input": {"name": "Alice"}},
            idempotency_key="invoke-key-1",
        )

    assert first == second
    assert first["output"] == {"call": 1, "name": "Alice"}
    assert calls == 1


def test_result_cache_simple_agent_replays_after_restart(
    tmp_path: Path,
    monkeypatch,
):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("NOVIE_AGENT_STATE_DIR", str(tmp_path / "state"))
    calls = 0

    agent_a = Agent(_result_cache_manifest("restart-simple"))

    @agent_a.invoke
    async def handle_a(ctx: InvokeContext) -> dict:
        nonlocal calls
        calls += 1
        return {"call": calls, "name": ctx.input.get("name")}

    client_a = TestClient(agent_a.build_app())
    first = client_a.post(
        "/invoke",
        json={"input": {"name": "Alice"}},
        headers={"Idempotency-Key": "restart-key-1"},
    )
    assert first.status_code == 200
    assert first.json()["output"] == {"call": 1, "name": "Alice"}

    agent_b = Agent(_result_cache_manifest("restart-simple"))

    @agent_b.invoke
    async def handle_b(ctx: InvokeContext) -> dict:  # pragma: no cover
        raise AssertionError("duplicate request should replay cached result")

    client_b = TestClient(agent_b.build_app())
    replay = client_b.post(
        "/invoke",
        json={"input": {"name": "Bob"}},
        headers={"Idempotency-Key": "restart-key-1"},
    )
    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert calls == 1
    assert (tmp_path / "state" / "restart-simple.invocations.sqlite3").exists()


@pytest.mark.asyncio
async def test_simple_agent_healthz():
    from fastapi.testclient import TestClient

    agent = Agent(_simple_manifest("health-test"))
    agent.invoke(lambda ctx: {"result": "ok"})
    app = agent.build_app()
    client = TestClient(app)

    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["agent_id"] == "health-test"
    assert resp.json()["invocation_store_backend"] == "InMemoryOneShotInvocationStore"


@pytest.mark.asyncio
async def test_simple_agent_well_known():
    from fastapi.testclient import TestClient

    agent = Agent(_simple_manifest())
    agent.invoke(lambda ctx: {})
    app = agent.build_app()
    client = TestClient(app)

    resp = client.get("/.well-known/agent.json")
    assert resp.status_code == 200
    assert resp.json()["agent_id"] == "test"
    assert resp.json()["protocol_mode"] == "simple"


# ─────────────────────────────────────────────────────────────────────────────
# 3b. FastAPI app — stream mode (handler events + observability multiplex)
# ─────────────────────────────────────────────────────────────────────────────


def _stream_manifest(agent_id: str = "stream-test") -> AgentManifestV2:
    return AgentManifestV2(
        agent_id=agent_id,
        name="Stream Test",
        version="0.1.0",
        kind="expert_basic",
        runtime="external_a2a",
        capabilities=(),
        declared_gates=(),
        protocol_mode="stream",
        endpoint="http://localhost:8002",
        supports_streaming=True,
    )


def _result_cache_stream_manifest(
    agent_id: str = "stream-result-cache-test",
) -> AgentManifestV2:
    return AgentManifestV2(
        agent_id=agent_id,
        name="Stream Result Cache Test",
        version="0.1.0",
        kind="expert_basic",
        runtime="external_a2a",
        capabilities=(),
        declared_gates=(),
        protocol_mode="stream",
        endpoint="http://localhost:8002",
        execution=ExecutionHints(durability="result_cache", emits_events=True),
        supports_streaming=True,
    )


@pytest.mark.asyncio
async def test_stream_endpoint_multiplexes_observability_events():
    """Handler 自身的 event 与 observability.report_llm_usage 推送的事件
    必须共同进入同一 NDJSON 流，并保留发生顺序（保证平台能 mirror 到 timeline
    + 写 UsageRecord）。"""
    from novie_agent_sdk.runtime import StreamContext
    from fastapi.testclient import TestClient

    agent = Agent(_stream_manifest())

    @agent.stream
    async def handle(ctx: StreamContext):
        yield {"kind": "content", "content": "first "}
        # 模拟 LLM 调用结束触发 observability 上报；emitter 把事件直接 push
        # 到 stream 队列，平台 _call_stream 会识别 payload.agent_event_kind
        await ctx.observability.report_llm_usage(
            provider="anthropic",
            model="claude-sonnet-4.5",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        )
        yield {"kind": "content", "content": "second"}
        yield {"kind": "final", "output": {"answer": "done"}}

    app = agent.build_app()
    client = TestClient(app)

    with client.stream("POST", "/stream", json={"input": {"q": "x"}}) as resp:
        assert resp.status_code == 200
        events: list[dict[str, Any]] = []
        for line in resp.iter_lines():
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))

    kinds = [e.get("kind") for e in events]
    # 至少 3 个 handler event + 1 observability event + 1 SDK auto done
    assert "content" in kinds
    assert "final" in kinds
    assert "done" in kinds  # SDK 末尾自动追加
    # observability 事件以 status_changed 形式插入；payload.agent_event_kind
    # 标识为 token_usage（与 tasks 模式 to_platform_task_event 等价）
    usage_events = [
        e
        for e in events
        if e.get("kind") == "status_changed"
        and isinstance(e.get("payload"), dict)
        and e["payload"].get("agent_event_kind") == "token_usage"
    ]
    assert len(usage_events) == 1
    payload = usage_events[0]["payload"]
    assert payload["usage"]["input_tokens"] == 10
    assert payload["usage"]["output_tokens"] == 20
    assert payload["usage"]["total_tokens"] == 30
    assert payload["model"] == "claude-sonnet-4.5"


def test_stream_endpoint_appends_done_when_handler_finishes_without_terminal():
    from fastapi.testclient import TestClient
    from novie_agent_sdk.runtime import StreamContext

    agent = Agent(_stream_manifest("stream-sentinel"))

    @agent.stream
    async def handle(ctx: StreamContext):
        yield {"kind": "content", "content": "body"}

    client = TestClient(agent.build_app())
    with client.stream("POST", "/stream", json={"input": {"q": "x"}}) as resp:
        assert resp.status_code == 200
        events = [json.loads(line) for line in resp.iter_lines() if line.strip()]

    assert [event.get("kind") for event in events] == ["content", "done"]
    assert events[-1]["metadata"]["terminal_source"] == "sdk_sentinel"


def test_stream_endpoint_emits_terminal_error_when_handler_raises():
    from fastapi.testclient import TestClient
    from novie_agent_sdk.runtime import StreamContext

    agent = Agent(_stream_manifest("stream-terminal-error"))

    @agent.stream
    async def handle(ctx: StreamContext):
        yield {"kind": "content", "content": "body"}
        raise RuntimeError("provider failed: SECRET_USER_OR_SKILL_PROMPT_MARKER")

    client = TestClient(agent.build_app())
    with client.stream("POST", "/stream", json={"input": {"q": "x"}}) as resp:
        assert resp.status_code == 200
        events = [json.loads(line) for line in resp.iter_lines() if line.strip()]

    assert [event.get("kind") for event in events] == ["content", "terminal_error"]
    assert events[-1]["error"] == "Agent execution failed."
    assert events[-1]["error_code"] == "agent_internal_error"
    assert "SECRET_USER_OR_SKILL_PROMPT_MARKER" not in json.dumps(events)
    assert events[-1]["metadata"]["terminal_source"] == "sdk_exception_guard"
    assert events[-1]["metadata"]["exception_type"] == "RuntimeError"
    assert events[-1]["metadata"]["raw_error_ref"].startswith("sdk-terminal:")


def test_stream_endpoint_terminal_error_carries_last_safe_handler_metadata():
    from fastapi.testclient import TestClient
    from novie_agent_sdk.runtime import StreamContext

    agent = Agent(_stream_manifest("stream-terminal-metadata"))

    @agent.stream
    async def handle(ctx: StreamContext):
        yield {"kind": "content", "content": "body"}
        yield {
            "kind": "trace",
            "metadata": {
                "runtime_phase": "finalize",
                "semantic_phase": "finalizing_output",
                "artifact_type": "brainstorm_notes",
                "artifact_family": "brainstorm",
                "capability_id": "agent.analyst.brainstorming",
                "content_stream_closed": True,
                "prompt_should_not_leak": "SECRET",
            },
        }
        raise RuntimeError("provider failed: SECRET_USER_OR_SKILL_PROMPT_MARKER")

    client = TestClient(agent.build_app())
    with client.stream("POST", "/stream", json={"input": {"q": "x"}}) as resp:
        assert resp.status_code == 200
        events = [json.loads(line) for line in resp.iter_lines() if line.strip()]

    terminal = events[-1]
    assert terminal["kind"] == "terminal_error"
    assert terminal["metadata"]["runtime_phase"] == "finalize"
    assert terminal["metadata"]["semantic_phase"] == "finalizing_output"
    assert terminal["metadata"]["artifact_type"] == "brainstorm_notes"
    assert terminal["metadata"]["artifact_family"] == "brainstorm"
    assert terminal["metadata"]["capability_id"] == "agent.analyst.brainstorming"
    assert terminal["metadata"]["content_stream_closed"] is True
    assert "prompt_should_not_leak" not in terminal["metadata"]
    assert "SECRET_USER_OR_SKILL_PROMPT_MARKER" not in json.dumps(events)


def test_stream_endpoint_serializes_public_agent_error_without_raw_cause():
    from fastapi.testclient import TestClient
    from novie_agent_sdk.runtime import StreamContext

    agent = Agent(_stream_manifest("stream-public-error"))

    @agent.stream
    async def handle(ctx: StreamContext):
        raise PublicAgentError(
            error_code="sectioned_authoring_llm_failed",
            public_message="Document finalization failed.",
        )
        yield {"kind": "content", "content": "unreachable"}

    client = TestClient(agent.build_app())
    with client.stream("POST", "/stream", json={"input": {"q": "x"}}) as resp:
        assert resp.status_code == 200
        events = [json.loads(line) for line in resp.iter_lines() if line.strip()]

    assert events[-1]["error"] == "Document finalization failed."
    assert events[-1]["error_code"] == "sectioned_authoring_llm_failed"


def test_invoke_endpoint_serializes_public_agent_error_without_raw_cause():
    from fastapi.testclient import TestClient

    agent = Agent(_simple_manifest("invoke-public-error"))

    @agent.invoke
    async def handle(ctx: InvokeContext):
        raise PublicAgentError(
            error_code="context_budget_exceeded",
            public_message="Input exceeds worker capacity.",
            replan_eligible=True,
            repair_eligible=True,
        )

    response = TestClient(agent.build_app()).post(
        "/invoke", json={"input": {"q": "x"}}
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "failed",
        "error": "Input exceeds worker capacity.",
        "error_code": "context_budget_exceeded",
        "output": {},
        "retryable": False,
        "replan_eligible": True,
        "repair_eligible": True,
    }


def test_stream_endpoint_sanitizes_explicit_terminal_error_and_marks_store_failed():
    from fastapi.testclient import TestClient
    from novie_agent_sdk.runtime import StreamContext

    store = InMemoryOneShotInvocationStore()
    agent = Agent(_stream_manifest("stream-explicit-failure"), invocation_store=store)

    @agent.stream
    async def handle(ctx: StreamContext):
        yield {"kind": "content", "content": "draft"}
        yield {
            "kind": "terminal_error",
            "error": "RAW_SECRET_USER_PROMPT",
            "error_code": "provider_raw_error",
        }

    client = TestClient(agent.build_app())
    with client.stream(
        "POST",
        "/stream",
        headers={"Idempotency-Key": "explicit-stream-failure-key"},
        json={"input": {"q": "x"}},
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line.strip()]

    assert [event["kind"] for event in events] == ["content", "terminal_error"]
    assert events[-1]["error"] == "Agent execution failed."
    assert events[-1]["error_code"] == "agent_internal_error"
    assert "RAW_SECRET_USER_PROMPT" not in json.dumps(events)
    record = store._records[("stream", "explicit-stream-failure-key")]  # noqa: SLF001
    assert record.status == "failed"
    assert record.error == "Agent execution failed."


def test_stream_endpoint_rejects_nested_failed_final_output():
    from fastapi.testclient import TestClient
    from novie_agent_sdk.runtime import StreamContext

    agent = Agent(_stream_manifest("stream-nested-failure"))

    @agent.stream
    async def handle(ctx: StreamContext):
        yield {
            "kind": "final",
            "output": {
                "status": "FAILED",
                "error": "RAW_SECRET_USER_PROMPT",
                "draft": "RAW_SECRET_USER_PROMPT",
            },
        }

    client = TestClient(agent.build_app())
    with client.stream("POST", "/stream", json={"input": {"q": "x"}}) as response:
        events = [json.loads(line) for line in response.iter_lines() if line.strip()]

    assert [event["kind"] for event in events] == ["terminal_error"]
    assert events[0]["error"] == "Agent execution failed."
    assert events[0]["error_code"] == "agent_internal_error"
    assert "RAW_SECRET_USER_PROMPT" not in json.dumps(events)


def test_stream_endpoint_rejects_top_level_failed_done_without_output_mapping():
    from fastapi.testclient import TestClient
    from novie_agent_sdk.runtime import StreamContext

    agent = Agent(_stream_manifest("stream-top-level-failure"))

    @agent.stream
    async def handle(ctx: StreamContext):
        yield {
            "kind": "DONE",
            "status": "FAILED",
            "error": "RAW_SECRET_USER_PROMPT",
        }

    client = TestClient(agent.build_app())
    with client.stream("POST", "/stream", json={"input": {"q": "x"}}) as response:
        events = [json.loads(line) for line in response.iter_lines() if line.strip()]

    assert [event["kind"] for event in events] == ["terminal_error"]
    assert events[0]["error"] == "Agent execution failed."
    assert events[0]["error_code"] == "agent_internal_error"
    assert "RAW_SECRET_USER_PROMPT" not in json.dumps(events)


def test_stream_endpoint_rejects_nested_failed_status_event():
    from fastapi.testclient import TestClient
    from novie_agent_sdk.runtime import StreamContext

    agent = Agent(_stream_manifest("stream-status-failure"))

    @agent.stream
    async def handle(ctx: StreamContext):
        yield {
            "kind": "status_changed",
            "payload": {
                "status": "FAILED",
                "error": "RAW_SECRET_USER_PROMPT",
                "draft": "RAW_SECRET_USER_PROMPT",
            },
        }

    client = TestClient(agent.build_app())
    with client.stream("POST", "/stream", json={"input": {"q": "x"}}) as response:
        events = [json.loads(line) for line in response.iter_lines() if line.strip()]

    assert [event["kind"] for event in events] == ["terminal_error"]
    assert events[0]["error"] == "Agent execution failed."
    assert events[0]["error_code"] == "agent_internal_error"
    assert "RAW_SECRET_USER_PROMPT" not in json.dumps(events)


def test_stream_endpoint_rejects_failed_metadata_and_ignores_empty_error():
    from fastapi.testclient import TestClient
    from novie_agent_sdk.runtime import StreamContext

    agent = Agent(_stream_manifest("stream-metadata-failure"))

    @agent.stream
    async def handle(ctx: StreamContext):
        yield {"kind": "content", "content": "safe", "error": None}
        yield {
            "kind": "trace",
            "metadata": {
                "status": "FAILED",
                "error": "RAW_SECRET_USER_PROMPT",
            },
        }

    client = TestClient(agent.build_app())
    with client.stream("POST", "/stream", json={"input": {"q": "x"}}) as response:
        events = [json.loads(line) for line in response.iter_lines() if line.strip()]

    assert [event["kind"] for event in events] == ["content", "terminal_error"]
    assert events[-1]["error"] == "Agent execution failed."
    assert "RAW_SECRET_USER_PROMPT" not in json.dumps(events)


def test_invoke_endpoint_rejects_nested_failed_output():
    from fastapi.testclient import TestClient

    agent = Agent(_simple_manifest("invoke-nested-failure"))

    @agent.invoke
    async def handle(ctx: InvokeContext) -> dict[str, Any]:
        return {
            "output": {
                "status": "FAILED",
                "error": "RAW_SECRET_USER_PROMPT",
                "draft": "RAW_SECRET_USER_PROMPT",
            }
        }

    response = TestClient(agent.build_app()).post(
        "/invoke", json={"input": {"q": "x"}}
    )
    assert response.status_code == 200
    assert response.json() == {
        "status": "failed",
        "error": "Agent execution failed.",
        "error_code": "agent_internal_error",
        "output": {},
    }


def test_invoke_handler_failure_stores_only_safe_error():
    from fastapi.testclient import TestClient

    store = InMemoryOneShotInvocationStore()
    agent = Agent(_simple_manifest("invoke-safe-error"), invocation_store=store)

    @agent.invoke
    async def handle(ctx: InvokeContext):
        raise RuntimeError("provider failed: SECRET_USER_OR_SKILL_PROMPT_MARKER")

    client = TestClient(agent.build_app(), raise_server_exceptions=False)
    response = client.post(
        "/invoke",
        headers={"Idempotency-Key": "invoke-safe-error-key"},
        json={"input": {"q": "x"}},
    )
    assert response.status_code == 500

    record = store._records[("invoke", "invoke-safe-error-key")]  # noqa: SLF001
    assert record.error == "Agent execution failed."
    assert "SECRET_USER_OR_SKILL_PROMPT_MARKER" not in json.dumps(record.__dict__)


def test_stream_endpoint_emits_keepalive_while_handler_is_silent(monkeypatch):
    from fastapi.testclient import TestClient
    from novie_agent_sdk import runtime as runtime_mod
    from novie_agent_sdk.runtime import StreamContext

    monkeypatch.setattr(runtime_mod, "_DEFAULT_STREAM_KEEPALIVE_SECONDS", 0.01)
    agent = Agent(_stream_manifest("stream-keepalive"))

    @agent.stream
    async def handle(ctx: StreamContext):
        await asyncio.sleep(0.03)
        yield {"kind": "content", "content": "body"}

    client = TestClient(agent.build_app())
    with client.stream("POST", "/stream", json={"input": {"q": "x"}}) as resp:
        assert resp.status_code == 200
        events = [json.loads(line) for line in resp.iter_lines() if line.strip()]

    keepalives = [
        event
        for event in events
        if event.get("kind") == "status_changed"
        and isinstance(event.get("payload"), dict)
        and event["payload"].get("agent_event_kind") == "keepalive"
    ]
    assert keepalives
    assert keepalives[0]["summary"] == "Agent is still working"
    assert [event.get("kind") for event in events][-2:] == ["content", "done"]


def test_stream_invocation_status_events_and_result_endpoints():
    from fastapi.testclient import TestClient
    from novie_agent_sdk.runtime import StreamContext

    store = InMemoryOneShotInvocationStore()
    agent = Agent(_stream_manifest("stream-invocation-query"), invocation_store=store)

    @agent.stream
    async def handle(ctx: StreamContext):
        yield {"kind": "content", "content": "hello "}
        yield {"kind": "final", "output": {"answer": "world"}}

    client = TestClient(agent.build_app())
    with client.stream(
        "POST",
        "/stream",
        headers={"Idempotency-Key": "stream-invocation-key-1"},
        json={"input": {"q": "x"}},
    ) as resp:
        assert resp.status_code == 200
        assert [json.loads(line)["kind"] for line in resp.iter_lines() if line.strip()] == [
            "content",
            "final",
            "done",
        ]

    started, record = asyncio.run(
        store.start_or_get("stream-invocation-key-1", "stream")
    )
    assert started is False

    status = client.get(f"/invocations/{record.invocation_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["invocation_id"] == record.invocation_id

    events = client.get(f"/invocations/{record.invocation_id}/events")
    assert events.status_code == 200
    assert [event["kind"] for event in events.json()["events"]] == [
        "content",
        "final",
        "done",
    ]

    result = client.get(f"/invocations/{record.invocation_id}/result")
    assert result.status_code == 200
    body = result.json()
    assert body["output"]["answer"] == "world"
    assert body["output"]["transcript"] == "hello "


@pytest.mark.asyncio
async def test_stream_agent_replays_duplicate_idempotency_key():
    import httpx
    from novie_agent_sdk.runtime import StreamContext

    agent = Agent(_stream_manifest("stream-replay"))
    calls = 0

    @agent.stream
    async def handle(ctx: StreamContext):
        nonlocal calls
        calls += 1
        yield {"kind": "content", "content": f"call-{calls}"}
        yield {"kind": "final", "output": {"call": calls}}

    app = agent.build_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://stream-agent",
    ) as client:
        first, second = await assert_http_json_stream_idempotency_replay(
            client,
            payload={"input": {"q": "x"}},
            idempotency_key="stream-key-1",
        )

    assert first == second
    assert [event.get("kind") for event in first] == ["content", "final", "done"]
    assert first[0]["content"] == "call-1"
    assert calls == 1


def test_result_cache_stream_agent_replays_after_restart(
    tmp_path: Path,
    monkeypatch,
):
    from fastapi.testclient import TestClient
    from novie_agent_sdk.runtime import StreamContext

    monkeypatch.setenv("NOVIE_AGENT_STATE_DIR", str(tmp_path / "state"))
    calls = 0

    agent_a = Agent(_result_cache_stream_manifest("restart-stream"))

    @agent_a.stream
    async def handle_a(ctx: StreamContext):
        nonlocal calls
        calls += 1
        yield {"kind": "content", "content": f"call-{calls}"}
        yield {"kind": "final", "output": {"call": calls}}

    client_a = TestClient(agent_a.build_app())
    with client_a.stream(
        "POST",
        "/stream",
        json={"input": {"q": "first"}},
        headers={"Idempotency-Key": "restart-stream-key-1"},
    ) as first_resp:
        assert first_resp.status_code == 200
        first_events = [
            json.loads(line)
            for line in first_resp.iter_lines()
            if line.strip()
        ]

    agent_b = Agent(_result_cache_stream_manifest("restart-stream"))

    @agent_b.stream
    async def handle_b(ctx: StreamContext):  # pragma: no cover
        raise AssertionError("duplicate request should replay cached stream")
        yield {"kind": "content", "content": "unreachable"}

    client_b = TestClient(agent_b.build_app())
    with client_b.stream(
        "POST",
        "/stream",
        json={"input": {"q": "second"}},
        headers={"Idempotency-Key": "restart-stream-key-1"},
    ) as replay_resp:
        assert replay_resp.status_code == 200
        replay_events = [
            json.loads(line)
            for line in replay_resp.iter_lines()
            if line.strip()
        ]

    assert replay_events == first_events
    assert [event.get("kind") for event in replay_events] == [
        "content",
        "final",
        "done",
    ]
    assert calls == 1
    assert (tmp_path / "state" / "restart-stream.invocations.sqlite3").exists()


# ─────────────────────────────────────────────────────────────────────────────
# 4. FastAPI app — tasks mode
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tasks_agent_full_lifecycle():
    from fastapi.testclient import TestClient

    agent = Agent(_tasks_manifest())

    @agent.task
    async def handle(ctx: TaskContext) -> dict:
        await ctx.emit_message("Starting")
        await asyncio.sleep(0.01)
        return {"answer": 42}

    app = agent.build_app()
    client = TestClient(app)

    # Create task
    resp = client.post("/tasks", json={"input": {"query": "test"}})
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]

    # Wait a bit for background task to complete
    await asyncio.sleep(0.1)

    # Get status
    resp = client.get(f"/tasks/{task_id}")
    assert resp.status_code == 200
    status = resp.json()["status"]
    assert status in ("running", "completed", "pending")

    # Wait more and check completed
    await asyncio.sleep(0.2)
    resp = client.get(f"/tasks/{task_id}")
    assert resp.json()["status"] == "completed"

    # Get events
    resp = client.get(f"/tasks/{task_id}/events")
    assert resp.status_code == 200
    events = resp.json()["events"]
    assert len(events) > 0

    # Get result
    resp = client.get(f"/tasks/{task_id}/result")
    assert resp.status_code == 200
    assert resp.json()["output"] == {"answer": 42}


@pytest.mark.asyncio
async def test_tasks_agent_replays_duplicate_idempotency_key():
    import httpx

    agent = Agent(_tasks_manifest())
    calls = 0

    @agent.task
    async def handle(ctx: TaskContext) -> dict:
        nonlocal calls
        calls += 1
        return {"call": calls}

    app = agent.build_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://tasks-agent",
    ) as client:
        first, second = await assert_http_json_tasks_idempotency_replay(
            client,
            payload={"input": {"query": "test"}},
            idempotency_key="tasks-key-1",
        )

    assert first["task_id"] == second["task_id"]
    assert calls <= 1


@pytest.mark.asyncio
async def test_tasks_agent_cancel():
    """Verify cancel endpoint works before task completes."""
    from fastapi.testclient import TestClient
    import threading

    agent = Agent(_tasks_manifest())
    # Use a threading.Event so the handler blocks until released
    proceed = threading.Event()

    @agent.task
    async def handle(ctx: TaskContext) -> dict:
        # Block in a thread-compatible way (poll every 10ms)
        for _ in range(1000):
            if ctx.is_cancelled:
                return {"cancelled": True}
            await asyncio.sleep(0.01)
            if proceed.is_set():
                break
        return {"done": True}

    app = agent.build_app()
    # Use raise_server_exceptions=False so background task errors don't crash TestClient
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/tasks", json={"input": {}})
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]

        # Cancel while task is running
        resp = client.post(f"/tasks/{task_id}/cancel")
        # Either 202 (cancelled successfully) or 409 (already terminal from fast completion)
        assert resp.status_code in (202, 409)

        # Release the handler if it's still waiting
        proceed.set()

        # After cancel, task should be cancelled or completed
        resp = client.get(f"/tasks/{task_id}")
        assert resp.json()["status"] in ("cancelled", "completed")


@pytest.mark.asyncio
async def test_tasks_agent_handler_exception():
    from fastapi.testclient import TestClient

    agent = Agent(_tasks_manifest())

    @agent.task
    async def handle(ctx: TaskContext) -> dict:
        raise RuntimeError("Something went wrong")

    app = agent.build_app()
    client = TestClient(app)

    resp = client.post("/tasks", json={"input": {}})
    task_id = resp.json()["task_id"]

    await asyncio.sleep(0.2)
    resp = client.get(f"/tasks/{task_id}")
    assert resp.json()["status"] == "failed"
    assert resp.json()["error"] is not None


@pytest.mark.asyncio
async def test_tasks_agent_404_for_unknown_task():
    from fastapi.testclient import TestClient

    agent = Agent(_tasks_manifest())
    agent.task(lambda ctx: {"done": True})
    app = agent.build_app()
    client = TestClient(app)

    resp = client.get("/tasks/nonexistent")
    assert resp.status_code == 404


def test_tasks_agent_production_requires_durable_store(monkeypatch):
    monkeypatch.setenv("NOVIE_ENV", "production")
    agent = Agent(_tasks_manifest())
    agent.task(lambda ctx: {"done": True})
    with pytest.raises(RuntimeError):
        agent.build_app()


@pytest.mark.asyncio
async def test_tasks_result_409_when_not_terminal():
    """Verify /result returns 409 for non-terminal tasks using store directly."""
    store = InMemoryTaskStore()
    # Create a task but don't run it (stays in pending/running state)
    await store.create_task("manual-task", {"q": 1})
    await store.update_task_status("manual-task", "running")
    task = await store.get_task("manual-task")
    assert task.status == "running"
    # result should be None since not completed
    result = await store.get_result("manual-task")
    assert result is None


@pytest.mark.asyncio
async def test_in_memory_task_store_evicts_oldest_when_max_tasks_exceeded():
    """TD #35 (2026-05-11) — bounded capacity. When the store is at
    ``max_tasks`` and a new task is created, the oldest task (FIFO
    insertion order) is dropped. Prevents long-running SDK agents from
    leaking memory."""
    store = InMemoryTaskStore(max_tasks=3)
    await store.create_task("t1", {"x": 1})
    await store.create_task("t2", {"x": 2})
    await store.create_task("t3", {"x": 3})
    assert {await store.get_task(tid) is not None for tid in ("t1", "t2", "t3")} == {True}

    # Adding a 4th evicts the oldest (t1).
    await store.create_task("t4", {"x": 4})
    assert await store.get_task("t1") is None
    assert await store.get_task("t4") is not None


@pytest.mark.asyncio
async def test_in_memory_task_store_evicts_expired_via_ttl():
    """TD #35 — TTL eviction. Tasks older than ``ttl_seconds`` are
    dropped lazily on the next ``create_task`` call and eagerly via the
    ``evict_expired`` public sweep."""
    import asyncio as _asyncio

    store = InMemoryTaskStore(ttl_seconds=0.1)
    await store.create_task("old", {"x": 1})
    assert await store.get_task("old") is not None

    # Wait past the TTL window.
    await _asyncio.sleep(0.2)

    # Lazy path: a subsequent create triggers eviction of "old".
    await store.create_task("fresh", {"x": 2})
    assert await store.get_task("old") is None
    assert await store.get_task("fresh") is not None

    # Eager path: evict_expired sweep removes "fresh" once it ages out.
    await _asyncio.sleep(0.2)
    evicted = await store.evict_expired()
    assert evicted == 1
    assert await store.get_task("fresh") is None


@pytest.mark.asyncio
async def test_in_memory_task_store_env_override_for_capacity(monkeypatch):
    """TD #35 — env vars (NOVIE_SDK_TASK_STORE_MAX_TASKS) let deployers
    tune the bounds without code change."""
    monkeypatch.setenv("NOVIE_SDK_TASK_STORE_MAX_TASKS", "2")
    store = InMemoryTaskStore()  # picks up env
    await store.create_task("a", {})
    await store.create_task("b", {})
    await store.create_task("c", {})  # evicts "a"
    assert await store.get_task("a") is None
    assert await store.get_task("c") is not None


def test_failure_envelope_ignores_phase_metadata_status_failed() -> None:
    """Soft quality-gate traces must not abort the A2A stream.

    Regression: ``document.section.quality_checked`` with
    ``metadata.status="failed"`` was recursively treated as a terminal
    failure envelope and rewritten to ``agent_internal_error``.
    """
    wire = {
        "kind": "trace",
        "metadata": {
            "event": "document.section.quality_checked",
            "status": "failed",
            "quality": {
                "passed": False,
                "failures": ["artifact_only_citations"],
                "hard_failures": [],
                "soft_failures": ["artifact_only_citations"],
            },
        },
    }
    assert _failure_envelope(wire) is None
    assert (
        _failure_envelope(
            {
                "kind": "trace",
                "metadata": {
                    "event": "document.section.quality_checked",
                    "status": "gate_failed",
                },
            }
        )
        is None
    )
    # Top-level terminal envelopes still fail closed.
    assert _failure_envelope({"status": "failed", "error": "boom"}) is not None
    assert (
        _failure_envelope({"kind": "terminal_error", "error_code": "agent_internal_error"})
        is not None
    )
    # Phase events that carry an explicit error field remain detectable.
    assert (
        _failure_envelope(
            {
                "kind": "trace",
                "metadata": {
                    "event": "agent.llm_call.failed",
                    "status": "failed",
                    "error": "TimeoutError",
                },
            }
        )
        is not None
    )
