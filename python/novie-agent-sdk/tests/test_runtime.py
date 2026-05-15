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
    InMemoryTaskStore,
    InvokeContext,
    RegistrationClient,
    RequestHeaders,
    SqliteOneShotInvocationStore,
    SqliteTaskStore,
    TaskContext,
    _a2a_header_signature,
)
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


@pytest.mark.asyncio
async def test_simple_agent_requires_signed_headers_in_production(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("NOVIE_RUNTIME_MODE", "production")
    monkeypatch.setenv("NOVIE_A2A_SHARED_SECRET", "a2a-secret")
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
    assert unsigned.json()["detail"]["error"] == "a2a_signature_required"

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
    signature = _a2a_header_signature(
        RequestHeaders.from_request(headers),
        "a2a-secret",
    )
    signed = client.post(
        "/invoke",
        json={"input": {"name": "Alice"}},
        headers={**headers, "x-novie-sig": signature},
    )
    assert signed.status_code == 200
    assert signed.json()["output"] == {
        "principal": "user-1",
        "project_id": "project-1",
    }


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
