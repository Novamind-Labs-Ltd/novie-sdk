from __future__ import annotations

from types import SimpleNamespace

from novie_agent_sdk import (
    RequestHeaders,
    context_block_from_sdk_request,
    execution_context_from_block,
    execution_context_from_runtime_block,
    execution_context_from_sdk_request,
    legacy_request_from_sdk_context,
    resolve_capability_id,
    resolve_runtime_capability_id,
)


def _ctx(input_: dict) -> SimpleNamespace:
    return SimpleNamespace(
        input=input_,
        headers=RequestHeaders(
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            project_id="project-1",
            user_id="user-1",
            session_id="session-1",
            request_id="request-1",
            step_id="step-1",
            raw={
                "x-novie-thread-id": "thread-1",
                "x-novie-workflow-id": "workflow-1",
                "x-novie-project-id": "project-1",
            },
        ),
    )


def test_resolve_capability_id_prefers_direct_value() -> None:
    assert (
        resolve_capability_id(
            {
                "capability_id": "direct",
                "capability_grants": [{"capability_id": "grant"}],
            }
        )
        == "direct"
    )


def test_resolve_capability_id_falls_back_to_grants() -> None:
    assert resolve_capability_id({"capability_grants": [{"capability_id": "grant"}]}) == "grant"


def test_context_block_from_sdk_request_projects_headers() -> None:
    block = context_block_from_sdk_request(_ctx({}), agent_id="demo")

    assert block["request_id"] == "request-1"
    assert block["session_id"] == "session-1"
    assert block["thread_id"] == "thread-1"
    assert block["workflow_id"] == "workflow-1"
    assert block["parent_step_id"] == "step-1"
    assert block["tenant"]["tenant_id"] == "tenant-1"
    assert block["tenant"]["workspace_id"] == "workspace-1"
    assert block["tenant"]["project_id"] == "project-1"
    assert block["identity"]["principal_id"] == "user-1"


def test_execution_context_from_sdk_request_preserves_project_id() -> None:
    exec_ctx = execution_context_from_sdk_request(_ctx({}), agent_id="demo")

    assert exec_ctx.request_id == "request-1"
    assert exec_ctx.tenant.tenant_id == "tenant-1"
    assert exec_ctx.tenant.workspace_id == "workspace-1"
    assert exec_ctx.tenant.project_id == "project-1"


def test_top_level_execution_context_from_block_keeps_document_signature() -> None:
    exec_ctx = execution_context_from_block({})

    assert exec_ctx.request_id == "req-agent-local"
    assert exec_ctx.session_id == "sess-agent-local"


def test_runtime_bridge_block_converter_has_explicit_name() -> None:
    exec_ctx = execution_context_from_runtime_block({}, agent_id="demo")

    assert exec_ctx.request_id == "req-demo-local"
    assert exec_ctx.session_id == "sess-demo-local"


def test_runtime_capability_resolver_has_explicit_name() -> None:
    assert (
        resolve_runtime_capability_id(
            {"capability_grants": [{"capability_id": "runtime-grant"}]}
        )
        == "runtime-grant"
    )


def test_legacy_request_from_sdk_context_shapes_runtime_inputs() -> None:
    ctx = _ctx({"capability_id": "cap-1", "question": "Ship it?"})

    request = legacy_request_from_sdk_context(ctx, agent_id="demo")

    assert request.capability_id == "cap-1"
    assert request.inputs["question"] == "Ship it?"
    assert request.execution_context.thread_id == "thread-1"
    assert request.llm_context.headers is ctx.headers
    assert hasattr(request.llm_context, "platform_ns")
