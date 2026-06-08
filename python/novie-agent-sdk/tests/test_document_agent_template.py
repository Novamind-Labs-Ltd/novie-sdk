from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk

from novie_agent_sdk import (
    DocumentAgentTemplate,
    DocumentCapabilitySpec,
    external_agent_checkpoint_service,
    put_external_agent_checkpoint,
    compile_skill_scope,
    resolve_document_agent_input,
)


class _FakeGraph:
    async def astream(self, _inputs: dict[str, Any], stream_mode: Any = "values", **_kwargs: Any):
        assert stream_mode == ["messages", "values"]
        yield "messages", (AIMessageChunk(content="Draft"), {"node": "agent"})
        yield "values", {"messages": [AIMessage(content="Final narrative.")]}


def _metadata(
    *,
    runtime_phase: str,
    semantic_phase: str | None,
    mode: str,
    phase: str,
    capability_id: str | None,
) -> dict[str, Any]:
    return {
        "runtime_phase": runtime_phase,
        "semantic_phase": semantic_phase,
        "mode": mode,
        "phase": phase,
        "capability_id": capability_id,
    }


def _last_message_text(value: Any) -> str:
    messages = value.get("messages") if isinstance(value, dict) else None
    if not messages:
        return ""
    return str(getattr(messages[-1], "content", "") or "")


def test_compile_skill_scope_loads_prompt_hint(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "demo" / "shared"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: demo-shared
description: Shared demo guidance.
allowed-tools: search_project_wiki, fetch_artifact
---

# Demo Skill

## Output Expectations

Write a concise document.
""",
        encoding="utf-8",
    )
    spec = DocumentCapabilitySpec(
        capability_id="agent.demo.write",
        skill_sources=["/skills/demo/shared/"],
        mode="write",
        phase="default",
        artifact_type="demo_document",
        artifact_family="demo",
        package_root=tmp_path,
    )

    scope = compile_skill_scope(spec)

    assert scope.skill_sources == ["/skills/demo/shared/"]
    assert scope.allowed_tools == ("fetch_artifact", "search_project_wiki")
    assert "demo-shared: Shared demo guidance." in scope.prompt_hint
    assert "Write a concise document." in scope.prompt_hint


def test_resolve_document_agent_input_hides_upstream_when_access_is_none(tmp_path: Path) -> None:
    spec = DocumentCapabilitySpec(
        capability_id="agent.demo.write",
        skill_sources=[],
        mode="write",
        phase="default",
        artifact_type="demo_document",
        artifact_family="demo",
        package_root=tmp_path,
        artifact_access="none",
    )

    resolved = resolve_document_agent_input(
        spec,
        brief={"title": "Demo"},
        upstream={"s1": {"summary": "Hidden"}},
    )

    assert resolved.upstream == {}
    assert resolved.uses_upstream_summary is False


def test_document_agent_template_streams_graph_content_and_result() -> None:
    template = DocumentAgentTemplate(
        owner_agent_id="demo",
        phase_metadata=_metadata,
        keepalive_env_var="NOVIE_TEST_KEEPALIVE_INTERVAL_S",
        context_budget_source="test",
    )

    async def _collect():
        events = []
        async for event in template.stream_graph_run(
            graph=_FakeGraph(),
            prompt="Write",
            callbacks=[],
            runtime_phase="draft",
            semantic_phase="drafting",
            mode="write",
            phase="default",
            capability_id="agent.demo.write",
            content_metadata={"capability_id": "agent.demo.write"},
            extract_values_text=_last_message_text,
        ):
            events.append(event)
        return events

    events = asyncio.run(_collect())

    assert events[0].kind == "content"
    assert events[0].content == "Draft"
    assert events[-1].narrative == "Draft\nFinal narrative."


def test_external_checkpoint_helpers_support_new_service_shape() -> None:
    calls: list[dict[str, Any]] = []

    class _Service:
        async def put(self, ctx: Any, **kwargs: Any) -> Any:
            calls.append({"ctx": ctx, **kwargs})
            return type("Record", (), {"checkpoint_id": "ckpt-1"})()

    ctx = type(
        "Ctx",
        (),
        {
            "thread_id": "thread-1",
        },
    )()
    services = type("Services", (), {"external_agent_checkpoints": _Service()})()

    service = external_agent_checkpoint_service(services)
    record = asyncio.run(
        put_external_agent_checkpoint(
            service,
            ctx,
            owner_agent_id="demo",
            payload={"current_phase": "finalize"},
            workflow_id="workflow-1",
            step_id="step-1",
            summary="draft complete",
            metadata={"capability_id": "agent.demo.write"},
        )
    )

    assert record.checkpoint_id == "ckpt-1"
    assert calls[0]["owner_agent_id"] == "demo"
    assert calls[0]["thread_id"] == "thread-1"
    assert calls[0]["payload"]["current_phase"] == "finalize"
