from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from novie_agent_sdk import (
    DocumentAgentTemplate,
    DocumentCapabilitySpec,
    build_document_deliverable_event,
    external_agent_checkpoint_service,
    put_external_agent_checkpoint,
    compile_skill_scope,
    resolve_document_agent_input,
    resolve_document_runtime_profile,
)


class _FakeGraph:
    async def astream(self, _inputs: dict[str, Any], stream_mode: Any = "values", **_kwargs: Any):
        assert stream_mode == ["messages", "values"]
        yield "values", {
            "messages": [
                HumanMessage(content="# Capability navigation\nSECRET SKILL PROMPT")
            ],
        }
        yield "values", {
            "messages": [SimpleNamespace(type="developer", content="DEVELOPER PROMPT")],
        }
        yield "values", {"messages": [SimpleNamespace(role="user", content="USER PROMPT")]}
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


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (HumanMessage(content="input"), False),
        (SystemMessage(content="system"), False),
        (SimpleNamespace(type="developer", content="developer"), False),
        (SimpleNamespace(role="user", content="user"), False),
        (AIMessage(content="assistant"), True),
        (AIMessageChunk(content="assistant"), True),
    ],
)
def test_is_assistant_message(message: Any, expected: bool) -> None:
    from novie_agent_sdk.document_agent_template import _is_assistant_message

    assert _is_assistant_message(message) is expected


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


def test_resolve_document_runtime_profile_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="demo capability_id is required"):
        resolve_document_runtime_profile(
            agent_name="demo",
            inputs={},
            resolve_capability=lambda _capability_id: None,
        )

    with pytest.raises(RuntimeError, match="unknown demo capability_id"):
        resolve_document_runtime_profile(
            agent_name="demo",
            capability_id="agent.demo.unknown",
            resolve_capability=lambda _capability_id: None,
        )


def test_resolve_document_runtime_profile_loads_required_skill_contract(
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "skills" / "demo" / "write"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: demo-write
metadata:
  novie:
    runtime_contract:
      version: 1
      runtime:
        strategy: sectioned_longform
---

# Demo Write
""",
        encoding="utf-8",
    )
    spec = DocumentCapabilitySpec(
        capability_id="agent.demo.write",
        skill_sources=["/skills/demo/write/"],
        mode="write",
        phase="default",
        artifact_type="demo_document",
        artifact_family="demo",
        package_root=tmp_path,
    )

    profile = resolve_document_runtime_profile(
        agent_name="demo",
        capability_id="agent.demo.write",
        resolve_capability=lambda _capability_id: spec,
        require_skill_contract=True,
    )

    assert profile.capability_id == "agent.demo.write"
    assert profile.mode == "write"
    assert profile.artifact_family == "demo"
    assert profile.skill_contract is not None
    assert profile.skill_contract.strategy == "sectioned_longform"


class _Recovery(BaseModel):
    fallback_used: bool = False
    fallback_reason: str = ""
    resumed_from_checkpoint: bool = False
    checkpoint_id: str = ""
    finalize_attempts: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class _FinalPayload(BaseModel):
    plan_id: str
    final_markdown: str
    structured_output: dict[str, Any] = Field(default_factory=dict)
    degraded_flags: list[str] = Field(default_factory=list)
    recovery: _Recovery = Field(default_factory=_Recovery)
    metadata: dict[str, Any] = Field(default_factory=dict)


class _Structured(BaseModel):
    summary: str


def test_build_document_deliverable_event_builds_common_envelope() -> None:
    event = build_document_deliverable_event(
        card=None,
        structured=_Structured(summary="Demo"),
        artifact_type="demo_document",
        artifact_family="demo",
        capability_id="agent.demo.write",
        analysis="# Demo",
        narrative="Draft narrative",
        final_payload_type=_FinalPayload,
        recovery_type=_Recovery,
        mode_key="demo_mode",
        mode="write",
        phase_key="demo_phase",
        phase="default",
        finalize_strategy="sectioned_longform",
        finalize_attempts=2,
        degraded_flags=["tool.degraded"],
        checkpoint_id="ckpt-1",
        quality={"quality_status": "skipped"},
        authoring_ledger={"sections": 3},
        skill_contract={"strategy": "sectioned_longform"},
    )

    assert event.kind == "final"
    assert event.output["kind"] == "document_deliverable"
    assert event.output["analysis"] == "# Demo"
    assert event.output["final_markdown"] == "# Demo"
    assert event.output["demo_mode"] == "write"
    payload = event.output["final_payload"]
    assert payload["plan_id"] == "agent.demo.write"
    assert payload["recovery"]["checkpoint_id"] == "ckpt-1"
    assert payload["recovery"]["finalize_attempts"] == 2
    assert payload["recovery"]["metadata"]["authoring_ledger"] == {"sections": 3}
    assert event.metadata["skill_contract"] == {"strategy": "sectioned_longform"}


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
    assert "# Capability navigation" not in events[-1].narrative
    assert "SECRET SKILL PROMPT" not in events[-1].narrative
    assert "DEVELOPER PROMPT" not in events[-1].narrative
    assert "USER PROMPT" not in events[-1].narrative


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
