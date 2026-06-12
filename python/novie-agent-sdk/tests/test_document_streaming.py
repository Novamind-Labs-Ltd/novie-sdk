from __future__ import annotations

import asyncio

import pytest
from novie_protocol.agents import AgentStreamEvent

from novie_agent_sdk.document_streaming import (
    SubtaskEventMapper,
    SubtaskIdleTimeoutError,
    with_subtask_keepalive,
)


@pytest.mark.asyncio
async def test_subtask_keepalive_raises_after_active_subtask_idle_timeout() -> None:
    mapper = SubtaskEventMapper(base_metadata={"analysis_mode": "research"})

    async def source():
        yield AgentStreamEvent(
            kind="tool_call",
            tool_name="task",
            tool_call_id="task-timeout",
            tool_args={
                "subagent_type": "researcher",
                "description": "Research a bounded question.",
            },
        )
        await asyncio.sleep(0.1)

    events: list[AgentStreamEvent] = []
    with pytest.raises(SubtaskIdleTimeoutError) as exc_info:
        async for item in with_subtask_keepalive(
            source(),
            subtask_events=mapper,
            phase_metadata=lambda **kwargs: kwargs,
            runtime_phase="collect_evidence",
            mode="research",
            phase="market_map",
            capability_id="agent.analyst.market_research",
            interval_seconds=0.01,
            subtask_idle_timeout_seconds=0.02,
        ):
            if isinstance(item, AgentStreamEvent):
                if item.kind in {"tool_call", "tool_result"}:
                    events.extend(mapper.map_tool_event(item))
                else:
                    events.append(item)

    timeout_events = [
        event
        for event in events
        if event.metadata.get("event") == "subtask.idle_timeout"
    ]
    assert timeout_events
    assert timeout_events[0].metadata["subtask_id"] == "task-timeout"
    assert exc_info.value.subtask["subtask_id"] == "task-timeout"
