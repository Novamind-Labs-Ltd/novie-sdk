from __future__ import annotations

import asyncio
from types import SimpleNamespace

from novie_agent_sdk import (
    agent_run_id,
    langchain_runnable_config,
    notify_usage_callbacks,
    runnable_run_id,
)


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="req-1",
        session_id="session-1",
        thread_id="thread-1",
        workflow_id="workflow-1",
        parent_step_id="s1",
    )


def test_agent_run_id_is_stable_for_same_segment() -> None:
    first = agent_run_id(
        agent_id="analyst",
        ctx=_ctx(),
        runtime_phase="draft",
        capability_id="agent.analyst.report_synthesis",
        stage="outline",
    )
    second = agent_run_id(
        agent_id="analyst",
        ctx=_ctx(),
        runtime_phase="draft",
        capability_id="agent.analyst.report_synthesis",
        stage="outline",
    )

    assert first == second


def test_langchain_runnable_config_carries_metadata_and_run_id() -> None:
    config = langchain_runnable_config(
        agent_id="analyst",
        ctx=_ctx(),
        callbacks=[],
        runtime_phase="draft",
        capability_id="agent.analyst.report_synthesis",
        mode="research",
        phase="synthesis",
        stage="section_1",
        metadata={"custom": "value"},
    )

    assert runnable_run_id(config) == config["run_id"]
    assert config["metadata"]["agent_id"] == "analyst"
    assert config["metadata"]["runtime_phase"] == "draft"
    assert config["metadata"]["custom"] == "value"


def test_notify_usage_callbacks_calls_on_llm_end_with_run_id() -> None:
    seen = []

    class _Callback:
        async def on_llm_end(self, response, *, run_id):
            seen.append((run_id, response.llm_output["token_usage"]))

    config = langchain_runnable_config(
        agent_id="analyst",
        ctx=_ctx(),
        callbacks=[_Callback()],
        runtime_phase="draft",
    )

    asyncio.run(notify_usage_callbacks(config, {"input_tokens": 10}))

    assert seen == [(config["run_id"], {"input_tokens": 10})]
