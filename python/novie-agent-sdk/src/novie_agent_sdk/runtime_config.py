from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5


def agent_run_id(
    *,
    agent_id: str,
    ctx: Any,
    runtime_phase: str,
    capability_id: str | None = None,
    mode: str | None = None,
    phase: str | None = None,
    stage: str | None = None,
) -> UUID:
    parts = [
        str(agent_id or "agent"),
        str(getattr(ctx, "request_id", "") or ""),
        str(getattr(ctx, "session_id", "") or ""),
        str(getattr(ctx, "thread_id", "") or ""),
        str(getattr(ctx, "workflow_id", "") or ""),
        str(getattr(ctx, "parent_step_id", "") or ""),
        capability_id or "",
        mode or "",
        phase or "",
        runtime_phase,
        stage or "",
    ]
    return uuid5(NAMESPACE_URL, "|".join(parts))


def langchain_runnable_config(
    *,
    agent_id: str,
    ctx: Any,
    callbacks: list[Any] | tuple[Any, ...] | None,
    runtime_phase: str,
    capability_id: str | None = None,
    mode: str | None = None,
    phase: str | None = None,
    stage: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = agent_run_id(
        agent_id=agent_id,
        ctx=ctx,
        runtime_phase=runtime_phase,
        capability_id=capability_id,
        mode=mode,
        phase=phase,
        stage=stage,
    )
    runnable_metadata = {
        "agent_id": agent_id,
        "request_id": getattr(ctx, "request_id", None),
        "session_id": getattr(ctx, "session_id", None),
        "thread_id": getattr(ctx, "thread_id", None),
        "workflow_id": getattr(ctx, "workflow_id", None),
        "parent_step_id": getattr(ctx, "parent_step_id", None),
        "capability_id": capability_id,
        "mode": mode,
        "phase": phase,
        "runtime_phase": runtime_phase,
        "stage": stage,
        **(metadata or {}),
    }
    return {
        "callbacks": list(callbacks or []),
        "run_id": run_id,
        "metadata": {
            key: value for key, value in runnable_metadata.items() if value is not None
        },
    }


def runnable_run_id(config: Any) -> UUID:
    if isinstance(config, dict):
        raw = config.get("run_id")
        if raw is None:
            metadata = config.get("metadata")
            if isinstance(metadata, dict):
                raw = metadata.get("run_id") or metadata.get("novie_run_id")
        if raw is None:
            configurable = config.get("configurable")
            if isinstance(configurable, dict):
                raw = configurable.get("run_id")
        if isinstance(raw, UUID):
            return raw
        if isinstance(raw, str):
            try:
                return UUID(raw)
            except ValueError:
                return uuid5(NAMESPACE_URL, raw)
    return uuid4()


async def notify_usage_callbacks(
    config: Any,
    usage_metadata: dict[str, Any],
) -> None:
    callbacks = []
    if isinstance(config, dict):
        raw = config.get("callbacks") or []
        callbacks = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    if not callbacks:
        return
    run_id = runnable_run_id(config)
    response = SimpleNamespace(
        llm_output={"token_usage": dict(usage_metadata or {})},
        generations=[],
    )
    for callback in callbacks:
        on_llm_end = getattr(callback, "on_llm_end", None)
        if callable(on_llm_end):
            result = on_llm_end(response, run_id=run_id)
            if hasattr(result, "__await__"):
                await result


__all__ = [
    "agent_run_id",
    "langchain_runnable_config",
    "notify_usage_callbacks",
    "runnable_run_id",
]
