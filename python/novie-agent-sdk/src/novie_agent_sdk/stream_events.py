from __future__ import annotations

from typing import Any

from novie_protocol.agents import AgentStreamEvent


def progress_event(
    label: str,
    *,
    metadata: dict[str, Any] | None = None,
    runtime_phase: str = "working",
    capability_id: str | None = None,
    phase: str | None = None,
    mode: str | None = None,
) -> AgentStreamEvent:
    """Create a platform-readable progress trace event."""
    event_metadata = dict(metadata or {})
    event_metadata["runtime_phase"] = runtime_phase
    event_metadata["progress_label"] = str(label or runtime_phase)
    if capability_id:
        event_metadata["capability_id"] = capability_id
    if phase:
        event_metadata["phase"] = phase
    if mode:
        event_metadata["mode"] = mode
    return AgentStreamEvent(kind="trace", metadata=event_metadata)


def tool_call_event(
    tool_name: str,
    *,
    tool_call_id: str | None = None,
    args: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentStreamEvent:
    event_metadata = dict(metadata or {})
    event_metadata.update(
        {
            "event": "tool_call",
            "tool_name": str(tool_name or ""),
            "tool_args": dict(args or {}),
        }
    )
    if tool_call_id:
        event_metadata["tool_call_id"] = tool_call_id
    return AgentStreamEvent(kind="trace", metadata=event_metadata)


def tool_result_event(
    tool_name: str,
    *,
    result: str = "",
    tool_call_id: str | None = None,
    ok: bool = True,
    metadata: dict[str, Any] | None = None,
    visibility: str = "internal",
) -> AgentStreamEvent:
    event_metadata = dict(metadata or {})
    normalized_visibility = str(visibility or "internal").strip().lower() or "internal"
    event_metadata.update(
        {
            "event": "tool_result",
            "tool_name": str(tool_name or ""),
            "tool_ok": bool(ok),
            "visibility": normalized_visibility,
            "tool_result_visibility": normalized_visibility,
            "tool_result_chars": len(str(result or "")),
        }
    )
    if tool_call_id:
        event_metadata["tool_call_id"] = tool_call_id
    return AgentStreamEvent(
        kind="trace",
        content=str(result or ""),
        metadata=event_metadata,
    )


def content_delta_event(
    content: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentStreamEvent:
    return AgentStreamEvent(
        kind="content",
        content=str(content or ""),
        metadata=dict(metadata or {}),
    )
