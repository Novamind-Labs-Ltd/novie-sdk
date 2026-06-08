from __future__ import annotations

from typing import Any

from novie_protocol.agents import AgentStreamEvent


def workpad_entry_event(
    *,
    kind: str,
    title: str,
    content: str,
    base_metadata: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    content_type: str = "text/markdown",
    artifact_type: str | None = None,
) -> AgentStreamEvent:
    """Build a standard Execution Workpad checkpoint stream event."""
    entry_kind = str(kind or "").strip()
    return AgentStreamEvent(
        kind="trace",
        metadata={
            **dict(base_metadata or {}),
            "event": "execution_workpad_entry",
            "workpad_entry": {
                "kind": entry_kind,
                "title": str(title or entry_kind).strip(),
                "content": str(content or ""),
                "content_type": content_type or "text/markdown",
                "artifact_type": artifact_type or f"execution_workpad.{entry_kind}",
                "metadata": dict(metadata or {}),
            },
        },
    )


def workpad_checkpoint_event(
    *,
    kind: str,
    title: str,
    content: str,
    base_metadata: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    content_type: str = "text/markdown",
    artifact_type: str | None = None,
) -> AgentStreamEvent:
    """Alias with checkpoint wording for long-running agent workpads."""
    return workpad_entry_event(
        kind=kind,
        title=title,
        content=content,
        base_metadata=base_metadata,
        metadata=metadata,
        content_type=content_type,
        artifact_type=artifact_type,
    )


def execution_workpad_context(inputs: dict[str, Any] | None) -> dict[str, Any]:
    """Return the compact Execution Workpad injected by the platform.

    The platform may provide the same payload either directly on `inputs` or
    under `inputs["platform_context"]`. Agents should treat this as a bounded
    index and fetch referenced artifacts for details.
    """
    source = inputs if isinstance(inputs, dict) else {}
    direct = source.get("execution_workpad")
    if isinstance(direct, dict):
        return dict(direct)
    platform_context = source.get("platform_context")
    if isinstance(platform_context, dict):
        nested = platform_context.get("execution_workpad")
        if isinstance(nested, dict):
            return dict(nested)
    return {}


def upstream_context(inputs: dict[str, Any] | None) -> dict[str, Any]:
    """Return the standard upstream_context.v1 payload injected by platform."""
    source = inputs if isinstance(inputs, dict) else {}
    direct = source.get("upstream_context")
    if isinstance(direct, dict):
        return dict(direct)
    platform_context = source.get("platform_context")
    if isinstance(platform_context, dict):
        nested = platform_context.get("upstream_context")
        if isinstance(nested, dict):
            return dict(nested)
    return {}


def execution_workpad_entries(workpad: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return normalized Execution Workpad entries from a compact workpad."""
    if not isinstance(workpad, dict):
        return []
    raw_entries = workpad.get("entries")
    if not isinstance(raw_entries, list):
        return []
    return [dict(entry) for entry in raw_entries if isinstance(entry, dict)]


def workpad_entries_by_kind(
    workpad: dict[str, Any] | None,
    kind: str,
) -> list[dict[str, Any]]:
    """Return workpad entries matching a semantic kind."""
    expected = str(kind or "").strip()
    if not expected:
        return []
    return [
        entry
        for entry in execution_workpad_entries(workpad)
        if str(entry.get("kind") or "").strip() == expected
    ]


def latest_workpad_entry(
    workpad: dict[str, Any] | None,
    *,
    kind: str | None = None,
) -> dict[str, Any] | None:
    """Return the newest workpad entry, optionally filtered by kind."""
    entries = (
        workpad_entries_by_kind(workpad, kind)
        if kind
        else execution_workpad_entries(workpad)
    )
    if not entries:
        return None
    return entries[-1]


def bounded_workpad_text(text: str, *, limit: int = 8000) -> str:
    """Bound workpad text without changing its media type."""
    value = str(text or "")
    if limit <= 0 or len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"{value[:limit]}\n\n...[truncated {omitted} chars]"


__all__ = [
    "bounded_workpad_text",
    "execution_workpad_entries",
    "execution_workpad_context",
    "latest_workpad_entry",
    "upstream_context",
    "workpad_entries_by_kind",
    "workpad_checkpoint_event",
    "workpad_entry_event",
]
