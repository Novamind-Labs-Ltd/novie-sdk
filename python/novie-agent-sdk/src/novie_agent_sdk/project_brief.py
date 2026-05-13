"""ProjectBrief helpers for agent prompts.

Dispatch injects ``ProjectBrief`` under ``inputs["__project_brief__"]``. Agents can
use ``extract_project_brief`` + ``render_brief_for_prompt`` without hand-parsing.

Both helpers are pure, tolerate missing keys / malformed shapes, and never raise:
they return ``None`` or empty guidance so callers can fall back to wiki search,
staging-backed context from the platform, or capability-approved pulls.
"""
from __future__ import annotations

import logging
from typing import Any

from novie_protocol.contracts import ProjectBrief

__all__ = [
    "extract_project_brief",
    "render_brief_for_prompt",
]

_log = logging.getLogger(__name__)

_INPUT_KEY = "__project_brief__"


def extract_project_brief(inputs: dict[str, Any] | None) -> ProjectBrief | None:
    """Deserialize ``ProjectBrief`` from an invoke/task ``inputs`` mapping."""
    if not inputs:
        return None
    raw = inputs.get(_INPUT_KEY)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        _log.warning(
            "%s present in inputs but not a dict (got %s); ignoring",
            _INPUT_KEY,
            type(raw).__name__,
        )
        return None
    try:
        return ProjectBrief.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        _log.warning("failed to parse %s: %s; ignoring", _INPUT_KEY, exc)
        return None


def render_brief_for_prompt(
    brief: ProjectBrief,
    *,
    header: str = "# Project Briefing",
    include_meta: bool = False,
) -> str:
    """Render ``ProjectBrief`` as markdown suitable for a system prompt."""
    if brief.minimal:
        reason = brief.degraded_reason or "no additional context"
        return (
            f"{header}\n\n_Project briefing is not available ({reason}). "
            "Fall back to ``services.wiki.search`` for curated knowledge, rely on "
            "platform-injected runtime context (including staged Member/PMS "
            "snapshots), or use approved capability pulls when needed._"
        )

    parts: list[str] = [header]

    if include_meta:
        parts.append(
            f"_project_id=`{brief.project_id}` · "
            f"tenant_id=`{brief.tenant_id}` · "
            f"generated_at=`{brief.generated_at.isoformat()}`_"
        )

    if brief.summary.strip():
        parts.append("## Summary\n\n" + brief.summary.strip())

    if brief.key_constraints:
        parts.append(_bullet_block("Key constraints", brief.key_constraints))

    if brief.recent_focus:
        parts.append(_bullet_block("Recent focus", brief.recent_focus))

    if brief.open_questions:
        parts.append(_bullet_block("Open questions", brief.open_questions))

    if len(parts) == 1:
        return (
            f"{header}\n\n_Project briefing is empty; "
            "use ``services.wiki.search`` or capability-backed fetch tools for "
            "specific facts._"
        )
    return "\n\n".join(parts)


def _bullet_block(title: str, items: tuple[str, ...]) -> str:
    lines = [f"## {title}", ""]
    lines.extend(f"- {item}" for item in items if item.strip())
    return "\n".join(lines)
