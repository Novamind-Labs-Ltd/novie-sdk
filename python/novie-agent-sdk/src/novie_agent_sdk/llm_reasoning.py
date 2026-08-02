"""Typed reasoning controls shared by SDK LLM surfaces."""
from __future__ import annotations

from typing import Any, Literal

ReasoningMode = Literal["default", "disabled"]
ReasoningEffort = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh", "max",
]
ReasoningWorkload = Literal[
    "outline",
    "extraction",
    "review",
    "document",
    "task_decomposition",
    "synthesis",
    "reception",
    "planning",
    "deep_research",
]

_WORKLOAD_EFFORT: dict[ReasoningWorkload, ReasoningEffort] = {
    "outline": "none",
    "extraction": "none",
    "review": "none",
    "document": "none",
    "task_decomposition": "medium",
    "synthesis": "medium",
    "reception": "low",
    "planning": "medium",
    "deep_research": "high",
}


def add_reasoning_arguments(
    args: dict[str, Any],
    *,
    mode: ReasoningMode,
    effort: ReasoningEffort | None,
    workload: ReasoningWorkload | None,
) -> None:
    if mode == "disabled" and effort not in (None, "none"):
        raise ValueError("reasoning_mode=disabled conflicts with reasoning_effort")
    if mode == "disabled":
        args["reasoning_mode"] = mode
    if effort is not None:
        args["reasoning_effort"] = effort
    if workload is not None:
        args["reasoning_workload"] = workload


def byok_reasoning_effort(
    model_id: str,
    mode: ReasoningMode,
    effort: ReasoningEffort | None,
    workload: ReasoningWorkload | None,
) -> ReasoningEffort | None:
    model = model_id.lower().rsplit("/", 1)[-1]
    supports_policy = (
        model == "gpt-5.6"
        or model.startswith("gpt-5.6-")
        or model in {"gpt-5.4", "gpt-5.5"}
        or model.startswith(("gpt-5.4-", "gpt-5.5-"))
    )
    if not supports_policy:
        return None
    if mode == "disabled":
        if effort not in (None, "none"):
            raise ValueError("reasoning_mode=disabled conflicts with reasoning_effort")
        return "none"
    return effort or (_WORKLOAD_EFFORT.get(workload) if workload else None)


__all__ = [
    "ReasoningEffort",
    "ReasoningMode",
    "ReasoningWorkload",
    "add_reasoning_arguments",
    "byok_reasoning_effort",
]
