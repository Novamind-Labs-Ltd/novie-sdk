"""Context-budget primitives for document and artifact agents."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any


GENERIC_DEFAULT_CONTEXT_BUDGET: dict[str, Any] = {
    "max_input_tokens": 180000,
    "max_output_tokens": 16000,
    "max_total_tokens": 200000,
    "max_wall_clock_seconds": 720,
    "phase_timeouts": {
        "draft": 120,
        "finalize": 180,
        "quality_review": 180,
        "revise": 240,
    },
    "max_revision_rounds": 1,
    "max_artifact_bytes_inline": 65536,
    "source": "sdk_default",
}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def default_context_budget(
    *,
    env_prefix: str = "NOVIE_AGENT",
    source: str = "sdk_default",
    max_revision_rounds: int = 1,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prefix = env_prefix.rstrip("_")
    budget = {
        **GENERIC_DEFAULT_CONTEXT_BUDGET,
        "max_input_tokens": env_int(
            f"{prefix}_MAX_INPUT_TOKENS",
            int(GENERIC_DEFAULT_CONTEXT_BUDGET["max_input_tokens"]),
        ),
        "max_output_tokens": env_int(
            f"{prefix}_MAX_OUTPUT_TOKENS",
            int(GENERIC_DEFAULT_CONTEXT_BUDGET["max_output_tokens"]),
        ),
        "max_total_tokens": env_int(
            f"{prefix}_MAX_TOTAL_TOKENS",
            int(GENERIC_DEFAULT_CONTEXT_BUDGET["max_total_tokens"]),
        ),
        "max_revision_rounds": max(1, int(max_revision_rounds or 1)),
        "phase_timeouts": dict(GENERIC_DEFAULT_CONTEXT_BUDGET["phase_timeouts"]),
        "source": source,
    }
    if overrides:
        phase_timeouts = dict(budget["phase_timeouts"])
        if isinstance(overrides.get("phase_timeouts"), dict):
            phase_timeouts.update(overrides["phase_timeouts"])
        budget.update(overrides)
        budget["phase_timeouts"] = phase_timeouts
    return budget


def context_budget_from_inputs(
    inputs: dict[str, Any] | None,
    *,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = inputs.get("context_budget") if isinstance(inputs, dict) else None
    if not isinstance(raw, dict):
        raw = {}
    base = dict(defaults or GENERIC_DEFAULT_CONTEXT_BUDGET)
    phase_timeouts = dict(base.get("phase_timeouts") or {})
    if isinstance(raw.get("phase_timeouts"), dict):
        phase_timeouts.update(raw["phase_timeouts"])
    return {
        **base,
        **raw,
        "phase_timeouts": phase_timeouts,
    }


def estimated_tokens(value: Any) -> int:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            text = str(value)
    return max(1, len(text) // 4)


def budget_limit(budget: dict[str, Any], key: str, default: int) -> int:
    try:
        return max(0, int(budget.get(key, default)))
    except (TypeError, ValueError):
        return default


def max_revision_rounds_from_budget(budget: dict[str, Any], default: int) -> int:
    return min(default, budget_limit(budget, "max_revision_rounds", default))


def budget_status(
    *,
    context_budget: dict[str, Any],
    estimated_input_tokens: int,
    estimated_output_tokens: int,
) -> str:
    max_input = budget_limit(context_budget, "max_input_tokens", 12000)
    max_output = budget_limit(context_budget, "max_output_tokens", 8000)
    max_total = budget_limit(context_budget, "max_total_tokens", 40000)
    if estimated_input_tokens > max_input:
        return "input_over_budget"
    if estimated_output_tokens > max_output:
        return "output_over_budget"
    if estimated_input_tokens + estimated_output_tokens > max_total:
        return "total_over_budget"
    return "within_budget"


def is_over_budget_status(status: str) -> bool:
    return status != "within_budget"


def phase_timeout_seconds(
    context_budget: dict[str, Any],
    phase_name: str,
    default: float,
) -> float:
    phase_timeouts = context_budget.get("phase_timeouts")
    raw: Any = None
    if isinstance(phase_timeouts, dict):
        raw = phase_timeouts.get(phase_name)
        aliases = {
            "completion_long_report": "finalize",
            "revise_sections": "revise",
            "final_editor": "finalize",
        }
        if raw is None and phase_name in aliases:
            raw = phase_timeouts.get(aliases[phase_name])
    if raw is None:
        raw = default
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return max(1.0, default)


def wall_clock_deadline(context_budget: dict[str, Any]) -> float | None:
    seconds = budget_limit(context_budget, "max_wall_clock_seconds", 0)
    if seconds <= 0:
        return None
    return asyncio.get_running_loop().time() + float(seconds)


def effective_phase_timeout_seconds(
    context_budget: dict[str, Any],
    phase_name: str,
    default: float,
    wall_clock_deadline: float | None,
) -> float:
    phase_timeout = phase_timeout_seconds(context_budget, phase_name, default)
    if wall_clock_deadline is None:
        return phase_timeout
    remaining = wall_clock_deadline - asyncio.get_running_loop().time()
    return max(1.0, min(phase_timeout, remaining))


def adaptive_phase_timeout_seconds(
    context_budget: dict[str, Any],
    phase_name: str,
    default: float,
    wall_clock_deadline: float | None,
    *,
    estimated_input_tokens: int = 0,
) -> float:
    base = phase_timeout_seconds(context_budget, phase_name, default)
    if phase_name in {"quality_review", "revise_sections", "revise", "final_editor"}:
        scaled = max(base, min(420.0, 90.0 + (max(0, estimated_input_tokens) / 90.0)))
        if phase_name == "final_editor":
            scaled = max(base, min(1200.0, 180.0 + (max(0, estimated_input_tokens) / 40.0)))
    else:
        scaled = base
    if wall_clock_deadline is None:
        return max(1.0, scaled)
    remaining = wall_clock_deadline - asyncio.get_running_loop().time()
    return max(1.0, min(scaled, remaining))


def budget_summary(
    *,
    context_budget: dict[str, Any],
    estimated_input_tokens: int,
    estimated_output_tokens: int,
    degraded_codes: list[str] | None = None,
) -> dict[str, Any]:
    codes = list(degraded_codes or [])
    return {
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_total_tokens": estimated_input_tokens + estimated_output_tokens,
        "max_input_tokens": budget_limit(context_budget, "max_input_tokens", 12000),
        "max_output_tokens": budget_limit(context_budget, "max_output_tokens", 8000),
        "max_total_tokens": budget_limit(context_budget, "max_total_tokens", 40000),
        "max_revision_rounds": budget_limit(context_budget, "max_revision_rounds", 1),
        "degraded": bool(codes),
        "degradation_codes": codes,
    }


__all__ = [
    "GENERIC_DEFAULT_CONTEXT_BUDGET",
    "adaptive_phase_timeout_seconds",
    "budget_limit",
    "budget_status",
    "budget_summary",
    "context_budget_from_inputs",
    "default_context_budget",
    "effective_phase_timeout_seconds",
    "env_int",
    "estimated_tokens",
    "is_over_budget_status",
    "max_revision_rounds_from_budget",
    "phase_timeout_seconds",
    "wall_clock_deadline",
]
