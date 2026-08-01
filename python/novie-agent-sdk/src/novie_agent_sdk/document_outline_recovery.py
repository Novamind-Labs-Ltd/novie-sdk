"""Deterministic recovery helpers for structured document outlines."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from typing import Any

from .platform_namespace import PlatformLlmCallError


def outline_output_token_budget(
    contract: Any,
    *,
    acceptance_criteria: Sequence[str] = (),
    brief: Mapping[str, Any] | None = None,
    attempt: int = 0,
) -> int:
    """Size outline calls from requested structure, not document prose length."""
    criterion_chars = sum(len(item) for item in acceptance_criteria)
    brief_units = min(40, _brief_leaf_count(brief or {}))
    base = (
        900
        + contract.max_outline_sections * 180
        + len(acceptance_criteria) * 120
        + criterion_chars // 4
        + brief_units * 40
    )
    multiplier = (1.0, 1.6, 2.4)[min(max(attempt, 0), 2)]
    return min(12000, max(2400, int(base * multiplier)))


def _brief_leaf_count(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(_brief_leaf_count(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_brief_leaf_count(child) for child in value)
    return int(bool(str(value or "").strip()))


def is_recoverable_outline_error(exc: Exception) -> bool:
    if not isinstance(exc, PlatformLlmCallError):
        return False
    return bool(
        exc.retryable
        or exc.is_transient
        or exc.reason_code
        in {
            "platform_llm_structured_empty_stream",
            "platform_llm_structured_timeout",
        }
    )


def merge_outline_plans(
    accepted: Sequence[Any],
    candidate: Sequence[Any],
    *,
    maximum: int,
) -> list[Any]:
    merged = list(accepted)
    positions = {plan.section_id: index for index, plan in enumerate(merged)}
    for plan in candidate:
        if plan.section_id in positions:
            merged[positions[plan.section_id]] = plan
        elif len(merged) < maximum:
            positions[plan.section_id] = len(merged)
            merged.append(plan)
    return merged


def outline_retry_context(
    accepted: Sequence[Any],
    acceptance_criteria: Sequence[str],
    *,
    json_block: Callable[..., str],
) -> str:
    if not accepted:
        return ""
    flattened = "\n".join(
        text
        for plan in accepted
        for text in (plan.title, plan.objective, *plan.required_points)
    )
    missing = [
        f"AC-{index}"
        for index in range(1, len(acceptance_criteria) + 1)
        if re.search(rf"(?<![A-Z0-9-])AC-{index}(?!\d)", flattened) is None
    ]
    return (
        "\n\nRecovery pass: retain or improve these already valid sections and "
        "return only additional/replacement sections needed to complete the "
        f"outline. Missing acceptance IDs: {', '.join(missing) or 'none'}.\n"
        f"Accepted sections:\n{json_block([asdict(plan) for plan in accepted], limit=8000)}"
    )


def repair_outline_deterministically(
    accepted: Sequence[Any],
    *,
    acceptance_criteria: Sequence[str],
    contract: Any,
    brief: Mapping[str, Any],
    plan_factory: Callable[..., Any],
    slugger: Callable[..., str],
) -> tuple[Any, ...]:
    plans = list(accepted[: contract.max_outline_sections])
    titles = ("Purpose and Scope", "Findings and Evidence", "Recommendations")
    task_title = str(brief.get("title") or "Document").strip()
    while len(plans) < contract.min_outline_sections:
        index = len(plans)
        title = (
            titles[index] if index < len(titles) else f"{task_title} Part {index + 1}"
        )
        plans.append(
            plan_factory(
                section_id=slugger(title, fallback=f"section-{index + 1}"),
                title=title,
                objective=f"Complete the {title.lower()} for {task_title}.",
                evidence_query=f"{task_title} {title}",
                min_words=contract.default_section_words,
                required_points=(f"Cover the required {title.lower()} content.",),
            )
        )
    flattened = "\n".join(
        text
        for plan in plans
        for text in (plan.title, plan.objective, *plan.required_points)
    )
    for index, criterion in enumerate(acceptance_criteria, start=1):
        ac_id = f"AC-{index}"
        if re.search(rf"(?<![A-Z0-9-]){re.escape(ac_id)}(?!\d)", flattened):
            continue
        target = (index - 1) % len(plans)
        plans[target] = replace(
            plans[target],
            required_points=plans[target].required_points + (f"{ac_id}: {criterion}",),
        )
    return tuple(plans)
