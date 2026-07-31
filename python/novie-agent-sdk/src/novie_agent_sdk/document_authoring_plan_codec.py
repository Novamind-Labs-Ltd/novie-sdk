"""Durable encoding helpers for ADR-115 authoring execution plans."""
from __future__ import annotations

from typing import Any, Mapping

from .document_authoring_plan import (
    AuthoringExecutionPlan,
    PlannedPart,
    SectionExecutionPlan,
)


def authoring_execution_plan_from_mapping(
    value: Mapping[str, Any],
) -> AuthoringExecutionPlan:
    sections: list[SectionExecutionPlan] = []
    for raw_section in value.get("sections") or []:
        if not isinstance(raw_section, Mapping):
            raise ValueError("invalid authoring execution section")
        parts = tuple(
            PlannedPart(
                **{
                    **dict(raw_part),
                    "plan_revisions": tuple(raw_part.get("plan_revisions") or (1,)),
                }
            )
            for raw_part in raw_section.get("parts") or []
            if isinstance(raw_part, Mapping)
        )
        sections.append(
            SectionExecutionPlan(
                section_id=str(raw_section.get("section_id") or ""),
                title=str(raw_section.get("title") or ""),
                objective=str(raw_section.get("objective") or ""),
                target_information_units=int(
                    raw_section.get("target_information_units") or 0
                ),
                max_information_units=int(
                    raw_section.get("max_information_units") or 0
                ),
                parts=parts,
                required_points=tuple(
                    str(item)
                    for item in raw_section.get("required_points") or ()
                    if str(item)
                ),
            )
        )
    return AuthoringExecutionPlan(
        schema=str(value.get("schema") or ""),
        revision=int(value.get("revision") or 1),
        outline_digest=str(value.get("outline_digest") or ""),
        sections=tuple(sections),
        max_authoring_llm_calls=int(value.get("max_authoring_llm_calls") or 0),
        max_authoring_compaction_calls=int(
            value.get("max_authoring_compaction_calls") or 0
        ),
        max_section_review_calls=int(value.get("max_section_review_calls") or 0),
        provider_output_ceiling=(
            int(value["provider_output_ceiling"])
            if value.get("provider_output_ceiling") is not None
            else None
        ),
        estimated_authoring_seconds=float(
            value.get("estimated_authoring_seconds") or 0
        ),
        deadline_feasible=bool(value.get("deadline_feasible", True)),
        input_digest=str(value.get("input_digest") or ""),
    )


__all__ = ["authoring_execution_plan_from_mapping"]
