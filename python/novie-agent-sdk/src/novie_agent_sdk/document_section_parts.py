"""Integration seam between SectionedLongformAuthor and ADR-115 part execution."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .document_authoring_plan import (
    AuthoringExecutionPlan,
    PlannedPart,
    SectionExecutionPlan,
)
from .document_part_assembly import AcceptedPart, assemble_section
from .document_part_execution import (
    AcceptedPartArtifact,
    PartExecutionResult,
    execute_planned_parts,
)


@dataclass(frozen=True, slots=True)
class SectionPartResult:
    markdown: str
    execution: PartExecutionResult


async def author_planned_section(
    owner: Any,
    *,
    brief: Mapping[str, Any],
    section_plan: Any,
    execution_plan: AuthoringExecutionPlan,
    execution_section: SectionExecutionPlan,
    section_index: int,
    previous_drafts: Sequence[Any],
    evidence_pack: Mapping[str, Any],
    authoring_summary: Mapping[str, Any] | None,
    workflow_id: str | None,
    thread_id: str | None,
    agent_id: str | None,
    checkpoint_base: Mapping[str, Any],
    resumed_parts: Sequence[AcceptedPartArtifact] = (),
) -> SectionPartResult:
    scope = {
        "tenant_id": str(owner._context_budget.get("tenant_id") or ""),
        "workspace_id": str(owner._context_budget.get("workspace_id") or ""),
        "workflow_id": str(workflow_id or ""),
        "step_id": str(owner._step_id or ""),
        "capability_id": str(owner._capability_id or ""),
    }

    async def generate(part: Any, cursor: Any, handoff: str) -> tuple[str, bool]:
        return await owner._draft_section(
            brief=brief,
            plan=section_plan,
            section_index=section_index,
            previous=list(previous_drafts),
            evidence_pack=evidence_pack,
            planned_part=part,
            coverage_cursor=cursor,
            previous_part_handoff=handoff,
            authoring_summary=authoring_summary,
            output_slots_remaining=(
                execution_plan.part_count - len(cursor.accepted_part_identities) + 1
            ),
        )

    async def persist(accepted: AcceptedPart) -> Mapping[str, Any]:
        return await owner._record_part(
            section_plan,
            accepted,
            section_index=section_index,
            workflow_id=workflow_id,
            thread_id=thread_id,
            agent_id=agent_id,
        )

    async def checkpoint(
        accepted: Sequence[AcceptedPartArtifact],
        cursor: Any,
        revision: int,
    ) -> None:
        payload = dict(checkpoint_base)
        call_budget = getattr(owner, "_authoring_call_budget", None)
        if call_budget is not None:
            payload.update(call_budget.metadata())
        await owner._checkpoint(
            **payload,
            checkpoint_schema="document-authoring-checkpoint.v2",
            authoring_execution_plan=execution_plan.to_mapping(),
            current_section_id=execution_section.section_id,
            accepted_parts=[item.to_checkpoint() for item in accepted],
            section_coverage_cursor=asdict(cursor),
            authoring_plan_revision=revision,
        )

    async def emit(event: str, metadata: Mapping[str, Any]) -> None:
        await owner._emit(event, **dict(metadata))

    result = await execute_planned_parts(
        execution_section,
        scope=scope,
        evidence=evidence_pack,
        generate=generate,
        persist=persist,
        checkpoint=checkpoint,
        emit=emit,
        resumed=resumed_parts,
        max_plan_revisions=_positive_int(
            owner._context_budget.get("max_authoring_plan_revisions_per_section"),
            2,
        ),
        min_part_information_units=_positive_int(
            owner._context_budget.get("min_part_information_units"),
            20,
        ),
    )
    markdown = assemble_section(
        section_title=execution_section.title,
        parts=[item.accepted for item in result.parts],
    )
    return SectionPartResult(markdown=markdown, execution=result)


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


async def resumed_parts_from_checkpoint(
    owner: Any,
    state: Mapping[str, Any],
    *,
    section_id: str,
) -> tuple[AcceptedPartArtifact, ...]:
    if str(state.get("current_section_id") or "") != section_id:
        return ()
    raw_parts = state.get("accepted_parts")
    if not isinstance(raw_parts, list):
        return ()
    resumed: list[AcceptedPartArtifact] = []
    for raw in raw_parts:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("plan"), Mapping):
            return ()
        try:
            plan = PlannedPart(**dict(raw["plan"]))
        except (TypeError, ValueError):
            return ()
        artifact_ref = (
            dict(raw.get("artifact_ref") or {})
            if isinstance(raw.get("artifact_ref"), Mapping)
            else {}
        )
        markdown = await owner._read_resume_artifact_text(artifact_ref)
        digest = hashlib.sha256(markdown.encode()).hexdigest()
        expected_digest = str(raw.get("content_digest") or "")
        if not markdown.strip() or not expected_digest or digest != expected_digest:
            return ()
        accepted = AcceptedPart(
            plan=plan,
            markdown=markdown,
            information_units=_positive_int(raw.get("information_units"), 1),
            artifact_ref=str(artifact_ref.get("artifact_ref") or ""),
            content_digest=digest,
        )
        resumed.append(
            AcceptedPartArtifact(
                accepted=accepted,
                artifact_ref=artifact_ref,
                handoff=str(raw.get("handoff") or ""),
            )
        )
    return tuple(resumed)


__all__ = [
    "SectionPartResult",
    "author_planned_section",
    "resumed_parts_from_checkpoint",
]
