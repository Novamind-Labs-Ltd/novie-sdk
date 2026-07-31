"""Bounded Planned Part execution with durable checkpoint progression."""
from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

from .document_authoring_plan import (
    PlannedPart,
    SectionExecutionPlan,
)
from .document_authoring_recovery import (
    SectionCoverageCursor,
    coalesce_remaining_parts,
    repartition_remaining_parts,
)
from .document_part_assembly import AcceptedPart, PlannedPartRejected, accept_part

GeneratePart = Callable[
    [PlannedPart, SectionCoverageCursor, str],
    Awaitable[tuple[str, bool]],
]
PersistPart = Callable[[AcceptedPart], Awaitable[Mapping[str, Any]]]
CheckpointPart = Callable[
    [Sequence["AcceptedPartArtifact"], SectionCoverageCursor, int],
    Awaitable[None],
]
EmitPart = Callable[[str, Mapping[str, Any]], Awaitable[None]]
PartMaximum = Callable[[PlannedPart], int]


class PartCompletionExhausted(RuntimeError):
    code = "document_part_completion_exhausted"


@dataclass(frozen=True, slots=True)
class AcceptedPartArtifact:
    accepted: AcceptedPart
    artifact_ref: Mapping[str, Any]
    handoff: str

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "plan": asdict(self.accepted.plan),
            "artifact_ref": dict(self.artifact_ref),
            "content_digest": self.accepted.content_digest,
            "information_units": self.accepted.information_units,
            "handoff": self.handoff,
        }


@dataclass(frozen=True, slots=True)
class PartExecutionResult:
    parts: tuple[AcceptedPartArtifact, ...]
    cursor: SectionCoverageCursor
    plan_revision: int


def _content_digest(markdown: str) -> str:
    return hashlib.sha256(markdown.encode()).hexdigest()


def _handoff(markdown: str, *, limit: int = 1600) -> str:
    text = " ".join(markdown.split())
    return text[-limit:]


async def execute_planned_parts(
    section: SectionExecutionPlan,
    *,
    scope: Mapping[str, Any],
    evidence: Any,
    generate: GeneratePart,
    persist: PersistPart,
    checkpoint: CheckpointPart,
    emit: EmitPart,
    max_acceptable_information_units: PartMaximum | None = None,
    resumed: Sequence[AcceptedPartArtifact] = (),
    max_plan_revisions: int = 2,
    min_part_information_units: int = 20,
    allow_over_maximum: bool = False,
) -> PartExecutionResult:
    """Generate, persist, then checkpoint every complete part."""
    accepted = list(resumed)
    parts = tuple(section.parts)
    revision = 1
    resume_mismatch = False
    for resumed_part in accepted:
        ordinal = resumed_part.accepted.plan.ordinal
        if ordinal < 1 or ordinal > len(parts):
            resume_mismatch = True
            break
        expected = parts[ordinal - 1]
        if (
            expected.objective_digest
            != resumed_part.accepted.plan.objective_digest
        ):
            resume_mismatch = True
            break
    if resume_mismatch:
        await emit(
            "document.part.resume_invalidated",
            {
                "section_id": section.section_id,
                "reason": "document_part_resume_identity_mismatch",
                "resumed_part_count": len(accepted),
            },
        )
        accepted = []
    cursor = SectionCoverageCursor(
        section_id=section.section_id,
        accepted_part_identities=tuple(
            item.accepted.plan.part_identity for item in accepted
        ),
        covered_objective_digests=tuple(
            item.accepted.plan.objective_digest for item in accepted
        ),
        remaining_objective_digests=tuple(
            item.objective_digest for item in parts[len(accepted) :]
        ),
        next_part_ordinal=len(accepted) + 1,
        plan_revision=revision,
    )
    index = len(accepted)
    while index < len(parts):
        planned = parts[index].with_evidence(evidence, scope=scope)
        parts = (*parts[:index], planned, *parts[index + 1 :])
        previous_handoff = accepted[-1].handoff if accepted else ""
        await emit(
            "document.part.started",
            {
                "section_id": section.section_id,
                "part_identity": planned.part_identity,
                "part_ordinal": planned.ordinal,
                "part_total": planned.total,
                "plan_revision": revision,
            },
        )
        markdown, truncated = await generate(planned, cursor, previous_handoff)
        try:
            accepted_part = accept_part(
                planned,
                markdown,
                section_title=section.title,
                truncated=truncated,
                hard_max_information_units=(
                    max_acceptable_information_units(planned)
                    if max_acceptable_information_units is not None
                    else None
                ),
                # Generation already owns the bounded three-rewrite policy.
                # The size ceiling triggers those retries; it is not a second
                # fail-closed admission gate after the final model result.
                allow_over_maximum=allow_over_maximum,
            )
        except PlannedPartRejected as exc:
            rejection = str(exc)
            issue_code = next(
                (
                    code
                    for code in (
                        "output_truncated",
                        "part_over_maximum",
                        "empty_part",
                    )
                    if code in rejection
                ),
                "",
            )
            if not issue_code:
                raise
            if revision >= max_plan_revisions:
                if issue_code == "empty_part":
                    # Preserve the precise terminal reason after the bounded
                    # retry so callers can distinguish provider-empty output
                    # from plan-shape exhaustion.
                    raise
                raise PartCompletionExhausted(
                    "document_part_completion_exhausted"
                ) from exc
            revision += 1
            try:
                if issue_code == "empty_part":
                    recovery_event = "document.part.empty_output_retried"
                elif issue_code == "part_over_maximum" and len(parts) - index > 1:
                    parts = coalesce_remaining_parts(
                        parts,
                        failed_index=index,
                        revision=revision,
                        scope=scope,
                    )
                    recovery_event = "document.part.coalesced"
                else:
                    parts = repartition_remaining_parts(
                        parts,
                        failed_index=index,
                        revision=revision,
                        scope=scope,
                        min_part_information_units=min_part_information_units,
                    )
                    recovery_event = "document.part.repartitioned"
            except ValueError as split_error:
                raise PartCompletionExhausted(
                    "document_part_completion_exhausted"
                ) from split_error
            cursor = replace(
                cursor,
                remaining_objective_digests=tuple(
                    item.objective_digest for item in parts[index:]
                ),
                blocking_issue_code=issue_code,
                plan_revision=revision,
            )
            await emit(
                recovery_event,
                {
                    "section_id": section.section_id,
                    "failed_part_identity": planned.part_identity,
                    "reason": issue_code,
                    "remaining_part_count": len(parts) - index,
                    "plan_revision": revision,
                },
            )
            continue
        accepted_part = replace(
            accepted_part,
            content_digest=_content_digest(accepted_part.markdown),
        )
        ref = dict(await persist(accepted_part))
        artifact = AcceptedPartArtifact(
            accepted=accepted_part,
            artifact_ref=ref,
            handoff=_handoff(accepted_part.markdown),
        )
        accepted.append(artifact)
        cursor = cursor.accept(planned)
        cursor = replace(
            cursor,
            remaining_objective_digests=tuple(
                item.objective_digest for item in parts[index + 1 :]
            ),
            plan_revision=revision,
        )
        await checkpoint(accepted, cursor, revision)
        await emit(
            "document.part.completed",
            {
                "section_id": section.section_id,
                "part_identity": planned.part_identity,
                "part_ordinal": planned.ordinal,
                "part_total": planned.total,
                "artifact_ref": ref,
                "content_digest": accepted_part.content_digest,
                "plan_revision": revision,
            },
        )
        index += 1
    return PartExecutionResult(tuple(accepted), cursor, revision)


__all__ = [
    "AcceptedPartArtifact",
    "PartCompletionExhausted",
    "PartExecutionResult",
    "execute_planned_parts",
]
