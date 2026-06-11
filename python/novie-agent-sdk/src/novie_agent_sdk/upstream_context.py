"""Budgeted upstream artifact resolution for agent runtimes.

The platform passes compact upstream handoff envelopes to agents. Those
envelopes are intentionally lossy: large artifacts stay behind artifact refs and
must be read through ``platform.artifacts.read`` so byte budgets and audit trails
are enforced. This module turns that protocol shape into a resolved, promptable
context before the LLM runs, instead of relying on the model to notice refs and
call tools on its own.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


ResolvedStatus = Literal["complete", "partial", "summary_only", "unavailable", "skipped"]

_DETAIL_OMITTED_FIELDS = frozenset(
    {
        "analysis",
        "content",
        "events_snapshot",
        "final_markdown",
        "final_payload",
        "raw_evidence",
        "structured_output",
        "transcript",
        "work_item_draft_graph",
    }
)

_DEFAULT_MAX_TOTAL_BYTES = 48_000
_DEFAULT_MAX_ARTIFACT_BYTES = 24_000
_DEFAULT_MAX_CHUNK_BYTES = 12_000


@dataclass(frozen=True, slots=True)
class UpstreamResolutionBudget:
    """Byte limits for SDK-owned upstream context resolution."""

    max_total_bytes: int = _DEFAULT_MAX_TOTAL_BYTES
    max_artifact_bytes: int = _DEFAULT_MAX_ARTIFACT_BYTES
    max_chunk_bytes: int = _DEFAULT_MAX_CHUNK_BYTES

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "UpstreamResolutionBudget":
        raw = dict(value or {})
        inline = _positive_int(raw.get("max_artifact_bytes_inline"), 65_536)
        return cls(
            max_total_bytes=_positive_int(
                raw.get("max_upstream_artifact_bytes_total"),
                min(inline, _DEFAULT_MAX_TOTAL_BYTES),
            ),
            max_artifact_bytes=_positive_int(
                raw.get("max_upstream_artifact_bytes"),
                min(inline, _DEFAULT_MAX_ARTIFACT_BYTES),
            ),
            max_chunk_bytes=min(
                64_000,
                _positive_int(raw.get("max_upstream_artifact_chunk_bytes"), _DEFAULT_MAX_CHUNK_BYTES),
            ),
        )


@dataclass(frozen=True, slots=True)
class ResolvedArtifactContext:
    """One artifact ref resolved through the platform budgeted read API."""

    artifact_id: str
    artifact_type: str = ""
    ref: str = ""
    status: ResolvedStatus = "skipped"
    retrieval_mode: str = "summary"
    summary: str = ""
    content: str = ""
    bytes_returned: int = 0
    total_bytes: int | None = None
    warnings: tuple[str, ...] = ()
    source_ref: Mapping[str, Any] = field(default_factory=dict)

    def to_prompt_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "ref": self.ref or f"artifact://{self.artifact_id}",
            "status": self.status,
            "retrieval_mode": self.retrieval_mode,
            "summary": self.summary,
            "bytes_returned": self.bytes_returned,
            "warnings": list(self.warnings),
        }
        if self.content:
            out["content"] = self.content
        if self.total_bytes is not None:
            out["total_bytes"] = self.total_bytes
        return out


@dataclass(frozen=True, slots=True)
class ResolvedUpstreamContext:
    """Resolved upstream context plus diagnostics for prompt assembly."""

    steps: Mapping[str, Mapping[str, Any]]
    items: tuple[ResolvedArtifactContext, ...] = ()
    warnings: tuple[str, ...] = ()
    budget: UpstreamResolutionBudget = field(default_factory=UpstreamResolutionBudget)

    def to_prompt_input(self) -> dict[str, Any]:
        """Return an upstream dict compatible with existing prompt builders."""

        out: dict[str, Any] = {}
        by_step: dict[str, list[dict[str, Any]]] = {}
        for item in self.items:
            step_id = str(item.source_ref.get("source_step_id") or "").strip()
            if step_id:
                by_step.setdefault(step_id, []).append(item.to_prompt_dict())

        for step_id, raw in self.steps.items():
            entry = dict(raw)
            resolved = by_step.get(str(step_id))
            if resolved:
                entry["resolved_artifacts"] = resolved
            out[str(step_id)] = entry
        if self.warnings:
            out["_upstream_resolution"] = {
                "warnings": list(self.warnings),
                "budget": {
                    "max_total_bytes": self.budget.max_total_bytes,
                    "max_artifact_bytes": self.budget.max_artifact_bytes,
                    "max_chunk_bytes": self.budget.max_chunk_bytes,
                },
            }
        return out

    @property
    def has_unavailable_required_context(self) -> bool:
        return any(item.status == "unavailable" for item in self.items)


async def resolve_upstream_context(
    *,
    platform: Any,
    upstream: Mapping[str, Any] | None,
    purpose: str = "",
    required_artifact_types: set[str] | None = None,
    budget: Mapping[str, Any] | UpstreamResolutionBudget | None = None,
) -> ResolvedUpstreamContext:
    """Resolve upstream artifact refs through SDK/platform budgeted reads.

    ``platform`` is expected to be the SDK ``PlatformNamespace`` or any object
    with an ``artifacts`` namespace that implements ``summarize`` and
    ``read_chunks``. If the namespace is unavailable, refs are surfaced as
    ``unavailable`` instead of silently disappearing.
    """

    steps = {
        str(step_id): dict(raw) if isinstance(raw, Mapping) else {"value": raw}
        for step_id, raw in dict(upstream or {}).items()
    }
    resolution_budget = (
        budget
        if isinstance(budget, UpstreamResolutionBudget)
        else UpstreamResolutionBudget.from_mapping(budget)
    )
    required_types = {item.strip() for item in (required_artifact_types or set()) if item.strip()}
    artifacts_ns = getattr(platform, "artifacts", None)
    resolved: list[ResolvedArtifactContext] = []
    warnings: list[str] = []
    remaining_total = resolution_budget.max_total_bytes

    for step_id, raw in steps.items():
        refs = _artifact_refs_for_step(step_id, raw)
        if not refs:
            continue
        detail_needed = _requires_detail_resolution(raw, purpose=purpose)
        for ref in refs:
            artifact_id = str(ref.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            artifact_type = str(ref.get("artifact_type") or raw.get("artifact_type") or "").strip()
            is_required = not required_types or artifact_type in required_types
            if not detail_needed and not is_required:
                resolved.append(
                    ResolvedArtifactContext(
                        artifact_id=artifact_id,
                        artifact_type=artifact_type,
                        ref=str(ref.get("ref") or ""),
                        status="skipped",
                        warnings=("detail_not_required",),
                        source_ref=ref,
                    )
                )
                continue
            if artifacts_ns is None:
                resolved.append(
                    ResolvedArtifactContext(
                        artifact_id=artifact_id,
                        artifact_type=artifact_type,
                        ref=str(ref.get("ref") or ""),
                        status="unavailable",
                        warnings=("platform_artifacts_unavailable",),
                        source_ref=ref,
                    )
                )
                warnings.append(f"{step_id}:{artifact_id}: platform_artifacts_unavailable")
                continue
            if remaining_total <= 0:
                resolved.append(
                    ResolvedArtifactContext(
                        artifact_id=artifact_id,
                        artifact_type=artifact_type,
                        ref=str(ref.get("ref") or ""),
                        status="summary_only",
                        warnings=("upstream_artifact_budget_exhausted",),
                        source_ref=ref,
                    )
                )
                warnings.append(f"{step_id}:{artifact_id}: upstream_artifact_budget_exhausted")
                continue

            item = await _resolve_one_artifact(
                artifacts_ns,
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                ref=str(ref.get("ref") or ""),
                source_ref=ref,
                budget=resolution_budget,
                remaining_total=remaining_total,
                purpose=purpose,
            )
            resolved.append(item)
            remaining_total -= item.bytes_returned
            if item.status in {"partial", "summary_only", "unavailable"}:
                warnings.extend(f"{step_id}:{artifact_id}: {warning}" for warning in item.warnings)

    return ResolvedUpstreamContext(
        steps=steps,
        items=tuple(resolved),
        warnings=tuple(dict.fromkeys(warnings)),
        budget=resolution_budget,
    )


async def _resolve_one_artifact(
    artifacts_ns: Any,
    *,
    artifact_id: str,
    artifact_type: str,
    ref: str,
    source_ref: Mapping[str, Any],
    budget: UpstreamResolutionBudget,
    remaining_total: int,
    purpose: str,
) -> ResolvedArtifactContext:
    summary = ""
    warnings: list[str] = []
    try:
        summary_result = await artifacts_ns.summarize(
            artifact_id,
            purpose=purpose or "resolve upstream artifact summary",
        )
    except Exception as exc:  # noqa: BLE001 - SDK helpers must degrade predictably.
        summary_result = {"available": False, "error": type(exc).__name__}
    if isinstance(summary_result, Mapping):
        summary = str(summary_result.get("summary") or "").strip()
        if summary_result.get("available") is False:
            warnings.append(str(summary_result.get("error") or "artifact_summary_unavailable"))

    max_for_artifact = min(budget.max_artifact_bytes, remaining_total)
    if max_for_artifact <= 0:
        return ResolvedArtifactContext(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            ref=ref,
            status="summary_only",
            retrieval_mode="summary",
            summary=summary,
            warnings=tuple(warnings or ["upstream_artifact_budget_exhausted"]),
            source_ref=source_ref,
        )

    chunks: list[str] = []
    bytes_returned = 0
    total_bytes: int | None = None
    next_offset: int | None = 0
    while next_offset is not None and bytes_returned < max_for_artifact:
        chunk_limit = min(budget.max_chunk_bytes, max_for_artifact - bytes_returned)
        if chunk_limit <= 0:
            break
        try:
            chunk_result = await artifacts_ns.read_chunks(
                artifact_id,
                offset=next_offset,
                max_bytes=chunk_limit,
                purpose=purpose or "resolve upstream artifact content",
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(type(exc).__name__)
            break
        if not isinstance(chunk_result, Mapping) or chunk_result.get("available") is False:
            error = (
                str(chunk_result.get("error") or "artifact_chunk_unavailable")
                if isinstance(chunk_result, Mapping)
                else "artifact_chunk_unavailable"
            )
            warnings.append(error)
            break
        content = _content_to_text(chunk_result.get("content"))
        if content:
            chunks.append(content)
            bytes_returned += len(content.encode("utf-8"))
        metadata = chunk_result.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        if isinstance(metadata.get("total_bytes"), int):
            total_bytes = int(metadata["total_bytes"])
        raw_next = metadata.get("next_offset")
        next_offset = int(raw_next) if isinstance(raw_next, int) else None
        if not content and next_offset is None:
            break

    content = "\n".join(part for part in chunks if part).strip()
    if content:
        complete = next_offset is None
        return ResolvedArtifactContext(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            ref=ref,
            status="complete" if complete else "partial",
            retrieval_mode="chunks",
            summary=summary,
            content=content,
            bytes_returned=bytes_returned,
            total_bytes=total_bytes,
            warnings=tuple(warnings + ([] if complete else ["artifact_content_truncated_by_budget"])),
            source_ref=source_ref,
        )
    return ResolvedArtifactContext(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        ref=ref,
        status="summary_only" if summary else "unavailable",
        retrieval_mode="summary",
        summary=summary,
        warnings=tuple(warnings or ["artifact_content_unavailable"]),
        source_ref=source_ref,
    )


def _artifact_refs_for_step(step_id: str, raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for candidate in raw.get("artifact_refs"), _mapping(raw.get("handoff_envelope")).get("artifact_refs"):
        if isinstance(candidate, list):
            refs.extend(dict(item) for item in candidate if isinstance(item, Mapping))
    provides = raw.get("provides_artifacts")
    if isinstance(provides, Mapping):
        refs.extend(dict(item) for item in provides.values() if isinstance(item, Mapping))
    if raw.get("artifact_id"):
        refs.append(
            {
                "artifact_id": raw.get("artifact_id"),
                "artifact_type": raw.get("artifact_type"),
                "ref": raw.get("artifact_ref"),
                "bytes": raw.get("bytes"),
            }
        )

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in refs:
        artifact_id = str(ref.get("artifact_id") or "").strip()
        uri = str(ref.get("ref") or "").strip()
        key = artifact_id or uri
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({**ref, "source_step_id": step_id})
    return out


def _requires_detail_resolution(raw: Mapping[str, Any], *, purpose: str) -> bool:
    if str(purpose).strip() in {"report_synthesis", "synthesis"}:
        if not any(raw.get(key) for key in ("analysis", "final_payload", "structured_output")):
            return True
    handoff = _mapping(raw.get("handoff_envelope"))
    handoff_meta = _mapping(raw.get("handoff_metadata"))
    compaction = _mapping(handoff.get("compaction"))
    if bool(handoff_meta.get("truncated")) or bool(raw.get("truncated")):
        return True
    if str(handoff_meta.get("summary_mode") or compaction.get("mode") or "").strip() == "deterministic_fallback":
        return True
    omitted = set()
    for source in (raw.get("omitted_fields"), handoff.get("omitted_fields"), handoff_meta.get("omitted_fields")):
        if isinstance(source, list):
            omitted.update(str(item) for item in source)
    return bool(omitted & _DETAIL_OMITTED_FIELDS)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)
