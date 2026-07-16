"""Artifact/workpad helpers for long-running SDK agents.

Agents should keep workpad state compact: record artifact refs and summaries,
then rebuild bounded evidence packs by reading refs through platform artifacts.
This module centralizes that pattern so business agents do not hand-roll
platform write/read loops.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Character budgets for prompt-facing evidence packs."""

    max_total_chars: int = 24_000
    max_item_chars: int = 6_000
    max_refs: int = 24
    max_workpad_entries: int = 24

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ContextBudget":
        raw = dict(value or {})
        return cls(
            max_total_chars=_positive_int(raw.get("max_evidence_pack_chars"), 24_000),
            max_item_chars=_positive_int(raw.get("max_evidence_item_chars"), 6_000),
            max_refs=_positive_int(raw.get("max_evidence_refs"), 24),
            max_workpad_entries=_positive_int(raw.get("max_workpad_entries"), 24),
        )


@dataclass(frozen=True, slots=True)
class EvidencePackItem:
    """One bounded artifact excerpt or summary in an evidence pack."""

    artifact_id: str
    artifact_type: str = ""
    ref: str = ""
    title: str = ""
    summary: str = ""
    content: str = ""
    source: str = ""
    bytes_returned: int = 0
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_prompt_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "ref": self.ref or f"artifact://{self.artifact_id}",
            "title": self.title,
            "summary": self.summary,
            "source": self.source,
            "warnings": list(self.warnings),
        }
        if self.content:
            out["content"] = self.content
        if self.bytes_returned:
            out["bytes_returned"] = self.bytes_returned
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out


@dataclass(frozen=True, slots=True)
class EvidencePack:
    """Bounded context rebuilt from workpad/upstream artifact refs."""

    items: tuple[EvidencePackItem, ...]
    warnings: tuple[str, ...] = ()
    budget: ContextBudget = field(default_factory=ContextBudget)

    @property
    def total_chars(self) -> int:
        return sum(len(item.content or item.summary) for item in self.items)

    def to_prompt_input(self) -> dict[str, Any]:
        return {
            "items": [item.to_prompt_dict() for item in self.items],
            "warnings": list(self.warnings),
            "budget": {
                "max_total_chars": self.budget.max_total_chars,
                "max_item_chars": self.budget.max_item_chars,
                "max_refs": self.budget.max_refs,
                "max_workpad_entries": self.budget.max_workpad_entries,
            },
            "total_chars": self.total_chars,
        }


class ArtifactLedger:
    """Small SDK facade for artifact creation and workpad ref recording."""

    def __init__(self, platform: Any) -> None:
        self._platform = platform
        self._artifacts = getattr(platform, "artifacts", None)
        self._workpads = getattr(platform, "workpads", None)

    @property
    def is_available(self) -> bool:
        return self._artifacts is not None and self._workpads is not None

    async def create_artifact(
        self,
        *,
        artifact_type: str,
        content: Any,
        content_type: str = "text/markdown",
        summary: str = "",
        workflow_id: str | None = None,
        thread_id: str | None = None,
        step_id: str | None = None,
        agent_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        create = getattr(self._artifacts, "create", None)
        if not callable(create):
            return {"available": False, "error": "artifact_create_unavailable"}
        result = await create(
            artifact_type=artifact_type,
            content=content,
            content_type=content_type,
            summary=summary,
            workflow_id=workflow_id,
            thread_id=thread_id,
            step_id=step_id,
            agent_id=agent_id,
            metadata=metadata,
        )
        return dict(result or {})

    async def record_entry(
        self,
        *,
        kind: str,
        title: str = "",
        workflow_id: str | None = None,
        step_id: str | None = None,
        agent_id: str | None = None,
        capability_id: str | None = None,
        artifact_result: Mapping[str, Any] | None = None,
        content_ref: str = "",
        content_preview: str = "",
        artifact_refs: Sequence[Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = getattr(self._workpads, "record_entry", None)
        if not callable(record):
            return {"available": False, "error": "workpad_record_unavailable"}
        refs = [dict(item) for item in (artifact_refs or ())]
        if artifact_result:
            refs.append(_artifact_ref_from_result(artifact_result, kind=kind))
            content_ref = content_ref or str(artifact_result.get("artifact_ref") or "")
        result = await record(
            kind=kind,
            title=title,
            workflow_id=workflow_id,
            step_id=step_id,
            agent_id=agent_id,
            capability_id=capability_id,
            content_ref=content_ref,
            content_preview=content_preview,
            artifact_refs=refs,
            metadata=metadata,
        )
        return dict(result or {})

    async def create_and_record(
        self,
        *,
        artifact_type: str,
        content: Any,
        kind: str,
        title: str = "",
        content_type: str = "text/markdown",
        summary: str = "",
        workflow_id: str | None = None,
        thread_id: str | None = None,
        step_id: str | None = None,
        agent_id: str | None = None,
        capability_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        workpad_metadata: Mapping[str, Any] | None = None,
        strict: bool = False,
    ) -> dict[str, Any]:
        artifact = await self.create_artifact(
            artifact_type=artifact_type,
            content=content,
            content_type=content_type,
            summary=summary,
            workflow_id=workflow_id,
            thread_id=thread_id,
            step_id=step_id,
            agent_id=agent_id,
            metadata=metadata,
        )
        if artifact.get("available", True) is False:
            if strict:
                raise RuntimeError(
                    "artifact_create_failed:"
                    f"{artifact.get('error') or 'artifact_create_unavailable'}"
                )
            return {"artifact": artifact, "workpad": None}
        preview = summary or _preview(content)
        workpad = await self.record_entry(
            kind=kind,
            title=title,
            workflow_id=workflow_id,
            step_id=step_id,
            agent_id=agent_id,
            capability_id=capability_id,
            artifact_result=artifact,
            content_preview=preview,
            metadata=workpad_metadata,
        )
        if isinstance(workpad, Mapping) and workpad.get("available", True) is False:
            artifact = {
                **artifact,
                "workpad_record": {
                    "degraded": True,
                    "error": workpad.get("error") or "workpad_record_unavailable",
                    "message": workpad.get("message") or "",
                },
            }
        return {"artifact": artifact, "workpad": workpad}

    async def set_final_deliverable(
        self,
        artifact_ref: str,
        *,
        workflow_id: str | None = None,
        step_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        setter = getattr(self._workpads, "set_final_deliverable", None)
        if not callable(setter):
            return {"available": False, "error": "workpad_final_unavailable"}
        result = await setter(
            artifact_ref,
            workflow_id=workflow_id,
            step_id=step_id,
            metadata=metadata,
        )
        return dict(result or {})


class EvidencePackBuilder:
    """Rebuild bounded prompt context from workpad and upstream artifact refs."""

    def __init__(
        self,
        platform: Any,
        *,
        budget: ContextBudget | Mapping[str, Any] | None = None,
    ) -> None:
        self._platform = platform
        self._artifacts = getattr(platform, "artifacts", None)
        self._workpads = getattr(platform, "workpads", None)
        self._budget = budget if isinstance(budget, ContextBudget) else ContextBudget.from_mapping(budget)

    async def build(
        self,
        *,
        workflow_id: str | None = None,
        upstream: Mapping[str, Any] | None = None,
        query: str = "",
        purpose: str = "",
        artifact_type_prefixes: set[str] | None = None,
        exclude_workpad_step_ids: set[str] | None = None,
    ) -> EvidencePack:
        refs: list[dict[str, Any]] = []
        warnings: list[str] = []
        snapshot = await self._snapshot(workflow_id=workflow_id)
        entries = snapshot.get("entries") if isinstance(snapshot, Mapping) else None
        if isinstance(entries, list):
            refs.extend(
                _refs_from_workpad_entries(
                    entries,
                    exclude_step_ids=exclude_workpad_step_ids or set(),
                )
            )
        elif snapshot and snapshot.get("available") is False:
            warnings.append(str(snapshot.get("error") or "workpad_snapshot_unavailable"))
        refs.extend(_refs_from_upstream(upstream or {}))

        prefixes = {item for item in (artifact_type_prefixes or set()) if item}
        deduped = _dedupe_refs(refs, prefixes=prefixes)[: self._budget.max_refs]
        if self._artifacts is None:
            return EvidencePack(
                items=(),
                warnings=tuple(warnings + ["platform_artifacts_unavailable"]),
                budget=self._budget,
            )

        items: list[EvidencePackItem] = []
        remaining = self._budget.max_total_chars
        empty_ignored = 0
        for ref in deduped:
            if remaining <= 0:
                warnings.append("evidence_pack_budget_exhausted")
                break
            item = await self._resolve_ref(
                ref,
                query=query,
                purpose=purpose,
                max_chars=min(self._budget.max_item_chars, remaining),
            )
            if item is None:
                if str(ref.get("artifact_id") or "").strip():
                    empty_ignored += 1
                continue
            items.append(item)
            remaining -= len(item.content or item.summary)

        if empty_ignored:
            # Empty wiki/artifact payloads must not participate in authoring.
            warnings.append("empty_evidence_ignored")
        return EvidencePack(
            items=tuple(items),
            warnings=tuple(dict.fromkeys(warnings)),
            budget=self._budget,
        )

    async def _snapshot(self, *, workflow_id: str | None) -> dict[str, Any]:
        snapshot = getattr(self._workpads, "snapshot", None)
        if not callable(snapshot):
            return {}
        result = await snapshot(
            workflow_id=workflow_id,
            limit=self._budget.max_workpad_entries,
        )
        return dict(result or {})

    async def _resolve_ref(
        self,
        ref: Mapping[str, Any],
        *,
        query: str,
        purpose: str,
        max_chars: int,
    ) -> EvidencePackItem | None:
        artifact_id = str(ref.get("artifact_id") or "").strip()
        if not artifact_id:
            return None
        artifact_type = str(ref.get("artifact_type") or "").strip()
        summary = ""
        content = ""
        warnings: list[str] = []
        bytes_returned = 0

        summarize = getattr(self._artifacts, "summarize", None)
        if callable(summarize):
            result = await summarize(
                artifact_id,
                purpose=purpose or "build evidence pack",
            )
            if isinstance(result, Mapping) and result.get("available", True) is not False:
                summary = str(result.get("summary") or "")[:max_chars]

        if query:
            search = getattr(self._artifacts, "search", None)
            if callable(search):
                result = await search(
                    artifact_id,
                    query,
                    purpose=purpose or "build evidence pack",
                    max_bytes=max_chars,
                )
                if isinstance(result, Mapping) and result.get("available", True) is not False:
                    content = _content_from_read_result(result)[:max_chars]
                    bytes_returned = _bytes_from_read_result(result)
        if not content:
            read_chunks = getattr(self._artifacts, "read_chunks", None)
            if callable(read_chunks):
                result = await read_chunks(
                    artifact_id,
                    purpose=purpose or "build evidence pack",
                    offset=0,
                    max_bytes=max_chars,
                )
                if isinstance(result, Mapping) and result.get("available", True) is not False:
                    content = _content_from_read_result(result)[:max_chars]
                    bytes_returned = _bytes_from_read_result(result)
                    metadata = result.get("metadata")
                    if isinstance(metadata, Mapping) and metadata.get("next_offset") is not None:
                        warnings.append("artifact_truncated_by_evidence_budget")

        # Empty wiki / unavailable artifact bodies: skip entirely so they do
        # not enter the section draft evidence pack or quality gates.
        if not str(summary or "").strip() and not str(content or "").strip():
            return None
        return EvidencePackItem(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            ref=str(ref.get("ref") or ""),
            title=str(ref.get("title") or ref.get("kind") or ""),
            summary=summary,
            content=content,
            source=str(ref.get("source") or ""),
            bytes_returned=bytes_returned,
            warnings=tuple(warnings),
            metadata={key: value for key, value in ref.items() if key not in {"artifact_id", "artifact_type", "ref"}},
        )


def _artifact_ref_from_result(result: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    return {
        "artifact_id": result.get("artifact_id"),
        "artifact_type": result.get("artifact_type"),
        "ref": result.get("artifact_ref"),
        "bytes": result.get("bytes"),
        "kind": kind,
    }


def _refs_from_workpad_entries(
    entries: Sequence[Any],
    *,
    exclude_step_ids: set[str],
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        entry_step_id = str(entry.get("step_id") or "").strip()
        if entry_step_id and entry_step_id in exclude_step_ids:
            continue
        artifact_refs = entry.get("artifact_refs")
        if not isinstance(artifact_refs, Sequence) or isinstance(artifact_refs, (str, bytes)):
            continue
        for ref in artifact_refs:
            if not isinstance(ref, Mapping):
                continue
            item = dict(ref)
            item.setdefault("source", "workpad")
            item.setdefault("title", entry.get("title") or "")
            if entry_step_id:
                item.setdefault("source_step_id", entry_step_id)
            refs.append(item)
    return refs


def _refs_from_upstream(upstream: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for step_id, raw in upstream.items():
        if not isinstance(raw, Mapping):
            continue
        artifact_refs = raw.get("artifact_refs")
        if isinstance(artifact_refs, Sequence) and not isinstance(artifact_refs, (str, bytes)):
            for ref in artifact_refs:
                if isinstance(ref, Mapping):
                    item = dict(ref)
                    item.setdefault("source", "upstream")
                    item.setdefault("source_step_id", str(step_id))
                    refs.append(item)
        artifact_id = str(raw.get("artifact_id") or "").strip()
        if artifact_id:
            refs.append(
                {
                    "artifact_id": artifact_id,
                    "artifact_type": raw.get("artifact_type") or "",
                    "ref": raw.get("artifact_ref") or f"artifact://{artifact_id}",
                    "source": "upstream",
                    "source_step_id": str(step_id),
                }
            )
    return refs


def _dedupe_refs(
    refs: Sequence[Mapping[str, Any]],
    *,
    prefixes: set[str],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for raw in refs:
        artifact_id = str(raw.get("artifact_id") or "").strip()
        if not artifact_id or artifact_id in seen:
            continue
        artifact_type = str(raw.get("artifact_type") or "").strip()
        if prefixes and not any(artifact_type.startswith(prefix) for prefix in prefixes):
            continue
        seen.add(artifact_id)
        out.append(dict(raw))
    return out


def _content_from_read_result(result: Mapping[str, Any]) -> str:
    for key in ("content", "text", "excerpt"):
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    excerpts = result.get("excerpts")
    if isinstance(excerpts, Sequence) and not isinstance(excerpts, (str, bytes)):
        parts = []
        for item in excerpts:
            if isinstance(item, Mapping):
                text = item.get("content") or item.get("text") or item.get("excerpt")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n\n".join(parts)
    return ""


def _bytes_from_read_result(result: Mapping[str, Any]) -> int:
    metadata = result.get("metadata")
    if isinstance(metadata, Mapping):
        return _positive_int(metadata.get("bytes"), 0)
    return _positive_int(result.get("bytes"), 0)


def _preview(content: Any, *, limit: int = 900) -> str:
    text = content if isinstance(content, str) else str(content)
    normalised = " ".join(text.split())
    if len(normalised) <= limit:
        return normalised
    return normalised[: limit - 1].rstrip() + "..."


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
