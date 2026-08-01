"""Reference-only terminal contract for sectioned document authoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .public_errors import PublicAgentError

DOCUMENT_FINALIZATION_KIND = "document_finalization_manifest"
DOCUMENT_FINALIZATION_VERSION = "document-finalization.v1"


class DocumentFinalizationManifestError(ValueError):
    """Raised when accepted section artifacts cannot form a finalization manifest."""


def build_document_finalization_manifest(
    *,
    artifact_type: str,
    authoring_ledger: Mapping[str, Any],
    language: str = "",
) -> dict[str, Any]:
    """Build a bounded manifest from the durable outline and section refs."""
    outline = _required_ref(authoring_ledger.get("outline_ref"), role="outline")
    raw_refs = authoring_ledger.get("artifact_refs")
    if not isinstance(raw_refs, Sequence) or isinstance(raw_refs, (str, bytes)):
        raise DocumentFinalizationManifestError(
            "document_finalization_manifest_missing_section_refs"
        )

    sections: list[dict[str, Any]] = []
    for raw in raw_refs:
        if not isinstance(raw, Mapping) or raw.get("role") != "section_draft":
            continue
        ref = _required_ref(raw, role="section")
        artifact_index = _mapping(raw.get("artifact"))
        metadata = _mapping(raw.get("metadata") or artifact_index.get("metadata"))
        index = _positive_int(metadata.get("section_index"))
        section_id = str(
            raw.get("section_id") or metadata.get("section_id") or ""
        ).strip()
        title = str(metadata.get("section_title") or "").strip()
        if not section_id or index is None:
            raise DocumentFinalizationManifestError(
                "document_finalization_manifest_invalid_section_identity"
            )
        quality = _mapping(metadata.get("quality"))
        sections.append(
            {
                "section_id": section_id,
                "section_index": index,
                "section_title": title,
                **ref,
                "artifact_revision": _positive_int(
                    raw.get("artifact_revision")
                    or raw.get("revision")
                    or artifact_index.get("revision")
                )
                or 1,
                "quality_status": (
                    "degraded" if bool(quality.get("degraded")) else "passed"
                ),
            }
        )

    sections.sort(key=lambda item: item["section_index"])
    if not sections or [item["section_index"] for item in sections] != list(
        range(1, len(sections) + 1)
    ):
        raise DocumentFinalizationManifestError(
            "document_finalization_manifest_invalid_section_order"
        )
    if len({item["section_id"] for item in sections}) != len(sections):
        raise DocumentFinalizationManifestError(
            "document_finalization_manifest_duplicate_section"
        )

    degraded = list(authoring_ledger.get("degraded_sections") or [])
    execution_control_action = str(
        authoring_ledger.get("execution_control_action") or ""
    ).strip()
    published_partial = execution_control_action == "finish_current"
    return {
        "kind": DOCUMENT_FINALIZATION_KIND,
        "version": DOCUMENT_FINALIZATION_VERSION,
        "artifact_type": str(artifact_type or "").strip(),
        "outline_ref": outline["artifact_ref"],
        "outline_artifact_type": outline["artifact_type"],
        "outline_sha256": outline["content_sha256"],
        "sections": sections,
        "assembly": {
            "mode": "ordered_markdown_join",
            "separator": "\n\n",
            **({"language": language} if language else {}),
        },
        "quality": {
            "status": "degraded" if degraded else "passed",
            "degraded_sections": degraded,
        },
        **(
            {
                "publication_state": "published_partial",
                "execution_control_action": execution_control_action,
                "incomplete": True,
            }
            if published_partial
            else {"publication_state": "published_final"}
        ),
    }


def build_typed_document_finalization_manifest(
    *,
    artifact_type: str,
    authoring_ledger: Mapping[str, Any],
    language: str = "",
) -> dict[str, Any]:
    """Build a manifest or expose a stable repairable contract failure."""
    try:
        return build_document_finalization_manifest(
            artifact_type=artifact_type,
            authoring_ledger=authoring_ledger,
            language=language,
        )
    except DocumentFinalizationManifestError as exc:
        raise PublicAgentError(
            error_code="document_finalization_manifest_invalid",
            public_message="Accepted document sections could not form a final deliverable.",
            retryable=False,
            replan_eligible=False,
            repair_eligible=True,
        ) from exc


def authoring_ledger_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild the ref-only ledger from a durable sectioned checkpoint."""
    refs: list[dict[str, Any]] = []
    drafts = checkpoint.get("drafts")
    if isinstance(drafts, Sequence) and not isinstance(drafts, (str, bytes)):
        for raw in drafts:
            draft = _mapping(raw)
            artifact = _mapping(draft.get("artifact_ref"))
            plan = _mapping(draft.get("plan"))
            if artifact:
                refs.append(
                    {
                        "role": "section_draft",
                        "section_id": str(plan.get("section_id") or ""),
                        **artifact,
                    }
                )
    return {
        "enabled": True,
        "status": "resumed",
        "outline_ref": _mapping(checkpoint.get("outline_ref")),
        "artifact_refs": refs,
        "section_count": len(refs),
        "completed_section_count": len(refs),
        "degraded": bool(checkpoint.get("degraded_sections")),
        "degraded_sections": list(checkpoint.get("degraded_sections") or []),
        "execution_control_action": str(
            checkpoint.get("execution_control_action") or ""
        ),
    }


def _required_ref(value: Any, *, role: str) -> dict[str, Any]:
    raw = _mapping(value)
    artifact_ref = str(raw.get("artifact_ref") or raw.get("ref") or "").strip()
    digest = str(raw.get("content_sha256") or raw.get("artifact_sha256") or "").strip()
    artifact_type = str(raw.get("artifact_type") or "").strip()
    byte_size = _positive_int(raw.get("bytes") or raw.get("byte_size"))
    if not artifact_ref.startswith("artifact://") or not digest or not artifact_type:
        raise DocumentFinalizationManifestError(
            f"document_finalization_manifest_invalid_{role}_ref"
        )
    return {
        "artifact_ref": artifact_ref,
        "artifact_type": artifact_type,
        "content_sha256": digest,
        **({"byte_size": byte_size} if byte_size is not None else {}),
    }


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


__all__ = [
    "DOCUMENT_FINALIZATION_KIND",
    "DOCUMENT_FINALIZATION_VERSION",
    "DocumentFinalizationManifestError",
    "authoring_ledger_from_checkpoint",
    "build_document_finalization_manifest",
    "build_typed_document_finalization_manifest",
]
