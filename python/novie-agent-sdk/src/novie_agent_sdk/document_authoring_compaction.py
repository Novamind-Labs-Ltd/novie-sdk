"""Structured, bounded ADR-115 authoring-context compaction."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .document_authoring_context import deterministic_authoring_summary


AUTHORING_CONTEXT_SUMMARY_SCHEMA: dict[str, Any] = {
    "title": "AuthoringContextSummary",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "covered_objectives": {"type": "array", "items": {"type": "string"}},
        "unresolved_objectives": {"type": "array", "items": {"type": "string"}},
        "established_facts": {"type": "array", "items": {"type": "string"}},
        "terminology": {"type": "array", "items": {"type": "string"}},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "unresolved_questions": {"type": "array", "items": {"type": "string"}},
        "continuity_notes": {"type": "array", "items": {"type": "string"}},
        "previous_part_handoff": {"type": "string"},
    },
    "required": [
        "covered_objectives",
        "unresolved_objectives",
        "established_facts",
        "terminology",
        "decisions",
        "unresolved_questions",
        "continuity_notes",
        "previous_part_handoff",
    ],
}

_LIST_FIELDS = (
    "covered_objectives",
    "unresolved_objectives",
    "established_facts",
    "terminology",
    "decisions",
    "unresolved_questions",
    "continuity_notes",
)


@dataclass(frozen=True, slots=True)
class AuthoringCompactionResult:
    summary: Mapping[str, Any]
    mode: str
    source_digest: str
    error: str = ""


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _validated(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    out: dict[str, Any] = {}
    for field in _LIST_FIELDS:
        value = payload.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            return None
        out[field] = [item.strip() for item in value if item.strip()]
    handoff = payload.get("previous_part_handoff")
    if not isinstance(handoff, str):
        return None
    out["previous_part_handoff"] = handoff.strip()
    return out


async def compact_authoring_context(
    llm_facade: Any,
    *,
    prior_summary: Mapping[str, Any] | None,
    accepted_part_markdown: str,
    covered_objectives: Sequence[str],
    unresolved_objectives: Sequence[str],
    evidence_refs: Sequence[Mapping[str, Any]],
    model: str | None = None,
    max_output_tokens: int = 800,
    timeout_seconds: float | None = None,
) -> AuthoringCompactionResult:
    source = {
        "prior_summary": dict(prior_summary or {}),
        "accepted_part_markdown": accepted_part_markdown,
        "covered_objectives": list(covered_objectives),
        "unresolved_objectives": list(unresolved_objectives),
        "evidence_refs": [dict(item) for item in evidence_refs],
    }
    source_digest = _digest(source)
    fallback = deterministic_authoring_summary(
        covered_objectives=covered_objectives,
        unresolved_objectives=unresolved_objectives,
        evidence_refs=evidence_refs,
        continuity_notes=(
            str((prior_summary or {}).get("previous_part_handoff") or ""),
        ),
    )
    structured = getattr(llm_facade, "structured", None)
    if not callable(structured):
        return AuthoringCompactionResult(
            fallback,
            "deterministic_fallback",
            source_digest,
            "structured_llm_unavailable",
        )
    try:
        result = await structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Compact document-authoring navigation state. Preserve objective "
                        "coverage, exact evidence refs, terminology, decisions, unresolved "
                        "questions, and continuity. Do not recreate or rewrite canonical "
                        "document prose."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(source, ensure_ascii=False, default=str),
                },
            ],
            output_schema=AUTHORING_CONTEXT_SUMMARY_SCHEMA,
            temperature=0,
            method="json_schema",
            strict=True,
            model=model,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        )
        parsed = _validated(
            result.get("structured") if isinstance(result, Mapping) else None
        )
        if parsed is None:
            raise ValueError("invalid structured authoring summary")
    except Exception as exc:
        return AuthoringCompactionResult(
            fallback,
            "deterministic_fallback",
            source_digest,
            type(exc).__name__,
        )
    return AuthoringCompactionResult(
        {
            "schema": "authoring-context-summary.v1",
            "profile": "authoring_context_v1",
            **parsed,
            "evidence_refs": [dict(item) for item in evidence_refs],
            "mode": "semantic_llm",
        },
        "semantic_llm",
        source_digest,
    )


__all__ = [
    "AUTHORING_CONTEXT_SUMMARY_SCHEMA",
    "AuthoringCompactionResult",
    "compact_authoring_context",
]
