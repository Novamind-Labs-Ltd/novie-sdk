"""Shared LLM-backed profile selection for document-writing capabilities."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .skill_contracts import SkillRuntimeContract


async def select_document_length_profile(
    *,
    inputs: Mapping[str, Any],
    brief: Mapping[str, Any],
    contract: SkillRuntimeContract,
    llm_facade: Any,
) -> dict[str, str]:
    """Select one declared length profile without semantic keyword routing."""
    supported = set(contract.document.length_profiles) or {"short", "medium", "long"}
    explicit_input = _normalise_length_profile(inputs.get("length_profile"))
    if explicit_input:
        return _checked_length_profile(
            explicit_input,
            supported=supported,
            source="user_input",
            confidence="confirmed",
        )
    explicit_brief = _normalise_length_profile(brief.get("length_profile"))
    if explicit_brief:
        return _checked_length_profile(
            explicit_brief,
            supported=supported,
            source="brief",
            confidence="confirmed",
        )

    defaults = dict(contract.task_profile.defaults or {})
    default_profile = _normalise_length_profile(defaults.get("length_profile"))
    if default_profile and default_profile != "adaptive":
        return _checked_length_profile(
            default_profile,
            supported=supported,
            source="skill_default",
            confidence="confirmed",
        )

    structured_call = getattr(llm_facade, "structured", None)
    if not callable(structured_call):
        return _fallback_length_profile(supported)

    result = await structured_call(
        messages=[
            {
                "role": "user",
                "content": (
                    "Select the document length profile for this capability. "
                    "Use only the profiles declared by the skill contract. "
                    "Prefer medium unless the requested deliverable clearly "
                    "requires a compact or deep document. Return only the "
                    "structured selection.\n\n"
                    f"Available profiles:\n{_json_preview(sorted(supported), limit=1000)}\n\n"
                    f"Skill default length_profile: "
                    f"{defaults.get('length_profile') or 'adaptive'}\n\n"
                    f"Brief:\n{_json_preview(brief, limit=6000)}"
                ),
            }
        ],
        output_schema={
            "title": "document_length_profile_selection",
            "type": "object",
            "additionalProperties": False,
            "required": ["length_profile", "confidence", "reason"],
            "properties": {
                "length_profile": {"type": "string", "enum": sorted(supported)},
                "confidence": {
                    "type": "string",
                    "enum": ["confirmed", "inferred"],
                },
                "reason": {"type": "string"},
            },
        },
        temperature=0.0,
    )
    structured = result.get("structured") if isinstance(result, Mapping) else None
    selected = (
        str(structured.get("length_profile") or "").strip().lower()
        if isinstance(structured, Mapping)
        else ""
    )
    confidence = (
        str(structured.get("confidence") or "inferred").strip().lower()
        if isinstance(structured, Mapping)
        else "inferred"
    )
    return _checked_length_profile(
        selected or "medium",
        supported=supported,
        source="inferred",
        confidence="confirmed" if confidence == "confirmed" else "inferred",
    )


def _fallback_length_profile(supported: set[str]) -> dict[str, str]:
    """Keep document delivery available if structured selection is unavailable."""
    return {
        "profile": "medium" if "medium" in supported else sorted(supported)[0],
        "source": "runtime_fallback",
        "confidence": "inferred",
    }


def _normalise_length_profile(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return raw if raw in {"short", "medium", "long", "ultra", "adaptive"} else ""


def _json_preview(value: Any, *, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _checked_length_profile(
    profile: str,
    *,
    supported: set[str],
    source: str,
    confidence: str,
) -> dict[str, str]:
    if profile == "adaptive":
        profile = "medium"
    if profile not in supported:
        raise RuntimeError(f"unsupported_document_length_profile:{profile}")
    return {
        "profile": profile,
        "source": source,
        "confidence": confidence,
    }


__all__ = ["select_document_length_profile"]
