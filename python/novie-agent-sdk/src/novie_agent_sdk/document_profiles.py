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
) -> dict[str, Any]:
    """Resolve a structured profile and optional user-requested unit range."""
    supported = set(contract.document.length_profiles) or {"short", "medium", "long"}
    input_range = _explicit_requested_range(inputs)
    brief_range = _explicit_requested_range(brief)
    requested_range = input_range if any(input_range) else brief_range
    requested_source = "user_input" if any(input_range) else "brief"
    explicit_input = _declared_length_profile(inputs)
    if explicit_input:
        return _checked_length_profile(
            explicit_input,
            supported=supported,
            source="user_input",
            confidence="confirmed",
            reason="Explicit length profile in capability input.",
            requested_minimum=requested_range[0],
            requested_maximum=requested_range[1],
        )
    explicit_brief = _declared_length_profile(brief)
    if explicit_brief:
        return _checked_length_profile(
            explicit_brief,
            supported=supported,
            source="brief",
            confidence="confirmed",
            reason="Explicit length profile in the normalized brief.",
            requested_minimum=requested_range[0],
            requested_maximum=requested_range[1],
        )

    defaults = dict(contract.task_profile.defaults or {})
    default_profile = _normalise_length_profile(defaults.get("length_profile"))
    if default_profile and default_profile != "adaptive":
        return _checked_length_profile(
            default_profile,
            supported=supported,
            source=requested_source if any(requested_range) else "skill_default",
            confidence="confirmed",
            reason="Fixed length profile declared by the selected skill.",
            requested_minimum=requested_range[0],
            requested_maximum=requested_range[1],
        )

    original_user_request = _original_user_request(inputs, brief)
    if not original_user_request and any(requested_range):
        original_user_request = (
            "Explicit structured document-wide range: "
            f"{requested_range[0]} to {requested_range[1]} information units."
        )
    if not original_user_request:
        return _fallback_length_profile(
            supported,
            requested_range=requested_range,
            source=(
                requested_source
                if any(requested_range)
                else "adaptive_default"
            ),
            reason=(
                "No authoritative original user request was available for "
                "descriptive length classification."
            ),
        )

    structured_call = getattr(llm_facade, "structured", None)
    if not callable(structured_call):
        return _fallback_length_profile(
            supported,
            requested_range=requested_range,
            source=requested_source if any(requested_range) else "runtime_fallback",
        )

    result = await structured_call(
        messages=[
            {
                "role": "user",
                "content": (
                    "Select the document length profile for this capability. "
                    "Use only the profiles declared by the skill contract. "
                    "Extract any explicit requested document-wide substantive "
                    "information-unit range from the brief. Use 0 when no bound "
                    "was requested. Do not invent a numeric range. Mark "
                    "request_basis as explicit_length_request only when the user "
                    "explicitly asks for a particular length, brevity, depth, or "
                    "numeric range. Completeness, audience, topic complexity, and "
                    "the number of requested sections are quality/scope signals, "
                    "not length requests. Use unspecified when no explicit length "
                    "intent exists; that case must use medium. "
                    "Return only the structured selection.\n\n"
                    f"Available profiles:\n{_json_preview(sorted(supported), limit=1000)}\n\n"
                    f"Skill default length_profile: "
                    f"{defaults.get('length_profile') or 'adaptive'}\n\n"
                    "Authoritative original user request (the only text whose "
                    "descriptive length intent may be classified):\n"
                    f"{original_user_request}"
                ),
            }
        ],
        output_schema={
            "title": "document_length_profile_selection",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "length_profile",
                "confidence",
                "reason",
                "request_basis",
                "requested_min_information_units",
                "requested_max_information_units",
            ],
            "properties": {
                "length_profile": {"type": "string", "enum": sorted(supported)},
                "confidence": {
                    "type": "string",
                    "enum": ["confirmed", "inferred"],
                },
                "reason": {"type": "string"},
                "request_basis": {
                    "type": "string",
                    "enum": ["explicit_length_request", "unspecified"],
                },
                "requested_min_information_units": {"type": "integer"},
                "requested_max_information_units": {"type": "integer"},
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
    request_basis = (
        str(structured.get("request_basis") or "unspecified").strip().lower()
        if isinstance(structured, Mapping)
        else "unspecified"
    )
    has_explicit_length_request = request_basis == "explicit_length_request"
    inferred_range = (
        _validated_requested_range(structured)
        if has_explicit_length_request
        else (0, 0)
    )
    requested_minimum, requested_maximum = (
        requested_range if any(requested_range) else inferred_range
    )
    if not any(requested_range) and not has_explicit_length_request:
        selected = "medium" if "medium" in supported else sorted(supported)[0]
    return _checked_length_profile(
        selected or "medium",
        supported=supported,
        source=(
            requested_source
            if any(requested_range)
            else "explicit_description"
            if has_explicit_length_request
            else "adaptive_default"
        ),
        confidence=(
            "confirmed"
            if any(requested_range) or has_explicit_length_request
            else "inferred"
        ),
        reason=(
            str(structured.get("reason") or "").strip()
            if isinstance(structured, Mapping)
            else ""
        ),
        requested_minimum=requested_minimum,
        requested_maximum=requested_maximum,
    )


def _fallback_length_profile(
    supported: set[str],
    *,
    requested_range: tuple[int, int] = (0, 0),
    source: str = "runtime_fallback",
    reason: str = "Structured length selection was unavailable.",
) -> dict[str, Any]:
    """Keep document delivery available if structured selection is unavailable."""
    return {
        "profile": "medium" if "medium" in supported else sorted(supported)[0],
        "source": source,
        "confidence": "confirmed" if any(requested_range) else "inferred",
        "reason": reason,
        "requested_min_information_units": requested_range[0],
        "requested_max_information_units": requested_range[1],
    }


def _normalise_length_profile(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw == "deep":
        return "long"
    return raw if raw in {"short", "medium", "long", "ultra", "adaptive"} else ""


def _declared_length_profile(source: Mapping[str, Any]) -> str:
    """Read the platform and SDK vocabularies without semantic re-inference."""
    for key in (
        "length_profile",
        "document_length_profile",
        "length_or_detail_level",
    ):
        profile = _normalise_length_profile(source.get(key))
        if profile:
            return profile
    raw_metadata = source.get("raw_metadata")
    if isinstance(raw_metadata, Mapping):
        for key in (
            "length_profile",
            "document_length_profile",
            "length_or_detail_level",
        ):
            profile = _normalise_length_profile(raw_metadata.get(key))
            if profile:
                return profile
        snapshot = raw_metadata.get("known_facts_snapshot")
        if isinstance(snapshot, Mapping):
            profile = _normalise_length_profile(
                snapshot.get("length_or_detail_level")
            )
            if profile:
                return profile
    return ""


def _json_preview(value: Any, *, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _checked_length_profile(
    profile: str,
    *,
    supported: set[str],
    source: str,
    confidence: str,
    reason: str,
    requested_minimum: int = 0,
    requested_maximum: int = 0,
) -> dict[str, Any]:
    if profile == "adaptive":
        profile = "medium"
    if profile not in supported:
        raise RuntimeError(f"unsupported_document_length_profile:{profile}")
    return {
        "profile": profile,
        "source": source,
        "confidence": confidence,
        "reason": reason,
        "requested_min_information_units": requested_minimum,
        "requested_max_information_units": requested_maximum,
    }


def _validated_requested_range(value: Any) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        return 0, 0
    try:
        minimum = max(0, int(value.get("requested_min_information_units") or 0))
        maximum = max(0, int(value.get("requested_max_information_units") or 0))
    except (TypeError, ValueError):
        return 0, 0
    if minimum and maximum and minimum > maximum:
        return 0, 0
    return minimum, maximum


def _explicit_requested_range(value: Mapping[str, Any]) -> tuple[int, int]:
    minimum = (
        value.get("requested_min_information_units")
        or value.get("min_information_units")
        or 0
    )
    maximum = (
        value.get("requested_max_information_units")
        or value.get("max_information_units")
        or 0
    )
    return _validated_requested_range(
        {
            "requested_min_information_units": minimum,
            "requested_max_information_units": maximum,
        }
    )


def _original_user_request(
    inputs: Mapping[str, Any],
    brief: Mapping[str, Any],
) -> str:
    """Return only a structurally identified original user utterance.

    Planner/reception briefs may add an inferred word range while normalizing
    quality requirements. Those projections are useful planning context, but
    they are not authoritative evidence that the user requested a long
    document.
    """
    authoritative_keys = (
        "raw_query",
        "raw_user_query",
        "original_user_request",
        "original_request",
        "user_prompt",
    )

    def find(value: Any) -> str:
        if isinstance(value, Mapping):
            for key in authoritative_keys:
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            for nested in value.values():
                candidate = find(nested)
                if candidate:
                    return candidate
        elif isinstance(value, (list, tuple)):
            for nested in value:
                candidate = find(nested)
                if candidate:
                    return candidate
        return ""

    return find(inputs) or find(brief)


__all__ = ["select_document_length_profile"]
