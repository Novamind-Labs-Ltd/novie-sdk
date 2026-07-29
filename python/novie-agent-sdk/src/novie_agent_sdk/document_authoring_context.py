"""ADR-115 deterministic Authoring Context Envelope construction."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .context_budget import estimated_tokens


class AuthoringContextOverflow(RuntimeError):
    code = "document_authoring_context_exceeded"


@dataclass(frozen=True, slots=True)
class ContextSlot:
    name: str
    content: str
    required: bool
    estimated_tokens: int


@dataclass(frozen=True, slots=True)
class AuthoringContextEnvelope:
    schema: str
    maximum_input_tokens: int
    estimated_input_tokens: int
    pressure: str
    slots: tuple[ContextSlot, ...]
    decisions: tuple[str, ...] = ()

    def render(self) -> str:
        return "\n\n".join(
            f"[{slot.name}]\n{slot.content}" for slot in self.slots if slot.content
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "slots": [
                {
                    "name": slot.name,
                    "required": slot.required,
                    "estimated_tokens": slot.estimated_tokens,
                }
                for slot in self.slots
            ],
        }


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _bounded(value: Any, token_limit: int) -> str:
    text = _text(value)
    if token_limit <= 0 or estimated_tokens(text) <= token_limit:
        return text
    return text[: max(1, token_limit * 4)].rstrip()


def _slot(name: str, value: Any, *, required: bool, limit: int = 0) -> ContextSlot:
    content = _bounded(value, limit)
    return ContextSlot(name, content, required, estimated_tokens(content))


def build_authoring_context_envelope(
    *,
    context_budget: Mapping[str, Any],
    authoring_contract: Any,
    objective: Any,
    required_evidence: Any,
    coverage_cursor: Any,
    previous_part_handoff: Any = "",
    authoring_summary: Any = "",
    optional_evidence: Any = "",
) -> AuthoringContextEnvelope:
    """Build a prompt envelope without provider-side silent truncation."""
    # Match the SDK's authoritative default when callers provide only an output
    # ceiling. Treating a partial budget as a tiny 12k model incorrectly makes
    # a large output reservation consume the entire context window.
    max_context = _positive_int(context_budget.get("max_input_tokens"), 180_000)
    reserved_output = _positive_int(context_budget.get("max_output_tokens"), 2_000)
    schema_headroom = _positive_int(
        context_budget.get("structured_output_headroom_tokens"),
        512,
    )
    safety_margin = _positive_int(context_budget.get("context_safety_margin_tokens"), 512)
    maximum_input = max_context - reserved_output - schema_headroom - safety_margin
    if maximum_input <= 0:
        raise AuthoringContextOverflow(
            "document_authoring_context_exceeded: no input budget remains after output reservation"
        )
    required_evidence_text = _text(required_evidence)
    required_evidence_slot = _slot(
        "required_evidence",
        required_evidence_text,
        required=True,
        limit=_positive_int(
            context_budget.get("max_required_evidence_tokens"),
            4_000,
        ),
    )
    slots = [
        _slot("authoring_contract", authoring_contract, required=True),
        _slot("current_objective", objective, required=True),
        required_evidence_slot,
        _slot("coverage_cursor", coverage_cursor, required=True),
        _slot(
            "previous_part_handoff",
            previous_part_handoff,
            required=False,
            limit=_positive_int(context_budget.get("max_part_handoff_tokens"), 400),
        ),
        _slot(
            "authoring_context_summary",
            authoring_summary,
            required=False,
            limit=_positive_int(context_budget.get("running_summary_max_tokens"), 800),
        ),
        _slot("optional_evidence", optional_evidence, required=False),
    ]
    decisions: list[str] = []
    if required_evidence_slot.estimated_tokens < estimated_tokens(
        required_evidence_text
    ):
        decisions.append("bound_required_evidence")

    def total() -> int:
        return sum(item.estimated_tokens for item in slots if item.content)

    if total() > maximum_input and slots[-1].content:
        slots[-1] = _slot("optional_evidence", "", required=False)
        decisions.append("drop_optional_evidence")
    if total() > maximum_input and slots[4].content:
        slots[4] = _slot(
            "previous_part_handoff",
            slots[4].content,
            required=False,
            limit=max(64, slots[4].estimated_tokens // 2),
        )
        decisions.append("reduce_previous_part_handoff")
    if total() > maximum_input and slots[5].content:
        slots[5] = _slot(
            "authoring_context_summary",
            slots[5].content,
            required=False,
            limit=max(128, slots[5].estimated_tokens // 2),
        )
        decisions.append("reduce_authoring_summary")
    required_tokens = sum(item.estimated_tokens for item in slots if item.required)
    if required_tokens > maximum_input or total() > maximum_input:
        raise AuthoringContextOverflow(
            "document_authoring_context_exceeded: minimum required envelope exceeds provider limit"
        )
    used = total()
    ratio = used / maximum_input
    pressure = "hard" if ratio >= 0.9 else "soft" if ratio >= 0.7 else "normal"
    return AuthoringContextEnvelope(
        schema="authoring-context-envelope.v1",
        maximum_input_tokens=maximum_input,
        estimated_input_tokens=used,
        pressure=pressure,
        slots=tuple(slots),
        decisions=tuple(decisions),
    )


def deterministic_authoring_summary(
    *,
    covered_objectives: Sequence[str],
    unresolved_objectives: Sequence[str],
    evidence_refs: Sequence[Mapping[str, Any]],
    continuity_notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Lossy navigation state; canonical text always remains behind refs."""
    return {
        "schema": "authoring-context-summary.v1",
        "profile": "authoring_context_v1:deterministic",
        "covered_objectives": list(dict.fromkeys(covered_objectives)),
        "unresolved_objectives": list(dict.fromkeys(unresolved_objectives)),
        "established_facts": [],
        "terminology": [],
        "decisions": [],
        "unresolved_questions": [],
        "continuity_notes": list(dict.fromkeys(continuity_notes)),
        "evidence_refs": [dict(item) for item in evidence_refs],
        "previous_part_handoff": continuity_notes[-1] if continuity_notes else "",
        "mode": "deterministic",
    }


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


__all__ = [
    "AuthoringContextEnvelope",
    "AuthoringContextOverflow",
    "ContextSlot",
    "build_authoring_context_envelope",
    "deterministic_authoring_summary",
]
