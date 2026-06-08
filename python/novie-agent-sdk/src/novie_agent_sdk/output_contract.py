from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

UPSTREAM_HANDOFF = "upstream_handoff"
TERMINAL_DELIVERABLE = "terminal_deliverable"
DEFAULT_HANDOFF_MAX_BYTES = 12_000
MIN_HANDOFF_MAX_BYTES = 1_024
MAX_HANDOFF_MAX_BYTES = 64_000


@dataclass(frozen=True)
class StepRunPolicy:
    """Runtime policy derived from the platform step output contract.

    The workflow DAG decides whether a step is an internal handoff producer or
    a user-facing terminal deliverable. Agents should consume that decision as
    behavior, not only as metadata.
    """

    step_role: str
    output_contract: dict[str, Any]
    user_visible_content_stream: bool
    quality_loop_enabled: bool
    final_deliverable_enabled: bool
    max_output_bytes: int

    @property
    def is_upstream_handoff(self) -> bool:
        return self.step_role == UPSTREAM_HANDOFF

    @property
    def is_terminal_deliverable(self) -> bool:
        return self.step_role == TERMINAL_DELIVERABLE


def output_contract(inputs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the platform-provided output contract for this step.

    The platform may decide from the workflow DAG whether a step should produce
    a bounded upstream handoff or a user-visible final deliverable. Agents
    consume that decision through this SDK helper instead of re-parsing platform
    payload shapes themselves.
    """
    source = inputs if isinstance(inputs, Mapping) else {}
    direct = source.get("output_contract")
    if isinstance(direct, Mapping):
        return dict(direct)
    platform_context = source.get("platform_context")
    if isinstance(platform_context, Mapping):
        nested = platform_context.get("output_contract")
        if isinstance(nested, Mapping):
            return dict(nested)
    return {}


def resolve_step_role(inputs: Mapping[str, Any] | None) -> str:
    """Resolve this step's role from platform contract fields."""
    source = inputs if isinstance(inputs, Mapping) else {}
    contract = output_contract(source)
    value = str(contract.get("step_role") or "").strip()
    if value:
        return value
    kind = str(contract.get("kind") or "").strip()
    if kind == "bounded_handoff":
        return UPSTREAM_HANDOFF
    if kind == "final_deliverable":
        return TERMINAL_DELIVERABLE
    value = str(source.get("step_role") or "").strip()
    if value:
        return value
    platform_context = source.get("platform_context")
    if isinstance(platform_context, Mapping):
        return str(platform_context.get("step_role") or "").strip()
    return ""


def is_upstream_handoff(inputs: Mapping[str, Any] | None) -> bool:
    """True when the current step should emit a bounded downstream handoff."""
    return resolve_step_role(inputs) == UPSTREAM_HANDOFF


def is_terminal_deliverable(inputs: Mapping[str, Any] | None) -> bool:
    """True when the current step is expected to produce a final user artifact."""
    return resolve_step_role(inputs) == TERMINAL_DELIVERABLE


def handoff_max_bytes(inputs: Mapping[str, Any] | None) -> int:
    """Return the bounded handoff byte budget from the platform contract."""
    source = inputs if isinstance(inputs, Mapping) else {}
    contract = output_contract(source)
    raw = contract.get("max_bytes") or source.get("max_handoff_bytes")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_HANDOFF_MAX_BYTES
    return max(MIN_HANDOFF_MAX_BYTES, min(value, MAX_HANDOFF_MAX_BYTES))


def step_run_policy(inputs: Mapping[str, Any] | None) -> StepRunPolicy:
    """Return the agent runtime policy implied by the platform contract."""
    contract = output_contract(inputs)
    role = resolve_step_role(inputs)
    if role == UPSTREAM_HANDOFF:
        max_bytes = handoff_max_bytes(inputs)
        return StepRunPolicy(
            step_role=role,
            output_contract=contract,
            user_visible_content_stream=False,
            quality_loop_enabled=False,
            final_deliverable_enabled=False,
            max_output_bytes=max_bytes,
        )
    return StepRunPolicy(
        step_role=role or TERMINAL_DELIVERABLE,
        output_contract=contract,
        user_visible_content_stream=True,
        quality_loop_enabled=True,
        final_deliverable_enabled=True,
        max_output_bytes=0,
    )


def fit_text_to_utf8_bytes(
    value: str,
    *,
    max_bytes: int,
    marker: str = "\n\n[Handoff truncated to platform byte budget.]",
) -> str:
    """Trim text to a UTF-8 byte budget without splitting codepoints."""
    text = str(value or "")
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    marker_bytes = marker.encode("utf-8")
    budget = max(0, int(max_bytes) - len(marker_bytes))
    clipped = encoded[:budget]
    while clipped:
        try:
            return clipped.decode("utf-8").rstrip() + marker
        except UnicodeDecodeError:
            clipped = clipped[:-1]
    return marker.strip()


__all__ = [
    "DEFAULT_HANDOFF_MAX_BYTES",
    "MAX_HANDOFF_MAX_BYTES",
    "MIN_HANDOFF_MAX_BYTES",
    "TERMINAL_DELIVERABLE",
    "UPSTREAM_HANDOFF",
    "StepRunPolicy",
    "fit_text_to_utf8_bytes",
    "handoff_max_bytes",
    "is_terminal_deliverable",
    "is_upstream_handoff",
    "output_contract",
    "resolve_step_role",
    "step_run_policy",
]
