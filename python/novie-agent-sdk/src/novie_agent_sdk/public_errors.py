"""Stable, prompt-safe errors exposed by the agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any


@dataclass(frozen=True, slots=True)
class PublicErrorFields:
    error_code: str
    public_message: str


class PublicAgentError(RuntimeError):
    """An intentional public failure with no raw-cause serialization."""

    def __init__(
        self,
        *,
        error_code: str,
        public_message: str,
        retryable: bool = False,
        replan_eligible: bool = False,
        repair_eligible: bool = False,
    ) -> None:
        super().__init__(public_message)
        self.error_code = error_code
        self.public_message = public_message
        self.retryable = retryable
        self.replan_eligible = replan_eligible
        self.repair_eligible = repair_eligible


_SAFE_ENVELOPE_MESSAGES = {
    "agent_internal_error": "Agent execution failed.",
    "agent_finalize_failed": "Document finalization failed.",
    "artifact_contract_violation": (
        "The generated artifact did not satisfy its required structure."
    ),
    "provider_timeout": "Document finalization timed out.",
    "sectioned_authoring_llm_failed": "Document finalization failed.",
    "sectioned_authoring_finalize_timeout": "Document finalization timed out.",
}


def public_error_fields(error: BaseException) -> PublicErrorFields:
    if isinstance(error, PublicAgentError):
        return PublicErrorFields(error.error_code, error.public_message)
    return PublicErrorFields("agent_internal_error", "Agent execution failed.")


def public_error_fields_from_envelope(payload: Mapping[str, Any]) -> PublicErrorFields:
    """Return only an allow-listed public error for handler-provided envelopes."""
    code = str(payload.get("error_code") or "")
    message = _SAFE_ENVELOPE_MESSAGES.get(code)
    if message is None:
        return PublicErrorFields("agent_internal_error", "Agent execution failed.")
    return PublicErrorFields(code, message)
