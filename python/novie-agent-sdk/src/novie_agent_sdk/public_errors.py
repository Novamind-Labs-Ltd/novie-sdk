"""Stable, prompt-safe errors exposed by the agent runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PublicErrorFields:
    error_code: str
    public_message: str


class PublicAgentError(RuntimeError):
    """An intentional public failure with no raw-cause serialization."""

    def __init__(self, *, error_code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.error_code = error_code
        self.public_message = public_message


def public_error_fields(error: BaseException) -> PublicErrorFields:
    if isinstance(error, PublicAgentError):
        return PublicErrorFields(error.error_code, error.public_message)
    return PublicErrorFields("agent_internal_error", "Agent execution failed.")
