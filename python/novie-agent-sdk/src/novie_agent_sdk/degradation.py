"""Degradation tracking helpers for long-running external agents."""
from __future__ import annotations

from .platform_namespace import (
    CapabilityCallDiagnostics,
    DegradationKind,
    classify_envelope_error,
)


class DegradationTracker:
    """Per-run accumulator for symbolic degradation flags."""

    __slots__ = ("_flags",)

    def __init__(self) -> None:
        self._flags: list[str] = []

    def mark(self, prefix: str, kind: DegradationKind | str) -> None:
        if not prefix or not kind:
            return
        flag = f"{prefix}.{kind}"
        if flag not in self._flags:
            self._flags.append(flag)

    def mark_diagnostics(
        self,
        prefix: str,
        diagnostics: CapabilityCallDiagnostics | object,
    ) -> None:
        kind = getattr(diagnostics, "kind", None)
        ok = bool(getattr(diagnostics, "ok", False))
        if kind is None:
            return
        if ok and kind != "no_results":
            return
        self.mark(prefix, str(kind))

    def flags(self) -> list[str]:
        return list(self._flags)

    def has(self, flag_or_prefix: str) -> bool:
        if not flag_or_prefix:
            return False
        return any(
            flag == flag_or_prefix or flag.startswith(f"{flag_or_prefix}.")
            for flag in self._flags
        )


__all__ = [
    "CapabilityCallDiagnostics",
    "DegradationKind",
    "DegradationTracker",
    "classify_envelope_error",
]
