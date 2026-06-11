"""Manifest projection helpers for external agents.

These helpers deliberately avoid platform ontology or task-class inference.
They only project what an agent manifest explicitly declares.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def provided_artifacts_for_capability(
    manifest: Any,
    *,
    capability_id: str,
    artifact_type: str = "",
    structured_output: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Project manifest ``provides`` entries into final output metadata."""
    names: list[str] = []
    for entry in _capability_manifest_entries(manifest):
        entry_capability_id = _entry_value(entry, "capability_id")
        if entry_capability_id != capability_id:
            continue
        provides = _entry_value(entry, "provides") or ()
        if isinstance(provides, (list, tuple, set)):
            names.extend(str(item) for item in provides)
        break
    if artifact_type:
        names.append(str(artifact_type))

    payload = dict(structured_output or {})
    provided: dict[str, dict[str, Any]] = {}
    for name in names:
        artifact_name = str(name).strip()
        if not artifact_name:
            continue
        provided.setdefault(artifact_name, {"structured_output": payload})
    return provided


def _capability_manifest_entries(manifest: Any) -> list[Any] | tuple[Any, ...]:
    if isinstance(manifest, Mapping):
        entries = manifest.get("capability_manifest") or ()
    else:
        entries = getattr(manifest, "capability_manifest", ()) or ()
    return entries if isinstance(entries, (list, tuple)) else ()


def _entry_value(entry: Any, key: str) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(key)
    return getattr(entry, key, None)


__all__ = ["provided_artifacts_for_capability"]
