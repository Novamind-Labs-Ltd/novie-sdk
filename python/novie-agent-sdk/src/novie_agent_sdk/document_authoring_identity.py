"""Stable scoped identities for completion-oriented document artifacts."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def part_identity(
    *,
    scope: Mapping[str, Any],
    section_id: str,
    objective_digest: str,
    evidence_digest: str,
) -> str:
    """Derive stable identity without coupling it to plan revision."""
    identity_scope = {
        key: str(scope.get(key) or "")
        for key in (
            "tenant_id",
            "workspace_id",
            "workflow_id",
            "step_id",
            "capability_id",
        )
    }
    return _digest(
        {
            **identity_scope,
            "section_id": section_id,
            "objective_digest": objective_digest,
            "evidence_digest": evidence_digest,
        }
    )


__all__ = ["part_identity"]
