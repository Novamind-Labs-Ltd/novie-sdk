"""Schema + loader for the ``semantics.yaml`` sidecar.

OpenAPI carries the *mechanical* contract (operations, parameters,
schemas, auth schemes); the sidecar carries the *judgmental* fields
Novie's capability layer needs but OpenAPI cannot express:

- ``intent_label``         — Reception/Planner-facing dotted id (e.g. ``pms.issue.move_lane``)
- ``risk_level``           — ``read`` / ``write`` / ``dangerous`` (no auto-inference: POST is not always a write)
- ``side_effect``          — ``none`` / ``session`` / ``tenant`` / ``external`` / ``irreversible``
- ``resource_binding``     — list of ResourceGraph nodes the operation consumes
- ``confirmation_default`` — ``optional`` / ``required`` / ``gated`` (required for ``write`` / ``dangerous``)
- ``dry_run_support``      — ``none`` / ``preview`` / ``planner_eval`` (alternative to confirmation gate)
- ``when_to_use`` / ``when_not_to_use`` — routing-hint prose for Reception/Planner
- ``generator_skip``       — explicitly don't expose this operation

The loader is lenient: extra keys are preserved (so future fields don't
break old generators) but unknown values for typed fields are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

# Mirror of ``novie_protocol.contracts.universal_capability._KNOWN_*`` —
# the schema enforced by ``novie providers validate``. We don't import the
# protocol package directly so the generator stays runnable in a stripped
# SDK install (CI, novie-cli scaffold), but the values must match the
# validator literals exactly so generated YAML round-trips through it.
ALLOWED_RISK_LEVELS = frozenset({"read", "write", "dangerous"})
ALLOWED_SIDE_EFFECTS = frozenset(
    {"none", "session", "tenant", "external", "irreversible"}
)
ALLOWED_CONFIRMATION = frozenset({"auto", "required", "gated"})
ALLOWED_DRY_RUN = frozenset(
    {
        "none",
        "preview_only",
        "preview_with_diff",
        "preview_with_side_effect_simulation",
    }
)
ALLOWED_KINDS = frozenset({"query", "command", "workflow", "stream", "task"})

# Sentinel string that ``bootstrap_semantics_from_openapi`` writes into
# the judgmental fields so the generator immediately fails strict mode
# until a human edits the file. Keeps the placeholder consistent across
# the generator, the CLI, and CI diagnostics.
TODO_MARKER = "__TODO__"


@dataclass
class OperationSemantics:
    """Sidecar entry for one OpenAPI operation."""

    intent_label: str
    risk_level: str
    side_effect: str = "external"
    resource_binding: tuple[str, ...] = ()
    confirmation_default: str | None = None
    dry_run_support: str = "none"
    when_to_use: str | None = None
    when_not_to_use: str | None = None
    kind: str | None = None  # auto-derived from risk_level when unset

    generator_skip: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_unfilled(self) -> bool:
        """``True`` when the bootstrap stub still has TODO placeholders.

        Strict-mode generation fails on these so CI catches them before a
        half-filled folder is committed.
        """
        if self.generator_skip:
            return False
        if self.intent_label == TODO_MARKER or self.risk_level == TODO_MARKER:
            return True
        return False


@dataclass
class SemanticsHints:
    """The full ``semantics.yaml`` content keyed by ``operationId``."""

    operations: dict[str, OperationSemantics] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    """Alias map ``old_operation_id`` → ``new_operation_id`` so a rename in
    OpenAPI doesn't trip the strict mapping check during a transition."""

    def find(self, operation_id: str) -> OperationSemantics | None:
        if operation_id in self.operations:
            return self.operations[operation_id]
        aliased = self.aliases.get(operation_id)
        if aliased:
            return self.operations.get(aliased)
        return None

    def known_operation_ids(self) -> Iterable[str]:
        return self.operations.keys()


def load_semantics(path: Path | str | dict[str, Any]) -> SemanticsHints:
    """Load a semantics sidecar.

    Accepts a Path, a string path, or a pre-parsed dict (used by tests).
    Validates that typed-field values are in the allowed sets.
    """
    if isinstance(path, dict):
        raw = path
    else:
        text = Path(path).read_text(encoding="utf-8")
        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"semantics.yaml must be a mapping at top level (got {type(loaded).__name__})")
        raw = loaded

    operations_raw = raw.get("operations") or {}
    if not isinstance(operations_raw, dict):
        raise ValueError("semantics.yaml `operations` must be a mapping")
    operations: dict[str, OperationSemantics] = {}
    for op_id, entry in operations_raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"semantics for {op_id!r} must be a mapping")
        operations[str(op_id)] = _coerce_entry(str(op_id), entry)

    aliases_raw = raw.get("aliases") or {}
    if not isinstance(aliases_raw, dict):
        raise ValueError("semantics.yaml `aliases` must be a mapping")
    aliases = {str(k): str(v) for k, v in aliases_raw.items()}

    return SemanticsHints(operations=operations, aliases=aliases)


def _coerce_entry(op_id: str, entry: dict[str, Any]) -> OperationSemantics:
    intent_label = str(entry.get("intent_label") or TODO_MARKER)
    risk_level = str(entry.get("risk_level") or TODO_MARKER)
    side_effect = str(entry.get("side_effect") or "external")
    confirmation_default = entry.get("confirmation_default")
    dry_run_support = str(entry.get("dry_run_support") or "none")
    kind_raw = entry.get("kind")
    kind = str(kind_raw) if kind_raw else None
    generator_skip = bool(entry.get("generator_skip") or False)

    resource_binding_raw = entry.get("resource_binding") or []
    if not isinstance(resource_binding_raw, (list, tuple)):
        raise ValueError(
            f"semantics for {op_id!r}: resource_binding must be a list of resource type strings"
        )
    resource_binding = tuple(str(r) for r in resource_binding_raw)

    if not generator_skip:
        if risk_level not in ALLOWED_RISK_LEVELS and risk_level != TODO_MARKER:
            raise ValueError(
                f"semantics for {op_id!r}: risk_level must be one of "
                f"{sorted(ALLOWED_RISK_LEVELS)} (got {risk_level!r})"
            )
        if side_effect not in ALLOWED_SIDE_EFFECTS:
            raise ValueError(
                f"semantics for {op_id!r}: side_effect must be one of "
                f"{sorted(ALLOWED_SIDE_EFFECTS)} (got {side_effect!r})"
            )
        if confirmation_default is not None and confirmation_default not in ALLOWED_CONFIRMATION:
            raise ValueError(
                f"semantics for {op_id!r}: confirmation_default must be one of "
                f"{sorted(ALLOWED_CONFIRMATION)} (got {confirmation_default!r})"
            )
        if dry_run_support not in ALLOWED_DRY_RUN:
            raise ValueError(
                f"semantics for {op_id!r}: dry_run_support must be one of "
                f"{sorted(ALLOWED_DRY_RUN)} (got {dry_run_support!r})"
            )
        if kind is not None and kind not in ALLOWED_KINDS:
            raise ValueError(
                f"semantics for {op_id!r}: kind must be one of "
                f"{sorted(ALLOWED_KINDS)} (got {kind!r})"
            )

    return OperationSemantics(
        intent_label=intent_label,
        risk_level=risk_level,
        side_effect=side_effect,
        resource_binding=resource_binding,
        confirmation_default=str(confirmation_default) if confirmation_default else None,
        dry_run_support=dry_run_support,
        when_to_use=_optional_str(entry.get("when_to_use")),
        when_not_to_use=_optional_str(entry.get("when_not_to_use")),
        kind=kind,
        generator_skip=generator_skip,
        metadata=dict(entry.get("metadata") or {}),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def bootstrap_semantics_from_openapi(openapi: dict[str, Any]) -> dict[str, Any]:
    """Produce a ``semantics.yaml``-shaped dict pre-populated with TODO stubs.

    Used by the W2 ``--bootstrap`` CLI flag so first-time integrators get
    a file with one entry per operation and only have to fill in the
    judgmental fields (intent_label, risk_level, resource_binding, …).
    """
    operations: dict[str, dict[str, Any]] = {}
    for path, methods in (openapi.get("paths") or {}).items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method.lower() not in ("get", "post", "put", "patch", "delete"):
                continue
            if not isinstance(op, dict):
                continue
            op_id = op.get("operationId")
            if not op_id:
                continue
            operations[str(op_id)] = {
                "intent_label": TODO_MARKER,
                "risk_level": TODO_MARKER,
                "resource_binding": [],
                "side_effect": "external",
                # ``kind`` left unset; the generator derives it from
                # ``risk_level`` (read → query, write/dangerous → command)
                # once the integrator fills the sidecar.
                "when_to_use": op.get("summary") or op.get("description") or "",
                "metadata": {
                    "openapi_method": method.lower(),
                    "openapi_path": path,
                },
            }
    return {"operations": operations, "aliases": {}}
