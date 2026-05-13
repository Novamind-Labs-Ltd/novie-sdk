"""Structured exceptions raised by the OpenAPI → capabilities generator.

The generator is invoked both from Python tests and from the W2 CLI;
each consumer wants different rendering, so we expose plain dataclasses
on the exceptions and let callers format the messages.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UnmappedOperationDescriptor:
    """One OpenAPI operation that has no entry in ``semantics.yaml``."""

    operation_id: str
    http_method: str
    http_path: str


class GeneratorError(Exception):
    """Base class for all generator-emitted errors."""


@dataclass
class UnmappedOperationsError(GeneratorError):
    """Raised when ``--strict`` mode finds unmapped operations.

    Carries the full list so the CLI / CI can surface a single readable
    diagnostic instead of forcing the operator to re-run for each
    missing entry.
    """

    operations: list[UnmappedOperationDescriptor] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__(self._render())

    def _render(self) -> str:
        if not self.operations:
            return "no unmapped operations"
        lines = [
            f"{len(self.operations)} OpenAPI operation(s) missing from semantics.yaml:",
        ]
        for op in self.operations:
            lines.append(
                f"  - {op.operation_id}  ({op.http_method.upper()} {op.http_path})"
            )
        lines.append(
            "Add an entry per operation under `operations:` in semantics.yaml, "
            "or set `generator_skip: true` if the operation is intentionally "
            "not exposed as a Novie capability."
        )
        return "\n".join(lines)


@dataclass
class MissingSemanticsFieldError(GeneratorError):
    """A specific operation has its semantics entry but a required field
    (e.g. ``risk_level`` for a non-GET op) is unset."""

    operation_id: str
    field_name: str
    reason: str = ""

    def __post_init__(self) -> None:
        msg = f"semantics for {self.operation_id!r} is missing required field {self.field_name!r}"
        if self.reason:
            msg = f"{msg}: {self.reason}"
        super().__init__(msg)


@dataclass
class UnsupportedOpenAPIError(GeneratorError):
    """An OpenAPI feature the generator does not yet model.

    Common cases: OpenAPI 2.0 (Swagger) input, deeply nested ``oneOf``
    references that exceed our resolution depth, ``$ref`` to a missing
    component. The author can fix by upgrading the spec or marking the
    affected op with ``generator_skip: true``.
    """

    detail: str = ""

    def __post_init__(self) -> None:
        super().__init__(self.detail or "unsupported OpenAPI construct")
