"""OpenAPI → Universal-provider folder generator.

This package translates an OpenAPI 3.0 / 3.1 spec into a Novie Universal
``capabilities.yaml`` document, paired with a hand-maintained
``semantics.yaml`` sidecar that supplies the judgmental fields OpenAPI
cannot provide (idempotency, resource binding, intent label, routing
hints).

Status: W1 of ``OPENAPI_PROVIDER_AUTOGEN_BACKLOG.md`` —
- ``generator.py``: pure-Python ``generate_capabilities`` entry point.
- ``semantics.py``: schema for the sidecar plus a single-pass discovery
  helper that lists every OpenAPI operation missing semantics rather
  than failing on the first one.
- ``errors.py``: structured exceptions consumers can ``except`` on
  (e.g. for the W2 CLI to render readable diagnostics).

Future slices add a CLI surface (W2), a CI drift gate (W3), and the
first real adopter — replacing ``pms_mock`` with a real ``pms`` provider
folder generated from the .NET PMS service's OpenAPI document (W4).
"""
from __future__ import annotations

from .errors import (
    GeneratorError,
    MissingSemanticsFieldError,
    UnmappedOperationDescriptor,
    UnmappedOperationsError,
    UnsupportedOpenAPIError,
)
from .generator import generate_capabilities, generate_provider_files, operation_ids
from .live_shape import (
    CapabilitySampleResult,
    FieldDrift,
    LiveShapeOutcome,
    build_live_shape_conformance_probe,
    diff_schemas,
    infer_json_schema,
    run_live_shape_probe,
)
from .semantics import (
    OperationSemantics,
    SemanticsHints,
    bootstrap_semantics_from_openapi,
    load_semantics,
)

__all__ = [
    "CapabilitySampleResult",
    "FieldDrift",
    "GeneratorError",
    "LiveShapeOutcome",
    "MissingSemanticsFieldError",
    "OperationSemantics",
    "SemanticsHints",
    "UnmappedOperationDescriptor",
    "UnmappedOperationsError",
    "UnsupportedOpenAPIError",
    "bootstrap_semantics_from_openapi",
    "build_live_shape_conformance_probe",
    "diff_schemas",
    "generate_capabilities",
    "generate_provider_files",
    "infer_json_schema",
    "load_semantics",
    "operation_ids",
    "run_live_shape_probe",
]
