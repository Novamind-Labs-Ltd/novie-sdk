"""OpenAPI 3.x → Universal-provider ``capabilities.yaml`` generator.

Pure-Python entry points used by:
- W1 unit tests (this slice)
- W2 CLI ``novie providers generate-from-openapi``
- W3 CI drift gate ``novie providers diff``
- W4 first adopter (real PMS provider folder)

Design notes:

- **No business-semantics inference.** The generator never guesses
  ``risk_level`` from HTTP method ("POST = write" is wrong as often as
  it's right — POST /search is read-only, GET /actions/start is a
  side-effecting RPC in many wild APIs). Every operation must have a
  matching entry in ``semantics.yaml``.
- **$ref inlining bounded at depth 6.** Beyond that, the schema is
  emitted as ``{"$ref": "<original>"}`` so the resulting
  ``capabilities.yaml`` is self-describing for the conformance probes
  but does not expand cyclical types unbounded.
- **Components/schemas not exported.** Capability ``input_schema`` /
  ``output_schema`` are inlined per-capability so each entry is
  self-contained; that's how the existing handwritten provider folders
  shape their schemas (see ``tests/fixtures/providers/observability``).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .errors import (
    UnmappedOperationDescriptor,
    UnmappedOperationsError,
    UnsupportedOpenAPIError,
)
from .semantics import (
    SemanticsHints,
    OperationSemantics,
    TODO_MARKER,
)

_HTTP_METHODS = ("get", "post", "put", "patch", "delete")
_REF_DEPTH_LIMIT = 6

# Default caller affordances. Mirrors the defaults in
# ``novie_agent_sdk.provider_authoring._project_capability``.
_DEFAULT_CALLER_TYPES = ("reception", "planner", "executor")
_DEFAULT_CALLER_MODES = ("plan", "execute")


def generate_capabilities(
    openapi: dict[str, Any],
    semantics: SemanticsHints,
    *,
    provider_id: str,
    strict: bool = True,
) -> dict[str, Any]:
    """Translate an OpenAPI 3.x document + semantics sidecar into a
    ``capabilities.yaml``-shaped dict.

    Args:
        openapi: parsed OpenAPI document (3.0 or 3.1).
        semantics: parsed ``semantics.yaml`` hints.
        provider_id: dotted provider id used to namespace ``capability_id``
            when an operation's semantics entry doesn't override it.
        strict: when True (default), raise on unmapped operations or
            stub TODO placeholders. When False, unmapped ops are silently
            skipped — useful for the W2 ``--bootstrap`` flow where the
            sidecar is being authored.

    Returns:
        A dict shaped as ``{"capabilities": [...]}`` ready to be
        ``yaml.safe_dump``ed.

    Raises:
        UnmappedOperationsError: ``strict=True`` and one or more
            operationIds in OpenAPI lack a semantics entry.
        UnsupportedOpenAPIError: spec is not OpenAPI 3.x or has a
            structural problem the generator can't recover from.
    """
    _ensure_openapi_3(openapi)

    components_schemas = (openapi.get("components") or {}).get("schemas") or {}
    capabilities: list[dict[str, Any]] = []
    unmapped: list[UnmappedOperationDescriptor] = []
    unfilled_ops: list[UnmappedOperationDescriptor] = []

    for path, path_item in (openapi.get("paths") or {}).items():
        if not isinstance(path_item, dict):
            continue
        path_level_params = path_item.get("parameters") or []
        for method, op in path_item.items():
            if method.lower() not in _HTTP_METHODS:
                continue
            if not isinstance(op, dict):
                continue
            op_id = op.get("operationId")
            if not op_id:
                # Skip silently — operations without operationId are not
                # addressable in a stable way and shouldn't become a
                # capability. The OpenAPI author needs to add one.
                continue

            hint = semantics.find(str(op_id))
            if hint is None:
                unmapped.append(
                    UnmappedOperationDescriptor(
                        operation_id=str(op_id),
                        http_method=method.lower(),
                        http_path=str(path),
                    )
                )
                continue
            if hint.generator_skip:
                continue
            if hint.is_unfilled():
                unfilled_ops.append(
                    UnmappedOperationDescriptor(
                        operation_id=str(op_id),
                        http_method=method.lower(),
                        http_path=str(path),
                    )
                )
                continue

            cap = _build_capability(
                op_id=str(op_id),
                method=method.lower(),
                path=str(path),
                op=op,
                path_level_params=path_level_params,
                hint=hint,
                provider_id=provider_id,
                components_schemas=components_schemas,
            )
            capabilities.append(cap)

    if strict and (unmapped or unfilled_ops):
        raise UnmappedOperationsError(operations=unmapped + unfilled_ops)

    return {"capabilities": capabilities}


def generate_provider_files(
    openapi: dict[str, Any],
    semantics: SemanticsHints,
    *,
    provider_id: str,
    display_name: str | None = None,
    version: str = "0.1.0",
    strict: bool = True,
) -> dict[str, dict[str, Any]]:
    """Produce all three Universal-provider files at once.

    The output is a mapping of ``filename → parsed dict`` so callers
    (W2 CLI, tests) can decide whether to write to disk, diff against
    existing files, or feed straight into ``provider_conformance``.

    ``provider.yaml`` — base structure with ``transport.kind=openapi``
    and an ``auth`` placeholder block derived from the OpenAPI
    ``securitySchemes``. The integrator fills in the env-var name +
    declared scopes once.

    ``capabilities.yaml`` — generated; the workhorse output.

    ``resources.yaml`` — left to the integrator (capabilities.yaml's
    ``consumes_resources`` references resource type names that live
    here). We seed it with the union of ``resource_binding`` values
    seen in ``semantics.yaml`` so first-time scaffolds aren't blank.
    """
    capabilities = generate_capabilities(
        openapi, semantics, provider_id=provider_id, strict=strict
    )
    provider = _build_provider_yaml(
        openapi=openapi,
        provider_id=provider_id,
        display_name=display_name,
        version=version,
    )
    resources = _seed_resources_yaml(semantics)
    return {
        "provider.yaml": provider,
        "capabilities.yaml": capabilities,
        "resources.yaml": resources,
    }


# ---------------------------------------------------------------------------
# Capability projection
# ---------------------------------------------------------------------------


def _build_capability(
    *,
    op_id: str,
    method: str,
    path: str,
    op: dict[str, Any],
    path_level_params: list[Any],
    hint: OperationSemantics,
    provider_id: str,
    components_schemas: dict[str, Any],
) -> dict[str, Any]:
    capability_id = (
        hint.intent_label
        if hint.intent_label and hint.intent_label != TODO_MARKER
        else f"{provider_id}.{op_id}"
    )
    capability: dict[str, Any] = {
        "capability_id": capability_id,
        "kind": hint.kind or _default_kind_for_risk(hint.risk_level),
        "risk_level": hint.risk_level,
        "side_effect": hint.side_effect,
        "input_schema": _build_input_schema(
            op=op,
            path_level_params=path_level_params,
            components_schemas=components_schemas,
        ),
        "output_schema": _build_output_schema(op, components_schemas),
        "consumes_resources": list(hint.resource_binding),
        "caller_types": list(_DEFAULT_CALLER_TYPES),
        "caller_modes": list(_DEFAULT_CALLER_MODES),
        "routing_hints": _build_routing_hints(op, hint),
        "metadata": _build_metadata(op_id, method, path, op, hint),
    }
    if hint.confirmation_default:
        capability["confirmation_default"] = hint.confirmation_default
    if hint.dry_run_support and hint.dry_run_support != "none":
        capability["dry_run_support"] = hint.dry_run_support
    return capability


def _build_input_schema(
    *,
    op: dict[str, Any],
    path_level_params: list[Any],
    components_schemas: dict[str, Any],
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []

    # Path / query / header parameters. ``op.parameters`` overrides
    # path-level entries with the same (in, name) pair per the OpenAPI
    # spec, but for capability-input purposes each parameter is its own
    # named field — so simple union is sufficient.
    seen: set[tuple[str, str]] = set()
    for param in list(path_level_params) + list(op.get("parameters") or []):
        if not isinstance(param, dict):
            continue
        loc = str(param.get("in") or "")
        name = str(param.get("name") or "")
        if not name:
            continue
        key = (loc, name)
        if key in seen:
            continue
        seen.add(key)
        schema = _resolve_schema(
            param.get("schema") or {"type": "string"}, components_schemas
        )
        if param.get("description"):
            schema = {**schema, "description": str(param["description"])}
        properties[name] = schema
        if param.get("required"):
            required.append(name)

    # Request body — flatten application/json schema into the input.
    body = op.get("requestBody")
    if isinstance(body, dict):
        content = body.get("content") or {}
        json_content = content.get("application/json") or {}
        body_schema_raw = json_content.get("schema")
        if body_schema_raw is not None:
            body_schema = _resolve_schema(body_schema_raw, components_schemas)
            properties["body"] = body_schema
            if body.get("required"):
                required.append("body")

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = sorted(set(required))
    return schema


def _build_output_schema(
    op: dict[str, Any], components_schemas: dict[str, Any]
) -> dict[str, Any]:
    responses = op.get("responses") or {}
    success = (
        responses.get("200")
        or responses.get("201")
        or responses.get("default")
    )
    if not isinstance(success, dict):
        return {"type": "object", "properties": {}}
    content = success.get("content") or {}
    json_content = content.get("application/json") or {}
    schema_raw = json_content.get("schema")
    if schema_raw is None:
        return {"type": "object", "properties": {}}
    return _resolve_schema(schema_raw, components_schemas)


def _build_routing_hints(
    op: dict[str, Any], hint: OperationSemantics
) -> dict[str, Any]:
    summary = (op.get("summary") or "").strip()
    description = (op.get("description") or "").strip()
    when_to_use = hint.when_to_use or summary or description or None
    when_not_to_use = hint.when_not_to_use
    out: dict[str, Any] = {}
    if when_to_use:
        out["when_to_use"] = when_to_use
    if when_not_to_use:
        out["when_not_to_use"] = when_not_to_use
    return out


def _build_metadata(
    op_id: str,
    method: str,
    path: str,
    op: dict[str, Any],
    hint: OperationSemantics,
) -> dict[str, Any]:
    md: dict[str, Any] = {
        "openapi_operation_id": op_id,
        "openapi_method": method,
        "openapi_path": path,
        "generated_by": "novie_agent_sdk.openapi_provider",
    }
    tags = op.get("tags") or []
    if isinstance(tags, list) and tags:
        md["tags"] = [str(t) for t in tags]
    if hint.metadata:
        # Hand-authored metadata wins; useful for sample-call hints used
        # by the W3 conformance live-shape probe.
        md.update(hint.metadata)
    return md


# ---------------------------------------------------------------------------
# provider.yaml + resources.yaml seeding
# ---------------------------------------------------------------------------


def _build_provider_yaml(
    *,
    openapi: dict[str, Any],
    provider_id: str,
    display_name: str | None,
    version: str,
) -> dict[str, Any]:
    info = openapi.get("info") or {}
    servers = openapi.get("servers") or []
    base_url = ""
    if servers and isinstance(servers, list) and isinstance(servers[0], dict):
        base_url = str(servers[0].get("url") or "")

    auth_block = _derive_auth(openapi)

    provider: dict[str, Any] = {
        "id": provider_id,
        "type": "openapi",
        "display_name": display_name or info.get("title") or provider_id,
        "version": version,
        "transport": {
            "kind": "openapi",
            "base_url": base_url,
            "openapi_source_url": "",  # set by integrator for CI drift gate
        },
    }
    if auth_block is not None:
        provider["auth"] = auth_block
    return {"provider": provider}


def _derive_auth(openapi: dict[str, Any]) -> dict[str, Any] | None:
    """Project ``components.securitySchemes`` to a Novie auth template.

    Returns ``None`` when no security is declared so the integrator
    knows to add one explicitly. Multiple schemes → prefer the first
    bearer/oauth2 one; the integrator can hand-edit if a different
    scheme is required.
    """
    schemes = (
        (openapi.get("components") or {}).get("securitySchemes") or {}
    )
    if not isinstance(schemes, dict) or not schemes:
        return None
    # Stable iteration so generator output is deterministic.
    for scheme_name in sorted(schemes.keys()):
        scheme = schemes[scheme_name]
        if not isinstance(scheme, dict):
            continue
        scheme_type = str(scheme.get("type") or "").lower()
        if scheme_type == "http" and str(scheme.get("scheme") or "").lower() == "bearer":
            return {
                "type": "bearer_token",
                "token_env": _env_name_for(scheme_name),
                "scope_declared": [],
                "openapi_scheme_name": scheme_name,
            }
        if scheme_type == "apikey":
            return {
                "type": "api_key",
                "key_env": _env_name_for(scheme_name),
                "header_name": str(scheme.get("name") or ""),
                "openapi_scheme_name": scheme_name,
            }
        if scheme_type == "oauth2":
            return {
                "type": "oauth2_client_credentials",
                "token_env": _env_name_for(scheme_name),
                "scope_declared": [],
                "openapi_scheme_name": scheme_name,
            }
    return None


def _env_name_for(scheme_name: str) -> str:
    cleaned = "".join(c.upper() if c.isalnum() else "_" for c in scheme_name)
    return f"PROVIDER_{cleaned}_TOKEN"


def _seed_resources_yaml(semantics: SemanticsHints) -> dict[str, Any]:
    seen: set[str] = set()
    for entry in semantics.operations.values():
        if entry.generator_skip:
            continue
        for resource in entry.resource_binding:
            if resource:
                seen.add(resource)
    if not seen:
        return {"resource_types": []}
    return {"resource_types": sorted(seen)}


# ---------------------------------------------------------------------------
# $ref resolution
# ---------------------------------------------------------------------------


def _resolve_schema(
    schema: Any,
    components_schemas: dict[str, Any],
    *,
    depth: int = 0,
) -> dict[str, Any]:
    """Inline ``$ref`` references up to a bounded depth.

    Returns a plain dict suitable for embedding inside the capability
    schema. Beyond ``_REF_DEPTH_LIMIT`` we keep the ``$ref`` string in
    place so cyclic types don't blow the stack — the conformance probes
    treat unresolved refs as "ok" since they're explicit boundaries.
    """
    if not isinstance(schema, dict):
        return {"type": "string"}  # OpenAPI allows boolean schemas; we don't.
    if "$ref" in schema:
        if depth >= _REF_DEPTH_LIMIT:
            return {"$ref": str(schema["$ref"])}
        target = _follow_ref(schema["$ref"], components_schemas)
        if target is None:
            raise UnsupportedOpenAPIError(detail=f"unresolved $ref: {schema['$ref']!r}")
        return _resolve_schema(target, components_schemas, depth=depth + 1)
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            out["properties"] = {
                str(k): _resolve_schema(v, components_schemas, depth=depth + 1)
                for k, v in value.items()
            }
        elif key == "items":
            out["items"] = _resolve_schema(value, components_schemas, depth=depth + 1)
        elif key in ("oneOf", "anyOf", "allOf") and isinstance(value, list):
            out[key] = [
                _resolve_schema(v, components_schemas, depth=depth + 1) for v in value
            ]
        else:
            out[key] = value
    return out


def _follow_ref(ref: Any, components_schemas: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(ref, str) or not ref.startswith("#/components/schemas/"):
        return None
    name = ref[len("#/components/schemas/"):]
    target = components_schemas.get(name)
    if not isinstance(target, dict):
        return None
    return target


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _ensure_openapi_3(openapi: dict[str, Any]) -> None:
    version = str(openapi.get("openapi") or "")
    if version.startswith("3."):
        return
    if openapi.get("swagger"):
        raise UnsupportedOpenAPIError(
            detail="OpenAPI 2.0 / Swagger is not supported; convert to 3.0+ first"
        )
    raise UnsupportedOpenAPIError(
        detail=f"unrecognised spec version (got openapi={version!r}); 3.x required"
    )


def _default_kind_for_risk(risk_level: str) -> str:
    """Pick a Universal-capability ``kind`` consistent with the risk level.

    The Universal-provider validator allows ``query | command | workflow |
    stream | task``. For an OpenAPI HTTP request the natural mapping is:
    read → ``query``, write/dangerous → ``command``. Long-running async
    operations should set ``kind: workflow`` or ``kind: task`` explicitly
    in semantics.yaml so this default doesn't apply.
    """
    if risk_level == "read":
        return "query"
    return "command"


def operation_ids(openapi: dict[str, Any]) -> Iterable[str]:
    """Helper: list every ``operationId`` in the OpenAPI document.

    Used by the W2 CLI ``--bootstrap`` mode and by tests.
    """
    for path_item in (openapi.get("paths") or {}).values():
        if not isinstance(path_item, dict):
            continue
        for method, op in path_item.items():
            if method.lower() not in _HTTP_METHODS:
                continue
            if not isinstance(op, dict):
                continue
            op_id = op.get("operationId")
            if op_id:
                yield str(op_id)
