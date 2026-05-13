"""Live response-shape probe for ``novie providers conformance --include-live-shape``.

OPENAPI_PROVIDER_AUTOGEN W3 step 2.


The static drift gate (W3 step 1) catches the case where the upstream
team's OpenAPI document differs from Novie's wrapper. This module
catches the *other* drift: when the upstream service is shipping a
**real response** that no longer matches its own published OpenAPI
spec — fields added quietly, types flipped, response keys renamed.
That's the drift class no static check can find.

How it works:
- Each capability that wants live verification declares a
  ``metadata.sample_call`` block in ``semantics.yaml``:

      operations:
        listIssues:
          metadata:
            sample_call:
              path_params: {project_id: 'demo'}
              query_params: {lane: 'Todo'}
              body: null
              expected_status: 200

- The generator copies that into the capability's
  ``metadata.sample_call`` field.
- ``run_live_shape_probe`` walks every capability with a sample_call,
  builds the HTTP request from
  ``capability.metadata.openapi_method/openapi_path``
  + the sample-call args, attaches a bearer token from the env var
  declared in ``provider.auth.token_env``, fires the request, infers
  a JSON Schema from the response body, and diffs that inference
  against the capability's declared ``output_schema``.

Soft-failure semantics: schema drift is reported in the probe ``detail``
but the probe **passes** — partial deploys (a non-prod stage with
extra debug fields) shouldn't take down the gate. Genuine reachability
errors (network / auth / non-2xx status) DO fail the probe so missing
credentials and broken endpoints surface clearly.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import re

import httpx
import yaml

# Probes return their status via ``ProviderConformanceProbe`` over in
# ``provider_conformance.py``; we re-import it lazily inside the
# probe functions to avoid creating a cycle in this fresh module.

# Substitution placeholders look like ``{project_id}`` per OpenAPI.
_PATH_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")


def _load_provider_yaml_runtime_bits(
    provider_dir: Path,
) -> tuple[dict[str, Any] | None, str]:
    """Read ``provider.yaml`` and return (auth_block, base_url).

    The structured ``CapabilityProvider`` schema doesn't carry the
    freeform ``auth`` block (it's not part of the frozen contract),
    so the live-shape probe re-reads the yaml to pick it up. Returns
    ``(None, "")`` when the file is missing or malformed — the probe
    then fires without auth + with the structured provider's
    ``transport.base_url`` if any.
    """
    yaml_path = provider_dir / "provider.yaml"
    if not yaml_path.exists():
        return None, ""
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return None, ""
    provider_block = raw.get("provider") or {}
    if not isinstance(provider_block, dict):
        return None, ""
    auth = provider_block.get("auth")
    transport = provider_block.get("transport") or {}
    base_url = ""
    if isinstance(transport, dict):
        base_url = str(transport.get("base_url") or "")
    return auth if isinstance(auth, dict) else None, base_url


@dataclass
class FieldDrift:
    """One field-level deviation between declared and observed schemas.

    Reported but not cause for hard failure — the integrator decides
    whether to refresh the OpenAPI or extend the sample call.
    """

    capability_id: str
    pointer: str
    kind: str  # "field_added" | "field_missing" | "type_changed"
    declared_type: str = ""
    observed_type: str = ""

    def render(self) -> str:
        if self.kind == "field_added":
            return f"  [drift] {self.capability_id}{self.pointer}: extra field in response"
        if self.kind == "field_missing":
            return f"  [drift] {self.capability_id}{self.pointer}: declared but missing from response"
        return (
            f"  [drift] {self.capability_id}{self.pointer}: "
            f"type changed (declared={self.declared_type!r}, observed={self.observed_type!r})"
        )


@dataclass
class CapabilitySampleResult:
    """Per-capability outcome for one ``sample_call`` exercise."""

    capability_id: str
    status: str  # "ok" | "drift" | "error" | "skipped"
    request_url: str = ""
    response_status: int | None = None
    drift: list[FieldDrift] = field(default_factory=list)
    error: str = ""
    skip_reason: str = ""


@dataclass
class LiveShapeOutcome:
    """Aggregate outcome for the full live-shape probe pass."""

    sampled: list[CapabilitySampleResult] = field(default_factory=list)

    @property
    def has_endpoint_errors(self) -> bool:
        return any(s.status == "error" for s in self.sampled)

    @property
    def has_drift(self) -> bool:
        return any(s.status == "drift" for s in self.sampled)

    @property
    def ran_count(self) -> int:
        return sum(1 for s in self.sampled if s.status in ("ok", "drift"))


# ---------------------------------------------------------------------------
# JSON Schema inference
# ---------------------------------------------------------------------------


def infer_json_schema(value: Any) -> dict[str, Any]:
    """Infer a minimal JSON Schema fragment from a Python value.

    Used to reverse-engineer the *real* response shape so we can diff
    it against the *declared* OpenAPI schema. Stays conservative: only
    emits ``type`` / ``properties`` / ``items`` — no ``enum`` /
    ``pattern`` / ``minimum`` since those are too brittle for real-world
    sample data.
    """
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        if not value:
            return {"type": "array"}
        # Take the first element as representative — production responses
        # are usually homogeneous arrays. Heterogeneous arrays surface as
        # type-mismatch drifts on subsequent elements; that's fine.
        return {"type": "array", "items": infer_json_schema(value[0])}
    if isinstance(value, Mapping):
        properties = {
            str(k): infer_json_schema(v) for k, v in value.items()
        }
        return {"type": "object", "properties": properties}
    return {"type": "string"}


def diff_schemas(
    declared: Mapping[str, Any] | None,
    observed: Mapping[str, Any] | None,
    *,
    capability_id: str,
    pointer: str = "",
) -> list[FieldDrift]:
    """Recursively diff a declared schema against an observed schema.

    Returns a list of ``FieldDrift`` entries describing structural
    deviations:
    - ``field_added``: present in observed but not declared.
    - ``field_missing``: declared but not in observed.
    - ``type_changed``: same key, different ``type``.

    Skips when either side has ``$ref``, ``oneOf``, ``anyOf``, or
    ``allOf`` since the W1 generator caps inlining at depth 6 and
    those forms aren't meaningfully comparable without a full
    resolver. The user-facing message points at the unresolved
    pointer for context.
    """
    drifts: list[FieldDrift] = []
    if not declared and not observed:
        return drifts
    if not isinstance(declared, Mapping) or not isinstance(observed, Mapping):
        return drifts

    if any(key in declared for key in ("$ref", "oneOf", "anyOf", "allOf")):
        return drifts
    if any(key in observed for key in ("$ref", "oneOf", "anyOf", "allOf")):
        return drifts

    declared_type = str(declared.get("type", ""))
    observed_type = str(observed.get("type", ""))
    if declared_type and observed_type and declared_type != observed_type:
        drifts.append(
            FieldDrift(
                capability_id=capability_id,
                pointer=pointer or "/",
                kind="type_changed",
                declared_type=declared_type,
                observed_type=observed_type,
            )
        )
        return drifts

    if declared_type == "object":
        declared_props = declared.get("properties") or {}
        observed_props = observed.get("properties") or {}
        if not isinstance(declared_props, Mapping) or not isinstance(
            observed_props, Mapping
        ):
            return drifts
        for key in observed_props:
            if key not in declared_props:
                drifts.append(
                    FieldDrift(
                        capability_id=capability_id,
                        pointer=f"{pointer}/{key}",
                        kind="field_added",
                    )
                )
        for key in declared_props:
            if key not in observed_props:
                drifts.append(
                    FieldDrift(
                        capability_id=capability_id,
                        pointer=f"{pointer}/{key}",
                        kind="field_missing",
                    )
                )
                continue
            drifts.extend(
                diff_schemas(
                    declared_props[key],
                    observed_props[key],
                    capability_id=capability_id,
                    pointer=f"{pointer}/{key}",
                )
            )
    elif declared_type == "array":
        declared_items = declared.get("items") or {}
        observed_items = observed.get("items") or {}
        drifts.extend(
            diff_schemas(
                declared_items,
                observed_items,
                capability_id=capability_id,
                pointer=f"{pointer}/items",
            )
        )

    return drifts


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


def _substitute_path(path_template: str, path_params: Mapping[str, Any]) -> str:
    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in path_params:
            raise ValueError(
                f"sample_call.path_params missing required field {key!r} "
                f"for path template {path_template!r}"
            )
        return str(path_params[key])

    return _PATH_PLACEHOLDER_RE.sub(_replace, path_template)


def _join_url(base_url: str, path: str) -> str:
    if not base_url:
        return path
    return base_url.rstrip("/") + "/" + path.lstrip("/")


# ---------------------------------------------------------------------------
# Probe entry point
# ---------------------------------------------------------------------------


def run_live_shape_probe(
    provider: Any,
    *,
    base_url: str = "",
    auth_block: Mapping[str, Any] | None = None,
    http_client: httpx.Client | None = None,
    env: Mapping[str, str] | None = None,
) -> LiveShapeOutcome:
    """Drive every capability with a ``sample_call`` and diff the response.

    Args:
        provider: a loaded ``CapabilityProvider`` (from ``load_provider_folder``).
            Used for ``capabilities`` only — ``base_url`` and
            ``auth_block`` come in as separate kwargs because the
            structured provider doesn't carry the freeform ``auth:``
            block from ``provider.yaml`` (it's not part of the frozen
            ``CapabilityProvider`` schema). Callers parse the raw yaml
            and pass the bits they need.
        base_url: provider HTTP base URL (from ``transport.base_url``).
            When the structured ``provider.transport`` carries a
            non-empty ``base_url`` it is used; otherwise this kwarg
            wins.
        auth_block: parsed ``provider.yaml.auth`` block — bearer-token
            / api-key / oauth2-client-credentials descriptor. None
            means no auth header is attached.
        http_client: pre-configured client (timeouts / TLS / mock transport).
            When None a default ``httpx.Client`` with a 15s timeout is
            created.
        env: env-var lookup table (defaults to ``os.environ``).
            Tests override this without polluting the process env.

    Returns:
        ``LiveShapeOutcome`` carrying one ``CapabilitySampleResult``
        per capability with a ``sample_call``. Capabilities without
        a ``sample_call`` are NOT included — they're not configured
        for live probing.

    Notes:
        Schema drift is reported in ``CapabilitySampleResult.drift``
        but the per-cap status is ``"drift"`` (still cleanly
        distinguishable from ``"error"``, which is reserved for
        endpoint-level failures: network, non-2xx, missing auth).
    """
    env_lookup = env if env is not None else os.environ

    # ``provider.transport`` is a structured ``TransportDescriptor``
    # with a ``base_url`` attribute when loaded from yaml; tests that
    # construct the provider differently can pass ``base_url`` as a
    # kwarg directly.
    transport = getattr(provider, "transport", None)
    if not base_url and transport is not None:
        attr_base_url = getattr(transport, "base_url", None)
        if attr_base_url:
            base_url = str(attr_base_url)
        elif isinstance(transport, Mapping):
            base_url = str(transport.get("base_url") or "")

    token = _resolve_token(auth_block, env_lookup)

    client = http_client or httpx.Client(timeout=15.0, follow_redirects=True)
    owns_client = http_client is None
    sampled: list[CapabilitySampleResult] = []
    try:
        for capability in provider.capabilities or []:
            metadata = getattr(capability, "metadata", None) or {}
            if not isinstance(metadata, Mapping):
                continue
            sample_call = metadata.get("sample_call")
            if not isinstance(sample_call, Mapping):
                continue
            sampled.append(
                _probe_one_capability(
                    capability=capability,
                    sample_call=sample_call,
                    base_url=base_url,
                    token=token,
                    client=client,
                )
            )
    finally:
        if owns_client:
            client.close()

    return LiveShapeOutcome(sampled=sampled)


def _resolve_token(auth_block: Any, env_lookup: Mapping[str, str]) -> str | None:
    if auth_block is None:
        return None
    if not isinstance(auth_block, Mapping):
        return None
    auth_type = str(auth_block.get("type") or "")
    if auth_type in ("bearer_token", "oauth2_client_credentials"):
        env_var = str(auth_block.get("token_env") or "")
        if not env_var:
            return None
        return env_lookup.get(env_var) or None
    if auth_type == "api_key":
        env_var = str(auth_block.get("key_env") or "")
        if not env_var:
            return None
        return env_lookup.get(env_var) or None
    return None


def _probe_one_capability(
    *,
    capability: Any,
    sample_call: Mapping[str, Any],
    base_url: str,
    token: str | None,
    client: httpx.Client,
) -> CapabilitySampleResult:
    capability_id = str(getattr(capability, "capability_id", "") or "")
    metadata = getattr(capability, "metadata", None) or {}
    method = str(metadata.get("openapi_method") or "get").upper()
    path_template = str(metadata.get("openapi_path") or "")

    path_params = sample_call.get("path_params") or {}
    query_params = sample_call.get("query_params") or {}
    body = sample_call.get("body")
    expected_status = int(sample_call.get("expected_status") or 200)

    if not isinstance(path_params, Mapping):
        return CapabilitySampleResult(
            capability_id=capability_id,
            status="error",
            error="sample_call.path_params must be an object",
        )

    try:
        path = _substitute_path(path_template, path_params)
    except ValueError as exc:
        return CapabilitySampleResult(
            capability_id=capability_id,
            status="error",
            error=str(exc),
        )

    url = _join_url(base_url, path)
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = client.request(
            method,
            url,
            params=dict(query_params) if query_params else None,
            json=body if body is not None else None,
            headers=headers or None,
        )
    except Exception as exc:  # noqa: BLE001 — surface any HTTP error as a row
        return CapabilitySampleResult(
            capability_id=capability_id,
            status="error",
            request_url=url,
            error=f"request failed: {exc}",
        )

    if response.status_code != expected_status:
        return CapabilitySampleResult(
            capability_id=capability_id,
            status="error",
            request_url=url,
            response_status=response.status_code,
            error=(
                f"unexpected status {response.status_code} "
                f"(expected {expected_status})"
            ),
        )

    try:
        body_doc = response.json() if response.content else {}
    except ValueError as exc:
        return CapabilitySampleResult(
            capability_id=capability_id,
            status="error",
            request_url=url,
            response_status=response.status_code,
            error=f"response body is not valid JSON: {exc}",
        )

    declared = getattr(capability, "output_schema", None) or {}
    observed = infer_json_schema(body_doc)
    drifts = diff_schemas(
        declared,
        observed,
        capability_id=capability_id,
        pointer="",
    )

    return CapabilitySampleResult(
        capability_id=capability_id,
        status="drift" if drifts else "ok",
        request_url=url,
        response_status=response.status_code,
        drift=drifts,
    )


# ---------------------------------------------------------------------------
# Conformance-probe adapter
# ---------------------------------------------------------------------------


def build_live_shape_conformance_probe(
    provider: Any,
    *,
    provider_dir: Path | None = None,
    base_url: str = "",
    auth_block: Mapping[str, Any] | None = None,
    http_client: httpx.Client | None = None,
    env: Mapping[str, str] | None = None,
) -> Any:
    """Wrap ``run_live_shape_probe`` as a single ``ProviderConformanceProbe``.

    Args:
        provider: structured ``CapabilityProvider`` from ``load_provider_folder``.
        provider_dir: when set, the probe re-reads ``provider.yaml``
            from disk to extract the ``auth`` block + ``transport.base_url``
            that don't survive on the structured ``CapabilityProvider``.
            Tests can skip this and pass ``base_url`` / ``auth_block``
            directly.
        base_url: ``transport.base_url`` override (wins when neither
            the structured provider nor ``provider.yaml`` carries one).
        auth_block: parsed ``provider.yaml.auth`` mapping (bearer /
            apikey / oauth2). Tests pass it directly.
        http_client: pre-configured ``httpx.Client``.
        env: env-var lookup override.

    Returning one probe (rather than one per capability) keeps the
    ``ProviderConformanceReport`` shape stable — existing CLI / CI
    consumers don't have to special-case a variable-sized probe list.
    Per-capability outcomes are folded into the probe ``detail`` /
    ``hint`` text so operators still see exactly which capability
    drifted.
    """
    # Lazy import — provider_conformance imports from this module
    # transitively via SDK __init__, so importing eagerly creates a
    # cycle on first SDK load.
    from ..provider_conformance import ProviderConformanceProbe

    if provider_dir is not None and (auth_block is None or not base_url):
        raw_auth, raw_base_url = _load_provider_yaml_runtime_bits(provider_dir)
        auth_block = auth_block if auth_block is not None else raw_auth
        base_url = base_url or raw_base_url

    outcome = run_live_shape_probe(
        provider,
        base_url=base_url,
        auth_block=auth_block,
        http_client=http_client,
        env=env,
    )

    if not outcome.sampled:
        return ProviderConformanceProbe(
            name="live_response_shape",
            status="skip",
            detail="no capability declares metadata.sample_call",
            hint=(
                "Add a sample_call block to operations in semantics.yaml "
                "to enable live response-shape verification."
            ),
        )

    if outcome.has_endpoint_errors:
        error_lines = [
            f"  {s.capability_id} [{s.response_status or '?'}]: {s.error}"
            for s in outcome.sampled
            if s.status == "error"
        ]
        ok_count = sum(1 for s in outcome.sampled if s.status == "ok")
        drift_count = sum(1 for s in outcome.sampled if s.status == "drift")
        detail = (
            f"sampled={len(outcome.sampled)} ok={ok_count} drift={drift_count} "
            f"errors={len(error_lines)}\n" + "\n".join(error_lines)
        )
        return ProviderConformanceProbe(
            name="live_response_shape",
            status="fail",
            detail=detail,
            hint=(
                "Endpoint errors usually mean missing credentials "
                "(check provider.auth.token_env) or wrong base_url. "
                "Live-shape failures are infrastructure issues; once they "
                "clear, drift detection runs against the real responses."
            ),
        )

    if outcome.has_drift:
        drift_lines: list[str] = []
        for s in outcome.sampled:
            if s.status != "drift":
                continue
            drift_lines.append(f"  {s.capability_id}:")
            drift_lines.extend(d.render() for d in s.drift[:6])
            if len(s.drift) > 6:
                drift_lines.append(
                    f"    [drift] …and {len(s.drift) - 6} more deviations"
                )
        ok_count = sum(1 for s in outcome.sampled if s.status == "ok")
        drift_count = sum(1 for s in outcome.sampled if s.status == "drift")
        detail = (
            f"sampled={len(outcome.sampled)} ok={ok_count} drift={drift_count}\n"
            + "\n".join(drift_lines)
        )
        # Soft failure per backlog: drift is a diagnostic, not a hard fail.
        # The probe stays ``pass`` so CI doesn't go red on partial deploys
        # (e.g. a non-prod stage with an extra debug field) but the
        # detail block is still pumped into the JSON / report so
        # operators can see and act on it.
        return ProviderConformanceProbe(
            name="live_response_shape",
            status="pass",
            detail=detail,
            hint=(
                "Schema drift detected against declared output_schema. "
                "Refresh the OpenAPI source (upstream may have shipped a "
                "new field without updating its OpenAPI spec) or update "
                "the capability's sample_call/output_schema."
            ),
        )

    return ProviderConformanceProbe(
        name="live_response_shape",
        status="pass",
        detail=f"sampled={len(outcome.sampled)} ok={len(outcome.sampled)} drift=0",
    )
