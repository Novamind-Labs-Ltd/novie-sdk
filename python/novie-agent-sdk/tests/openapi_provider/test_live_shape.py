"""W3 step 2 — live response-shape probe unit tests.

Drives ``run_live_shape_probe`` and ``build_live_shape_conformance_probe``
against an in-memory ``httpx.MockTransport`` so we never make real
network calls. Covers:

- JSON Schema inference for objects, arrays, primitives, null.
- Schema diff: field added in response, field missing from response,
  type changed.
- Probe success when sample_call's response matches output_schema.
- Probe drift when response carries an extra field.
- Probe error when endpoint returns 500 / non-2xx.
- Probe error when bearer-token env var is set in provider.yaml but
  not present in the env lookup.
- Capabilities without ``sample_call`` are dropped from the outcome
  (not surfaced as errors).
- Conformance probe wrapper status mapping (skip / pass / fail).
- Soft-failure semantics: drift surfaces in detail but probe stays
  ``pass`` per the backlog contract.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from novie_agent_sdk.openapi_provider import (
    FieldDrift,
    build_live_shape_conformance_probe,
    diff_schemas,
    infer_json_schema,
    run_live_shape_probe,
)
from novie_agent_sdk.provider_authoring import load_provider_folder


SDK_FIXTURE_DIR = (
    Path(__file__).parent.parent / "fixtures" / "openapi"
)


# ---------------------------------------------------------------------------
# Inference + diff (pure functions)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected_type",
    [
        (None, "null"),
        (True, "boolean"),
        (1, "integer"),
        (1.5, "number"),
        ("hello", "string"),
    ],
)
def test_infer_json_schema_primitive_types(value: Any, expected_type: str) -> None:
    assert infer_json_schema(value) == {"type": expected_type}


def test_infer_json_schema_array_uses_first_element() -> None:
    schema = infer_json_schema([{"id": "a"}, {"id": "b"}])
    assert schema == {
        "type": "array",
        "items": {"type": "object", "properties": {"id": {"type": "string"}}},
    }


def test_infer_json_schema_empty_array_omits_items() -> None:
    assert infer_json_schema([]) == {"type": "array"}


def test_infer_json_schema_nested_object() -> None:
    schema = infer_json_schema(
        {"id": "x", "labels": ["a", "b"], "owner": {"name": "alice"}}
    )
    assert schema == {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "labels": {"type": "array", "items": {"type": "string"}},
            "owner": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        },
    }


def test_diff_schemas_detects_field_added_in_response() -> None:
    declared = {"type": "object", "properties": {"id": {"type": "string"}}}
    observed = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "priority_v2": {"type": "integer"},  # extra field
        },
    }
    drifts = diff_schemas(declared, observed, capability_id="x.y")
    assert len(drifts) == 1
    assert drifts[0].kind == "field_added"
    assert drifts[0].pointer == "/priority_v2"


def test_diff_schemas_detects_field_missing_from_response() -> None:
    declared = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
        },
    }
    observed = {"type": "object", "properties": {"id": {"type": "string"}}}
    drifts = diff_schemas(declared, observed, capability_id="x.y")
    assert len(drifts) == 1
    assert drifts[0].kind == "field_missing"
    assert drifts[0].pointer == "/title"


def test_diff_schemas_detects_type_change() -> None:
    declared = {"type": "object", "properties": {"id": {"type": "string"}}}
    observed = {"type": "object", "properties": {"id": {"type": "integer"}}}
    drifts = diff_schemas(declared, observed, capability_id="x.y")
    assert len(drifts) == 1
    assert drifts[0].kind == "type_changed"
    assert drifts[0].declared_type == "string"
    assert drifts[0].observed_type == "integer"


def test_diff_schemas_no_false_positive_when_aligned() -> None:
    schema = {
        "type": "object",
        "properties": {"id": {"type": "string"}, "lane": {"type": "string"}},
    }
    drifts = diff_schemas(schema, schema, capability_id="x.y")
    assert drifts == []


def test_diff_schemas_recurses_into_nested_arrays() -> None:
    declared = {
        "type": "object",
        "properties": {
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                },
            }
        },
    }
    observed = {
        "type": "object",
        "properties": {
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "archived": {"type": "boolean"},
                    },
                },
            }
        },
    }
    drifts = diff_schemas(declared, observed, capability_id="x.y")
    assert len(drifts) == 1
    assert drifts[0].kind == "field_added"
    assert drifts[0].pointer == "/issues/items/archived"


def test_diff_schemas_skips_unresolved_refs() -> None:
    """W1's generator caps $ref inlining at depth 6; deeper refs come
    through as ``{"$ref": ...}`` which we can't meaningfully diff
    without a full resolver. Make sure we don't drown the report in
    spurious deviations on those boundaries."""
    declared = {"$ref": "#/components/schemas/Deep"}
    observed = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert diff_schemas(declared, observed, capability_id="x.y") == []


# ---------------------------------------------------------------------------
# Probe runner — uses MockTransport so no real network
# ---------------------------------------------------------------------------


def _build_provider_folder(
    tmp_path: Path,
    *,
    capabilities: list[dict[str, Any]],
    auth_token_env: str | None = "PROVIDER_PMS_TOKEN",
    base_url: str = "https://pms.test/v1",
) -> Path:
    provider_dir = tmp_path / "pms"
    provider_dir.mkdir()
    auth_block: dict[str, Any] | None = None
    if auth_token_env:
        auth_block = {
            "type": "bearer_token",
            "token_env": auth_token_env,
            "scope_declared": [],
        }
    provider = {
        "provider": {
            "id": "pms",
            "type": "openapi",
            "display_name": "PMS Test",
            "version": "0.1.0",
            "transport": {
                "kind": "openapi",
                "base_url": base_url,
            },
        }
    }
    if auth_block:
        provider["provider"]["auth"] = auth_block
    (provider_dir / "provider.yaml").write_text(
        yaml.safe_dump(provider, sort_keys=False), encoding="utf-8"
    )
    (provider_dir / "capabilities.yaml").write_text(
        yaml.safe_dump({"capabilities": capabilities}, sort_keys=False),
        encoding="utf-8",
    )
    (provider_dir / "resources.yaml").write_text(
        yaml.safe_dump({"resource_types": []}, sort_keys=False),
        encoding="utf-8",
    )
    return provider_dir


def _list_issue_capability(*, with_sample: bool = True) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "openapi_method": "get",
        "openapi_path": "/projects/{project_id}/issues",
        "openapi_operation_id": "listIssues",
    }
    if with_sample:
        metadata["sample_call"] = {
            "path_params": {"project_id": "demo"},
            "query_params": {"lane": "Todo"},
            "expected_status": 200,
        }
    return {
        "capability_id": "pms.issue.list",
        "kind": "query",
        "risk_level": "read",
        "side_effect": "external",
        "input_schema": {"type": "object"},
        "output_schema": {
            "type": "object",
            "properties": {
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                        },
                    },
                }
            },
        },
        "consumes_resources": ["issue"],
        "caller_types": ["reception"],
        "caller_modes": ["execute"],
        "routing_hints": {},
        "metadata": metadata,
    }


def _mock_transport(handler: Any) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _runtime_bits(provider_dir: Path) -> dict[str, Any]:
    """Pull the auth_block + base_url out of provider.yaml for direct
    ``run_live_shape_probe`` calls. Mirrors what
    ``build_live_shape_conformance_probe(provider_dir=...)`` does
    internally so test assertions can stay focused on probe behavior
    rather than yaml plumbing."""
    raw = yaml.safe_load(
        (provider_dir / "provider.yaml").read_text(encoding="utf-8")
    ) or {}
    provider_block = raw.get("provider") or {}
    auth = provider_block.get("auth")
    transport = provider_block.get("transport") or {}
    return {
        "auth_block": auth if isinstance(auth, dict) else None,
        "base_url": str(transport.get("base_url") or ""),
    }


def test_probe_passes_when_response_matches_declared_schema(
    tmp_path: Path,
) -> None:
    provider_dir = _build_provider_folder(
        tmp_path, capabilities=[_list_issue_capability()]
    )
    folder = load_provider_folder(provider_dir)
    assert folder.provider is not None

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={"issues": [{"id": "I-1", "title": "Demo"}]},
        )

    client = httpx.Client(transport=_mock_transport(handler))
    outcome = run_live_shape_probe(
        folder.provider,
        base_url="https://pms.test/v1",
        auth_block={
            "type": "bearer_token",
            "token_env": "PROVIDER_PMS_TOKEN",
        },
        http_client=client,
        env={"PROVIDER_PMS_TOKEN": "secret"},
    )
    assert len(outcome.sampled) == 1
    sample = outcome.sampled[0]
    assert sample.status == "ok"
    assert sample.drift == []
    assert sample.response_status == 200
    assert "/projects/demo/issues" in captured["url"]
    assert "lane=Todo" in captured["url"]
    assert captured["auth"] == "Bearer secret"


def test_probe_reports_drift_when_response_has_extra_field(
    tmp_path: Path,
) -> None:
    provider_dir = _build_provider_folder(
        tmp_path, capabilities=[_list_issue_capability()]
    )
    folder = load_provider_folder(provider_dir)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "issues": [
                    {"id": "I-1", "title": "Demo", "priority_v2": 5}
                ]
            },
        )

    client = httpx.Client(transport=_mock_transport(handler))
    outcome = run_live_shape_probe(
        folder.provider,
        **_runtime_bits(provider_dir),
        http_client=client,
        env={"PROVIDER_PMS_TOKEN": "x"},
    )
    sample = outcome.sampled[0]
    assert sample.status == "drift"
    drift_kinds = {d.kind for d in sample.drift}
    assert "field_added" in drift_kinds
    assert any(d.pointer == "/issues/items/priority_v2" for d in sample.drift)
    assert outcome.has_drift
    assert not outcome.has_endpoint_errors


def test_probe_records_error_on_non_2xx_status(tmp_path: Path) -> None:
    provider_dir = _build_provider_folder(
        tmp_path, capabilities=[_list_issue_capability()]
    )
    folder = load_provider_folder(provider_dir)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = httpx.Client(transport=_mock_transport(handler))
    outcome = run_live_shape_probe(
        folder.provider,
        **_runtime_bits(provider_dir),
        http_client=client,
        env={"PROVIDER_PMS_TOKEN": "x"},
    )
    sample = outcome.sampled[0]
    assert sample.status == "error"
    assert sample.response_status == 500
    assert "unexpected status 500" in sample.error
    assert outcome.has_endpoint_errors
    assert not outcome.has_drift


def test_probe_records_error_on_invalid_json_response(tmp_path: Path) -> None:
    provider_dir = _build_provider_folder(
        tmp_path, capabilities=[_list_issue_capability()]
    )
    folder = load_provider_folder(provider_dir)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>")

    client = httpx.Client(transport=_mock_transport(handler))
    outcome = run_live_shape_probe(
        folder.provider,
        **_runtime_bits(provider_dir),
        http_client=client,
        env={"PROVIDER_PMS_TOKEN": "x"},
    )
    sample = outcome.sampled[0]
    assert sample.status == "error"
    assert "not valid JSON" in sample.error


def test_probe_skips_token_when_no_auth_block(tmp_path: Path) -> None:
    """Endpoints that don't need auth (declared via auth=None) shouldn't
    have an Authorization header injected."""
    provider_dir = _build_provider_folder(
        tmp_path,
        capabilities=[_list_issue_capability()],
        auth_token_env=None,
    )
    folder = load_provider_folder(provider_dir)

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200, json={"issues": [{"id": "x", "title": "y"}]}
        )

    client = httpx.Client(transport=_mock_transport(handler))
    outcome = run_live_shape_probe(
        folder.provider,
        **_runtime_bits(provider_dir),
        http_client=client,
        env={},
    )
    assert outcome.sampled[0].status == "ok"
    assert captured["auth"] is None


def test_probe_drops_capability_without_sample_call(tmp_path: Path) -> None:
    cap_no_sample = _list_issue_capability(with_sample=False)
    provider_dir = _build_provider_folder(tmp_path, capabilities=[cap_no_sample])
    folder = load_provider_folder(provider_dir)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be called")

    client = httpx.Client(transport=_mock_transport(handler))
    outcome = run_live_shape_probe(
        folder.provider,
        **_runtime_bits(provider_dir),
        http_client=client,
        env={"PROVIDER_PMS_TOKEN": "x"},
    )
    assert outcome.sampled == []
    assert not outcome.has_drift
    assert not outcome.has_endpoint_errors


def test_probe_records_error_on_missing_path_param(tmp_path: Path) -> None:
    cap = _list_issue_capability()
    cap["metadata"]["sample_call"]["path_params"] = {}  # forgot project_id
    provider_dir = _build_provider_folder(tmp_path, capabilities=[cap])
    folder = load_provider_folder(provider_dir)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be called — substitution should fail first")

    client = httpx.Client(transport=_mock_transport(handler))
    outcome = run_live_shape_probe(
        folder.provider,
        **_runtime_bits(provider_dir),
        http_client=client,
        env={"PROVIDER_PMS_TOKEN": "x"},
    )
    sample = outcome.sampled[0]
    assert sample.status == "error"
    assert "project_id" in sample.error


# ---------------------------------------------------------------------------
# Conformance-probe wrapper
# ---------------------------------------------------------------------------


def test_conformance_probe_skips_when_no_sample_calls(tmp_path: Path) -> None:
    cap = _list_issue_capability(with_sample=False)
    provider_dir = _build_provider_folder(tmp_path, capabilities=[cap])
    folder = load_provider_folder(provider_dir)

    probe = build_live_shape_conformance_probe(folder.provider)
    assert probe.name == "live_response_shape"
    assert probe.status == "skip"
    assert "sample_call" in probe.detail


def test_conformance_probe_passes_when_no_drift(tmp_path: Path) -> None:
    provider_dir = _build_provider_folder(
        tmp_path, capabilities=[_list_issue_capability()]
    )
    folder = load_provider_folder(provider_dir)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"issues": [{"id": "I-1", "title": "Demo"}]}
        )

    client = httpx.Client(transport=_mock_transport(handler))
    probe = build_live_shape_conformance_probe(
        folder.provider,
        provider_dir=provider_dir,
        http_client=client,
        env={"PROVIDER_PMS_TOKEN": "x"},
    )
    assert probe.status == "pass"
    assert "drift=0" in probe.detail


def test_conformance_probe_pass_with_drift_is_soft_failure(tmp_path: Path) -> None:
    """Per backlog: drift is a diagnostic, not a hard fail; probe stays
    ``pass`` so partial deploys (extra debug fields in non-prod) don't
    take down the gate. The drift detail still surfaces in the report."""
    provider_dir = _build_provider_folder(
        tmp_path, capabilities=[_list_issue_capability()]
    )
    folder = load_provider_folder(provider_dir)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "issues": [
                    {"id": "I-1", "title": "Demo", "priority_v2": 5}
                ]
            },
        )

    client = httpx.Client(transport=_mock_transport(handler))
    probe = build_live_shape_conformance_probe(
        folder.provider,
        provider_dir=provider_dir,
        http_client=client,
        env={"PROVIDER_PMS_TOKEN": "x"},
    )
    assert probe.status == "pass"  # soft failure
    assert "drift" in probe.detail.lower()
    assert "priority_v2" in probe.detail
    assert "Refresh the OpenAPI" in (probe.hint or "")


def test_conformance_probe_fail_on_endpoint_error(tmp_path: Path) -> None:
    """Genuine reachability errors (network / non-2xx) DO fail the probe
    so missing credentials / broken endpoints surface clearly. Soft
    failure is reserved for schema drift on a successful call."""
    provider_dir = _build_provider_folder(
        tmp_path, capabilities=[_list_issue_capability()]
    )
    folder = load_provider_folder(provider_dir)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "no token"})

    client = httpx.Client(transport=_mock_transport(handler))
    probe = build_live_shape_conformance_probe(
        folder.provider,
        provider_dir=provider_dir,
        http_client=client,
        env={},  # no token in env
    )
    assert probe.status == "fail"
    assert "401" in probe.detail
    assert "credentials" in (probe.hint or "")


def test_run_provider_conformance_appends_live_probe_when_flag_on(
    tmp_path: Path,
) -> None:
    """Top-level ``run_provider_conformance(include_live_shape=True)``
    appends one ``live_response_shape`` probe to the report. The legacy
    static probes still come first so existing CLI consumers see the
    same prefix."""
    from novie_agent_sdk.provider_conformance import run_provider_conformance

    provider_dir = _build_provider_folder(
        tmp_path, capabilities=[_list_issue_capability(with_sample=False)]
    )
    # Without flag — only static probes.
    static_only = run_provider_conformance(provider_dir)
    static_names = {p.name for p in static_only.probes}
    assert "live_response_shape" not in static_names

    # With flag — adds live_response_shape; sample_call missing → skip.
    with_live = run_provider_conformance(
        provider_dir, include_live_shape=True
    )
    names = [p.name for p in with_live.probes]
    assert names[-1] == "live_response_shape"
    last = with_live.probes[-1]
    assert last.status == "skip"
