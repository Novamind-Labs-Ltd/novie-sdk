"""W1 — OpenAPI → ``capabilities.yaml`` generator unit tests.

Driven by the synthetic ``tests/fixtures/openapi/pms-openapi.json`` +
``pms-semantics.yaml`` pair; locks in the contract that:

- Every OpenAPI op with a semantics entry becomes one capability.
- ``generator_skip: true`` ops are dropped silently.
- ``$ref`` schemas inline correctly.
- Path-level + op-level parameters merge into the input schema.
- Strict mode raises a structured error listing every unmapped op.
- Bootstrap mode produces a complete, but TODO-marked, sidecar.
- ``provider.yaml`` carries an auth template derived from
  ``securitySchemes`` so the integrator only fills the env-var name.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from novie_agent_sdk.openapi_provider import (
    OperationSemantics,
    SemanticsHints,
    UnmappedOperationsError,
    UnsupportedOpenAPIError,
    bootstrap_semantics_from_openapi,
    generate_capabilities,
    generate_provider_files,
    load_semantics,
    operation_ids,
)


FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "openapi"


@pytest.fixture
def pms_openapi() -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / "pms-openapi.json").read_text(encoding="utf-8"))


@pytest.fixture
def pms_semantics() -> SemanticsHints:
    return load_semantics(FIXTURE_DIR / "pms-semantics.yaml")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_generate_capabilities_emits_one_per_mapped_operation(
    pms_openapi: dict[str, Any], pms_semantics: SemanticsHints
) -> None:
    out = generate_capabilities(pms_openapi, pms_semantics, provider_id="pms")
    caps = out["capabilities"]
    capability_ids = sorted(c["capability_id"] for c in caps)

    # 4 mapped ops + healthcheck (skipped) = 4 capabilities.
    assert capability_ids == [
        "pms.issue.archive",
        "pms.issue.get",
        "pms.issue.list",
        "pms.issue.move_lane",
    ]


def test_capability_carries_required_contract_fields(
    pms_openapi: dict[str, Any], pms_semantics: SemanticsHints
) -> None:
    out = generate_capabilities(pms_openapi, pms_semantics, provider_id="pms")
    by_id = {c["capability_id"]: c for c in out["capabilities"]}

    list_cap = by_id["pms.issue.list"]
    # Default kind derives from risk_level (read → query, write/dangerous → command).
    assert list_cap["kind"] == "query"
    assert list_cap["risk_level"] == "read"
    assert list_cap["side_effect"] == "external"
    assert list_cap["consumes_resources"] == ["issue"]
    assert "when_to_use" in list_cap["routing_hints"]
    assert list_cap["metadata"]["openapi_method"] == "get"
    assert list_cap["metadata"]["openapi_path"] == "/projects/{project_id}/issues"

    move_cap = by_id["pms.issue.move_lane"]
    assert move_cap["kind"] == "command"
    assert move_cap["risk_level"] == "write"
    assert move_cap["side_effect"] == "tenant"
    assert move_cap["confirmation_default"] == "required"
    assert sorted(move_cap["consumes_resources"]) == ["issue", "lane"]

    archive_cap = by_id["pms.issue.archive"]
    assert archive_cap["kind"] == "command"
    assert archive_cap["risk_level"] == "dangerous"
    assert archive_cap["side_effect"] == "irreversible"
    assert archive_cap["confirmation_default"] == "required"


def test_input_schema_merges_path_query_and_body_params(
    pms_openapi: dict[str, Any], pms_semantics: SemanticsHints
) -> None:
    out = generate_capabilities(pms_openapi, pms_semantics, provider_id="pms")
    move = next(c for c in out["capabilities"] if c["capability_id"] == "pms.issue.move_lane")
    schema = move["input_schema"]
    props = schema["properties"]
    assert "project_id" in props
    assert "issue_id" in props
    # Body schema is inlined (resolved through $ref → MoveIssueRequest).
    assert "body" in props
    body = props["body"]
    assert body["type"] == "object"
    assert "target_lane" in body["properties"]
    assert sorted(schema["required"]) == ["body", "issue_id", "project_id"]


def test_output_schema_resolves_ref_chain(
    pms_openapi: dict[str, Any], pms_semantics: SemanticsHints
) -> None:
    out = generate_capabilities(pms_openapi, pms_semantics, provider_id="pms")
    move = next(c for c in out["capabilities"] if c["capability_id"] == "pms.issue.move_lane")
    output = move["output_schema"]
    # MoveIssueResponse → contains issue (Issue $ref).
    assert output["type"] == "object"
    issue_field = output["properties"]["issue"]
    assert issue_field["type"] == "object"
    assert "title" in issue_field["properties"]


def test_query_parameter_lands_in_input_schema(
    pms_openapi: dict[str, Any], pms_semantics: SemanticsHints
) -> None:
    out = generate_capabilities(pms_openapi, pms_semantics, provider_id="pms")
    listing = next(c for c in out["capabilities"] if c["capability_id"] == "pms.issue.list")
    props = listing["input_schema"]["properties"]
    assert "lane" in props


def test_skipped_op_is_dropped(
    pms_openapi: dict[str, Any], pms_semantics: SemanticsHints
) -> None:
    out = generate_capabilities(pms_openapi, pms_semantics, provider_id="pms")
    capability_ids = {c["capability_id"] for c in out["capabilities"]}
    assert "pms.healthcheck" not in capability_ids


# ---------------------------------------------------------------------------
# Strict-mode + error semantics
# ---------------------------------------------------------------------------


def test_strict_mode_raises_with_full_unmapped_list(
    pms_openapi: dict[str, Any]
) -> None:
    # Empty semantics → all 5 ops are unmapped (4 real + healthcheck).
    empty = SemanticsHints()
    with pytest.raises(UnmappedOperationsError) as exc_info:
        generate_capabilities(pms_openapi, empty, provider_id="pms")
    err = exc_info.value
    op_ids = sorted(o.operation_id for o in err.operations)
    assert op_ids == [
        "archiveIssue",
        "getIssue",
        "healthcheck",
        "listIssues",
        "moveIssueToLane",
    ]
    rendered = str(err)
    # Every op surfaces in the rendered diagnostic with its method+path.
    assert "moveIssueToLane" in rendered
    assert "POST /projects/{project_id}/issues/{issue_id}/move" in rendered


def test_strict_mode_off_drops_unmapped_silently(
    pms_openapi: dict[str, Any]
) -> None:
    out = generate_capabilities(pms_openapi, SemanticsHints(), provider_id="pms", strict=False)
    assert out["capabilities"] == []


def test_unfilled_todo_marker_fails_strict(pms_openapi: dict[str, Any]) -> None:
    """Bootstrap output (with TODO_MARKER fields) must fail strict-mode
    so CI catches half-filled folders before they're merged."""
    bootstrap = bootstrap_semantics_from_openapi(pms_openapi)
    # bootstrap_semantics_from_openapi returns dict; pass through load to
    # exercise the same parser path the CLI uses.
    semantics = load_semantics(bootstrap)
    with pytest.raises(UnmappedOperationsError):
        generate_capabilities(pms_openapi, semantics, provider_id="pms")


def test_unsupported_openapi_2_rejected() -> None:
    swagger = {"swagger": "2.0", "paths": {}}
    with pytest.raises(UnsupportedOpenAPIError, match="2.0"):
        generate_capabilities(swagger, SemanticsHints(), provider_id="x")


# ---------------------------------------------------------------------------
# generate_provider_files (the bundled output)
# ---------------------------------------------------------------------------


def test_generate_provider_files_emits_three_files(
    pms_openapi: dict[str, Any], pms_semantics: SemanticsHints
) -> None:
    out = generate_provider_files(
        pms_openapi,
        pms_semantics,
        provider_id="pms",
        display_name="PMS Service",
        version="0.1.0",
    )
    assert sorted(out.keys()) == sorted(
        ["provider.yaml", "capabilities.yaml", "resources.yaml"]
    )

    provider = out["provider.yaml"]["provider"]
    assert provider["id"] == "pms"
    assert provider["type"] == "openapi"
    assert provider["transport"]["kind"] == "openapi"
    assert provider["transport"]["base_url"] == "https://pms.internal.example/v1"

    auth = provider["auth"]
    assert auth["type"] == "bearer_token"
    assert auth["token_env"] == "PROVIDER_BEARERAUTH_TOKEN"
    assert auth["openapi_scheme_name"] == "bearerAuth"

    # resources.yaml seeded from semantics.resource_binding union.
    assert sorted(out["resources.yaml"]["resource_types"]) == ["issue", "lane"]


def test_generated_capabilities_yaml_round_trips_through_yaml(
    pms_openapi: dict[str, Any], pms_semantics: SemanticsHints
) -> None:
    out = generate_provider_files(
        pms_openapi, pms_semantics, provider_id="pms"
    )
    text = yaml.safe_dump(out["capabilities.yaml"], sort_keys=False)
    parsed = yaml.safe_load(text)
    assert parsed == out["capabilities.yaml"]


# ---------------------------------------------------------------------------
# bootstrap helpers + listing
# ---------------------------------------------------------------------------


def test_bootstrap_semantics_covers_every_operation(
    pms_openapi: dict[str, Any]
) -> None:
    seeded = bootstrap_semantics_from_openapi(pms_openapi)
    seeded_ops = set(seeded["operations"].keys())
    found_ops = set(operation_ids(pms_openapi))
    assert seeded_ops == found_ops
    # Every entry has TODO sentinels for the judgmental fields.
    for entry in seeded["operations"].values():
        assert entry["intent_label"] == "__TODO__"
        assert entry["risk_level"] == "__TODO__"
        # And path/method metadata so the CLI can show context.
        assert entry["metadata"]["openapi_method"] in (
            "get", "post", "put", "patch", "delete"
        )


def test_load_semantics_rejects_invalid_risk_level(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "operations:\n"
        "  someOp:\n"
        "    intent_label: foo.bar\n"
        "    risk_level: idempotent_write\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="risk_level"):
        load_semantics(bad)


def test_aliases_let_renamed_op_keep_mapping(
    pms_openapi: dict[str, Any]
) -> None:
    """If the upstream renames ``listIssues`` → ``getIssueList`` we can
    keep the integrator's intent_label by aliasing — without this, every
    rename would force a whole-file edit."""
    semantics = SemanticsHints(
        operations={
            "getIssueList": OperationSemantics(
                intent_label="pms.issue.list",
                risk_level="read",
                resource_binding=("issue",),
                generator_skip=False,
            ),
        },
        aliases={"listIssues": "getIssueList"},
    )
    fixture = {
        "openapi": "3.0.3",
        "components": pms_openapi.get("components") or {},
        "paths": {
            "/projects/{pid}/issues": {
                "get": pms_openapi["paths"]["/projects/{project_id}/issues"]["get"]
            }
        },
    }
    out = generate_capabilities(fixture, semantics, provider_id="pms")
    cap_ids = [c["capability_id"] for c in out["capabilities"]]
    assert cap_ids == ["pms.issue.list"]
