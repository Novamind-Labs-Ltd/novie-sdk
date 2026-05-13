"""UNIVERSAL_CAPABILITY W8 step 3 — provider conformance suite tests.

Locks the conformance probe set so external authors get a clear
green/red signal before registration.

Acceptance bullets locked:
- "Conformance tests prove health, discovery, resource resolution,
  invocation, auth denial, and audit behavior."
- "Failed conformance explains whether the problem is auth,
  protocol, durability, or result schema." (each probe carries
  ``hint``)
- "A write provider must declare dry-run/gate behavior or fail
  validation." (write_provider_gate probe)
"""
# ruff: noqa: I001
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import httpx

from novie_agent_sdk import (
    ProviderConformanceProbe,
    ProviderConformanceReport,
    ProviderRuntimeConformanceConfig,
    run_provider_conformance,
)


_VALID_PROVIDER_YAML = dedent(
    """
    provider:
      id: integration.github
      type: openapi
      display_name: GitHub Adapter
      version: 0.1.0
      transport:
        kind: openapi
        spec_url: https://api.github.com/openapi.json
        base_url: https://api.github.com
      health:
        kind: http_get
        url: https://api.github.com/rate_limit
    """
).strip()


_VALID_CAPABILITIES_READ = dedent(
    """
    capabilities:
      - capability_id: integration.github.repo.read
        kind: query
        risk_level: read
        side_effect: none
        input_schema:
          type: object
        output_schema:
          type: object
        consumes_resources:
          - repository
        routing_hints:
          when_to_use: Read GitHub repository metadata.
    """
).strip()


_VALID_RESOURCES = dedent(
    """
    resource_types:
      - repository
    """
).strip()

_RUNTIME_SAMPLE_CAPABILITIES_READ = dedent(
    """
    capabilities:
      - capability_id: integration.github.repo.read
        kind: query
        risk_level: read
        side_effect: none
        input_schema:
          type: object
          properties:
            owner:
              type: string
            repo:
              type: string
        output_schema:
          type: object
        consumes_resources:
          - repository
        routing_hints:
          when_to_use: Read GitHub repository metadata.
        metadata:
          runtime_sample_call:
            mode: execute
            inputs:
              owner: Novamind-Labs-Ltd
              repo: novie
            expected_status: ok
    """
).strip()

_RUNTIME_DENIAL_CAPABILITIES_READ = dedent(
    """
    capabilities:
      - capability_id: integration.github.repo.read
        kind: query
        risk_level: read
        side_effect: none
        input_schema:
          type: object
        output_schema:
          type: object
        consumes_resources:
          - repository
        routing_hints:
          when_to_use: Read GitHub repository metadata.
        metadata:
          runtime_auth_denial_sample:
            mode: execute
            inputs:
              owner: Novamind-Labs-Ltd
              repo: novie-private
            expected_status: denied
    """
).strip()


def _write_provider(
    root: Path,
    *,
    provider_yaml: str = _VALID_PROVIDER_YAML,
    capabilities_yaml: str = _VALID_CAPABILITIES_READ,
    resources_yaml: str = _VALID_RESOURCES,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "provider.yaml").write_text(provider_yaml + "\n", encoding="utf-8")
    (root / "capabilities.yaml").write_text(capabilities_yaml + "\n", encoding="utf-8")
    (root / "resources.yaml").write_text(resources_yaml + "\n", encoding="utf-8")
    return root


# ── Happy path ──────────────────────────────────────────────────────────────


def test_conformance_passes_on_valid_read_only_provider(tmp_path: Path) -> None:
    report = run_provider_conformance(_write_provider(tmp_path / "github"))
    assert isinstance(report, ProviderConformanceReport)
    assert report.ok, [(p.name, p.detail) for p in report.failures]
    assert report.provider_id == "integration.github"
    names = [p.name for p in report.probes]
    assert names == [
        "folder_validation",
        "health_metadata",
        "discovery",
        "capability_id_naming",
        "write_provider_gate",
        "resource_consistency",
        "auth_scope_declared",
        "audit_metadata",
    ]


def test_runtime_conformance_invokes_registered_provider_sample(
    tmp_path: Path,
) -> None:
    """W8 step 4: runtime probes are opt-in and call the canonical
    gateway ``/invocations`` route only when a runtime sample exists."""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://api.github.com/rate_limit":
            return httpx.Response(200, json={"ok": True})
        if str(request.url) == "https://gateway.test/invocations":
            body = json.loads(request.content.decode())
            assert body["capability_id"] == "integration.github.repo.read"
            assert body["provider_id"] == "integration.github"
            assert request.headers["x-novie-org-id"] == "tenant-a"
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "audit_id": "audit-1",
                    "trace_id": "trace-1",
                },
            )
        return httpx.Response(404)

    report = run_provider_conformance(
        _write_provider(
            tmp_path / "github",
            capabilities_yaml=_RUNTIME_SAMPLE_CAPABILITIES_READ,
        ),
        runtime_config=ProviderRuntimeConformanceConfig(
            gateway_url="https://gateway.test",
            org_id="tenant-a",
            project_id="project-a",
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert report.ok, [(p.name, p.detail) for p in report.failures]
    runtime_health = next(
        p for p in report.probes if p.name == "runtime_health_reachability"
    )
    runtime_invocation = next(
        p for p in report.probes if p.name == "runtime_invocation_round_trip"
    )
    runtime_denial = next(p for p in report.probes if p.name == "runtime_auth_denial")
    assert runtime_health.status == "pass"
    assert runtime_invocation.status == "pass"
    assert runtime_denial.status == "skip"


def test_runtime_conformance_fails_when_invocation_lacks_trace_id(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://api.github.com/rate_limit":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(200, json={"status": "ok", "audit_id": "audit-1"})

    report = run_provider_conformance(
        _write_provider(
            tmp_path / "github",
            capabilities_yaml=_RUNTIME_SAMPLE_CAPABILITIES_READ,
        ),
        runtime_config=ProviderRuntimeConformanceConfig(
            gateway_url="https://gateway.test",
            org_id="tenant-a",
            project_id="project-a",
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    runtime_invocation = next(
        p for p in report.probes if p.name == "runtime_invocation_round_trip"
    )
    assert runtime_invocation.status == "fail"
    assert "trace_id" in runtime_invocation.detail


def test_runtime_conformance_auth_denial_sample_requires_denied_status(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://api.github.com/rate_limit":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(
            200,
            json={
                "status": "denied",
                "audit_id": "audit-denied",
                "trace_id": "trace-denied",
            },
        )

    report = run_provider_conformance(
        _write_provider(
            tmp_path / "github",
            capabilities_yaml=_RUNTIME_DENIAL_CAPABILITIES_READ,
        ),
        runtime_config=ProviderRuntimeConformanceConfig(
            gateway_url="https://gateway.test",
            org_id="tenant-a",
            project_id="project-a",
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    runtime_denial = next(p for p in report.probes if p.name == "runtime_auth_denial")
    assert runtime_denial.status == "pass", runtime_denial.detail


def test_runtime_conformance_identity_headers_cannot_be_overridden() -> None:
    """W11 regression: runtime probes must not let extra headers
    rewrite the tenant/project identity sent to the gateway."""
    config = ProviderRuntimeConformanceConfig(
        gateway_url="https://gateway.test",
        org_id="tenant-a",
        project_id="project-a",
        workspace_id="workspace-a",
        user_id="u-1",
        headers={
            "authorization": "Bearer test",
            "x-novie-org-id": "tenant-b",
            "x-novie-project-id": "project-b",
            "x-novie-workspace-id": "workspace-b",
            "x-novie-user-id": "u-2",
        },
    )

    headers = config.identity_headers()
    assert headers["authorization"] == "Bearer test"
    assert headers["x-novie-org-id"] == "tenant-a"
    assert headers["x-novie-project-id"] == "project-a"
    assert headers["x-novie-workspace-id"] == "workspace-a"
    assert headers["x-novie-user-id"] == "u-1"


def test_conformance_skips_other_probes_when_folder_invalid(
    tmp_path: Path,
) -> None:
    """If the folder doesn't load, every other probe should be
    ``skip`` so the operator sees the actual broken layer instead of
    a wall of cascading failures."""
    target = tmp_path / "broken"
    target.mkdir()
    # Missing all three required files.
    report = run_provider_conformance(target)
    assert not report.ok
    assert report.probes[0].name == "folder_validation"
    assert report.probes[0].status == "fail"
    assert all(p.status == "skip" for p in report.probes[1:])


# ── health_metadata ─────────────────────────────────────────────────────────


def test_conformance_fails_when_health_kind_missing(tmp_path: Path) -> None:
    bad_provider = dedent(
        """
        provider:
          id: integration.github
          type: openapi
          display_name: GitHub Adapter
          version: 0.1.0
          transport:
            kind: openapi
            spec_url: https://api.github.com/openapi.json
          health: {}
        """
    ).strip()
    report = run_provider_conformance(
        _write_provider(tmp_path / "github", provider_yaml=bad_provider),
    )
    health = next(p for p in report.probes if p.name == "health_metadata")
    assert health.status == "fail"
    assert "kind" in health.detail


# ── discovery ───────────────────────────────────────────────────────────────


def test_conformance_fails_when_zero_capabilities(tmp_path: Path) -> None:
    empty_caps = "capabilities: []"
    # A provider with zero capabilities also fails the schema-level
    # ``capabilities_empty`` check upstream — but we can still gate
    # discovery on the conformance side. We need a folder that
    # passes folder_validation though, so add one capability that
    # the discovery probe will count.
    # ↑ folder_validation requires non-empty capabilities; conformance
    # will surface it via folder_validation, not discovery — and in
    # that case discovery is ``skip`` per the cascade rule.
    report = run_provider_conformance(
        _write_provider(tmp_path / "github", capabilities_yaml=empty_caps),
    )
    folder = next(p for p in report.probes if p.name == "folder_validation")
    assert folder.status == "fail"
    discovery = next(p for p in report.probes if p.name == "discovery")
    assert discovery.status == "skip"


# ── capability_id_naming ────────────────────────────────────────────────────


def test_conformance_fails_when_capability_id_off_convention(
    tmp_path: Path,
) -> None:
    off_caps = dedent(
        """
        capabilities:
          - capability_id: legacy.repo.read
            kind: query
            risk_level: read
            side_effect: none
            input_schema:
              type: object
            output_schema:
              type: object
            consumes_resources:
              - repository
            routing_hints:
              when_to_use: Legacy read.
        """
    ).strip()
    report = run_provider_conformance(
        _write_provider(tmp_path / "github", capabilities_yaml=off_caps),
    )
    naming = next(p for p in report.probes if p.name == "capability_id_naming")
    assert naming.status == "fail"
    assert "integration.github." in naming.detail
    assert "legacy.repo.read" in naming.detail
    assert naming.hint  # actionable


# ── write_provider_gate (acceptance bullet) ─────────────────────────────────
#
# The existing W8 step 1 validator already rejects write/dangerous
# capabilities that omit dry-run / confirmation / gate at
# folder_validation time, so the conformance probe here serves as a
# defense-in-depth check + a clearer "why" message when a provider
# does pass folder_validation but is missing one of the lighter
# governance signals.


def test_conformance_passes_when_write_capability_declares_full_gate(
    tmp_path: Path,
) -> None:
    """A valid write provider with both dry-run and gated confirmation
    passes the gate probe."""
    good_write = dedent(
        """
        capabilities:
          - capability_id: integration.github.repo.write
            kind: command
            risk_level: write
            side_effect: external_write
            input_schema:
              type: object
            output_schema:
              type: object
            consumes_resources:
              - repository
            produces_resources:
              - repository
            auth_scope:
              - write:repos
            dry_run_support: preview_with_diff
            confirmation_default: gated
            routing_hints:
              when_to_use: Write GitHub repository content.
        """
    ).strip()
    report = run_provider_conformance(
        _write_provider(tmp_path / "github", capabilities_yaml=good_write),
    )
    # If folder_validation fails on this fixture (because some
    # cross-validation upstream still rejects), surface the actual
    # blocker so tests fail loudly rather than silently skip.
    folder = next(p for p in report.probes if p.name == "folder_validation")
    assert folder.status == "pass", folder.detail
    gate = next(p for p in report.probes if p.name == "write_provider_gate")
    assert gate.status == "pass", gate.detail


def test_conformance_skips_gate_for_pure_read_provider(tmp_path: Path) -> None:
    """Read-only providers shouldn't be penalized for not declaring
    write gates."""
    report = run_provider_conformance(_write_provider(tmp_path / "github"))
    gate = next(p for p in report.probes if p.name == "write_provider_gate")
    assert gate.status == "skip"


def test_conformance_existing_validator_still_blocks_write_without_gate(
    tmp_path: Path,
) -> None:
    """The W8 step-1 validator has its own
    ``write_capability_missing_preview_or_gate`` rule. Conformance
    must still flag this scenario — it's surfaced via
    folder_validation rather than write_provider_gate, but the
    operator-facing report is still red."""
    bad_write = dedent(
        """
        capabilities:
          - capability_id: integration.github.repo.write
            kind: command
            risk_level: write
            side_effect: external_write
            input_schema:
              type: object
            output_schema:
              type: object
            consumes_resources:
              - repository
            produces_resources:
              - repository
            auth_scope:
              - write:repos
            routing_hints:
              when_to_use: Write GitHub repository content.
        """
    ).strip()
    report = run_provider_conformance(
        _write_provider(tmp_path / "github", capabilities_yaml=bad_write),
    )
    assert not report.ok
    folder = next(p for p in report.probes if p.name == "folder_validation")
    assert folder.status == "fail"


# ── resource_consistency ────────────────────────────────────────────────────


def test_conformance_fails_when_capability_refs_undeclared_resource(
    tmp_path: Path,
) -> None:
    bad_caps = dedent(
        """
        capabilities:
          - capability_id: integration.github.repo.read
            kind: query
            risk_level: read
            side_effect: none
            input_schema:
              type: object
            output_schema:
              type: object
            consumes_resources:
              - repository
              - issues
            routing_hints:
              when_to_use: Read GitHub data.
        """
    ).strip()
    report = run_provider_conformance(
        _write_provider(tmp_path / "github", capabilities_yaml=bad_caps),
    )
    resource = next(p for p in report.probes if p.name == "resource_consistency")
    assert resource.status == "fail"
    assert "issues" in resource.detail
    assert resource.hint


# ── auth_scope_declared ─────────────────────────────────────────────────────


def test_conformance_fails_when_write_capability_missing_auth_scope(
    tmp_path: Path,
) -> None:
    bad_write = dedent(
        """
        capabilities:
          - capability_id: integration.github.repo.write
            kind: command
            risk_level: write
            side_effect: external_write
            input_schema:
              type: object
            output_schema:
              type: object
            consumes_resources:
              - repository
            produces_resources:
              - repository
            dry_run_support: preview_with_diff
            confirmation_default: gated
            routing_hints:
              when_to_use: Write GitHub data.
        """
    ).strip()
    report = run_provider_conformance(
        _write_provider(tmp_path / "github", capabilities_yaml=bad_write),
    )
    folder = next(p for p in report.probes if p.name == "folder_validation")
    assert folder.status == "pass", folder.detail
    auth = next(p for p in report.probes if p.name == "auth_scope_declared")
    assert auth.status == "fail"
    assert "auth_scope" in auth.hint


def test_conformance_skips_auth_for_read_only_provider(tmp_path: Path) -> None:
    report = run_provider_conformance(_write_provider(tmp_path / "github"))
    auth = next(p for p in report.probes if p.name == "auth_scope_declared")
    assert auth.status == "skip"


# ── audit_metadata ──────────────────────────────────────────────────────────


def test_conformance_fails_when_capability_missing_when_to_use(
    tmp_path: Path,
) -> None:
    no_alias = dedent(
        """
        capabilities:
          - capability_id: integration.github.repo.read
            kind: query
            risk_level: read
            side_effect: none
            input_schema:
              type: object
            output_schema:
              type: object
            consumes_resources:
              - repository
        """
    ).strip()
    report = run_provider_conformance(
        _write_provider(tmp_path / "github", capabilities_yaml=no_alias),
    )
    audit = next(p for p in report.probes if p.name == "audit_metadata")
    assert audit.status == "fail"
    assert "when_to_use" in audit.hint


# ── Report serialization ────────────────────────────────────────────────────


def test_conformance_report_serializes_to_dict(tmp_path: Path) -> None:
    report = run_provider_conformance(_write_provider(tmp_path / "github"))
    data = report.to_dict()
    assert data["ok"] is True
    assert data["provider_id"] == "integration.github"
    assert isinstance(data["probes"], list)
    assert {p["name"] for p in data["probes"]} >= {
        "folder_validation",
        "health_metadata",
        "discovery",
        "capability_id_naming",
        "write_provider_gate",
        "resource_consistency",
        "auth_scope_declared",
        "audit_metadata",
    }


def test_probe_to_dict_round_trip() -> None:
    probe = ProviderConformanceProbe(
        name="x", status="fail", detail="d", hint="h",
    )
    assert probe.to_dict() == {
        "name": "x", "status": "fail", "detail": "d", "hint": "h",
    }
