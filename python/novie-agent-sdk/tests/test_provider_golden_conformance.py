"""UNIVERSAL_CAPABILITY W11 step 1 — golden provider conformance.

Locks the W8 step-3 conformance behavior against curated reference
provider fixtures so:

- The conformance suite stays green for shapes the platform claims
  to support (acceptance bullet "Conformance suite can be run
  against a third-party adapter before enabling it").
- Schema / probe drift fails CI loudly instead of breaking
  authoring silently.
- New authors have a working starting point — the fixtures are
  intentionally close to the real Langfuse-style observability and
  GitHub-write providers the W9 / W10 work will land.

Each fixture under ``tests/fixtures/providers/<name>/`` ships
``provider.yaml``, ``capabilities.yaml``, and ``resources.yaml``
mirroring the canonical W8 layout.
"""
# ruff: noqa: I001
from __future__ import annotations

from pathlib import Path

import pytest

from novie_agent_sdk import (
    ProviderConformanceReport,
    run_provider_conformance,
)


_FIXTURES = Path(__file__).parent / "fixtures" / "providers"


_GOLDEN_PROVIDERS: list[tuple[str, str]] = [
    ("observability", "platform.observability"),
    ("integration_github", "integration.github"),
]


@pytest.mark.parametrize("fixture_name,expected_provider_id", _GOLDEN_PROVIDERS)
def test_golden_provider_passes_conformance(
    fixture_name: str, expected_provider_id: str,
) -> None:
    """Acceptance bullet: every golden provider passes the W8 step-3
    conformance suite end to end.

    Failure here means the conformance contract drifted in a way
    that breaks a real-world provider shape — usually a missed
    governance signal or a renamed required field. Keep the
    fixture as-is and fix the conformance probe (or add
    documentation explaining the breaking change in the W11 spec)
    instead of editing fixtures around the failure.
    """
    fixture_dir = _FIXTURES / fixture_name
    assert fixture_dir.is_dir(), f"missing fixture: {fixture_dir}"
    report = run_provider_conformance(fixture_dir)
    assert isinstance(report, ProviderConformanceReport)
    assert report.provider_id == expected_provider_id
    assert report.ok, [(p.name, p.detail) for p in report.failures]


def test_observability_fixture_skips_write_probes() -> None:
    """The Langfuse-style observability provider is read-only by
    design. Conformance must skip (not fail) the write-only probes
    so the shape isn't penalized."""
    report = run_provider_conformance(_FIXTURES / "observability")
    assert report.ok
    write_gate = next(
        p for p in report.probes if p.name == "write_provider_gate"
    )
    auth = next(p for p in report.probes if p.name == "auth_scope_declared")
    assert write_gate.status == "skip"
    assert auth.status == "skip"


def test_github_fixture_exercises_write_gate() -> None:
    """GitHub fixture has both a read capability and a write
    capability. Conformance should:
    - pass write_provider_gate (write capability declares
      dry_run_support + confirmation_default + auth_scope +
      credential_refs)
    - pass auth_scope_declared
    - pass resource_consistency (capability resource refs match
      ``resources.yaml``)
    """
    report = run_provider_conformance(_FIXTURES / "integration_github")
    assert report.ok
    write_gate = next(
        p for p in report.probes if p.name == "write_provider_gate"
    )
    auth = next(p for p in report.probes if p.name == "auth_scope_declared")
    resources = next(
        p for p in report.probes if p.name == "resource_consistency"
    )
    assert write_gate.status == "pass", write_gate.detail
    assert auth.status == "pass", auth.detail
    assert resources.status == "pass", resources.detail


def test_golden_providers_have_no_warnings() -> None:
    """Golden fixtures should ship clean — even warning-severity
    probes should pass. Tighter than ``ok`` so the W11 regression
    layer catches softer drift earlier."""
    for fixture_name, _ in _GOLDEN_PROVIDERS:
        report = run_provider_conformance(_FIXTURES / fixture_name)
        for probe in report.probes:
            assert probe.status != "fail", (
                f"{fixture_name}: probe {probe.name!r} failed: {probe.detail}"
            )
