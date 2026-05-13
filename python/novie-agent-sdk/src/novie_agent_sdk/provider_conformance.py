"""UNIVERSAL_CAPABILITY W8 step 3 — universal-provider conformance suite.

Static conformance over a provider folder + its loaded
``CapabilityProvider`` so authors get a green/red gate before
registration. Probes cover the W8 spec's six conformance areas
(health / discovery / resource resolution / invocation / auth
denial / audit) at the **declarative** level — runtime probes
against a live provider deployment can layer on top via the same
``ConformanceProbe`` envelope.

Surface (locked by ``test_provider_conformance.py``):

- ``ProviderConformanceProbe`` / ``ProviderConformanceReport`` —
  structured pass/fail/skip per probe + actionable hint.
- ``run_provider_conformance(provider_dir) -> ProviderConformanceReport``
  — loads the folder via ``load_provider_folder`` and runs every
  probe in dependency order (folder validation must pass before
  capability-level probes can fire).
- Hint-rich failures so the W8 acceptance bullet "Failed
  conformance explains whether the problem is auth, protocol,
  durability, or result schema" is locked.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin

import httpx

from novie_protocol.contracts.universal_capability import (
    CapabilityContract,
    CapabilityProvider,
)

from .provider_authoring import (
    ProviderFolderOutcome,
    load_provider_folder,
)


ProbeStatus = Literal["pass", "fail", "skip"]


@dataclass(frozen=True, slots=True)
class ProviderConformanceProbe:
    """One conformance probe outcome."""

    name: str
    status: ProbeStatus
    detail: str = ""
    hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "hint": self.hint,
        }


@dataclass(frozen=True, slots=True)
class ProviderConformanceReport:
    """Aggregate of every probe in one conformance run."""

    provider_dir: Path
    probes: tuple[ProviderConformanceProbe, ...] = field(default_factory=tuple)
    provider_id: str = ""

    @property
    def ok(self) -> bool:
        return not any(p.status == "fail" for p in self.probes)

    @property
    def failures(self) -> tuple[ProviderConformanceProbe, ...]:
        return tuple(p for p in self.probes if p.status == "fail")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_dir": str(self.provider_dir),
            "provider_id": self.provider_id,
            "ok": self.ok,
            "probes": [p.to_dict() for p in self.probes],
        }


@dataclass(frozen=True, slots=True)
class ProviderRuntimeConformanceConfig:
    """HTTP runtime probe settings for a registered provider.

    Static conformance proves the provider folder is valid. Runtime
    conformance uses the gateway's canonical ``POST /invocations`` path
    to prove the registered provider can actually execute through the
    W4 middleware chain and return audit/trace identifiers.
    """

    gateway_url: str
    org_id: str
    project_id: str
    user_id: str = "provider-conformance"
    workspace_id: str = ""
    session_id: str = "provider-conformance"
    request_id: str = "provider-conformance"
    timeout_seconds: float = 30.0
    headers: Mapping[str, str] = field(default_factory=dict)
    verify_tls: bool = True

    @property
    def invocation_url(self) -> str:
        return urljoin(self.gateway_url.rstrip("/") + "/", "invocations")

    def identity_headers(self) -> dict[str, str]:
        headers = dict(self.headers)
        headers.update({
            "x-novie-org-id": self.org_id,
            "x-novie-project-id": self.project_id,
            "x-novie-workspace-id": self.workspace_id or self.project_id,
            "x-novie-user-id": self.user_id,
            "x-novie-session-id": self.session_id,
            "x-novie-request-id": self.request_id,
        })
        return headers


def _passed(name: str, *, detail: str = "") -> ProviderConformanceProbe:
    return ProviderConformanceProbe(name=name, status="pass", detail=detail)


def _failed(
    name: str, detail: str, *, hint: str = "",
) -> ProviderConformanceProbe:
    return ProviderConformanceProbe(
        name=name, status="fail", detail=detail, hint=hint,
    )


def _skipped(name: str, detail: str) -> ProviderConformanceProbe:
    return ProviderConformanceProbe(
        name=name, status="skip", detail=detail,
    )


# ── Probe implementations ───────────────────────────────────────────────────


def _probe_folder_validation(
    outcome: ProviderFolderOutcome,
) -> ProviderConformanceProbe:
    """First gate. The W8 step 1 ``load_provider_folder`` already
    runs schema + cross-file checks; conformance fails fast if it
    didn't pass."""
    if outcome.result.errors:
        first = outcome.result.errors[0]
        return _failed(
            "folder_validation",
            (
                f"{len(outcome.result.errors)} validation error(s); "
                f"first: {first.field_path} — {first.author_message}"
            ),
            hint="Run ``novie providers validate`` and fix every error before retrying conformance.",
        )
    return _passed(
        "folder_validation",
        detail=(
            f"warnings={len(outcome.result.warnings)} "
            f"infos={len(outcome.result.infos)}"
        ),
    )


def _probe_health_metadata(
    provider: CapabilityProvider,
) -> ProviderConformanceProbe:
    """Acceptance bullet: "Conformance tests prove health, …".
    Static check that the provider declares a health probe shape so
    runtime tooling has something to call."""
    health = provider.health
    if health is None:
        return _failed(
            "health_metadata",
            "provider.yaml missing ``health`` block",
            hint=(
                "Declare ``health`` in provider.yaml so platform "
                "operator tooling can probe the provider — at minimum "
                "``kind`` and either ``endpoint`` (HTTP) or ``check`` "
                "(internal)."
            ),
        )
    kind = getattr(health, "kind", "") or ""
    if not kind or kind == "none":
        return _failed(
            "health_metadata",
            f"provider.health.kind={kind!r} is not a usable probe",
            hint=(
                "Set ``health.kind`` to a real probe (e.g. ``http_get``, "
                "``internal``, ``llm_proxy``) so platform operator "
                "tooling can actually call the provider."
            ),
        )
    return _passed("health_metadata", detail=f"kind={kind}")


def _probe_discovery(
    provider: CapabilityProvider,
) -> ProviderConformanceProbe:
    """Acceptance bullet: "Conformance tests prove …, discovery, …".
    A provider with no capabilities is not discoverable — flag as
    a failure with a hint to add the first capability."""
    if not provider.capabilities:
        return _failed(
            "discovery",
            "provider declares zero capabilities",
            hint=(
                "Add at least one entry to capabilities.yaml. The "
                "platform's capability discovery filters skip "
                "providers with no capabilities."
            ),
        )
    return _passed(
        "discovery",
        detail=f"{len(provider.capabilities)} capabilit(y/ies) declared",
    )


def _probe_capability_id_naming(
    provider: CapabilityProvider,
) -> ProviderConformanceProbe:
    """Capability ids should follow ``<provider_id>.<...>`` so the
    planner's prefix search works. Off-convention ids are common for
    legacy providers — surfaced as a warning-shaped failure with the
    list of offenders."""
    expected_prefix = f"{provider.provider_id}."
    offenders = [
        c.capability_id for c in provider.capabilities
        if not c.capability_id.startswith(expected_prefix)
    ]
    if offenders:
        return _failed(
            "capability_id_naming",
            f"{len(offenders)} capability id(s) miss prefix {expected_prefix!r}: "
            + ", ".join(offenders[:3])
            + ("…" if len(offenders) > 3 else ""),
            hint=(
                f"Rename capabilities to ``{expected_prefix}<verb>`` so the "
                "planner's prefix search ranks them correctly. "
                "Off-convention ids still load but lose ranking quality."
            ),
        )
    return _passed(
        "capability_id_naming",
        detail=f"all {len(provider.capabilities)} capability ids use {expected_prefix!r} prefix",
    )


_READ_ONLY_SIDE_EFFECTS: frozenset[str] = frozenset({
    "",          # unset → conservative read-only default
    "none",
    "read",
    "read_only",
    "session",   # session-scoped state mutation; doesn't escape provider
})


def _capability_is_write(capability: CapabilityContract) -> bool:
    """A capability is a 'write' for W8 purposes when it has any
    side effect that escapes the provider's session. Read-only +
    session-only capabilities don't need dry-run/gate enforcement
    (acceptance bullet "A new read-only provider can be added with
    no Reception/Planner code change" — read providers must not
    be blocked by gate probes)."""
    side_effect = (capability.side_effect or "").strip().lower()
    risk_level = (capability.risk_level or "").strip().lower()
    if side_effect in _READ_ONLY_SIDE_EFFECTS and risk_level == "read":
        return False
    if side_effect in _READ_ONLY_SIDE_EFFECTS:
        return False
    return True


def _probe_write_provider_gate(
    provider: CapabilityProvider,
) -> ProviderConformanceProbe:
    """Acceptance bullet: "A write provider must declare dry-run/gate
    behavior or fail validation." Each non-read capability must
    expose either a dry-run path (``dry_run_support`` != ``none``)
    or a confirmation/gate (``confirmation_default`` ∈ ``required /
    gated`` OR a non-empty ``gate_policy`` OR
    ``governance.requires_human_gate``).
    """
    write_caps = [c for c in provider.capabilities if _capability_is_write(c)]
    if not write_caps:
        return _skipped(
            "write_provider_gate",
            "provider declares no write capabilities — gate enforcement "
            "is read-only-safe by default",
        )
    offenders: list[str] = []
    for cap in write_caps:
        has_dry_run = (cap.dry_run_support or "none") != "none"
        has_confirmation = cap.confirmation_default in ("required", "gated")
        has_gate_policy = bool(cap.gate_policy)
        has_human_gate = bool(getattr(cap.governance, "requires_human_gate", False))
        if not (has_dry_run or has_confirmation or has_gate_policy or has_human_gate):
            offenders.append(cap.capability_id)
    if offenders:
        return _failed(
            "write_provider_gate",
            f"{len(offenders)} write capabilit(y/ies) declare neither "
            f"dry-run nor a gate: " + ", ".join(offenders[:3])
            + ("…" if len(offenders) > 3 else ""),
            hint=(
                "Set ``dry_run_support`` to a non-``none`` value or "
                "``confirmation_default=required|gated`` (or add "
                "``governance.requires_human_gate=true``) on every "
                "write capability."
            ),
        )
    return _passed(
        "write_provider_gate",
        detail=f"all {len(write_caps)} write capabilit(y/ies) declare gate behavior",
    )


def _probe_resource_consistency(
    provider: CapabilityProvider,
) -> ProviderConformanceProbe:
    """Acceptance bullet: "Conformance tests prove …, resource
    resolution, …". Every resource referenced by a capability's
    ``consumes_resources`` / ``produces_resources`` must be declared
    in ``resources.yaml`` (i.e. appear in ``provider.resource_types``).
    """
    declared = set(provider.resource_types)
    missing: list[tuple[str, str]] = []
    for cap in provider.capabilities:
        for ref in (*cap.consumes_resources, *cap.produces_resources):
            if ref and ref not in declared:
                missing.append((cap.capability_id, ref))
    if missing:
        sample = ", ".join(f"{cap}→{ref}" for cap, ref in missing[:3])
        more = "…" if len(missing) > 3 else ""
        return _failed(
            "resource_consistency",
            f"{len(missing)} capability resource ref(s) not in resources.yaml: {sample}{more}",
            hint=(
                "Add the missing ``resource_types`` entries to "
                "resources.yaml or fix the capability's "
                "``consumes_resources`` / ``produces_resources`` to "
                "reference declared resource types."
            ),
        )
    return _passed(
        "resource_consistency",
        detail=f"{len(declared)} resource type(s) declared, all capability refs match",
    )


def _probe_auth_scope_declared(
    provider: CapabilityProvider,
) -> ProviderConformanceProbe:
    """Acceptance bullet: "Conformance tests prove …, auth denial,
    …". A provider with write capabilities but no auth scope
    declarations gives the platform nothing to deny on. Read-only
    providers are exempt."""
    write_caps = [c for c in provider.capabilities if _capability_is_write(c)]
    if not write_caps:
        return _skipped(
            "auth_scope_declared",
            "provider has no write capabilities — auth scopes optional",
        )
    offenders = [
        c.capability_id for c in write_caps
        if not c.auth_scope and not c.credential_refs
    ]
    if offenders:
        return _failed(
            "auth_scope_declared",
            (
                f"{len(offenders)} write capabilit(y/ies) declare neither "
                f"``auth_scope`` nor ``credential_refs``: "
                + ", ".join(offenders[:3])
                + ("…" if len(offenders) > 3 else "")
            ),
            hint=(
                "Add ``auth_scope`` (e.g. ``read:projects``, "
                "``write:tasks``) or ``credential_refs`` so the "
                "platform's binding/auth gate has a concrete scope to "
                "check before invocation."
            ),
        )
    return _passed(
        "auth_scope_declared",
        detail=f"all {len(write_caps)} write capabilit(y/ies) declare auth scopes",
    )


def _probe_audit_metadata(
    provider: CapabilityProvider,
) -> ProviderConformanceProbe:
    """Acceptance bullet: "Conformance tests prove …, audit
    behavior". Each capability must declare a ``risk_level`` so the
    audit pipeline can label the call (``read_only`` calls land in
    a separate audit lane than ``write`` / ``dangerous``).
    Read-only capabilities should additionally declare a
    ``description`` so audit records are human-readable.
    """
    no_risk = [c.capability_id for c in provider.capabilities if not c.risk_level]
    if no_risk:
        return _failed(
            "audit_metadata",
            f"{len(no_risk)} capabilit(y/ies) missing ``risk_level``: "
            + ", ".join(no_risk[:3])
            + ("…" if len(no_risk) > 3 else ""),
            hint=(
                "Set ``risk_level`` on every capability — audit "
                "downstream filters on this field. Acceptable values: "
                "``read``, ``write``, ``dangerous``."
            ),
        )
    no_alias = [
        c.capability_id for c in provider.capabilities
        if not c.routing_hints.when_to_use
    ]
    if no_alias:
        return _failed(
            "audit_metadata",
            f"{len(no_alias)} capabilit(y/ies) missing ``routing_hints.when_to_use``: "
            + ", ".join(no_alias[:3])
            + ("…" if len(no_alias) > 3 else ""),
            hint=(
                "Set ``routing_hints.when_to_use`` so audit records "
                "can describe why the capability fired."
            ),
        )
    return _passed(
        "audit_metadata",
        detail=f"all {len(provider.capabilities)} capabilit(y/ies) declare risk_level + when_to_use",
    )


def _probe_runtime_health_reachability(
    provider: CapabilityProvider,
    runtime_config: ProviderRuntimeConformanceConfig,
    *,
    http_client: httpx.Client | None,
) -> ProviderConformanceProbe:
    """Reach a declared HTTP health endpoint when one exists."""
    health = provider.health
    if health.kind != "http_get" or not health.url:
        return _skipped(
            "runtime_health_reachability",
            "provider does not declare health.kind=http_get with a URL",
        )

    try:
        response = _runtime_request(
            runtime_config,
            http_client=http_client,
            method="GET",
            url=health.url,
            headers={},
        )
    except Exception as exc:  # noqa: BLE001 - operator-facing probe detail
        return _failed(
            "runtime_health_reachability",
            f"health request failed: {type(exc).__name__}: {exc}",
            hint="Verify the registered provider health URL is reachable from this environment.",
        )
    if response.status_code < 200 or response.status_code >= 400:
        return _failed(
            "runtime_health_reachability",
            f"health endpoint returned HTTP {response.status_code}",
            hint="Fix the provider health endpoint or update provider.yaml health.url.",
        )
    return _passed(
        "runtime_health_reachability",
        detail=f"health endpoint returned HTTP {response.status_code}",
    )


def _probe_runtime_invocation_round_trip(
    provider: CapabilityProvider,
    runtime_config: ProviderRuntimeConformanceConfig,
    *,
    http_client: httpx.Client | None,
) -> ProviderConformanceProbe:
    samples = _capabilities_with_runtime_sample(provider, "runtime_sample_call")
    if not samples:
        return _skipped(
            "runtime_invocation_round_trip",
            "no capability metadata.runtime_sample_call blocks declared",
        )

    failures: list[str] = []
    for capability, sample in samples:
        failure = _exercise_invocation_sample(
            provider,
            capability,
            sample,
            runtime_config,
            http_client=http_client,
            default_expected_statuses=("ok", "dry_run_only", "needs_confirmation"),
        )
        if failure:
            failures.append(f"{capability.capability_id}: {failure}")

    if failures:
        return _failed(
            "runtime_invocation_round_trip",
            "; ".join(failures[:3]) + ("..." if len(failures) > 3 else ""),
            hint=(
                "Register the provider in the gateway runtime, ensure sample inputs are valid, "
                "and confirm the invocation result includes audit_id and trace_id."
            ),
        )
    return _passed(
        "runtime_invocation_round_trip",
        detail=f"{len(samples)} runtime sample(s) invoked through /invocations",
    )


def _probe_runtime_auth_denial(
    provider: CapabilityProvider,
    runtime_config: ProviderRuntimeConformanceConfig,
    *,
    http_client: httpx.Client | None,
) -> ProviderConformanceProbe:
    samples = _capabilities_with_runtime_sample(provider, "runtime_auth_denial_sample")
    if not samples:
        return _skipped(
            "runtime_auth_denial",
            "no capability metadata.runtime_auth_denial_sample blocks declared",
        )

    failures: list[str] = []
    for capability, sample in samples:
        failure = _exercise_invocation_sample(
            provider,
            capability,
            sample,
            runtime_config,
            http_client=http_client,
            default_expected_statuses=("denied",),
        )
        if failure:
            failures.append(f"{capability.capability_id}: {failure}")

    if failures:
        return _failed(
            "runtime_auth_denial",
            "; ".join(failures[:3]) + ("..." if len(failures) > 3 else ""),
            hint=(
                "Configure a runtime_auth_denial_sample that exercises a denied binding, "
                "policy, credential, or auth path while still returning audit_id and trace_id."
            ),
        )
    return _passed(
        "runtime_auth_denial",
        detail=f"{len(samples)} denial sample(s) returned denied with audit/trace ids",
    )


def _capabilities_with_runtime_sample(
    provider: CapabilityProvider,
    metadata_key: str,
) -> tuple[tuple[CapabilityContract, Mapping[str, Any]], ...]:
    out: list[tuple[CapabilityContract, Mapping[str, Any]]] = []
    for capability in provider.capabilities:
        sample = capability.metadata.get(metadata_key)
        if isinstance(sample, Mapping):
            out.append((capability, sample))
    return tuple(out)


def _exercise_invocation_sample(
    provider: CapabilityProvider,
    capability: CapabilityContract,
    sample: Mapping[str, Any],
    runtime_config: ProviderRuntimeConformanceConfig,
    *,
    http_client: httpx.Client | None,
    default_expected_statuses: tuple[str, ...],
) -> str:
    body = {
        "capability_id": capability.capability_id,
        "provider_id": capability.provider_id or provider.provider_id,
        "mode": str(sample.get("mode") or "execute"),
        "inputs": dict(sample.get("inputs") or {}),
        "resource_refs": list(sample.get("resource_refs") or []),
        "correlation": dict(sample.get("correlation") or {}),
        "metadata": {
            **dict(sample.get("metadata") or {}),
            "conformance_probe": "provider_runtime",
        },
    }
    expected_http_status = int(sample.get("expected_http_status") or 200)
    expected_statuses = tuple(sample.get("expected_statuses") or ())
    if not expected_statuses:
        expected_status = str(sample.get("expected_status") or "")
        expected_statuses = (expected_status,) if expected_status else default_expected_statuses

    try:
        response = _runtime_request(
            runtime_config,
            http_client=http_client,
            method="POST",
            url=runtime_config.invocation_url,
            headers=runtime_config.identity_headers(),
            json=body,
        )
    except Exception as exc:  # noqa: BLE001 - operator-facing probe detail
        return f"request failed: {type(exc).__name__}: {exc}"

    if response.status_code != expected_http_status:
        return f"expected HTTP {expected_http_status}, got {response.status_code}"

    try:
        payload = response.json()
    except ValueError as exc:
        return f"response was not JSON: {exc}"

    status = str(payload.get("status") or "")
    if status not in expected_statuses:
        return f"expected status in {expected_statuses!r}, got {status!r}"
    if not payload.get("audit_id"):
        return "result missing audit_id"
    if not payload.get("trace_id"):
        return "result missing trace_id"
    return ""


def _runtime_request(
    runtime_config: ProviderRuntimeConformanceConfig,
    *,
    http_client: httpx.Client | None,
    method: str,
    url: str,
    headers: Mapping[str, str],
    json: Mapping[str, Any] | None = None,
) -> httpx.Response:
    if http_client is not None:
        return http_client.request(method, url, headers=headers, json=json)
    with httpx.Client(
        timeout=runtime_config.timeout_seconds,
        verify=runtime_config.verify_tls,
        follow_redirects=True,
    ) as client:
        return client.request(method, url, headers=headers, json=json)


# ── Top-level runner ────────────────────────────────────────────────────────


def run_provider_conformance(
    provider_dir: Path,
    *,
    include_live_shape: bool = False,
    runtime_config: ProviderRuntimeConformanceConfig | None = None,
    http_client: Any = None,
    env: Any = None,
) -> ProviderConformanceReport:
    """Run the full universal-provider conformance suite over
    ``provider_dir``.

    Folder validation gates the rest of the suite — if the folder
    fails to load, capability-level probes are surfaced as ``skip``
    rather than ``fail`` so the report points at the actual broken
    layer (the folder) without drowning the operator in cascading
    capability failures.

    Args:
        provider_dir: provider folder to validate.
        include_live_shape: when True, append the
            ``live_response_shape`` probe (OPENAPI_PROVIDER_AUTOGEN
            W3 step 2). The probe drives every capability with a
            ``metadata.sample_call`` block, fires the request, and
            diffs the JSON response against the declared
            ``output_schema``. Off by default — opt-in because it
            performs real network I/O against the upstream service.
        http_client: pre-configured ``httpx.Client`` (timeouts /
            ``MockTransport`` for tests). Forwarded to the live
            shape probe; ignored when ``include_live_shape=False``.
        env: env-var lookup table override (defaults to ``os.environ``).
            Forwarded to the live shape probe so tests can stub
            credentials without polluting the process env.
        runtime_config: when supplied, append W8 step-4 runtime probes
            against a registered provider through the gateway's
            canonical ``POST /invocations`` route.
    """
    folder = load_provider_folder(provider_dir)
    probes: list[ProviderConformanceProbe] = []
    folder_probe = _probe_folder_validation(folder)
    probes.append(folder_probe)

    provider = folder.provider
    if folder_probe.status == "fail" or provider is None:
        for name in (
            "health_metadata",
            "discovery",
            "capability_id_naming",
            "write_provider_gate",
            "resource_consistency",
            "auth_scope_declared",
            "audit_metadata",
        ):
            probes.append(
                _skipped(
                    name,
                    "skipped because folder_validation failed",
                )
            )
        if include_live_shape:
            probes.append(
                _skipped(
                    "live_response_shape",
                    "skipped because folder_validation failed",
                )
            )
        if runtime_config is not None:
            for name in (
                "runtime_health_reachability",
                "runtime_invocation_round_trip",
                "runtime_auth_denial",
            ):
                probes.append(
                    _skipped(
                        name,
                        "skipped because folder_validation failed",
                    )
                )
        return ProviderConformanceReport(
            provider_dir=provider_dir.resolve(),
            probes=tuple(probes),
            provider_id="",
        )

    probes.append(_probe_health_metadata(provider))
    probes.append(_probe_discovery(provider))
    probes.append(_probe_capability_id_naming(provider))
    probes.append(_probe_write_provider_gate(provider))
    probes.append(_probe_resource_consistency(provider))
    probes.append(_probe_auth_scope_declared(provider))
    probes.append(_probe_audit_metadata(provider))

    if include_live_shape:
        # Lazy import — keeps a default conformance run free of any
        # httpx import overhead, and avoids loading the openapi_provider
        # package transitively in tests that don't need it.
        from .openapi_provider.live_shape import build_live_shape_conformance_probe

        probes.append(
            build_live_shape_conformance_probe(
                provider,
                provider_dir=provider_dir,
                http_client=http_client,
                env=env,
            )
        )

    if runtime_config is not None:
        probes.append(
            _probe_runtime_health_reachability(
                provider,
                runtime_config,
                http_client=http_client,
            )
        )
        probes.append(
            _probe_runtime_invocation_round_trip(
                provider,
                runtime_config,
                http_client=http_client,
            )
        )
        probes.append(
            _probe_runtime_auth_denial(
                provider,
                runtime_config,
                http_client=http_client,
            )
        )

    return ProviderConformanceReport(
        provider_dir=provider_dir.resolve(),
        probes=tuple(probes),
        provider_id=provider.provider_id,
    )


__all__ = [
    "ProbeStatus",
    "ProviderConformanceProbe",
    "ProviderConformanceReport",
    "ProviderRuntimeConformanceConfig",
    "run_provider_conformance",
]
