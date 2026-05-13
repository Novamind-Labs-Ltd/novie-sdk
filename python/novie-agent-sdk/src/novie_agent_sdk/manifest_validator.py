"""``agent.yaml`` + manifest validator
(EXPERT_AGENT_SDK W2 step 2 — validator side).

Two passes:

1. **Author-facing pass** (``validate_agent_yaml``): catches issues
   the schema can't, like the capability-id naming convention,
   missing ``outputs.provides`` for an artifact_agent, or
   suspicious risk/gate combinations. Diagnostics are written
   for the agent author who edits ``agent.yaml`` — concrete
   field paths and remediation hints.
2. **Platform-facing pass** (``validate_generated_manifest``):
   parses the generated dict through ``AgentManifestV2.from_dict``
   and surfaces the platform's structural ``.validate()`` errors
   as diagnostics. These are the same checks the
   ``ManifestRegistry`` runs at registration time, so a green
   validator output guarantees registration won't reject the
   manifest.

The ``ManifestValidationIssue`` envelope carries **both**
author-facing and platform-facing message variants so a CLI
(W2 step 3) can pick which audience to render to.

Severity contract
=================

- ``error`` — the platform's ``ManifestRegistry`` would reject
  the manifest. Author must fix before registering.
- ``warning`` — manifest will register but the value is
  suspicious or under-specified. Most useful early in agent
  development; can be suppressed.
- ``info`` — informational notes (deprecations, defaults that
  weren't explicitly set). Not yet emitted; reserved for future
  slices.
"""
# ruff: noqa: RUF001, RUF002, RUF003
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

from novie_protocol.contracts.agent_sdk_v2 import AgentManifestV2

from .agent_yaml import AgentYamlConfig


Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True, slots=True)
class ManifestValidationIssue:
    """Single diagnostic produced by the validator.

    ``code`` is a short stable identifier so callers can branch
    programmatically (e.g. CLI exit codes). ``field_path`` follows
    the same conventions ``AgentYamlConfig`` uses internally
    (dotted for sections, ``capabilities[0]`` for list entries).
    Both message variants are pre-rendered.
    """

    severity: Severity
    field_path: str
    code: str
    author_message: str
    platform_message: str


@dataclass(frozen=True, slots=True)
class ManifestValidationResult:
    """Aggregate of all diagnostics from one validation run."""

    issues: tuple[ManifestValidationIssue, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[ManifestValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[ManifestValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "warning")

    @property
    def infos(self) -> tuple[ManifestValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "info")

    @property
    def is_valid(self) -> bool:
        """True iff there are no ``error``-severity issues. Warnings
        and infos don't fail validation — they're surfaced for the
        CLI / IDE to render."""
        return not self.errors

    def merged(
        self, other: "ManifestValidationResult",
    ) -> "ManifestValidationResult":
        return ManifestValidationResult(
            issues=tuple(self.issues) + tuple(other.issues),
        )


# ── Author-facing checks ──────────────────────────────────────────────────────


def _check_capability_id_naming(
    config: AgentYamlConfig,
) -> list[ManifestValidationIssue]:
    """Acceptance bullet: 'capability ids follow ``agent.<agent_id>.
    <capability>`` convention'. Surface as a warning so existing
    agents that don't follow the convention can still register;
    upgrade to error in a future slice once existing agents are
    migrated."""
    out: list[ManifestValidationIssue] = []
    expected_prefix = f"agent.{config.agent.id}."
    for index, capability_id in enumerate(config.capabilities):
        if capability_id.startswith(expected_prefix):
            continue
        out.append(
            ManifestValidationIssue(
                severity="warning",
                field_path=f"capabilities[{index}]",
                code="capability_id_naming_off_convention",
                author_message=(
                    f"capability id {capability_id!r} doesn't follow "
                    f"the ``agent.<agent_id>.<capability>`` convention "
                    f"(expected prefix ``{expected_prefix}``). The "
                    "platform won't reject this, but the planner's "
                    "capability search ranks by-prefix matches higher."
                ),
                platform_message=(
                    f"capability_id {capability_id!r} does not match "
                    f"prefix {expected_prefix!r} for agent_id "
                    f"{config.agent.id!r}; ManifestRegistry accepts but "
                    "convention check warns."
                ),
            )
        )
    return out


def _check_artifact_agent_provides(
    config: AgentYamlConfig,
) -> list[ManifestValidationIssue]:
    """An ``artifact_agent`` whose ``outputs.provides`` is empty has
    no observable effect — flag as a warning."""
    if config.agent.type != "artifact_agent":
        return []
    if config.outputs.provides:
        return []
    return [
        ManifestValidationIssue(
            severity="warning",
            field_path="outputs.provides",
            code="artifact_agent_missing_provides",
            author_message=(
                "artifact_agent type declares no ``outputs.provides``. "
                "The planner can't wire this agent's output into a "
                "downstream step — list at least one resource type "
                "(e.g. ``analysis_artifact``) or use ``tool_agent`` "
                "type for sync read-only tools."
            ),
            platform_message=(
                "AgentManifestV2.kind=expert_basic with empty "
                "capability_manifest[*].provides. Manifest registers "
                "but is unwireable in plan compilation."
            ),
        )
    ]


def _check_worker_agent_duration(
    config: AgentYamlConfig,
) -> list[ManifestValidationIssue]:
    """A ``worker_agent`` with ``runtime.duration=<1s`` is almost
    certainly mis-typed as worker_agent — those are for long
    runs."""
    if config.agent.type != "worker_agent":
        return []
    if config.runtime.duration not in ("<1s",):
        return []
    return [
        ManifestValidationIssue(
            severity="warning",
            field_path="runtime.duration",
            code="worker_agent_too_short",
            author_message=(
                "worker_agent type with duration ``<1s`` looks "
                "mis-classified — worker_agent is for long-running "
                "tasks (cortex-style). Use ``tool_agent`` for sync "
                "<1s tools or pick a longer ``runtime.duration``."
            ),
            platform_message=(
                "AgentManifestV2.kind=expert_complex with "
                "execution.expected_duration_seconds<=1 violates the "
                "worker shape contract."
            ),
        )
    ]


def _resolved_durability(config: AgentYamlConfig) -> str:
    if config.runtime.durability is not None:
        return config.runtime.durability
    if config.agent.type == "worker_agent":
        return "task_store"
    return "none"


def _check_durability_matches_agent_shape(
    config: AgentYamlConfig,
) -> list[ManifestValidationIssue]:
    durability = _resolved_durability(config)
    if config.agent.type == "worker_agent" and durability != "task_store":
        return [
            ManifestValidationIssue(
                severity="error",
                field_path="runtime.durability",
                code="worker_agent_requires_task_store",
                author_message=(
                    "worker_agent must declare ``runtime.durability: "
                    "task_store`` because accepted async tasks can outlive "
                    "the initial HTTP request. Use artifact_agent/tool_agent "
                    "for stateless one-shot work."
                ),
                platform_message=(
                    "protocol_mode=tasks requires execution.durability="
                    "'task_store' for production-safe task resume."
                ),
            )
        ]
    if config.agent.type != "worker_agent" and durability == "task_store":
        return [
            ManifestValidationIssue(
                severity="warning",
                field_path="runtime.durability",
                code="oneshot_task_store_durability_unusual",
                author_message=(
                    "one-shot agents normally use ``none`` or "
                    "``result_cache`` durability. ``task_store`` is "
                    "reserved for async task agents; verify the agent "
                    "type is correct."
                ),
                platform_message=(
                    "protocol_mode is simple/stream but "
                    "execution.durability='task_store'."
                ),
            )
        ]
    return []


def _check_oneshot_side_effect_durability(
    config: AgentYamlConfig,
) -> list[ManifestValidationIssue]:
    if config.agent.type == "worker_agent":
        return []
    durability = _resolved_durability(config)
    has_side_effect = (
        config.governance.risk in ("write", "dangerous")
        or config.governance.side_effect != "none"
    )
    if not has_side_effect or durability != "none":
        return []
    return [
        ManifestValidationIssue(
            severity="warning",
            field_path="runtime.durability",
            code="oneshot_side_effect_without_result_cache",
            author_message=(
                "this one-shot agent declares side effects but no durable "
                "result cache. Set ``runtime.durability: result_cache`` "
                "when duplicate Idempotency-Key requests must replay "
                "after process restart, or keep ``none`` only if the "
                "operation is explicitly confirmed non-idempotent."
            ),
            platform_message=(
                "simple/stream side-effecting capability declares "
                "execution.durability='none'; production retries may not "
                "be replay-safe across restart."
            ),
        )
    ]


def _check_endpoint_port_consistency(
    config: AgentYamlConfig,
) -> list[ManifestValidationIssue]:
    """Acceptance bullet: 'validate endpoint/port consistency'.
    When the author overrides the endpoint via
    ``advanced.manifest_overrides``, warn if the URL doesn't
    contain ``runtime.port`` — usually a copy-paste mistake."""
    override = config.advanced.manifest_overrides.get("endpoint")
    if not override:
        return []
    if not isinstance(override, str):
        return [
            ManifestValidationIssue(
                severity="error",
                field_path="advanced.manifest_overrides.endpoint",
                code="endpoint_must_be_string",
                author_message=(
                    "``manifest_overrides.endpoint`` must be a string. "
                    f"Got: {type(override).__name__}."
                ),
                platform_message=(
                    "AgentManifestV2.endpoint must be str; got "
                    f"{type(override).__name__}."
                ),
            )
        ]
    parsed = urlparse(override)
    if parsed.port is None:
        # Endpoint without an explicit port (e.g. https://example.com)
        # is fine — operator may be running behind a reverse proxy.
        return []
    if parsed.port == config.runtime.port:
        return []
    return [
        ManifestValidationIssue(
            severity="warning",
            field_path="advanced.manifest_overrides.endpoint",
            code="endpoint_port_mismatch",
            author_message=(
                f"``manifest_overrides.endpoint`` port "
                f"{parsed.port} differs from ``runtime.port`` "
                f"{config.runtime.port}. Often a copy-paste mistake; "
                "if intentional (e.g. reverse proxy port), suppress "
                "with ``manifest_overrides.runtime_port_consistency: "
                "false`` (forthcoming flag)."
            ),
            platform_message=(
                "endpoint URL port "
                f"{parsed.port} != AgentYaml.runtime.port "
                f"{config.runtime.port}; manifest registers but "
                "operator-side port mapping must be in place."
            ),
        )
    ]


def _check_human_gate_with_read_risk(
    config: AgentYamlConfig,
) -> list[ManifestValidationIssue]:
    """``requires_human_gate=True`` on a ``read`` capability is
    unusual — gates exist to slow down side-effecting actions.
    Warn so the author confirms the intent."""
    if not config.governance.requires_human_gate:
        return []
    if config.governance.risk != "read":
        return []
    return [
        ManifestValidationIssue(
            severity="warning",
            field_path="governance.requires_human_gate",
            code="human_gate_on_read_risk",
            author_message=(
                "``requires_human_gate=true`` with ``risk=read`` is "
                "unusual — human gates exist to slow down "
                "side-effecting actions. If you genuinely want a "
                "manual checkpoint on a read-only agent, set "
                "``risk=write`` and ``side_effect=session`` for "
                "more accurate platform routing."
            ),
            platform_message=(
                "AgentManifestV2 declares declared_gates with "
                "capability_manifest[*].risk=read; platform routes "
                "this through the read fast-path which bypasses the "
                "gate."
            ),
        )
    ]


def _check_capabilities_non_empty(
    config: AgentYamlConfig,
) -> list[ManifestValidationIssue]:
    """An agent registering with zero capabilities is a no-op from
    the platform's perspective."""
    if config.capabilities:
        return []
    return [
        ManifestValidationIssue(
            severity="warning",
            field_path="capabilities",
            code="capabilities_empty",
            author_message=(
                "``capabilities`` is empty — the agent registers but "
                "exposes nothing for the planner to invoke. Add at "
                "least one capability_id like "
                f"``agent.{config.agent.id}.do_thing``."
            ),
            platform_message=(
                "AgentManifestV2.capability_manifest is empty; agent "
                "registers but contributes no rows to the W2 capability "
                "discovery surface."
            ),
        )
    ]


_AUTHOR_CHECKS: tuple = (
    _check_capability_id_naming,
    _check_artifact_agent_provides,
    _check_worker_agent_duration,
    _check_durability_matches_agent_shape,
    _check_oneshot_side_effect_durability,
    _check_endpoint_port_consistency,
    _check_human_gate_with_read_risk,
    _check_capabilities_non_empty,
)


def validate_agent_yaml(
    config: AgentYamlConfig,
) -> ManifestValidationResult:
    """Run author-facing checks on the ``AgentYamlConfig``.

    These are issues the Pydantic schema can't catch — naming
    conventions, semantic completeness, suspicious flag
    combinations. Diagnostics are written for the agent author.
    """
    issues: list[ManifestValidationIssue] = []
    for check in _AUTHOR_CHECKS:
        issues.extend(check(config))
    return ManifestValidationResult(issues=tuple(issues))


# ── Platform-facing checks ────────────────────────────────────────────────────


def validate_generated_manifest(
    manifest: dict[str, Any],
) -> ManifestValidationResult:
    """Run the platform's structural validation on a generated
    manifest dict and surface results as diagnostics.

    Parses through ``AgentManifestV2.from_dict``; calls the
    platform's ``.validate()``; converts each error string into
    an issue with both author and platform message variants. A
    parse failure (malformed dict shape) is reported as a single
    ``manifest_unparseable`` error.
    """
    try:
        parsed = AgentManifestV2.from_dict(manifest)
    except Exception as exc:
        return ManifestValidationResult(
            issues=(
                ManifestValidationIssue(
                    severity="error",
                    field_path="<manifest>",
                    code="manifest_unparseable",
                    author_message=(
                        "The generated manifest is malformed and "
                        "can't be parsed by the platform's "
                        "AgentManifestV2 contract. Usually means a "
                        "``manifest_overrides`` value has the wrong "
                        f"type. Underlying error: {exc!r}"
                    ),
                    platform_message=(
                        f"AgentManifestV2.from_dict raised: {exc!r}"
                    ),
                ),
            ),
        )
    issues: list[ManifestValidationIssue] = []
    for error_message in parsed.validate():
        issues.append(
            ManifestValidationIssue(
                severity="error",
                field_path="<platform_validate>",
                code="platform_validate_failure",
                author_message=(
                    "The generated manifest is structurally invalid: "
                    f"{error_message}. The platform's ManifestRegistry "
                    "would reject this; fix the corresponding "
                    "agent.yaml field before retrying."
                ),
                platform_message=(
                    f"AgentManifestV2.validate() reported: {error_message}"
                ),
            )
        )
    return ManifestValidationResult(issues=tuple(issues))


# ── Combined helper ───────────────────────────────────────────────────────────


def validate_agent_yaml_and_manifest(
    config: AgentYamlConfig,
    manifest: dict[str, Any],
) -> ManifestValidationResult:
    """Convenience: run both passes and merge the results.

    Author-facing diagnostics first, then platform-facing.
    Callers that want to short-circuit on author errors before
    spending time on platform checks should run them separately.
    """
    return validate_agent_yaml(config).merged(
        validate_generated_manifest(manifest),
    )


__all__ = [
    "ManifestValidationIssue",
    "ManifestValidationResult",
    "Severity",
    "validate_agent_yaml",
    "validate_agent_yaml_and_manifest",
    "validate_generated_manifest",
]
