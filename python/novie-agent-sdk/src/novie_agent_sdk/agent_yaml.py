"""``agent.yaml`` authoring schema (EXPERT_AGENT_SDK W1).

Authors write a small ``agent.yaml`` instead of a full
``agent-manifest-v2`` JSON. This module defines the Pydantic
models for the human-facing config; the manifest generator (W2)
projects this onto ``.well-known/agent.json``.

Design intent
=============

- Required fields are intentionally minimal so a hello-world agent
  needs ~10 lines of YAML.
- Optional fields default to platform-safe values
  (``read`` risk, no side effects, no human gate).
- ``advanced.manifest_overrides`` is the documented escape hatch
  for fields the simplified schema doesn't surface yet — power
  users can still emit any manifest-v2 field by name.
- Pydantic ``model_config = ConfigDict(extra="forbid")`` on every
  section so a typo in a field name surfaces as an actionable
  error with the field path.

Example minimal ``agent.yaml``
==============================

.. code-block:: yaml

    agent:
      id: analyst
      name: Analyst
      version: 0.2.0
      type: artifact_agent
    description: Structured requirement extraction.
    capabilities:
      - agent.analyst.requirement_extraction
    runtime:
      port: 8010
    routing:
      when_to_use: Structured requirement extraction.
"""
# ruff: noqa: RUF001, RUF002, RUF003
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Platform-aligned literals: keep these in sync with
# ``novie_protocol.contracts.capability``. Re-declared locally so a
# bad SDK author sees a pydantic error pointing at the YAML line,
# not an obscure import-time failure if the platform contracts move.
AgentType = Literal["artifact_agent", "worker_agent", "tool_agent"]
GovernanceRisk = Literal["read", "write", "dangerous"]
GovernanceSideEffect = Literal[
    "none", "session", "tenant", "external", "irreversible"
]
RuntimeDuration = Literal["<1s", "<1min", "<5min", "<1h", ">1h"]
RuntimeDurability = Literal["none", "result_cache", "task_store"]
InputContractSource = Literal[
    "user_input",
    "upstream_capability",
    "runtime_context",
    "platform_projection",
]


_AGENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_\-\.]{1,63}$")
_SEMVER_PATTERN = re.compile(
    r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z\-\.]+)?(?:\+[0-9A-Za-z\-\.]+)?$"
)
_CAPABILITY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_\-\.]{1,127}$")


class _StrictModel(BaseModel):
    """Base for every section. ``extra="forbid"`` so a typo
    surfaces as a clear ``unexpected_field`` error pointing at the
    offending YAML line."""

    model_config = ConfigDict(extra="forbid")


class AgentYamlIdentity(_StrictModel):
    """The ``agent`` block — identity fields."""

    id: str = Field(
        ...,
        description=(
            "Stable agent id. Lowercase letters / digits / dot / "
            "underscore / hyphen. 2-64 chars. Used as ``agent_id`` in "
            "the platform manifest registry."
        ),
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="Human-readable display name.",
    )
    version: str = Field(
        ...,
        description=(
            "SemVer-formatted version (``MAJOR.MINOR.PATCH`` plus "
            "optional pre-release / build metadata)."
        ),
    )
    type: AgentType = Field(
        ...,
        description=(
            "Agent shape: ``artifact_agent`` produces structured "
            "artifacts (analyst-style); ``worker_agent`` runs "
            "long-running tasks (cortex-style); ``tool_agent`` "
            "exposes one or more sync tools."
        ),
    )

    @field_validator("id")
    @classmethod
    def _id_format(cls, value: str) -> str:
        if not _AGENT_ID_PATTERN.match(value):
            raise ValueError(
                "agent.id must be lowercase, 2-64 chars, "
                "[a-z0-9_-.] only — got "
                f"{value!r}"
            )
        return value

    @field_validator("version")
    @classmethod
    def _version_format(cls, value: str) -> str:
        if not _SEMVER_PATTERN.match(value):
            raise ValueError(
                "agent.version must be SemVer (MAJOR.MINOR.PATCH "
                f"with optional -prerelease/+build) — got {value!r}"
            )
        return value


class AgentYamlInputContract(_StrictModel):
    """Source semantics for one consumed artifact."""

    artifact: str = Field(..., min_length=1)
    source: InputContractSource = "upstream_capability"
    provider: str = ""
    required: bool = True


class AgentYamlInputs(_StrictModel):
    """``inputs`` block — what the agent consumes."""

    consumes: list[str] | dict[str, list[str]] = Field(
        default_factory=list,
        description=(
            "Resource type names the agent reads as inputs (e.g. "
            "``project_document`` / ``analysis_artifact``). Plain "
            "list of strings, or a map keyed by capability suffix for "
            "multi-capability agents."
        ),
    )
    input_contracts: list[AgentYamlInputContract] = Field(
        default_factory=list,
        description=(
            "Optional typed source declarations for consumed artifacts. "
            "Use this when an input is supplied by user input, platform "
            "runtime context, or a platform projection instead of an "
            "upstream capability."
        ),
    )


class AgentYamlOutputs(_StrictModel):
    """``outputs`` block — what the agent produces."""

    provides: list[str] | dict[str, list[str]] = Field(
        default_factory=list,
        description=(
            "Resource type names the agent produces (e.g. "
            "``analysis_artifact`` / ``task_bundle`` / ``code_diff``). "
            "Plain list of strings, or a map keyed by capability suffix for "
            "multi-capability agents."
        ),
    )


class AgentYamlRuntime(_StrictModel):
    """``runtime`` block — how the agent runs."""

    port: int = Field(
        ...,
        ge=1,
        le=65535,
        description="Local TCP port the agent serves on.",
    )
    duration: RuntimeDuration = Field(
        default="<1min",
        description=(
            "Expected duration class. Drives the platform's task-vs-"
            "stream routing decision."
        ),
    )
    durability: RuntimeDurability | None = Field(
        default=None,
        description=(
            "Accepted-work durability claim. ``none`` means stateless "
            "one-shot; ``result_cache`` means one-shot idempotency "
            "results survive process restart; ``task_store`` means "
            "async task records/events/results survive restart. If "
            "omitted, worker_agent defaults to ``task_store`` and "
            "one-shot agents default to ``none``."
        ),
    )
    secrets: list[str] = Field(
        default_factory=list,
        description=(
            "Names of secrets the agent needs (e.g. "
            "``github_app_token``). The platform credential broker "
            "mints scoped leases at invocation time."
        ),
    )


class AgentYamlRouting(_StrictModel):
    """``routing`` block — natural-language hints for Reception/Planner."""

    when_to_use: str = Field(
        default="",
        max_length=2000,
        description=(
            "One- or two-sentence guidance on when this agent is the "
            "right tool. Surfaced to the planner LLM verbatim."
        ),
    )
    when_not_to_use: str = Field(
        default="",
        max_length=2000,
        description=(
            "Optional negative hint — when the agent should NOT be "
            "picked. Helps the planner LLM avoid mis-routing."
        ),
    )


class AgentYamlGovernance(_StrictModel):
    """``governance`` block — policy + safety hints."""

    risk: GovernanceRisk = Field(
        default="read",
        description=(
            "Risk classification. ``read`` capabilities can run "
            "without confirmation; ``write`` and ``dangerous`` go "
            "through preview / gate flow."
        ),
    )
    side_effect: GovernanceSideEffect = Field(
        default="none",
        description=(
            "Side-effect scope. ``none`` for pure reads; ``session`` "
            "for in-session state; ``tenant`` for cross-session "
            "tenant state; ``external`` for outbound API calls; "
            "``irreversible`` for permanent effects (always gated)."
        ),
    )
    requires_tracker_issue: bool = Field(
        default=False,
        description=(
            "Whether the platform should refuse invocation unless a "
            "linked PMS issue exists (audit trail requirement)."
        ),
    )
    requires_human_gate: bool = Field(
        default=False,
        description=(
            "Whether the platform must pause for human approval "
            "before execution. Defaults to True implicitly when "
            "``risk=dangerous``."
        ),
    )


class AgentYamlCapabilityOverride(_StrictModel):
    """Per-capability contract overrides for multi-capability agents."""

    consumes: list[str] | None = None
    provides: list[str] | None = None
    input_contracts: list[AgentYamlInputContract] | None = None
    caller_types: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentYamlAdvanced(_StrictModel):
    """``advanced`` block — escape hatch for fields the simplified
    schema doesn't surface yet.

    Power users emit any ``agent-manifest-v2`` field by name under
    ``manifest_overrides``. The W2 manifest generator merges this
    dict on top of the projected manifest with overrides winning,
    so an explicit override beats any value derived from the
    simplified fields."""

    manifest_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Free-form overrides applied on top of the generated "
            "``.well-known/agent.json``. Keys must match the "
            "manifest-v2 field names exactly. The generator does not "
            "validate the override values — that's the manifest "
            "validator's job."
        ),
    )
    capability_overrides: dict[str, AgentYamlCapabilityOverride] = Field(
        default_factory=dict,
        description=(
            "Optional per-capability contract overrides. Keys are "
            "capability ids. Values can override consumes, provides, "
            "input_contracts, caller_types, and metadata for that one "
            "capability entry."
        ),
    )


class AgentYamlConfig(_StrictModel):
    """Top-level ``agent.yaml`` schema.

    Wraps the seven sections plus the description / capabilities
    fields that live at the root because they're the most edited
    parts of an agent's authoring config.
    """

    agent: AgentYamlIdentity = Field(
        ...,
        description="Identity fields — required.",
    )
    description: str = Field(
        default="",
        max_length=4000,
        description=(
            "Free-form description of what the agent does. Shown to "
            "operators in the platform UI; not consumed by the LLM "
            "directly (use ``routing.when_to_use`` for that)."
        ),
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description=(
            "Capability ids the agent advertises. Each id matches "
            "the platform's ``CapabilityContract.capability_id`` "
            "convention (e.g. ``agent.analyst.requirement_extraction``). "
            "Lowercase + dotted; uniqueness enforced."
        ),
    )
    inputs: AgentYamlInputs = Field(
        default_factory=AgentYamlInputs,
        description="Resources the agent consumes.",
    )
    outputs: AgentYamlOutputs = Field(
        default_factory=AgentYamlOutputs,
        description="Resources the agent produces.",
    )
    runtime: AgentYamlRuntime = Field(
        ...,
        description="Runtime parameters — required (port at minimum).",
    )
    routing: AgentYamlRouting = Field(
        default_factory=AgentYamlRouting,
        description="Natural-language routing hints.",
    )
    governance: AgentYamlGovernance = Field(
        default_factory=AgentYamlGovernance,
        description="Risk / side-effect / gate policy.",
    )
    advanced: AgentYamlAdvanced = Field(
        default_factory=AgentYamlAdvanced,
        description="Escape hatch for fields not yet in the schema.",
    )

    @field_validator("capabilities")
    @classmethod
    def _capabilities_unique_and_well_formed(
        cls, value: list[str],
    ) -> list[str]:
        seen: set[str] = set()
        for cap in value:
            if not _CAPABILITY_ID_PATTERN.match(cap):
                raise ValueError(
                    "capabilities[*] must be lowercase, "
                    "[a-z0-9_-.] only, 2-128 chars — "
                    f"got {cap!r}"
                )
            if cap in seen:
                raise ValueError(
                    f"capabilities contains duplicate id {cap!r}"
                )
            seen.add(cap)
        return value


__all__ = [
    "AgentType",
    "AgentYamlAdvanced",
    "AgentYamlConfig",
    "AgentYamlCapabilityOverride",
    "AgentYamlGovernance",
    "AgentYamlIdentity",
    "AgentYamlInputContract",
    "AgentYamlInputs",
    "AgentYamlOutputs",
    "AgentYamlRouting",
    "AgentYamlRuntime",
    "GovernanceRisk",
    "GovernanceSideEffect",
    "RuntimeDuration",
    "RuntimeDurability",
]
