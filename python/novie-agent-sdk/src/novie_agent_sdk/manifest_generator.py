"""``agent.yaml`` → manifest-v2 generator
(EXPERT_AGENT_SDK W2 — generator side).

Projects a validated ``AgentYamlConfig`` (W1) onto the
``.well-known/agent.json`` shape consumed by the platform's
``ManifestRegistry``. The generator is **pure** — accepts a
config + optional endpoint, returns a plain JSON-compatible
``dict`` that round-trips through
``AgentManifestV2.from_dict``.

This slice lands the **library**. The CLI commands
(``novie agents generate-manifest`` / ``novie agents validate``)
and the W2 validator come in subsequent slices so each PR stays
small.

Projection rules
================

- ``agent.id``                              →  ``agent_id``
- ``agent.name``                            →  ``name``
- ``agent.version``                         →  ``version``
- ``agent.type``                            →  ``kind`` + ``protocol_mode`` defaults
   - ``artifact_agent``  →  ``kind=expert_basic``, ``protocol_mode=stream``
   - ``worker_agent``    →  ``kind=expert_complex``, ``protocol_mode=tasks``
   - ``tool_agent``      →  ``kind=expert_basic``, ``protocol_mode=simple``
- ``capabilities[]``                        →  ``capabilities`` + one
                                                ``capability_manifest`` entry per id
- ``inputs.consumes`` / ``outputs.provides``  →  consumes / provides on every
                                                  capability entry
- ``runtime.port``                          →  ``endpoint = http://localhost:{port}``
                                                (override via explicit
                                                ``endpoint`` arg or
                                                ``advanced.manifest_overrides``)
- ``runtime.duration``                      →  ``execution.expected_duration_seconds`` +
                                                ``max_duration_seconds`` +
                                                each entry's ``expected_duration_class``
- ``runtime.durability``                    →  ``execution.durability`` +
                                                ``metadata['durability']``
- ``runtime.secrets[]``                     →  ``required_secrets``
- ``routing.when_to_use``                   →  ``metadata['when_to_use']`` + each
                                                entry's ``natural_language_aliases``
                                                head
- ``routing.when_not_to_use``               →  ``metadata['when_not_to_use']``
- ``governance.risk`` /
  ``governance.side_effect``                →  every capability entry's
                                                ``risk`` / ``side_effect``
- ``governance.requires_tracker_issue`` /
  ``governance.requires_human_gate``        →  each capability entry's
                                                nested ``governance`` +
                                                ``declared_gates``
- ``advanced.capability_overrides.*.gates`` →  the matching capability entry
                                                only
- ``advanced.manifest_overrides``           →  applied last as a top-level
                                                ``dict.update`` so power users
                                                emit any manifest-v2 field by
                                                name. Nested keys (e.g.
                                                ``execution.idempotent``) use
                                                dotted-path semantics.
"""
# ruff: noqa: RUF001, RUF002, RUF003
from __future__ import annotations

from typing import Any, Literal

from .agent_yaml import (
    AgentType,
    AgentYamlConfig,
    RuntimeDuration,
    RuntimeDurability,
)


_KIND_FROM_TYPE: dict[AgentType, str] = {
    "artifact_agent": "expert_basic",
    "worker_agent": "expert_complex",
    "tool_agent": "expert_basic",
}


_PROTOCOL_MODE_FROM_TYPE: dict[AgentType, Literal["simple", "stream", "tasks"]] = {
    "artifact_agent": "stream",
    "worker_agent": "tasks",
    "tool_agent": "simple",
}


_EXEC_KIND_FROM_TYPE: dict[AgentType, str] = {
    "artifact_agent": "stream",
    "worker_agent": "async",
    "tool_agent": "sync",
}


# (expected_seconds, max_seconds) per duration class.
_DURATION_SECONDS: dict[RuntimeDuration, tuple[int, int]] = {
    "<1s": (1, 30),
    "<1min": (30, 300),
    "<5min": (300, 1800),
    "<1h": (1800, 3600),
    ">1h": (3600, 14400),
}


_MANIFEST_SCHEMA_URL = "https://novie.dev/schemas/agent-manifest-v2.json"
_STANDARD_INPUT_PROVIDERS = {
    "task_brief": ("user_input", "platform.user_input.task_brief"),
    "user_goal": ("user_input", "platform.user_input.user_goal"),
    "brief": ("user_input", "platform.user_input.brief"),
    "task_bundle": ("runtime_context", "platform.pms.ticket_execution"),
    "project.repo.default": ("runtime_context", "platform.project_context.repo_default"),
    "tracker_issue": ("platform_projection", "platform.tracker.ingestion"),
}


def _set_dotted_path(target: dict[str, Any], key: str, value: Any) -> None:
    """Apply a manifest_overrides key. Supports dotted paths
    (e.g. ``execution.idempotent``) so power users don't need to
    re-emit an entire nested object to flip one boolean."""
    if "." not in key:
        target[key] = value
        return
    parts = key.split(".")
    cursor: Any = target
    for part in parts[:-1]:
        if not isinstance(cursor.get(part), dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _consumes_for_capability(config: AgentYamlConfig, capability_id: str) -> list[str]:
    consumes = config.inputs.consumes
    if isinstance(consumes, dict):
        suffix = capability_id.rsplit(".", 1)[-1]
        return list(consumes.get(capability_id) or consumes.get(suffix) or [])
    return list(consumes)


def _provides_for_capability(config: AgentYamlConfig, capability_id: str) -> list[str]:
    provides = config.outputs.provides
    if isinstance(provides, dict):
        suffix = capability_id.rsplit(".", 1)[-1]
        return list(provides.get(capability_id) or provides.get(suffix) or [])
    return list(provides)


def _input_contracts_for_consumes(consumes: list[str]) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for artifact in consumes:
        source, provider = _STANDARD_INPUT_PROVIDERS.get(
            artifact,
            ("upstream_capability", ""),
        )
        contract: dict[str, Any] = {
            "artifact": artifact,
            "source": source,
            "required": True,
        }
        if provider:
            contract["provider"] = provider
        contracts.append(contract)
    return contracts


def _generate_capability_entry(
    *,
    capability_id: str,
    config: AgentYamlConfig,
) -> dict[str, Any]:
    """Project one capability id from ``capabilities[]`` onto the
    ``AgentCapabilityManifestEntry`` shape."""
    expected_seconds, _max_seconds = _DURATION_SECONDS[config.runtime.duration]
    override = config.advanced.capability_overrides.get(capability_id)
    consumes = (
        list(override.consumes)
        if override is not None and override.consumes is not None
        else _consumes_for_capability(config, capability_id)
    )
    provides = (
        list(override.provides)
        if override is not None and override.provides is not None
        else _provides_for_capability(config, capability_id)
    )
    input_contracts = []
    if override is not None and override.input_contracts is not None:
        input_contracts = [
            contract.model_dump(exclude_none=True)
            for contract in override.input_contracts
        ]
    elif config.inputs.input_contracts:
        input_contracts = [
            contract.model_dump(exclude_none=True)
            for contract in config.inputs.input_contracts
        ]
    else:
        input_contracts = _input_contracts_for_consumes(consumes)
    gates = (
        [
            {
                key: value
                for key, value in gate.model_dump(exclude_none=True).items()
                if key != "boundary_id" or value
            }
            for gate in override.gates
        ]
        if override is not None
        else []
    )
    has_required_gate = any(gate.required for gate in override.gates) if override else False
    entry = {
        "capability_id": capability_id,
        "version": config.agent.version,
        "display_name": _humanize_capability_id(capability_id),
        "description": config.description or capability_id,
        "input_schema": {},
        "output_schema": {},
        "risk": config.governance.risk,
        "side_effect": config.governance.side_effect,
        "exec_kind": _EXEC_KIND_FROM_TYPE[config.agent.type],
        "runtime_ref": "",
        "tags": [],
        "natural_language_aliases": (
            [config.routing.when_to_use[:120]]
            if config.routing.when_to_use
            else []
        ),
        "examples": [],
        "idempotent": False,
        "expected_duration_class": config.runtime.duration,
        "streamable": config.agent.type == "artifact_agent",
        "cancellation_supported": config.agent.type == "worker_agent",
        "progress_events": config.agent.type == "worker_agent",
        "dry_run_supported": False,
        "requires_confirmation": (
            config.governance.requires_human_gate
            or config.governance.risk in ("write", "dangerous")
            or any(
                gate.required and gate.timing != "post_step"
                for gate in (override.gates if override else ())
            )
        ),
        "requires": [],
        "conflicts": [],
        "provides": provides,
        "consumes": consumes,
        "input_contracts": input_contracts,
        "execution_lane": "direct",
        "risk_class": _risk_class_from_governance(config),
        "governance": {
            "requires_plan_review": False,
            "requires_tracker_issue": config.governance.requires_tracker_issue,
            "requires_human_gate": (
                config.governance.requires_human_gate or has_required_gate
            ),
        },
        "gates": gates,
    }
    if override is not None:
        if override.side_effect_boundaries:
            entry["side_effect_boundaries"] = list(
                override.side_effect_boundaries
            )
        if override.caller_types is not None:
            entry["caller_types"] = list(override.caller_types)
        if override.metadata:
            metadata = dict(entry.get("metadata") or {})
            metadata.update(override.metadata)
            entry["metadata"] = metadata
    return entry


def _humanize_capability_id(capability_id: str) -> str:
    """``agent.analyst.requirement_extraction`` →
    ``Requirement Extraction``. Last dotted segment with snake_case
    expanded into Title Case."""
    last = capability_id.rsplit(".", 1)[-1]
    return last.replace("_", " ").replace("-", " ").title()


def _risk_class_from_governance(
    config: AgentYamlConfig,
) -> str:
    """Project the simplified ``governance.risk`` /
    ``side_effect`` onto the platform's narrower ``risk_class``
    literal (``read_only`` / ``repo_mutation`` / ``external_write``)."""
    if config.governance.risk == "read":
        return "read_only"
    if config.governance.side_effect == "external":
        return "external_write"
    return "repo_mutation"


def _resolve_durability(config: AgentYamlConfig) -> RuntimeDurability:
    if config.runtime.durability is not None:
        return config.runtime.durability
    if config.agent.type == "worker_agent":
        return "task_store"
    return "none"


def _generate_execution_block(config: AgentYamlConfig) -> dict[str, Any]:
    expected_seconds, max_seconds = _DURATION_SECONDS[config.runtime.duration]
    return {
        "expected_duration_seconds": expected_seconds,
        "max_duration_seconds": max_seconds,
        "idempotent": False,
        "supports_cancel": config.agent.type == "worker_agent",
        "supports_resume": config.agent.type == "worker_agent",
        "emits_events": config.agent.type
        in ("artifact_agent", "worker_agent"),
        "durability": _resolve_durability(config),
    }


def _generate_metadata(config: AgentYamlConfig) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if config.routing.when_to_use:
        metadata["when_to_use"] = config.routing.when_to_use
    if config.routing.when_not_to_use:
        metadata["when_not_to_use"] = config.routing.when_not_to_use
    if config.description:
        metadata["description"] = config.description
    metadata["durability"] = _resolve_durability(config)
    return metadata


def _generate_declared_gates(config: AgentYamlConfig) -> list[str]:
    """Map ``governance.requires_human_gate`` onto a declared gate
    name. The platform doesn't enforce gate naming today, so the
    generator emits a stable convention: ``{agent_id}.human_review``
    when the flag is set."""
    if config.governance.requires_human_gate:
        return [f"{config.agent.id}.human_review"]
    return []


def generate_agent_manifest(
    config: AgentYamlConfig,
    *,
    endpoint: str | None = None,
) -> dict[str, Any]:
    """Project a validated ``AgentYamlConfig`` onto the
    ``.well-known/agent.json`` dict shape.

    Parameters
    ----------
    config
        The W1 ``AgentYamlConfig`` (already validated).
    endpoint
        Optional explicit endpoint. Defaults to
        ``http://localhost:{runtime.port}``. ``advanced.manifest_overrides
        ['endpoint']`` overrides whatever the caller passes.

    Returns
    -------
    dict
        JSON-compatible dict ready to be written to
        ``.well-known/agent.json``. Round-trips through
        ``AgentManifestV2.from_dict``.
    """
    resolved_endpoint = (
        endpoint
        if endpoint is not None
        else f"http://localhost:{config.runtime.port}"
    )

    capability_manifest = [
        _generate_capability_entry(
            capability_id=capability_id, config=config,
        )
        for capability_id in config.capabilities
    ]

    manifest: dict[str, Any] = {
        "$schema": _MANIFEST_SCHEMA_URL,
        "agent_id": config.agent.id,
        "name": config.agent.name,
        "version": config.agent.version,
        "kind": _KIND_FROM_TYPE[config.agent.type],
        "runtime": "external_a2a",
        "capabilities": list(config.capabilities),
        "capability_manifest": capability_manifest,
        "declared_gates": _generate_declared_gates(config),
        "protocol_mode": _PROTOCOL_MODE_FROM_TYPE[config.agent.type],
        "endpoint": resolved_endpoint,
        "execution": _generate_execution_block(config),
        "required_secrets": list(config.runtime.secrets),
        "supports_streaming": config.agent.type == "artifact_agent",
        "sandbox_isolation": "shared",
        "task_bundles_path": "",
        "metadata": _generate_metadata(config),
    }

    # ``advanced.manifest_overrides`` lands last so power users can
    # tweak any field. Dotted paths supported: ``execution.idempotent``
    # → ``manifest['execution']['idempotent']``.
    for key, value in config.advanced.manifest_overrides.items():
        _set_dotted_path(manifest, str(key), value)

    return manifest


__all__ = [
    "generate_agent_manifest",
]
