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
  ``governance.requires_human_gate``        →  each capability entry +
                                                ``declared_gates``
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
    "<1h": (1800, 3600),
    ">1h": (3600, 14400),
}


_MANIFEST_SCHEMA_URL = "https://novie.dev/schemas/agent-manifest-v2.json"


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


def _generate_capability_entry(
    *,
    capability_id: str,
    config: AgentYamlConfig,
) -> dict[str, Any]:
    """Project one capability id from ``capabilities[]`` onto the
    ``AgentCapabilityManifestEntry`` shape."""
    expected_seconds, _max_seconds = _DURATION_SECONDS[config.runtime.duration]
    return {
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
        ),
        "requires": [],
        "conflicts": [],
        "provides": list(config.outputs.provides),
        "consumes": list(config.inputs.consumes),
        "execution_lane": "direct",
        "risk_class": _risk_class_from_governance(config),
        "requires_plan_review": False,
        "requires_tracker_issue": config.governance.requires_tracker_issue,
        "requires_human_gate": config.governance.requires_human_gate,
    }


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
