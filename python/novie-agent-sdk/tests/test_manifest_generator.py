"""EXPERT_AGENT_SDK W2 — manifest generator tests.

Locks the ``agent.yaml`` → ``.well-known/agent.json`` projection:

- minimal config produces a complete manifest dict (round-trips
  through ``AgentManifestV2.from_dict``)
- agent.type → kind / protocol_mode / exec_kind mapping
- runtime.duration → execution seconds + expected_duration_class
- capabilities list → capability_manifest entries (one per id) with
  per-entry projection rules
- governance fields propagate to every capability entry
- routing.when_to_use → metadata + first natural_language_alias
- governance.requires_human_gate → declared_gates entry
- manifest_overrides escape hatch: top-level + dotted paths
- endpoint default + explicit override + manifest_overrides
  precedence
- generated manifest passes the platform's
  ``AgentManifestV2.validate()`` check
- generated manifest is **deterministic** — same input → same dict
"""
# ruff: noqa: I001
from __future__ import annotations

from copy import deepcopy

import pytest

from novie_protocol.contracts.agent_sdk_v2 import AgentManifestV2

from novie_agent_sdk.agent_yaml import AgentYamlConfig
from novie_agent_sdk.manifest_generator import generate_agent_manifest


def _minimal_config(**overrides) -> AgentYamlConfig:
    payload = {
        "agent": {
            "id": "analyst",
            "name": "Analyst",
            "version": "0.2.0",
            "type": "artifact_agent",
        },
        "runtime": {"port": 8010},
    }
    for key, value in overrides.items():
        payload[key] = value
    return AgentYamlConfig.model_validate(payload)


def _full_config() -> AgentYamlConfig:
    return AgentYamlConfig.model_validate({
        "agent": {
            "id": "analyst",
            "name": "Analyst",
            "version": "0.2.0",
            "type": "artifact_agent",
        },
        "description": "Structured requirement extraction.",
        "capabilities": [
            "agent.analyst.requirement_extraction",
            "agent.analyst.summarize",
        ],
        "inputs": {"consumes": ["project_document"]},
        "outputs": {"provides": ["analysis_artifact"]},
        "runtime": {
            "port": 8010,
            "duration": "<1min",
            "durability": "none",
            "secrets": ["github_app_token"],
        },
        "routing": {
            "when_to_use": "Structured requirement extraction.",
            "when_not_to_use": "Open-ended chat.",
        },
        "governance": {
            "risk": "read",
            "side_effect": "session",
            "requires_tracker_issue": True,
            "requires_human_gate": False,
        },
    })


# ── Round-trip + top-level shape ──────────────────────────────────────────────


def test_minimal_manifest_round_trips_through_protocol_dataclass() -> None:
    """The generated dict must be parseable by the platform's
    ``AgentManifestV2.from_dict`` so the manifest registry can
    consume it without translation."""
    manifest = generate_agent_manifest(_minimal_config())
    parsed = AgentManifestV2.from_dict(manifest)
    assert parsed.agent_id == "analyst"
    assert parsed.name == "Analyst"
    assert parsed.version == "0.2.0"
    assert parsed.runtime == "external_a2a"


def test_minimal_manifest_passes_platform_validate() -> None:
    """The platform's ``AgentManifestV2.validate()`` returns an
    empty list on a healthy manifest. The generator must never
    emit one that fails this check."""
    manifest = generate_agent_manifest(_minimal_config())
    parsed = AgentManifestV2.from_dict(manifest)
    assert parsed.validate() == []


def test_full_manifest_passes_platform_validate() -> None:
    manifest = generate_agent_manifest(_full_config())
    parsed = AgentManifestV2.from_dict(manifest)
    assert parsed.validate() == []


def test_manifest_carries_required_top_level_keys() -> None:
    manifest = generate_agent_manifest(_minimal_config())
    required = {
        "$schema",
        "agent_id",
        "name",
        "version",
        "kind",
        "runtime",
        "capabilities",
        "capability_manifest",
        "declared_gates",
        "protocol_mode",
        "endpoint",
        "execution",
        "required_secrets",
        "supports_streaming",
        "sandbox_isolation",
        "task_bundles_path",
        "metadata",
    }
    assert required.issubset(set(manifest.keys()))


# ── agent.type → kind / protocol_mode / exec_kind ────────────────────────────


def test_artifact_agent_type_maps_to_stream_basic() -> None:
    config = _minimal_config()
    manifest = generate_agent_manifest(config)
    assert manifest["kind"] == "expert_basic"
    assert manifest["protocol_mode"] == "stream"
    assert manifest["supports_streaming"] is True


def test_worker_agent_type_maps_to_tasks_complex() -> None:
    payload = _minimal_config().model_dump()
    payload["agent"]["type"] = "worker_agent"
    config = AgentYamlConfig.model_validate(payload)
    manifest = generate_agent_manifest(config)
    assert manifest["kind"] == "expert_complex"
    assert manifest["protocol_mode"] == "tasks"
    assert manifest["execution"]["supports_cancel"] is True
    assert manifest["execution"]["supports_resume"] is True
    assert manifest["execution"]["durability"] == "task_store"
    assert manifest["metadata"]["durability"] == "task_store"


def test_tool_agent_type_maps_to_simple_basic() -> None:
    payload = _minimal_config().model_dump()
    payload["agent"]["type"] = "tool_agent"
    config = AgentYamlConfig.model_validate(payload)
    manifest = generate_agent_manifest(config)
    assert manifest["kind"] == "expert_basic"
    assert manifest["protocol_mode"] == "simple"
    assert manifest["supports_streaming"] is False
    assert manifest["execution"]["supports_cancel"] is False
    assert manifest["execution"]["durability"] == "none"


# ── runtime.duration → execution timing + entry duration class ───────────────


@pytest.mark.parametrize(
    ("duration", "expected", "max_"),
    [
        ("<1s", 1, 30),
        ("<1min", 30, 300),
        ("<1h", 1800, 3600),
        (">1h", 3600, 14400),
    ],
)
def test_runtime_duration_maps_to_execution_seconds(
    duration: str, expected: int, max_: int,
) -> None:
    payload = _minimal_config().model_dump()
    payload["runtime"]["duration"] = duration
    config = AgentYamlConfig.model_validate(payload)
    manifest = generate_agent_manifest(config)
    assert manifest["execution"]["expected_duration_seconds"] == expected
    assert manifest["execution"]["max_duration_seconds"] == max_


def test_runtime_duration_propagates_to_capability_entries() -> None:
    config = AgentYamlConfig.model_validate({
        "agent": {
            "id": "analyst",
            "name": "Analyst",
            "version": "0.2.0",
            "type": "artifact_agent",
        },
        "capabilities": ["agent.analyst.do"],
        "runtime": {"port": 8010, "duration": "<1h"},
    })
    manifest = generate_agent_manifest(config)
    entry = manifest["capability_manifest"][0]
    assert entry["expected_duration_class"] == "<1h"


def test_runtime_durability_explicit_result_cache_propagates() -> None:
    payload = _minimal_config().model_dump()
    payload["runtime"]["durability"] = "result_cache"
    config = AgentYamlConfig.model_validate(payload)
    manifest = generate_agent_manifest(config)
    assert manifest["execution"]["durability"] == "result_cache"
    assert manifest["metadata"]["durability"] == "result_cache"


def test_worker_agent_defaults_to_task_store_durability() -> None:
    payload = _minimal_config().model_dump()
    payload["agent"]["type"] = "worker_agent"
    payload["runtime"]["durability"] = None
    config = AgentYamlConfig.model_validate(payload)
    manifest = generate_agent_manifest(config)
    assert manifest["execution"]["durability"] == "task_store"
    assert manifest["metadata"]["durability"] == "task_store"


def test_oneshot_agents_default_to_no_durability() -> None:
    manifest = generate_agent_manifest(_minimal_config())
    assert manifest["execution"]["durability"] == "none"
    assert manifest["metadata"]["durability"] == "none"


# ── capabilities → capability_manifest entries ───────────────────────────────


def test_each_capability_id_produces_one_entry() -> None:
    config = _full_config()
    manifest = generate_agent_manifest(config)
    entries = manifest["capability_manifest"]
    assert len(entries) == 2
    ids = [e["capability_id"] for e in entries]
    assert ids == [
        "agent.analyst.requirement_extraction",
        "agent.analyst.summarize",
    ]


def test_capability_entry_carries_per_entry_projection() -> None:
    config = _full_config()
    manifest = generate_agent_manifest(config)
    entry = manifest["capability_manifest"][0]
    assert entry["version"] == "0.2.0"
    assert entry["risk"] == "read"
    assert entry["side_effect"] == "session"
    assert entry["consumes"] == ["project_document"]
    assert entry["provides"] == ["analysis_artifact"]
    assert entry["requires_tracker_issue"] is True
    assert entry["requires_human_gate"] is False
    assert entry["exec_kind"] == "stream"  # artifact_agent default


def test_capability_entry_humanizes_id_for_display_name() -> None:
    config = _full_config()
    manifest = generate_agent_manifest(config)
    entry = manifest["capability_manifest"][0]
    # ``agent.analyst.requirement_extraction`` → ``Requirement Extraction``
    assert entry["display_name"] == "Requirement Extraction"


def test_capability_entry_description_falls_back_to_capability_id() -> None:
    """When ``description`` is empty on the config, each entry's
    description falls back to its capability_id so the manifest is
    never blank."""
    payload = _minimal_config().model_dump()
    payload["capabilities"] = ["agent.x.foo"]
    config = AgentYamlConfig.model_validate(payload)
    manifest = generate_agent_manifest(config)
    assert manifest["capability_manifest"][0]["description"] == "agent.x.foo"


def test_when_to_use_drops_into_natural_language_aliases() -> None:
    config = _full_config()
    manifest = generate_agent_manifest(config)
    entry = manifest["capability_manifest"][0]
    aliases = entry["natural_language_aliases"]
    assert len(aliases) == 1
    assert aliases[0] == "Structured requirement extraction."


def test_zero_capabilities_produces_empty_entry_list() -> None:
    config = _minimal_config()
    manifest = generate_agent_manifest(config)
    assert manifest["capabilities"] == []
    assert manifest["capability_manifest"] == []


# ── governance + risk_class projection ───────────────────────────────────────


def test_read_governance_maps_to_read_only_risk_class() -> None:
    config = _minimal_config()
    payload = config.model_dump()
    payload["capabilities"] = ["agent.x.read"]
    payload["governance"] = {"risk": "read", "side_effect": "none"}
    config = AgentYamlConfig.model_validate(payload)
    manifest = generate_agent_manifest(config)
    assert manifest["capability_manifest"][0]["risk_class"] == "read_only"


def test_external_side_effect_maps_to_external_write() -> None:
    payload = _minimal_config().model_dump()
    payload["capabilities"] = ["agent.x.write"]
    payload["governance"] = {"risk": "write", "side_effect": "external"}
    config = AgentYamlConfig.model_validate(payload)
    manifest = generate_agent_manifest(config)
    assert manifest["capability_manifest"][0]["risk_class"] == "external_write"


def test_write_with_session_side_effect_maps_to_repo_mutation() -> None:
    payload = _minimal_config().model_dump()
    payload["capabilities"] = ["agent.x.write"]
    payload["governance"] = {"risk": "write", "side_effect": "session"}
    config = AgentYamlConfig.model_validate(payload)
    manifest = generate_agent_manifest(config)
    assert manifest["capability_manifest"][0]["risk_class"] == "repo_mutation"


def test_human_gate_appears_on_declared_gates_and_entries() -> None:
    payload = _minimal_config().model_dump()
    payload["capabilities"] = ["agent.x.write"]
    payload["governance"] = {
        "risk": "write",
        "requires_human_gate": True,
    }
    config = AgentYamlConfig.model_validate(payload)
    manifest = generate_agent_manifest(config)
    assert manifest["declared_gates"] == ["analyst.human_review"]
    assert manifest["capability_manifest"][0]["requires_human_gate"] is True


def test_no_human_gate_keeps_declared_gates_empty() -> None:
    config = _minimal_config()
    manifest = generate_agent_manifest(config)
    assert manifest["declared_gates"] == []


def test_write_risk_implies_requires_confirmation() -> None:
    payload = _minimal_config().model_dump()
    payload["capabilities"] = ["agent.x.write"]
    payload["governance"] = {"risk": "write"}
    config = AgentYamlConfig.model_validate(payload)
    manifest = generate_agent_manifest(config)
    entry = manifest["capability_manifest"][0]
    assert entry["requires_confirmation"] is True


# ── routing → metadata ───────────────────────────────────────────────────────


def test_routing_fields_land_on_metadata() -> None:
    config = _full_config()
    manifest = generate_agent_manifest(config)
    md = manifest["metadata"]
    assert md["when_to_use"] == "Structured requirement extraction."
    assert md["when_not_to_use"] == "Open-ended chat."
    assert md["description"] == "Structured requirement extraction."


def test_empty_routing_omits_metadata_keys() -> None:
    config = _minimal_config()
    manifest = generate_agent_manifest(config)
    md = manifest["metadata"]
    assert "when_to_use" not in md
    assert "when_not_to_use" not in md


# ── secrets → required_secrets ───────────────────────────────────────────────


def test_runtime_secrets_propagate_to_required_secrets() -> None:
    config = _full_config()
    manifest = generate_agent_manifest(config)
    assert manifest["required_secrets"] == ["github_app_token"]


# ── endpoint resolution + override precedence ────────────────────────────────


def test_endpoint_defaults_to_localhost_with_runtime_port() -> None:
    config = AgentYamlConfig.model_validate({
        "agent": {
            "id": "analyst",
            "name": "Analyst",
            "version": "0.2.0",
            "type": "artifact_agent",
        },
        "runtime": {"port": 8765},
    })
    manifest = generate_agent_manifest(config)
    assert manifest["endpoint"] == "http://localhost:8765"


def test_explicit_endpoint_overrides_default() -> None:
    config = _minimal_config()
    manifest = generate_agent_manifest(
        config, endpoint="https://analyst.example.com",
    )
    assert manifest["endpoint"] == "https://analyst.example.com"


def test_manifest_overrides_endpoint_wins_over_explicit_arg() -> None:
    """``advanced.manifest_overrides`` is the final word — power
    users always win."""
    payload = _minimal_config().model_dump()
    payload["advanced"] = {
        "manifest_overrides": {
            "endpoint": "https://override.example.com",
        },
    }
    config = AgentYamlConfig.model_validate(payload)
    manifest = generate_agent_manifest(
        config, endpoint="https://explicit-arg.example.com",
    )
    assert manifest["endpoint"] == "https://override.example.com"


# ── manifest_overrides escape hatch ──────────────────────────────────────────


def test_manifest_overrides_top_level_key() -> None:
    payload = _minimal_config().model_dump()
    payload["advanced"] = {
        "manifest_overrides": {
            "task_bundles_path": "/var/lib/bundles",
        },
    }
    config = AgentYamlConfig.model_validate(payload)
    manifest = generate_agent_manifest(config)
    assert manifest["task_bundles_path"] == "/var/lib/bundles"


def test_manifest_overrides_dotted_path_writes_into_nested_dict() -> None:
    """Dotted-path semantics so power users don't need to re-emit
    an entire nested object to flip one boolean."""
    payload = _minimal_config().model_dump()
    payload["advanced"] = {
        "manifest_overrides": {
            "execution.idempotent": True,
        },
    }
    config = AgentYamlConfig.model_validate(payload)
    manifest = generate_agent_manifest(config)
    assert manifest["execution"]["idempotent"] is True
    # And the rest of execution stayed put.
    assert "expected_duration_seconds" in manifest["execution"]


def test_manifest_overrides_can_replace_kind() -> None:
    payload = _minimal_config().model_dump()
    payload["advanced"] = {"manifest_overrides": {"kind": "custom_kind"}}
    config = AgentYamlConfig.model_validate(payload)
    manifest = generate_agent_manifest(config)
    assert manifest["kind"] == "custom_kind"


# ── Determinism ──────────────────────────────────────────────────────────────


def test_manifest_is_deterministic() -> None:
    """Acceptance bullet: 'Generated manifests are deterministic
    and stable in git diffs.' Two calls with the same config →
    structurally equal output."""
    config = _full_config()
    manifest1 = generate_agent_manifest(config)
    manifest2 = generate_agent_manifest(config)
    assert manifest1 == manifest2


def test_manifest_round_trips_through_deepcopy() -> None:
    """Sanity: the dict has no shared mutable state between calls."""
    config = _full_config()
    manifest1 = generate_agent_manifest(config)
    manifest2 = deepcopy(manifest1)
    manifest2["capability_manifest"][0]["risk"] = "dangerous"
    # Re-generating from the same config matches the original,
    # not the mutated copy.
    manifest3 = generate_agent_manifest(config)
    assert manifest3 == manifest1
    assert manifest3 != manifest2


# ── Edge cases ───────────────────────────────────────────────────────────────


def test_runtime_external_a2a_constant() -> None:
    """SDK agents are always external_a2a — overriding via
    manifest_overrides is allowed but the default is fixed."""
    config = _minimal_config()
    manifest = generate_agent_manifest(config)
    assert manifest["runtime"] == "external_a2a"


def test_schema_url_present() -> None:
    config = _minimal_config()
    manifest = generate_agent_manifest(config)
    assert manifest["$schema"] == (
        "https://novie.dev/schemas/agent-manifest-v2.json"
    )
