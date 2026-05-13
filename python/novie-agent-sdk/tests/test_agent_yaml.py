"""EXPERT_AGENT_SDK W1 — ``agent.yaml`` authoring schema tests.

Locks the contract:

- minimal valid config (just identity + runtime.port)
- full config exercising every field
- invalid inputs surface field-path-aware error messages
- agent.id format / version SemVer / port range / capability id
  format / capability uniqueness
- ``extra="forbid"`` catches typos at any nesting level
- ``advanced.manifest_overrides`` accepts arbitrary keys without
  validating their shape (escape hatch contract)
- defaults match documented platform-safe baseline
  (read risk / no side effects / no human gate)
"""
# ruff: noqa: I001
from __future__ import annotations

import pytest
from pydantic import ValidationError

from novie_agent_sdk.agent_yaml import (
    AgentYamlConfig,
    AgentYamlIdentity,
    AgentYamlRuntime,
)


def _minimal_payload() -> dict:
    return {
        "agent": {
            "id": "analyst",
            "name": "Analyst",
            "version": "0.2.0",
            "type": "artifact_agent",
        },
        "runtime": {"port": 8010},
    }


def _full_payload() -> dict:
    return {
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
        "advanced": {
            "manifest_overrides": {
                "compat_range": ["agent-manifest-v2"],
                "tags": ["analyst"],
            },
        },
    }


# ── Minimal / full happy paths ────────────────────────────────────────────────


def test_minimal_payload_validates() -> None:
    config = AgentYamlConfig.model_validate(_minimal_payload())
    assert config.agent.id == "analyst"
    assert config.runtime.port == 8010
    # Defaults applied.
    assert config.governance.risk == "read"
    assert config.governance.side_effect == "none"
    assert config.governance.requires_human_gate is False
    assert config.runtime.duration == "<1min"
    assert config.capabilities == []
    assert config.advanced.manifest_overrides == {}


def test_full_payload_validates() -> None:
    config = AgentYamlConfig.model_validate(_full_payload())
    assert config.agent.type == "artifact_agent"
    assert config.capabilities == [
        "agent.analyst.requirement_extraction",
        "agent.analyst.summarize",
    ]
    assert config.inputs.consumes == ["project_document"]
    assert config.outputs.provides == ["analysis_artifact"]
    assert config.runtime.secrets == ["github_app_token"]
    assert config.runtime.durability == "none"
    assert config.governance.requires_tracker_issue is True
    assert config.advanced.manifest_overrides["tags"] == ["analyst"]


# ── agent.id format ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "agent_id",
    ["analyst", "agent.task_splitter", "expert-agent-1", "a1"],
)
def test_valid_agent_ids(agent_id: str) -> None:
    payload = _minimal_payload()
    payload["agent"]["id"] = agent_id
    AgentYamlConfig.model_validate(payload)


@pytest.mark.parametrize(
    "agent_id",
    [
        "Analyst",  # uppercase
        "1analyst",  # starts with digit
        "a",  # too short
        "agent with space",
        "agent/path",
        "x" * 65,  # too long
    ],
)
def test_invalid_agent_id_surfaces_actionable_error(agent_id: str) -> None:
    payload = _minimal_payload()
    payload["agent"]["id"] = agent_id
    with pytest.raises(ValidationError) as exc_info:
        AgentYamlConfig.model_validate(payload)
    errors = exc_info.value.errors()
    assert any("agent" in error["loc"] for error in errors)
    assert any("id" in error["loc"] for error in errors)


# ── version SemVer ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "version",
    ["0.0.1", "1.2.3", "10.20.30", "1.0.0-alpha", "1.0.0-rc.1+build.7"],
)
def test_valid_semver_versions(version: str) -> None:
    payload = _minimal_payload()
    payload["agent"]["version"] = version
    AgentYamlConfig.model_validate(payload)


@pytest.mark.parametrize(
    "version",
    ["1.0", "v1.0.0", "1.0.0.0", "latest", "1"],
)
def test_invalid_version_rejected(version: str) -> None:
    payload = _minimal_payload()
    payload["agent"]["version"] = version
    with pytest.raises(ValidationError) as exc_info:
        AgentYamlConfig.model_validate(payload)
    assert any("version" in error["loc"] for error in exc_info.value.errors())


# ── agent.type literal ────────────────────────────────────────────────────────


def test_valid_agent_types() -> None:
    for t in ("artifact_agent", "worker_agent", "tool_agent"):
        payload = _minimal_payload()
        payload["agent"]["type"] = t
        AgentYamlConfig.model_validate(payload)


def test_invalid_agent_type_rejected() -> None:
    payload = _minimal_payload()
    payload["agent"]["type"] = "magic_agent"
    with pytest.raises(ValidationError) as exc_info:
        AgentYamlConfig.model_validate(payload)
    assert any("type" in error["loc"] for error in exc_info.value.errors())


# ── runtime.port range ────────────────────────────────────────────────────────


@pytest.mark.parametrize("port", [1, 8080, 65535])
def test_valid_runtime_ports(port: int) -> None:
    payload = _minimal_payload()
    payload["runtime"]["port"] = port
    AgentYamlConfig.model_validate(payload)


@pytest.mark.parametrize("port", [0, -1, 65536, 100000])
def test_invalid_runtime_port_rejected(port: int) -> None:
    payload = _minimal_payload()
    payload["runtime"]["port"] = port
    with pytest.raises(ValidationError) as exc_info:
        AgentYamlConfig.model_validate(payload)
    assert any("port" in error["loc"] for error in exc_info.value.errors())


# ── runtime.duration literal ──────────────────────────────────────────────────


@pytest.mark.parametrize("duration", ["<1s", "<1min", "<1h", ">1h"])
def test_valid_runtime_durations(duration: str) -> None:
    payload = _minimal_payload()
    payload["runtime"]["duration"] = duration
    AgentYamlConfig.model_validate(payload)


def test_invalid_runtime_duration_rejected() -> None:
    payload = _minimal_payload()
    payload["runtime"]["duration"] = "5min"
    with pytest.raises(ValidationError):
        AgentYamlConfig.model_validate(payload)


# ── runtime.durability literal ───────────────────────────────────────────────


@pytest.mark.parametrize("durability", ["none", "result_cache", "task_store"])
def test_valid_runtime_durability_levels(durability: str) -> None:
    payload = _minimal_payload()
    payload["runtime"]["durability"] = durability
    config = AgentYamlConfig.model_validate(payload)
    assert config.runtime.durability == durability


def test_runtime_durability_defaults_to_none_for_generator_resolution() -> None:
    config = AgentYamlConfig.model_validate(_minimal_payload())
    assert config.runtime.durability is None


def test_invalid_runtime_durability_rejected() -> None:
    payload = _minimal_payload()
    payload["runtime"]["durability"] = "memory"
    with pytest.raises(ValidationError):
        AgentYamlConfig.model_validate(payload)


# ── governance literals ───────────────────────────────────────────────────────


@pytest.mark.parametrize("risk", ["read", "write", "dangerous"])
def test_valid_governance_risks(risk: str) -> None:
    payload = _minimal_payload()
    payload["governance"] = {"risk": risk}
    AgentYamlConfig.model_validate(payload)


@pytest.mark.parametrize(
    "side_effect",
    ["none", "session", "tenant", "external", "irreversible"],
)
def test_valid_governance_side_effects(side_effect: str) -> None:
    payload = _minimal_payload()
    payload["governance"] = {"side_effect": side_effect}
    AgentYamlConfig.model_validate(payload)


def test_invalid_governance_risk_rejected() -> None:
    payload = _minimal_payload()
    payload["governance"] = {"risk": "scary"}
    with pytest.raises(ValidationError):
        AgentYamlConfig.model_validate(payload)


# ── capability id format + uniqueness ────────────────────────────────────────


def test_capabilities_unique_required() -> None:
    payload = _minimal_payload()
    payload["capabilities"] = ["agent.analyst.foo", "agent.analyst.foo"]
    with pytest.raises(ValidationError) as exc_info:
        AgentYamlConfig.model_validate(payload)
    msg = str(exc_info.value)
    assert "duplicate" in msg


def test_capability_id_format_validated() -> None:
    payload = _minimal_payload()
    payload["capabilities"] = ["UPPERCASE"]
    with pytest.raises(ValidationError) as exc_info:
        AgentYamlConfig.model_validate(payload)
    msg = str(exc_info.value)
    assert "lowercase" in msg or "[a-z0-9_-.]" in msg


def test_empty_capabilities_list_allowed() -> None:
    payload = _minimal_payload()
    payload["capabilities"] = []
    config = AgentYamlConfig.model_validate(payload)
    assert config.capabilities == []


# ── extra="forbid" catches typos ─────────────────────────────────────────────


def test_top_level_typo_rejected() -> None:
    payload = _minimal_payload()
    payload["descriptionn"] = "typo"  # extra typo'd field
    with pytest.raises(ValidationError) as exc_info:
        AgentYamlConfig.model_validate(payload)
    assert any(
        "descriptionn" in str(error.get("loc", ()))
        or error["type"] == "extra_forbidden"
        for error in exc_info.value.errors()
    )


def test_section_typo_rejected() -> None:
    payload = _minimal_payload()
    payload["agent"]["versionn"] = "0.2.0"
    with pytest.raises(ValidationError) as exc_info:
        AgentYamlConfig.model_validate(payload)
    errors = exc_info.value.errors()
    assert any(error["type"] == "extra_forbidden" for error in errors)


def test_runtime_typo_rejected() -> None:
    payload = _minimal_payload()
    payload["runtime"]["porte"] = 8010
    with pytest.raises(ValidationError) as exc_info:
        AgentYamlConfig.model_validate(payload)
    assert any(
        error["type"] == "extra_forbidden" for error in exc_info.value.errors()
    )


# ── Required-field omissions ──────────────────────────────────────────────────


def test_missing_agent_section_rejected() -> None:
    payload = {"runtime": {"port": 8010}}
    with pytest.raises(ValidationError) as exc_info:
        AgentYamlConfig.model_validate(payload)
    assert any("agent" in error["loc"] for error in exc_info.value.errors())


def test_missing_runtime_section_rejected() -> None:
    payload = {
        "agent": {
            "id": "analyst",
            "name": "Analyst",
            "version": "0.2.0",
            "type": "artifact_agent",
        },
    }
    with pytest.raises(ValidationError) as exc_info:
        AgentYamlConfig.model_validate(payload)
    assert any("runtime" in error["loc"] for error in exc_info.value.errors())


def test_missing_runtime_port_rejected() -> None:
    payload = _minimal_payload()
    del payload["runtime"]["port"]
    with pytest.raises(ValidationError) as exc_info:
        AgentYamlConfig.model_validate(payload)
    assert any("port" in error["loc"] for error in exc_info.value.errors())


# ── Escape hatch ─────────────────────────────────────────────────────────────


def test_advanced_manifest_overrides_accepts_arbitrary_keys() -> None:
    """The escape hatch must NOT validate the shape of override
    values — power users use it to inject manifest-v2 fields the
    simplified schema doesn't surface yet. Validation happens in
    W2 against the manifest-v2 contract."""
    payload = _minimal_payload()
    payload["advanced"] = {
        "manifest_overrides": {
            "compat_range": ["agent-manifest-v2"],
            "experimental_field_x": {"nested": [1, 2, 3]},
            "another_field": None,
        },
    }
    config = AgentYamlConfig.model_validate(payload)
    assert (
        config.advanced.manifest_overrides["experimental_field_x"]
        == {"nested": [1, 2, 3]}
    )
    assert config.advanced.manifest_overrides["another_field"] is None


# ── Direct section instantiation ─────────────────────────────────────────────


def test_identity_section_constructs_directly() -> None:
    """Sections must be importable + constructible standalone so
    the W2 generator can build partial configs without a full
    config envelope."""
    identity = AgentYamlIdentity(
        id="analyst",
        name="Analyst",
        version="0.1.0",
        type="artifact_agent",
    )
    assert identity.id == "analyst"


def test_runtime_section_constructs_directly() -> None:
    runtime = AgentYamlRuntime(port=8080)
    assert runtime.port == 8080
    assert runtime.duration == "<1min"
    assert runtime.durability is None
    assert runtime.secrets == []


# ── Defaults baseline ────────────────────────────────────────────────────────


def test_defaults_are_platform_safe() -> None:
    """Acceptance bullet from W1: defaults should be platform-safe.
    A minimal config with no governance section gets read / none /
    no gate — the most conservative possible."""
    config = AgentYamlConfig.model_validate(_minimal_payload())
    assert config.governance.risk == "read"
    assert config.governance.side_effect == "none"
    assert config.governance.requires_tracker_issue is False
    assert config.governance.requires_human_gate is False


def test_capabilities_default_empty() -> None:
    """Defaults should match: a minimal agent with no capabilities
    block has empty capabilities list (W2 will raise if a generator
    tries to emit a manifest with zero capabilities)."""
    config = AgentYamlConfig.model_validate(_minimal_payload())
    assert config.capabilities == []
