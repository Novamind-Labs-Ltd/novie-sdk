"""EXPERT_AGENT_SDK W2 step 2 — manifest validator tests.

Locks the contract for ``validate_agent_yaml`` +
``validate_generated_manifest`` + their combined helper.

Author-facing pass:
- capability_id naming convention (warning, not error)
- artifact_agent without ``outputs.provides`` (warning)
- worker_agent with ``<1s`` duration (warning)
- endpoint port mismatch in manifest_overrides (warning)
- non-string endpoint override (error)
- requires_human_gate=True with risk=read (warning)
- empty capabilities list (warning)

Platform-facing pass:
- generated manifest from a valid config has zero errors
- malformed manifest dict → unparseable error
- platform-validate failures (e.g. supports_cancel without
  protocol_mode=tasks) → platform_validate_failure error

Result envelope:
- ``is_valid`` ignores warnings, fails only on errors
- ``errors`` / ``warnings`` / ``infos`` accessors
- ``merged`` combines two results
"""
# ruff: noqa: I001
from __future__ import annotations

import pytest

from novie_agent_sdk.agent_yaml import AgentYamlConfig
from novie_agent_sdk.manifest_generator import generate_agent_manifest
from novie_agent_sdk.manifest_validator import (
    ManifestValidationIssue,
    ManifestValidationResult,
    validate_agent_yaml,
    validate_agent_yaml_and_manifest,
    validate_generated_manifest,
)


def _config(**overrides) -> AgentYamlConfig:
    payload: dict = {
        "agent": {
            "id": "analyst",
            "name": "Analyst",
            "version": "0.2.0",
            "type": "artifact_agent",
        },
        "description": (
            "Test fixture analyst agent that extracts structured "
            "requirements from input documents."
        ),
        "capabilities": ["agent.analyst.do_thing"],
        "outputs": {"provides": ["analysis_artifact"]},
        "runtime": {"port": 8010},
    }
    for key, value in overrides.items():
        payload[key] = value
    return AgentYamlConfig.model_validate(payload)


# ── ManifestValidationResult envelope ────────────────────────────────────────


def test_empty_result_is_valid() -> None:
    result = ManifestValidationResult()
    assert result.is_valid is True
    assert result.errors == ()
    assert result.warnings == ()
    assert result.infos == ()


def test_result_with_only_warnings_still_valid() -> None:
    result = ManifestValidationResult(
        issues=(
            ManifestValidationIssue(
                severity="warning",
                field_path="x",
                code="x",
                author_message="x",
                platform_message="x",
            ),
        ),
    )
    assert result.is_valid is True
    assert len(result.warnings) == 1
    assert result.errors == ()


def test_result_with_error_is_invalid() -> None:
    result = ManifestValidationResult(
        issues=(
            ManifestValidationIssue(
                severity="error",
                field_path="x",
                code="x",
                author_message="x",
                platform_message="x",
            ),
        ),
    )
    assert result.is_valid is False
    assert len(result.errors) == 1


def test_result_merged_concatenates_issues() -> None:
    a = ManifestValidationResult(
        issues=(
            ManifestValidationIssue(
                severity="warning",
                field_path="x",
                code="a",
                author_message="a",
                platform_message="a",
            ),
        ),
    )
    b = ManifestValidationResult(
        issues=(
            ManifestValidationIssue(
                severity="error",
                field_path="y",
                code="b",
                author_message="b",
                platform_message="b",
            ),
        ),
    )
    merged = a.merged(b)
    assert len(merged.issues) == 2
    assert merged.is_valid is False  # b's error wins


# ── Author-facing: capability_id naming convention ───────────────────────────


def test_capability_id_off_convention_warns() -> None:
    config = _config(capabilities=["wrong.namespace.thing"])
    result = validate_agent_yaml(config)
    codes = [i.code for i in result.issues]
    assert "capability_id_naming_off_convention" in codes
    # Warning, not error — author can ignore until convention enforced.
    assert result.is_valid is True
    naming_issue = next(
        i for i in result.issues
        if i.code == "capability_id_naming_off_convention"
    )
    assert naming_issue.severity == "warning"
    assert "wrong.namespace.thing" in naming_issue.author_message
    assert "agent.analyst." in naming_issue.author_message


def test_capability_id_on_convention_no_warning() -> None:
    config = _config(capabilities=[
        "agent.analyst.do_thing",
        "agent.analyst.summarize",
    ])
    result = validate_agent_yaml(config)
    assert not [
        i for i in result.issues
        if i.code == "capability_id_naming_off_convention"
    ]


def test_partial_naming_off_emits_one_warning_per_offender() -> None:
    config = _config(capabilities=[
        "agent.analyst.do_thing",      # OK
        "wrong.one",                    # Off
        "another.bad",                  # Off
    ])
    result = validate_agent_yaml(config)
    naming_issues = [
        i for i in result.issues
        if i.code == "capability_id_naming_off_convention"
    ]
    assert len(naming_issues) == 2
    assert "capabilities[1]" in [i.field_path for i in naming_issues]
    assert "capabilities[2]" in [i.field_path for i in naming_issues]


# ── Author-facing: artifact_agent without provides ───────────────────────────


def test_artifact_agent_without_provides_warns() -> None:
    config = _config(outputs={"provides": []})
    result = validate_agent_yaml(config)
    codes = [i.code for i in result.issues]
    assert "artifact_agent_missing_provides" in codes


def test_artifact_agent_with_provides_no_warning() -> None:
    config = _config(outputs={"provides": ["analysis_artifact"]})
    result = validate_agent_yaml(config)
    assert "artifact_agent_missing_provides" not in [
        i.code for i in result.issues
    ]


def test_tool_agent_without_provides_no_warning() -> None:
    """Tool agents commonly have no produces — they're sync read
    tools. Don't fire the warning for them."""
    payload = {
        "agent": {
            "id": "tool",
            "name": "Tool",
            "version": "0.1.0",
            "type": "tool_agent",
        },
        "capabilities": ["agent.tool.do"],
        "runtime": {"port": 8010},
    }
    config = AgentYamlConfig.model_validate(payload)
    result = validate_agent_yaml(config)
    assert "artifact_agent_missing_provides" not in [
        i.code for i in result.issues
    ]


# ── Author-facing: worker_agent with too-short duration ──────────────────────


def test_worker_agent_under_1s_warns() -> None:
    payload = {
        "agent": {
            "id": "cortex",
            "name": "Cortex",
            "version": "0.1.0",
            "type": "worker_agent",
        },
        "capabilities": ["agent.cortex.run"],
        "runtime": {"port": 8010, "duration": "<1s"},
    }
    config = AgentYamlConfig.model_validate(payload)
    result = validate_agent_yaml(config)
    assert "worker_agent_too_short" in [i.code for i in result.issues]


def test_worker_agent_with_long_duration_no_warning() -> None:
    payload = {
        "agent": {
            "id": "cortex",
            "name": "Cortex",
            "version": "0.1.0",
            "type": "worker_agent",
        },
        "capabilities": ["agent.cortex.run"],
        "runtime": {"port": 8010, "duration": ">1h"},
    }
    config = AgentYamlConfig.model_validate(payload)
    result = validate_agent_yaml(config)
    assert "worker_agent_too_short" not in [i.code for i in result.issues]


# ── Author-facing: runtime durability ────────────────────────────────────────


def test_worker_agent_without_explicit_durability_uses_task_store_default() -> None:
    payload = {
        "agent": {
            "id": "cortex",
            "name": "Cortex",
            "version": "0.1.0",
            "type": "worker_agent",
        },
        "capabilities": ["agent.cortex.run"],
        "runtime": {"port": 8010, "duration": ">1h"},
    }
    config = AgentYamlConfig.model_validate(payload)
    result = validate_agent_yaml(config)
    assert "worker_agent_requires_task_store" not in [
        i.code for i in result.issues
    ]


def test_worker_agent_with_non_task_store_durability_errors() -> None:
    payload = {
        "agent": {
            "id": "cortex",
            "name": "Cortex",
            "version": "0.1.0",
            "type": "worker_agent",
        },
        "capabilities": ["agent.cortex.run"],
        "runtime": {
            "port": 8010,
            "duration": ">1h",
            "durability": "result_cache",
        },
    }
    config = AgentYamlConfig.model_validate(payload)
    result = validate_agent_yaml(config)
    assert result.is_valid is False
    assert "worker_agent_requires_task_store" in [
        i.code for i in result.errors
    ]


def test_oneshot_with_task_store_durability_warns() -> None:
    config = _config(runtime={"port": 8010, "durability": "task_store"})
    result = validate_agent_yaml(config)
    assert "oneshot_task_store_durability_unusual" in [
        i.code for i in result.warnings
    ]


def test_oneshot_side_effect_without_result_cache_warns() -> None:
    config = _config(governance={"risk": "write", "side_effect": "external"})
    result = validate_agent_yaml(config)
    assert "oneshot_side_effect_without_result_cache" in [
        i.code for i in result.warnings
    ]


def test_oneshot_side_effect_with_result_cache_no_warning() -> None:
    config = _config(
        runtime={"port": 8010, "durability": "result_cache"},
        governance={"risk": "write", "side_effect": "external"},
    )
    result = validate_agent_yaml(config)
    assert "oneshot_side_effect_without_result_cache" not in [
        i.code for i in result.issues
    ]


# ── Author-facing: endpoint/port consistency ─────────────────────────────────


def test_endpoint_port_match_no_warning() -> None:
    config = _config(advanced={
        "manifest_overrides": {"endpoint": "http://localhost:8010"},
    })
    result = validate_agent_yaml(config)
    assert "endpoint_port_mismatch" not in [i.code for i in result.issues]


def test_endpoint_port_mismatch_warns() -> None:
    config = _config(advanced={
        "manifest_overrides": {"endpoint": "http://example.com:9999"},
    })
    result = validate_agent_yaml(config)
    assert "endpoint_port_mismatch" in [i.code for i in result.issues]
    issue = next(
        i for i in result.issues if i.code == "endpoint_port_mismatch"
    )
    assert "9999" in issue.author_message
    assert "8010" in issue.author_message


def test_endpoint_without_port_no_warning() -> None:
    """https://example.com (no explicit port) is a typical
    reverse-proxy setup — don't warn."""
    config = _config(advanced={
        "manifest_overrides": {"endpoint": "https://analyst.example.com"},
    })
    result = validate_agent_yaml(config)
    assert "endpoint_port_mismatch" not in [i.code for i in result.issues]


def test_non_string_endpoint_override_errors() -> None:
    config = _config(advanced={
        "manifest_overrides": {"endpoint": 8080},
    })
    result = validate_agent_yaml(config)
    assert result.is_valid is False
    assert "endpoint_must_be_string" in [i.code for i in result.issues]


# ── Author-facing: human gate × read risk ────────────────────────────────────


def test_human_gate_with_read_risk_warns() -> None:
    config = _config(governance={
        "risk": "read",
        "requires_human_gate": True,
    })
    result = validate_agent_yaml(config)
    assert "human_gate_on_read_risk" in [i.code for i in result.issues]


def test_human_gate_with_write_risk_no_warning() -> None:
    config = _config(governance={
        "risk": "write",
        "requires_human_gate": True,
    })
    result = validate_agent_yaml(config)
    assert "human_gate_on_read_risk" not in [
        i.code for i in result.issues
    ]


# ── Author-facing: empty capabilities ────────────────────────────────────────


def test_empty_capabilities_warns() -> None:
    config = _config(capabilities=[])
    result = validate_agent_yaml(config)
    assert "capabilities_empty" in [i.code for i in result.issues]


def test_non_empty_capabilities_no_warning() -> None:
    config = _config()
    result = validate_agent_yaml(config)
    assert "capabilities_empty" not in [i.code for i in result.issues]


# ── Platform-facing: generated manifest ──────────────────────────────────────


def test_valid_config_generates_manifest_without_platform_errors() -> None:
    config = _config()
    manifest = generate_agent_manifest(config)
    result = validate_generated_manifest(manifest)
    assert result.is_valid is True
    assert result.errors == ()


def test_platform_validate_surfaces_supports_cancel_mismatch() -> None:
    """Inject a manifest that violates the platform's structural
    rule: ``supports_cancel=true`` requires ``protocol_mode=tasks``.
    The validator should surface this as a platform-facing error."""
    config = _config()
    manifest = generate_agent_manifest(config)
    # artifact_agent default is protocol_mode=stream, supports_cancel=False.
    # Force the inconsistency:
    manifest["execution"]["supports_cancel"] = True
    # protocol_mode stays "stream" → mismatch.
    result = validate_generated_manifest(manifest)
    assert result.is_valid is False
    assert "platform_validate_failure" in [i.code for i in result.errors]
    msg = " ".join(i.author_message for i in result.errors)
    assert "supports_cancel" in msg
    assert "protocol_mode" in msg


def test_platform_validate_surfaces_unparseable_manifest() -> None:
    """A manifest dict with the wrong type for a required field
    is reported as ``manifest_unparseable`` rather than swallowed."""
    bad_manifest = {"agent_id": ["not", "a", "string"]}
    result = validate_generated_manifest(bad_manifest)
    assert result.is_valid is False
    codes = [i.code for i in result.errors]
    # Either unparseable (raised at from_dict) or platform_validate
    # depending on how AgentManifestV2.from_dict handles the bad type
    # — both are acceptable error paths. Lock that the result is
    # invalid + at least one error fires.
    assert codes
    assert codes[0] in {
        "manifest_unparseable",
        "platform_validate_failure",
    }


def test_platform_validate_carries_both_message_variants() -> None:
    """Each diagnostic has both author and platform message
    populated so a CLI can pick the audience."""
    config = _config()
    manifest = generate_agent_manifest(config)
    manifest["execution"]["supports_cancel"] = True
    result = validate_generated_manifest(manifest)
    for issue in result.errors:
        assert issue.author_message
        assert issue.platform_message
        # Platform message references the platform contract directly.
        assert "AgentManifestV2" in issue.platform_message


# ── Combined helper ──────────────────────────────────────────────────────────


def test_combined_helper_runs_both_passes() -> None:
    config = _config(capabilities=[
        "wrong.namespace.thing",  # author warning
    ])
    manifest = generate_agent_manifest(config)
    manifest["execution"]["supports_cancel"] = True  # platform error
    result = validate_agent_yaml_and_manifest(config, manifest)
    codes = [i.code for i in result.issues]
    assert "capability_id_naming_off_convention" in codes
    assert "platform_validate_failure" in codes
    # Result is invalid because platform-validate fails.
    assert result.is_valid is False


def test_combined_helper_clean_config_returns_valid() -> None:
    config = _config()
    manifest = generate_agent_manifest(config)
    result = validate_agent_yaml_and_manifest(config, manifest)
    assert result.is_valid is True


# ── Severity contract ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("setup", "expected_severity"),
    [
        # Warning-only cases keep is_valid True
        (
            lambda: _config(capabilities=["wrong.id"]),
            "warning",
        ),
        (
            lambda: _config(outputs={"provides": []}),
            "warning",
        ),
        (
            lambda: _config(governance={
                "risk": "read",
                "requires_human_gate": True,
            }),
            "warning",
        ),
    ],
)
def test_warnings_keep_result_valid(setup, expected_severity: str) -> None:
    config = setup()
    result = validate_agent_yaml(config)
    matching = [i for i in result.issues if i.severity == expected_severity]
    assert matching  # at least one warning fires
    # Warnings alone don't fail validation.
    assert result.is_valid is True


def test_each_issue_has_short_stable_code() -> None:
    """Codes must be ascii-snake-case so callers can branch
    programmatically without parsing english messages."""
    config = _config(
        capabilities=["wrong.id"],
        governance={"risk": "read", "requires_human_gate": True},
    )
    result = validate_agent_yaml(config)
    for issue in result.issues:
        assert issue.code
        assert issue.code.replace("_", "").isalnum()
        assert " " not in issue.code
