from __future__ import annotations

from novie_agent_sdk import (
    DEFAULT_HANDOFF_MAX_BYTES,
    TERMINAL_DELIVERABLE,
    UPSTREAM_HANDOFF,
    fit_text_to_utf8_bytes,
    handoff_max_bytes,
    is_terminal_deliverable,
    is_upstream_handoff,
    output_contract,
    resolve_step_role,
    step_run_policy,
)


def test_resolves_upstream_handoff_from_output_contract_kind() -> None:
    inputs = {"output_contract": {"kind": "bounded_handoff", "max_bytes": 4096}}

    assert output_contract(inputs)["kind"] == "bounded_handoff"
    assert resolve_step_role(inputs) == UPSTREAM_HANDOFF
    assert is_upstream_handoff(inputs)
    assert handoff_max_bytes(inputs) == 4096


def test_resolves_terminal_deliverable_from_nested_platform_context() -> None:
    inputs = {"platform_context": {"output_contract": {"kind": "final_deliverable"}}}

    assert resolve_step_role(inputs) == TERMINAL_DELIVERABLE
    assert is_terminal_deliverable(inputs)


def test_handoff_budget_has_safe_defaults_and_bounds() -> None:
    assert handoff_max_bytes({}) == DEFAULT_HANDOFF_MAX_BYTES
    assert handoff_max_bytes({"output_contract": {"max_bytes": "16"}}) == 1024
    assert handoff_max_bytes({"output_contract": {"max_bytes": 999_999}}) == 64_000


def test_fit_text_to_utf8_bytes_preserves_utf8_boundary() -> None:
    value = "市场" * 200

    result = fit_text_to_utf8_bytes(value, max_bytes=80)

    assert len(result.encode("utf-8")) <= 80
    assert result.endswith("[Handoff truncated to platform byte budget.]")


def test_step_run_policy_disables_user_visible_generation_for_upstream_handoff() -> None:
    policy = step_run_policy(
        {"output_contract": {"kind": "bounded_handoff", "max_bytes": 4096}}
    )

    assert policy.step_role == UPSTREAM_HANDOFF
    assert policy.is_upstream_handoff
    assert policy.user_visible_content_stream is False
    assert policy.quality_loop_enabled is False
    assert policy.final_deliverable_enabled is False
    assert policy.max_output_bytes == 4096


def test_step_run_policy_defaults_to_terminal_deliverable_behavior() -> None:
    policy = step_run_policy({})

    assert policy.step_role == TERMINAL_DELIVERABLE
    assert policy.is_terminal_deliverable
    assert policy.user_visible_content_stream is True
    assert policy.quality_loop_enabled is True
    assert policy.final_deliverable_enabled is True
