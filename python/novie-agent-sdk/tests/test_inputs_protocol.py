"""TechDebt #7 (2026-05-11) — wire-protocol ``inputs`` vs ``input``.

Locks the Python SDK's ``_extract_inputs`` contract:

- Canonical key: ``inputs`` (plural) — matches the platform's
  ``_build_agent_payload`` and the Rust SDK's ``payload.inputs()``
  accessor. Production traffic from the Temporal A2A activity sends
  this shape.
- Legacy key: ``input`` (singular) — older callers / tests. Accepted
  as a fallback for backward compatibility; the next major SDK
  release should drop it.
- Both present → ``inputs`` wins (canonical key takes precedence so
  one consistent migration path forward exists).
- Neither present → empty dict. **NOT** the whole body (the pre-fix
  behaviour leaked ``context`` / ``capability_grants`` /
  ``credential_leases`` into the agent's input).
- Non-dict values → empty dict (defensive coercion so a malformed
  payload doesn't crash the agent handler).
"""
# ruff: noqa: PLC0415
from __future__ import annotations

import pytest


def test_canonical_inputs_key_wins() -> None:
    """Platform sends ``inputs`` — SDK extracts that dict, ignores
    everything else at the top level."""
    from novie_agent_sdk.runtime import _extract_inputs

    body = {
        "context": {"tenant_id": "t-a"},
        "inputs": {"query": "hello", "params": {"k": 1}},
        "capability_grants": [{"capability_id": "x"}],
        "credential_leases": {"github": "lease-1"},
    }
    extracted = _extract_inputs(body)
    assert extracted == {"query": "hello", "params": {"k": 1}}
    # Crucial: top-level wire fields don't leak into the agent's input.
    assert "context" not in extracted
    assert "capability_grants" not in extracted
    assert "credential_leases" not in extracted


def test_legacy_input_key_still_accepted() -> None:
    """Older callers sending ``input`` (singular) keep working —
    backward compatibility for the migration window."""
    from novie_agent_sdk.runtime import _extract_inputs

    body = {"input": {"query": "legacy"}}
    assert _extract_inputs(body) == {"query": "legacy"}


def test_canonical_inputs_beats_legacy_input_when_both_present() -> None:
    """When a buggy caller sends both, the canonical form wins so the
    migration direction is deterministic. Locks the precedence
    contract for the deprecation window."""
    from novie_agent_sdk.runtime import _extract_inputs

    body = {
        "inputs": {"q": "canonical"},
        "input": {"q": "legacy"},
    }
    assert _extract_inputs(body) == {"q": "canonical"}


def test_missing_inputs_returns_empty_dict_not_whole_body() -> None:
    """The actual bug TechDebt #7 was filed for: pre-fix code did
    ``body.get("input", body)`` which leaked wire metadata into the
    agent's input when neither key was present. The fix returns
    empty dict so agents see ``ctx.input == {}`` instead of a dict
    containing ``context`` / ``capability_grants`` etc."""
    from novie_agent_sdk.runtime import _extract_inputs

    body = {
        "context": {"tenant_id": "t-a"},
        "capability_grants": [{"capability_id": "x"}],
        "credential_leases": {"github": "lease-1"},
        # No inputs / input keys present.
    }
    extracted = _extract_inputs(body)
    assert extracted == {}
    # The leak that TechDebt #7 fixes:
    assert "context" not in extracted
    assert "capability_grants" not in extracted


def test_non_dict_inputs_coerces_to_empty_dict() -> None:
    """Defensive coercion: a malformed payload (``inputs: "string"``
    or ``inputs: null``) doesn't crash the agent handler — the
    handler always sees a dict."""
    from novie_agent_sdk.runtime import _extract_inputs

    assert _extract_inputs({"inputs": None}) == {}
    assert _extract_inputs({"inputs": "not a dict"}) == {}
    assert _extract_inputs({"inputs": ["array", "not", "dict"]}) == {}
    assert _extract_inputs({"inputs": 42}) == {}


def test_non_dict_legacy_input_coerces_to_empty_dict() -> None:
    """Same defensive coercion for the legacy fallback."""
    from novie_agent_sdk.runtime import _extract_inputs

    assert _extract_inputs({"input": None}) == {}
    assert _extract_inputs({"input": "raw string"}) == {}


def test_completely_empty_body() -> None:
    """Empty body (technically invalid wire shape) → empty dict.
    Belt-and-braces: ``_parse_json`` already rejects empty bodies
    at the HTTP layer, so this is purely defensive."""
    from novie_agent_sdk.runtime import _extract_inputs

    assert _extract_inputs({}) == {}


def test_inputs_dict_is_returned_by_reference_not_copy() -> None:
    """Performance / semantics check: the extracted dict is the same
    object the caller passed in. The handler may add fields (e.g.
    ``__agent_status_callback__``) and downstream code that reads
    the body again sees the same mutation.

    This matches the pre-fix behaviour where the agent received the
    body itself when neither key was present — production agents
    may rely on shared-reference semantics."""
    from novie_agent_sdk.runtime import _extract_inputs

    inputs_obj = {"query": "x"}
    body = {"inputs": inputs_obj}
    extracted = _extract_inputs(body)
    assert extracted is inputs_obj
