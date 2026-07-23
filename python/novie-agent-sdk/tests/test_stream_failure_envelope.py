"""Failure-envelope classification for the A2A stream guards.

Regression coverage for the 2026-07-23 analyst brainstorm outage: the
``sdk_envelope_guard`` classified a recoverable ``agent.llm_call.retrying``
observability event (metadata carried the retried call's ``error`` details)
as stream-terminal, killing the run before the SDK's own LLM retry executed.
Named phase events under ``metadata`` are observability signals — neither
their ``status`` nor their ``error`` fields may terminate the stream.
"""

from __future__ import annotations

from novie_agent_sdk.runtime import _failure_envelope


def _llm_call_retrying_event() -> dict[str, object]:
    """The exact production shape that killed the brainstorm stream."""
    return {
        "kind": "trace",
        "metadata": {
            "event": "agent.llm_call.retrying",
            "runtime_phase": "sectioned_authoring",
            "call_id": "llm-draft-section-0001",
            "llm_purpose": "draft_section",
            "attempt": 1,
            "max_attempts": 2,
            "status": "retrying",
            "error": "PlatformLlmCallError",
            "message": (
                "platform LLM capability 'platform.llm.chat' failed: "
                "kind=platform_unavailable error_code=llm_provider_error"
            ),
            "next_attempt": 2,
        },
    }


def test_llm_call_retrying_is_not_terminal() -> None:
    assert _failure_envelope(_llm_call_retrying_event()) is None


def test_tool_error_with_terminal_status_fails_closed() -> None:
    # A named event whose own status is terminal (or missing) keeps the
    # explicit ``error`` field detectable — ``agent.tool_error`` from a
    # genuinely failed tool call still terminates. Recoverable reports must
    # self-declare an in-flight status (``retrying``, ``running``, …).
    event = {
        "kind": "tool_result",
        "tool_name": "artifact.write",
        "tool_call_id": "tool-artifact-write-0001",
        "tool_result": "artifact_create_failed:artifact_ref_missing",
        "metadata": {
            "event": "agent.tool_error",
            "tool_name": "artifact.write",
            "status": "failed",
            "error": "RuntimeError",
            "message": "artifact_create_failed:artifact_ref_missing",
            "ok": False,
        },
    }
    assert _failure_envelope(event) is not None


def test_tool_retry_report_with_in_flight_status_is_not_terminal() -> None:
    event = {
        "kind": "tool_result",
        "metadata": {
            "event": "agent.tool_error",
            "tool_name": "evidence.build",
            "status": "retrying",
            "error": "TimeoutError",
            "message": "evidence build timed out; retrying",
        },
    }
    assert _failure_envelope(event) is None


def test_named_quality_gate_event_is_not_terminal() -> None:
    # Pins the pre-existing status exemption for named phase events.
    event = {
        "kind": "trace",
        "metadata": {
            "event": "document.section.quality_checked",
            "status": "failed",
            "section_id": "s1",
        },
    }
    assert _failure_envelope(event) is None


def test_top_level_failed_status_is_terminal() -> None:
    envelope = {"status": "failed", "error": "boom", "output": {}}
    assert _failure_envelope(envelope) is envelope


def test_terminal_kind_is_terminal() -> None:
    envelope = {"kind": "terminal_error", "error": "boom"}
    assert _failure_envelope(envelope) is envelope


def test_failure_nested_under_output_is_terminal() -> None:
    envelope = {"kind": "result", "output": {"status": "failed", "error": "boom"}}
    assert _failure_envelope(envelope) is not None


def test_error_nested_under_unnamed_metadata_is_terminal() -> None:
    # Only *named* phase events are exempt; anonymous metadata carrying an
    # error still terminates.
    envelope = {"kind": "trace", "metadata": {"error": "boom"}}
    assert _failure_envelope(envelope) is not None


def test_named_event_with_terminal_kind_still_terminates() -> None:
    # The exemption never gates ``kind`` — an explicit terminal kind inside a
    # named event's metadata stays terminal.
    envelope = {
        "kind": "trace",
        "metadata": {"event": "agent.custom", "kind": "terminal_error"},
    }
    assert _failure_envelope(envelope) is not None
