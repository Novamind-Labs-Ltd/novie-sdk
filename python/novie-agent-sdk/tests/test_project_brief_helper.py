"""Tests for ProjectBrief helper functions."""
from __future__ import annotations

from datetime import datetime, timezone

from novie_agent_sdk import extract_project_brief, render_brief_for_prompt
from novie_protocol.contracts import ProjectBrief


def _sample_brief(**overrides) -> ProjectBrief:
    defaults = dict(
        project_id="p-test",
        tenant_id="t-acme",
        generated_at=datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc),
        source_hash="deadbeef",
        summary="Project X is launching next sprint.",
        key_constraints=("must pass legal review",),
        recent_focus=("finalized UI copy",),
        open_questions=("who signs off release?",),
    )
    defaults.update(overrides)
    return ProjectBrief(**defaults)


# ---------------- extract ----------------


def test_extract_returns_none_on_empty_inputs() -> None:
    assert extract_project_brief(None) is None
    assert extract_project_brief({}) is None


def test_extract_returns_none_when_key_missing() -> None:
    assert extract_project_brief({"other": 1}) is None


def test_extract_returns_none_when_value_not_dict() -> None:
    assert extract_project_brief({"__project_brief__": "oops"}) is None
    assert extract_project_brief({"__project_brief__": 42}) is None


def test_extract_returns_brief_when_dict_valid() -> None:
    brief = _sample_brief()
    inputs = {"__project_brief__": brief.to_dict()}
    parsed = extract_project_brief(inputs)
    assert parsed is not None
    assert parsed.project_id == "p-test"
    assert parsed.summary.startswith("Project X")
    assert parsed.key_constraints == ("must pass legal review",)


def test_extract_returns_none_on_malformed_dict() -> None:
    """Missing required keys logs a warning and yields None (no exception)."""
    assert extract_project_brief({"__project_brief__": {"no": "fields"}}) is None


def test_extract_handles_minimal_brief() -> None:
    degraded = ProjectBrief.degraded(
        project_id="p",
        tenant_id="t",
        source_hash="x",
        reason="llm_failure: TimeoutError",
    )
    parsed = extract_project_brief({"__project_brief__": degraded.to_dict()})
    assert parsed is not None
    assert parsed.minimal is True
    assert parsed.degraded_reason is not None


# ---------------- render ----------------


def test_render_full_brief() -> None:
    out = render_brief_for_prompt(_sample_brief())
    assert out.startswith("# Project Briefing")
    assert "## Summary" in out
    assert "Project X is launching" in out
    assert "## Key constraints" in out
    assert "- must pass legal review" in out
    assert "## Recent focus" in out
    assert "- finalized UI copy" in out
    assert "## Open questions" in out
    assert "- who signs off release?" in out


def test_render_skips_empty_sections() -> None:
    brief = _sample_brief(
        key_constraints=(),
        recent_focus=(),
        open_questions=(),
    )
    out = render_brief_for_prompt(brief)
    assert "## Summary" in out
    assert "## Key constraints" not in out
    assert "## Recent focus" not in out
    assert "## Open questions" not in out


def test_render_minimal_brief_emits_fallback_hint() -> None:
    brief = ProjectBrief.degraded(
        project_id="p",
        tenant_id="t",
        source_hash="x",
        reason="llm_failure: Boom",
    )
    out = render_brief_for_prompt(brief)
    assert "not available" in out
    assert "llm_failure: Boom" in out
    assert "services.wiki.search" in out


def test_render_empty_brief_emits_pull_hint() -> None:
    """Empty summary and lists → hint to use wiki search / capability pulls."""
    brief = _sample_brief(
        summary="",
        key_constraints=(),
        recent_focus=(),
        open_questions=(),
    )
    out = render_brief_for_prompt(brief)
    assert "# Project Briefing" in out
    assert "empty" in out
    assert "services.wiki.search" in out


def test_render_with_meta_line() -> None:
    brief = _sample_brief()
    out = render_brief_for_prompt(brief, include_meta=True)
    assert "project_id=`p-test`" in out
    assert "tenant_id=`t-acme`" in out
    # Meta line only when explicitly requested.
    out_default = render_brief_for_prompt(brief)
    assert "project_id=`p-test`" not in out_default


def test_render_custom_header() -> None:
    out = render_brief_for_prompt(_sample_brief(), header="# Custom briefing")
    assert out.startswith("# Custom briefing")
    assert "# Project Briefing" not in out
