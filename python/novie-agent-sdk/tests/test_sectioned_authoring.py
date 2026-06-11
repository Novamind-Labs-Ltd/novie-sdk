from __future__ import annotations

from typing import Any

import pytest

from novie_agent_sdk import SectionedLongformAuthor


class _FakeLlm:
    async def structured(
        self,
        *,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
        temperature: float,
    ) -> dict[str, Any]:
        return {
            "structured": {
                "length_profile": "short",
                "sections": [
                    {
                        "section_id": "context",
                        "title": "Context",
                        "objective": "Explain the problem context.",
                        "evidence_query": "context",
                        "min_words": 5,
                    },
                    {
                        "section_id": "recommendation",
                        "title": "Recommendation",
                        "objective": "Recommend the next step.",
                        "evidence_query": "recommendation",
                        "min_words": 5,
                    },
                ],
            }
        }

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> dict[str, str]:
        prompt = messages[0]["content"]
        if "Polish the concatenated sections" in prompt:
            return {"content": "too short"}
        if '"section_id": "recommendation"' in prompt:
            return {"content": "## Recommendation\n\neta theta iota kappa lambda mu"}
        return {"content": "## Context\n\nalpha beta gamma delta epsilon zeta"}


class _FakeArtifacts:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        artifact_id = f"artifact-{len(self.created) + 1}"
        result = {
            "artifact_id": artifact_id,
            "artifact_ref": f"artifact://{artifact_id}",
            "artifact_type": kwargs["artifact_type"],
            "bytes": len(str(kwargs.get("content") or "")),
        }
        self.created.append({**kwargs, **result})
        return result


class _FakeWorkpads:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self.final_refs: list[str] = []

    async def snapshot(self, *, workflow_id: str | None, limit: int) -> dict[str, Any]:
        return {"entries": []}

    async def record_entry(self, **kwargs: Any) -> dict[str, Any]:
        self.entries.append(dict(kwargs))
        return {"entry_id": f"entry-{len(self.entries)}"}

    async def set_final_deliverable(
        self,
        artifact_ref: str,
        *,
        workflow_id: str | None,
        step_id: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.final_refs.append(artifact_ref)
        return {"ok": True}


class _FakePlatform:
    def __init__(self) -> None:
        self.artifacts = _FakeArtifacts()
        self.workpads = _FakeWorkpads()


@pytest.mark.asyncio
async def test_sectioned_author_records_outline_sections_and_final_ref() -> None:
    platform = _FakePlatform()
    author = SectionedLongformAuthor(
        llm_facade=_FakeLlm(),
        platform=platform,
        artifact_type="prd_document",
        step_id="s2",
        capability_id="agent.pm.prd_create",
        authoring_contract={
            "coverage_model": "prd_document",
            "min_outline_sections": 2,
            "max_outline_sections": 2,
            "min_section_words": 5,
            "default_section_words": 5,
            "max_section_words": 20,
            "final_retention_ratio": 0.8,
            "outline_artifact_type": "prd_document.outline",
            "section_artifact_type": "prd_document.section",
            "final_artifact_type": "prd_document",
        },
    )

    result = await author.author(
        brief={"title": "Payments PRD"},
        upstream={},
        workflow_id="workflow-1",
        thread_id="thread-1",
        agent_id="pm",
    )

    assert "## Context" in result.markdown
    assert "## Recommendation" in result.markdown
    assert result.ledger["created_count"] == 4
    assert [item["role"] for item in result.ledger["artifact_refs"]] == [
        "outline",
        "section_draft",
        "section_draft",
        "final_deliverable",
    ]
    assert [item["artifact_type"] for item in platform.artifacts.created] == [
        "prd_document.outline",
        "prd_document.section",
        "prd_document.section",
        "prd_document",
    ]
    assert platform.workpads.final_refs == ["artifact://artifact-4"]


def test_outline_schema_carries_function_title() -> None:
    """LangChain rejects title-less dict schemas in with_structured_output
    (incident 2026-06-11: 'Unsupported function ... must have a top-level
    title key'). The outline schema must always ship a title."""
    from novie_agent_sdk.sectioned_authoring import (
        SectionedAuthoringContract,
        _outline_schema,
    )

    schema = _outline_schema(SectionedAuthoringContract())
    assert schema.get("title"), "outline schema lost its top-level title"
