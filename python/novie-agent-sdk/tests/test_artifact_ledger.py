from __future__ import annotations

from typing import Any

import pytest

from novie_agent_sdk import ArtifactLedger, ContextBudget, EvidencePackBuilder


class _Artifacts:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.reads: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        self.created.append(kwargs)
        index = len(self.created)
        return {
            "artifact_id": f"art-{index}",
            "artifact_ref": f"artifact://art-{index}",
            "artifact_type": kwargs["artifact_type"],
            "bytes": len(str(kwargs["content"]).encode("utf-8")),
        }

    async def summarize(self, artifact_id: str, *, purpose: str = "") -> dict[str, Any]:
        self.reads.append({"method": "summarize", "artifact_id": artifact_id, "purpose": purpose})
        return {"available": True, "summary": f"summary for {artifact_id}"}

    async def search(
        self,
        artifact_id: str,
        query: str,
        *,
        purpose: str = "",
        max_bytes: int = 12000,
    ) -> dict[str, Any]:
        self.reads.append(
            {
                "method": "search",
                "artifact_id": artifact_id,
                "query": query,
                "purpose": purpose,
                "max_bytes": max_bytes,
            }
        )
        return {
            "available": True,
            "content": f"matched content for {artifact_id}: {query}",
            "metadata": {"bytes": 32},
        }

    async def read_chunks(
        self,
        artifact_id: str,
        *,
        purpose: str = "",
        offset: int = 0,
        max_bytes: int = 12000,
    ) -> dict[str, Any]:
        self.reads.append(
            {
                "method": "read_chunks",
                "artifact_id": artifact_id,
                "purpose": purpose,
                "offset": offset,
                "max_bytes": max_bytes,
            }
        )
        return {
            "available": True,
            "content": ("chunk-" + artifact_id + "-") * 20,
            "metadata": {"bytes": max_bytes, "next_offset": 10},
        }


class _Workpads:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self.final_refs: list[dict[str, Any]] = []

    async def record_entry(self, **kwargs: Any) -> dict[str, Any]:
        self.entries.append(kwargs)
        return {"available": True, "entry_id": f"entry-{len(self.entries)}"}

    async def snapshot(self, *, workflow_id: str | None = None, limit: int = 24) -> dict[str, Any]:
        return {
            "available": True,
            "workflow_id": workflow_id,
            "entries": self.entries[-limit:],
        }

    async def set_final_deliverable(
        self,
        artifact_ref: str,
        *,
        workflow_id: str | None = None,
        step_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.final_refs.append(
            {
                "artifact_ref": artifact_ref,
                "workflow_id": workflow_id,
                "step_id": step_id,
                "metadata": dict(metadata or {}),
            }
        )
        return {"available": True, "artifact_ref": artifact_ref}


class _FailingWorkpads(_Workpads):
    async def record_entry(self, **kwargs: Any) -> dict[str, Any]:
        self.entries.append(kwargs)
        return {"available": False, "error": "workpad_down"}


class _Platform:
    def __init__(self) -> None:
        self.artifacts = _Artifacts()
        self.workpads = _Workpads()


@pytest.mark.asyncio
async def test_artifact_ledger_creates_artifact_and_records_workpad_ref() -> None:
    platform = _Platform()
    ledger = ArtifactLedger(platform)

    result = await ledger.create_and_record(
        artifact_type="management_report.section",
        content="section body",
        kind="section_draft",
        title="Market",
        summary="section summary",
        workflow_id="wf-1",
        step_id="s2",
        capability_id="agent.analyst.report_synthesis",
        metadata={"section_id": "market"},
    )

    assert result["artifact"]["artifact_id"] == "art-1"
    assert platform.artifacts.created[0]["metadata"] == {"section_id": "market"}
    entry = platform.workpads.entries[0]
    assert entry["kind"] == "section_draft"
    assert entry["content_ref"] == "artifact://art-1"
    assert entry["content_preview"] == "section summary"
    assert entry["artifact_refs"] == [
        {
            "artifact_id": "art-1",
            "artifact_type": "management_report.section",
            "ref": "artifact://art-1",
            "bytes": 12,
            "kind": "section_draft",
        }
    ]


@pytest.mark.asyncio
async def test_artifact_ledger_strict_create_and_record_raises_on_workpad_failure() -> None:
    platform = _Platform()
    platform.workpads = _FailingWorkpads()
    ledger = ArtifactLedger(platform)

    with pytest.raises(RuntimeError, match="workpad_record_failed:workpad_down"):
        await ledger.create_and_record(
            artifact_type="management_report.section",
            content="section body",
            kind="section_draft",
            title="Market",
            workflow_id="wf-1",
            step_id="s2",
            strict=True,
        )

    assert platform.artifacts.created[0]["artifact_type"] == "management_report.section"
    assert platform.workpads.entries[0]["kind"] == "section_draft"


@pytest.mark.asyncio
async def test_artifact_ledger_sets_final_deliverable_ref() -> None:
    platform = _Platform()
    ledger = ArtifactLedger(platform)

    result = await ledger.set_final_deliverable(
        "artifact://final",
        workflow_id="wf-1",
        step_id="s2",
        metadata={"role": "final"},
    )

    assert result["artifact_ref"] == "artifact://final"
    assert platform.workpads.final_refs == [
        {
            "artifact_ref": "artifact://final",
            "workflow_id": "wf-1",
            "step_id": "s2",
            "metadata": {"role": "final"},
        }
    ]


@pytest.mark.asyncio
async def test_evidence_pack_builder_reads_deduped_refs_with_budget() -> None:
    platform = _Platform()
    platform.workpads.entries.append(
        {
            "title": "s1 market",
            "artifact_refs": [
                {
                    "artifact_id": "art-market",
                    "artifact_type": "market_analysis",
                    "ref": "artifact://art-market",
                }
            ],
        }
    )
    upstream = {
        "s1": {
            "artifact_refs": [
                {
                    "artifact_id": "art-market",
                    "artifact_type": "market_analysis",
                },
                {
                    "artifact_id": "art-web",
                    "artifact_type": "web_research_evidence",
                },
            ]
        }
    }
    builder = EvidencePackBuilder(
        platform,
        budget=ContextBudget(max_total_chars=45, max_item_chars=30, max_refs=4),
    )

    pack = await builder.build(
        workflow_id="wf-1",
        upstream=upstream,
        query="pricing",
        purpose="section draft",
    )

    assert [item.artifact_id for item in pack.items] == ["art-market", "art-web"]
    assert pack.items[0].content.startswith("matched content for art-market")
    assert pack.items[1].content.startswith("matched content")
    assert pack.total_chars <= 45
    assert [call["method"] for call in platform.artifacts.reads] == [
        "summarize",
        "search",
        "summarize",
        "search",
    ]
