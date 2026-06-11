from __future__ import annotations

from typing import Any

import pytest

from novie_agent_sdk import resolve_upstream_context


class _Artifacts:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []

    async def summarize(self, artifact_id: str, *, purpose: str = "") -> dict[str, Any]:
        self.calls.append({"method": "summarize", "artifact_id": artifact_id, "purpose": purpose})
        return {
            "available": True,
            "artifact_id": artifact_id,
            "summary": "stored platform summary",
        }

    async def read_chunks(
        self,
        artifact_id: str,
        *,
        purpose: str = "",
        offset: int = 0,
        max_bytes: int = 12000,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "read_chunks",
                "artifact_id": artifact_id,
                "purpose": purpose,
                "offset": offset,
                "max_bytes": max_bytes,
            }
        )
        index = 0 if offset <= 0 else 1
        content = self.chunks[index] if index < len(self.chunks) else ""
        next_offset = None if index + 1 >= len(self.chunks) else offset + len(content.encode("utf-8"))
        return {
            "available": True,
            "artifact_id": artifact_id,
            "content": content,
            "metadata": {
                "offset": offset,
                "bytes": len(content.encode("utf-8")),
                "total_bytes": sum(len(item.encode("utf-8")) for item in self.chunks),
                "next_offset": next_offset,
            },
        }


class _Platform:
    def __init__(self, artifacts: _Artifacts | None = None) -> None:
        if artifacts is not None:
            self.artifacts = artifacts


@pytest.mark.asyncio
async def test_resolve_upstream_context_reads_chunks_for_lossy_handoff() -> None:
    artifacts = _Artifacts(["market evidence part 1", "market evidence part 2"])
    upstream = {
        "s1": {
            "artifact_type": "market_analysis",
            "summary": "deterministic compact summary",
            "artifact_refs": [
                {
                    "artifact_id": "art-1",
                    "artifact_type": "market_analysis",
                    "bytes": 1024,
                }
            ],
            "handoff_envelope": {
                "compaction": {"mode": "deterministic_fallback"},
                "omitted_fields": ["analysis", "structured_output"],
            },
            "handoff_metadata": {
                "truncated": True,
                "summary_mode": "deterministic_fallback",
            },
        }
    }

    resolved = await resolve_upstream_context(
        platform=_Platform(artifacts),
        upstream=upstream,
        purpose="report_synthesis",
        required_artifact_types={"market_analysis"},
        budget={"max_artifact_bytes_inline": 4096},
    )

    assert len(resolved.items) == 1
    item = resolved.items[0]
    assert item.status == "complete"
    assert item.retrieval_mode == "chunks"
    assert item.summary == "stored platform summary"
    assert "market evidence part 1" in item.content
    assert "market evidence part 2" in item.content
    prompt_input = resolved.to_prompt_input()
    assert prompt_input["s1"]["resolved_artifacts"][0]["content"] == item.content
    assert [call["method"] for call in artifacts.calls] == [
        "summarize",
        "read_chunks",
        "read_chunks",
    ]


@pytest.mark.asyncio
async def test_resolve_upstream_context_marks_partial_when_budget_stops_chunks() -> None:
    artifacts = _Artifacts(["a" * 20, "b" * 20])

    resolved = await resolve_upstream_context(
        platform=_Platform(artifacts),
        upstream={
            "s1": {
                "artifact_type": "market_analysis",
                "artifact_id": "art-1",
                "handoff_metadata": {"truncated": True},
            }
        },
        purpose="report_synthesis",
        budget={
            "max_artifact_bytes_inline": 20,
            "max_upstream_artifact_chunk_bytes": 20,
        },
    )

    item = resolved.items[0]
    assert item.status == "partial"
    assert item.content == "a" * 20
    assert "artifact_content_truncated_by_budget" in item.warnings


@pytest.mark.asyncio
async def test_resolve_upstream_context_surfaces_unavailable_without_artifacts_namespace() -> None:
    resolved = await resolve_upstream_context(
        platform=_Platform(),
        upstream={
            "s1": {
                "artifact_type": "market_analysis",
                "artifact_id": "art-1",
                "handoff_metadata": {"truncated": True},
            }
        },
        purpose="report_synthesis",
        required_artifact_types={"market_analysis"},
    )

    assert resolved.items[0].status == "unavailable"
    assert "platform_artifacts_unavailable" in resolved.items[0].warnings
    assert resolved.has_unavailable_required_context is True
