from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from typing import Any

from novie_agent_sdk import ArtifactReader, format_artifact_read_result


class _ArtifactsNamespace:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def read(self, artifact_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"artifact_id": artifact_id, **kwargs})
        return self.response


class _ArtifactsTextNamespace:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def read_text(self, artifact_id: str, **kwargs: Any) -> str:
        self.calls.append({"artifact_id": artifact_id, **kwargs})
        return self.response


class _ArtifactsTextByModeNamespace:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def read_text(self, artifact_id: str, **kwargs: Any) -> str:
        self.calls.append({"artifact_id": artifact_id, **kwargs})
        mode = kwargs.get("mode")
        offset = kwargs.get("offset")
        query = kwargs.get("query")
        if mode == "search":
            return f"[artifact {artifact_id}] mode=search\n\nExcerpts:\n{query}"
        if mode == "chunks":
            return f"[artifact {artifact_id}] mode=chunks\n\nContent:\noffset={offset}"
        return f"[artifact {artifact_id}] mode=summary\n\nSummary:\nCompact facts."


def test_artifact_reader_prefers_read_text_and_normalizes_uri() -> None:
    artifacts = _ArtifactsTextNamespace("SDK rendered text")
    reader = ArtifactReader(SimpleNamespace(artifacts=artifacts), purpose="test read")

    result = asyncio.run(
        reader.read_text(
            "artifact://artifact-1",
            mode="chunks",
            offset=128,
            max_bytes=4096,
        )
    )

    assert result == "SDK rendered text"
    assert artifacts.calls == [
        {
            "artifact_id": "artifact-1",
            "mode": "chunks",
            "query": None,
            "offset": 128,
            "max_bytes": 4096,
            "purpose": "test read",
        }
    ]


def test_artifact_reader_caches_exact_reads_and_budgets_uncached_reads() -> None:
    artifacts = _ArtifactsNamespace(
        {
            "available": True,
            "artifact_id": "artifact-1",
            "mode": "summary",
            "summary": "Cached summary",
        }
    )
    reader = ArtifactReader(SimpleNamespace(artifacts=artifacts), max_uncached_reads=1)

    first = asyncio.run(reader.read_text("artifact-1", mode="summary"))
    cached = asyncio.run(reader.read_text("artifact://artifact-1", mode="summary"))
    exhausted = asyncio.run(reader.read_text("artifact-2", mode="summary"))

    assert first == cached
    assert "Cached summary" in cached
    assert "step budget exhausted" in exhausted
    assert len(artifacts.calls) == 1


def test_artifact_reader_semantic_dedupe_blocks_summary_then_chunk_zero() -> None:
    artifacts = _ArtifactsTextByModeNamespace()
    reader = ArtifactReader(SimpleNamespace(artifacts=artifacts), max_uncached_reads=4)

    summary = asyncio.run(reader.read_text("artifact-1", mode="summary"))
    duplicate = asyncio.run(reader.read_text("artifact-1", mode="chunks", offset=0))
    later_chunk = asyncio.run(reader.read_text("artifact-1", mode="chunks", offset=2048))
    search = asyncio.run(
        reader.read_text("artifact-1", mode="search", query="pricing assumptions")
    )

    assert "Compact facts" in summary
    assert "already provided in this step" in duplicate
    assert "offset=2048" in later_chunk
    assert "pricing assumptions" in search
    assert [call["mode"] for call in artifacts.calls] == ["summary", "chunks", "search"]
    assert artifacts.calls[1]["offset"] == 2048
    assert reader.remaining_uncached_reads == 1


def test_artifact_reader_formats_base64_json_and_next_offset() -> None:
    encoded = base64.b64encode(
        json.dumps(
            {
                "answer": "Market evidence",
                "results": [
                    {
                        "title": "Source",
                        "url": "https://example.test",
                        "content": "Relevant public evidence.",
                    }
                ],
            }
        ).encode("utf-8")
    ).decode("ascii")

    rendered = format_artifact_read_result(
        {
            "available": True,
            "artifact_id": "artifact-1",
            "mode": "chunks",
            "metadata": {
                "encoding": "base64",
                "content_type": "application/json",
                "next_offset": 1024,
            },
            "content": {"data": encoded},
            "excerpts": [{"offset": 64, "excerpt": "Bounded excerpt"}],
        }
    )

    assert "Answer: Market evidence" in rendered
    assert "Source" in rendered
    assert "Bounded excerpt" in rendered
    assert "Next offset: 1024" in rendered
    assert encoded not in rendered


def test_artifact_reader_calls_unavailable_callback() -> None:
    seen: list[dict[str, Any]] = []
    artifacts = _ArtifactsNamespace(
        {
            "available": False,
            "artifact_id": "artifact-1",
            "error": "not_found",
            "message": "Missing artifact",
        }
    )
    reader = ArtifactReader(
        SimpleNamespace(artifacts=artifacts),
        on_unavailable=lambda data: seen.append(data),
    )

    result = asyncio.run(reader.read_text("artifact-1"))

    assert "Missing artifact" in result
    assert seen and seen[0]["error"] == "not_found"
