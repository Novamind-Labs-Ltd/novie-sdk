from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from typing import Any

from novie_agent_sdk import (
    ArtifactReader,
    artifact_read_header,
    format_artifact_read_result,
    scrub_artifact_scaffolding,
)


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


def test_scrub_removes_the_header_the_renderer_emits() -> None:
    """Anti-drift: the scrubber is pinned to the emitter's actual output.

    If `artifact_read_header` ever changes shape, this fails rather than
    letting the scrub pattern silently stop matching.
    """
    rendered = format_artifact_read_result(
        {
            "artifact_id": "art-906227eab30943de",
            "mode": "chunks",
            "content": "Revenue grew 12%.",
        }
    )
    assert artifact_read_header("art-906227eab30943de", "chunks") in rendered

    scrubbed = scrub_artifact_scaffolding(rendered)

    assert scrubbed.removed == 1
    assert "[artifact" not in scrubbed.text
    assert "mode=chunks" not in scrubbed.text
    assert "Revenue grew 12%." in scrubbed.text


def test_scrub_is_a_no_op_without_a_marker() -> None:
    body = "\n## Findings\n\nRevenue grew.\n\n"

    scrubbed = scrub_artifact_scaffolding(body)

    assert scrubbed.removed == 0
    assert scrubbed.text == body  # byte-for-byte, no incidental reformatting


def test_scrub_keeps_prose_that_merely_mentions_an_artifact() -> None:
    """Only the whole-line machine header goes; inline mentions are prose."""
    body = "We read [artifact art-1] mode=chunks inline while drafting."

    scrubbed = scrub_artifact_scaffolding(body)

    assert scrubbed.removed == 0
    assert scrubbed.text == body


def test_scrub_keeps_the_renderer_labels_that_are_ordinary_english() -> None:
    """"Summary:"/"Content:" are words authors use; stripping them would harm."""
    body = "## Notes\n\nSummary:\nRevenue grew.\n\nContent:\nDetails follow."

    scrubbed = scrub_artifact_scaffolding(body)

    assert scrubbed.removed == 0
    assert scrubbed.text == body


def test_scrub_removes_every_marker_and_is_idempotent() -> None:
    body = (
        "[artifact art-1] mode=summary\n\n"
        "First point.\n\n"
        "  [artifact art-2] mode=search  \n\n"
        "Second point.\n"
    )

    once = scrub_artifact_scaffolding(body)
    twice = scrub_artifact_scaffolding(once.text)

    assert once.removed == 2
    assert twice.removed == 0
    assert twice.text == once.text
    assert "First point." in once.text
    assert "Second point." in once.text


def test_scrub_handles_empty_and_non_string_input() -> None:
    assert scrub_artifact_scaffolding("") == ("", 0)
    assert scrub_artifact_scaffolding(None) == ("", 0)
