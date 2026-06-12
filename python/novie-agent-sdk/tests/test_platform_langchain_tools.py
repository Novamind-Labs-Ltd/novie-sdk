from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from novie_agent_sdk import (
    PlatformToolConfig,
    PlatformToolDegradationFlags,
    build_platform_langchain_tools,
)


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(
        tenant=SimpleNamespace(project_id="project-1", workspace_id="workspace-1")
    )


class _Tracker:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def mark(self, flag: str, reason: str) -> None:
        self.events.append((flag, reason))


def test_build_platform_langchain_tools_filters_allowed_tools() -> None:
    tools = build_platform_langchain_tools(
        _ctx(),
        platform=None,
        allowed_tools=("fetch_artifact",),
    )

    assert [tool.name for tool in tools] == ["fetch_artifact"]


@pytest.mark.asyncio
async def test_web_research_caches_platform_results() -> None:
    calls: list[tuple[str, int]] = []

    class _Web:
        async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((query, int(kwargs["max_results"])))
            return {
                "answer": "cached answer",
                "results": [
                    {
                        "title": "Result",
                        "url": "https://example.com",
                        "content": "Evidence text",
                    }
                ],
            }

    platform = SimpleNamespace(web=_Web())
    tools = {
        tool.name: tool
        for tool in build_platform_langchain_tools(
            _ctx(),
            platform=platform,
            config=PlatformToolConfig(web_research_budget=1),
        )
    }

    first = await tools["web_research"].ainvoke(
        {"query": "  Novie   market ", "max_results": 5}
    )
    second = await tools["web_research"].ainvoke(
        {"query": "Novie market", "max_results": 5}
    )

    assert len(calls) == 1
    assert calls == [("Novie market", 5)]
    assert "[web_research]" in first
    assert "[web_research cached_result=true]" in second


@pytest.mark.asyncio
async def test_missing_platform_namespaces_mark_degradation_flags() -> None:
    tracker = _Tracker()
    tools = {
        tool.name: tool
        for tool in build_platform_langchain_tools(
            _ctx(),
            platform=None,
            tracker=tracker,
            flags=PlatformToolDegradationFlags(
                knowledge_search="knowledge_missing",
                web_research="web_missing",
                artifact_read="artifact_missing",
            ),
        )
    }

    wiki = await tools["search_project_wiki"].ainvoke({"query": "architecture"})
    web = await tools["web_research"].ainvoke({"query": "market"})

    assert "Project wiki search is unavailable" in wiki
    assert "External web search is unavailable" in web
    assert ("knowledge_missing", "unconfigured") in tracker.events
    assert ("web_missing", "unconfigured") in tracker.events

