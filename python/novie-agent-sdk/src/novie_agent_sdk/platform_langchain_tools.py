"""LangChain tools backed by Novie platform namespaces."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

from .artifact_text import ArtifactReader


@dataclass(frozen=True, slots=True)
class PlatformToolConfig:
    """Configuration for platform-backed LangChain tools."""

    fetch_artifact_budget: int = 8
    fetch_artifact_purpose: str = "evidence retrieval"
    fetch_artifact_unavailable_message: str = (
        "Artifact retrieval is unavailable because platform.artifacts is not available."
    )
    fetch_artifact_exhausted_message: str = (
        "fetch_artifact step budget exhausted. Use compact upstream context and "
        "already fetched excerpts."
    )
    web_research_budget: int = 8
    web_research_cache_entries: int = 64
    web_result_limit: int = 5
    web_answer_chars: int = 500
    web_snippet_chars: int = 260
    knowledge_unavailable_message: str = (
        "Project wiki search is unavailable because platform.knowledge.search is not available."
    )
    web_unavailable_message: str = (
        "External web search is unavailable because platform.web.search is not available."
    )
    web_budget_exhausted_message: str = (
        "web_research step budget exhausted. Stop calling public web retrieval "
        "tools in this step and return a bounded result from existing evidence."
    )


@dataclass(frozen=True, slots=True)
class PlatformToolDegradationFlags:
    """Symbolic degradation flag names used by the caller's tracker."""

    knowledge_search: str = "platform_knowledge_search"
    web_research: str = "web_research"
    artifact_read: str = "workflow_facts"


class TextResultCache:
    """Small LRU cache for text tool results within one agent step."""

    def __init__(self, *, max_entries: int = 64) -> None:
        self._max_entries = max(1, int(max_entries or 1))
        self._items: OrderedDict[tuple[Any, ...], str] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> str | None:
        value = self._items.get(key)
        if value is None:
            return None
        self._items.move_to_end(key)
        return value

    def set(self, key: tuple[Any, ...], value: str) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self._max_entries:
            self._items.popitem(last=False)


def build_platform_langchain_tools(
    ctx: Any,
    *,
    platform: Any | None,
    tracker: Any | None = None,
    allowed_tools: tuple[str, ...] | list[str] | set[str] | None = None,
    config: PlatformToolConfig | None = None,
    flags: PlatformToolDegradationFlags | None = None,
    format_wiki_results: Callable[[list[dict[str, Any]]], str] | None = None,
) -> list[Any]:
    """Build common platform-backed LangChain tools.

    The caller owns naming policy, budgets, and degradation flag names through
    ``config`` and ``flags``. The SDK owns the repeated mechanics: namespace
    checks, artifact reader wiring, per-step web budget, and result caching.
    """
    try:
        from langchain_core.tools import tool
    except ImportError as exc:
        raise RuntimeError(
            "langchain-core is required to build platform LangChain tools."
        ) from exc

    config = config or PlatformToolConfig()
    flags = flags or PlatformToolDegradationFlags()

    def _mark(flag: str, reason: str) -> None:
        if tracker is not None and hasattr(tracker, "mark"):
            tracker.mark(flag, reason)

    def _mark_artifact_unavailable(data: dict[str, Any]) -> None:
        _mark(flags.artifact_read, str(data.get("error") or "artifact_read_unavailable"))

    artifact_reader = ArtifactReader(
        platform,
        max_uncached_reads=config.fetch_artifact_budget,
        purpose=config.fetch_artifact_purpose,
        unavailable_message=config.fetch_artifact_unavailable_message,
        exhausted_message=config.fetch_artifact_exhausted_message,
        on_unavailable=_mark_artifact_unavailable,
    )
    web_research_cache = TextResultCache(max_entries=config.web_research_cache_entries)
    web_research_budget = {"remaining": max(1, int(config.web_research_budget or 1))}

    @tool
    async def search_project_wiki(query: str, top_k: int = 5) -> str:
        """Search project knowledge (wiki / curated corpus) for domain facts."""
        knowledge_namespace = getattr(platform, "knowledge", None)
        if knowledge_namespace is None:
            _mark(flags.knowledge_search, "unconfigured")
            return config.knowledge_unavailable_message
        proj_scope = getattr(getattr(ctx, "tenant", None), "project_id", None) or getattr(
            getattr(ctx, "tenant", None),
            "workspace_id",
            None,
        )
        results = await knowledge_namespace.search(
            query=query,
            top_k=top_k,
            project_id=proj_scope,
        )
        formatter = format_wiki_results or _format_wiki_results
        return formatter(results)

    @tool
    async def web_research(query: str, max_results: int = 5) -> str:
        """Search the public web for market, technical, or domain evidence."""
        web_namespace = getattr(platform, "web", None)
        if web_namespace is None:
            _mark(flags.web_research, "unconfigured")
            return config.web_unavailable_message
        normalized_query = " ".join(str(query or "").split())
        bounded_max_results = min(
            max(1, int(max_results or 1)),
            max(1, int(config.web_result_limit or 1)),
        )
        cache_key = (normalized_query.lower(), bounded_max_results)
        cached = web_research_cache.get(cache_key)
        if cached is not None:
            return cached + "\n\n[web_research cached_result=true]"
        if web_research_budget["remaining"] <= 0:
            _mark(flags.web_research, "step_budget_exhausted")
            return config.web_budget_exhausted_message
        web_research_budget["remaining"] -= 1
        data = await web_namespace.search(
            normalized_query,
            max_results=bounded_max_results,
            search_depth="advanced",
            include_answer=True,
        )
        rendered = _format_web_search_result(
            data,
            tracker=tracker,
            flag=flags.web_research,
            query=normalized_query,
            remaining_budget=web_research_budget["remaining"],
            result_limit=bounded_max_results,
            answer_chars=config.web_answer_chars,
            snippet_chars=config.web_snippet_chars,
        )
        web_research_cache.set(cache_key, rendered)
        return rendered

    @tool
    async def fetch_artifact(
        artifact_id: str,
        mode: str = "summary",
        query: str | None = None,
        offset: int = 0,
        max_bytes: int = 12000,
    ) -> str:
        """Read an artifact through platform-enforced budgets."""
        if getattr(platform, "artifacts", None) is None:
            _mark(flags.artifact_read, "artifact_read_unconfigured")
        return await artifact_reader.read_text(
            artifact_id,
            mode=mode,
            query=query,
            offset=offset,
            max_bytes=max_bytes,
        )

    tools = [search_project_wiki, web_research, fetch_artifact]
    if allowed_tools is None:
        return tools
    allowed = {str(name).strip() for name in allowed_tools if str(name).strip()}
    return [tool_item for tool_item in tools if getattr(tool_item, "name", "") in allowed]


def _format_wiki_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No relevant wiki entries found."
    lines: list[str] = []
    for idx, item in enumerate(results, start=1):
        title = str(item.get("title") or item.get("page_id") or f"wiki-{idx}")
        summary = str(item.get("summary") or item.get("content") or "").strip()
        lines.append(f"{idx}. {title}: {summary[:500]}")
    return "\n".join(lines)


def _format_web_search_result(
    data: dict[str, Any],
    *,
    tracker: Any | None = None,
    flag: str = "web_research",
    query: str = "",
    remaining_budget: int | None = None,
    result_limit: int = 5,
    answer_chars: int = 500,
    snippet_chars: int = 260,
) -> str:
    if data.get("available") is False:
        if tracker is not None and hasattr(tracker, "mark"):
            tracker.mark(flag, str(data.get("error") or "platform_unavailable"))
        message = str(data.get("message") or data.get("error") or "").strip()
        return (
            "External web search is unavailable through the platform."
            + (f" Reason: {message}" if message else "")
        )
    lines: list[str] = []
    header = "[web_research]"
    if query:
        header += f' query="{query.strip()}"'
    if remaining_budget is not None:
        header += f" remaining_budget={max(0, int(remaining_budget))}"
    lines.append(header)
    if answer := data.get("answer"):
        answer_text = str(answer).strip()
        if len(answer_text) > answer_chars:
            answer_text = answer_text[:answer_chars].rstrip() + "..."
        lines.append(f"Answer:\n{answer_text}")
    results = [
        item for item in (data.get("results") or [])
        if isinstance(item, dict)
    ][: max(1, result_limit)]
    if results:
        cards: list[str] = []
        for idx, item in enumerate(results, start=1):
            title = item.get("title") or f"Result {idx}"
            url = item.get("url") or ""
            content = str(
                item.get("content") or item.get("snippet") or ""
            ).strip().replace("\n", " ")
            if len(content) > snippet_chars:
                content = content[:snippet_chars].rstrip() + "..."
            cards.append(
                f"{idx}. {title}\nURL: {url}\nEvidence: {content}".strip()
            )
        lines.append("Evidence cards:\n" + "\n\n".join(cards))
    if len(lines) == 1 and tracker is not None and hasattr(tracker, "mark"):
        tracker.mark(flag, "no_results")
    return "\n\n".join(lines) if len(lines) > 1 else (
        "External web search returned no results."
    )


__all__ = [
    "PlatformToolConfig",
    "PlatformToolDegradationFlags",
    "TextResultCache",
    "build_platform_langchain_tools",
]

