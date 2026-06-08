from __future__ import annotations

import base64
import binascii
import json
from collections import OrderedDict
from collections.abc import Callable
from typing import Any


class ArtifactReadCache:
    """Small in-process LRU cache for repeated bounded artifact reads."""

    def __init__(self, *, max_entries: int = 64) -> None:
        self._max_entries = max(1, max_entries)
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


class ArtifactReader:
    """Prompt-safe platform artifact reader with cache and per-step budget.

    The reader is intentionally small: agents pass a platform namespace, then ask
    for text. It owns artifact id normalization, exact-read cache keys,
    platform ``read_text`` preference, legacy ``read`` formatting, and a bounded
    count of uncached reads.
    """

    def __init__(
        self,
        platform: Any | None,
        *,
        max_uncached_reads: int = 8,
        cache: ArtifactReadCache | None = None,
        purpose: str = "agent evidence retrieval",
        unavailable_message: str = (
            "Artifact retrieval is unavailable because platform.artifacts is not available."
        ),
        exhausted_message: str = (
            "fetch_artifact step budget exhausted. Use the compact upstream "
            "handoff, Execution Workpad, and already fetched excerpts. Do not "
            "continue artifact retrieval unless the platform starts a new step "
            "with a fresh budget."
        ),
        on_unavailable: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._platform = platform
        self._cache = cache or ArtifactReadCache()
        self._remaining = max(1, int(max_uncached_reads or 1))
        self._purpose = purpose
        self._unavailable_message = unavailable_message
        self._exhausted_message = exhausted_message
        self._on_unavailable = on_unavailable
        self._read_state: dict[str, dict[str, Any]] = {}

    @property
    def remaining_uncached_reads(self) -> int:
        return self._remaining

    async def read_text(
        self,
        artifact_id: str,
        *,
        mode: str = "summary",
        query: str | None = None,
        offset: int = 0,
        max_bytes: int = 12000,
    ) -> str:
        artifacts_namespace = getattr(self._platform, "artifacts", None)
        if artifacts_namespace is None:
            return self._unavailable_message
        normalized_artifact_id = normalize_artifact_id(artifact_id)
        normalized_mode = str(mode or "summary").strip().lower() or "summary"
        normalized_query = str(query or "").strip()
        normalized_offset = int(offset or 0)
        normalized_max_bytes = int(max_bytes or 12000)
        cache_key = artifact_read_cache_key(
            normalized_artifact_id,
            mode=normalized_mode,
            query=normalized_query or None,
            offset=normalized_offset,
            max_bytes=normalized_max_bytes,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        duplicate = self._semantic_duplicate_message(
            normalized_artifact_id,
            mode=normalized_mode,
            query=normalized_query,
            offset=normalized_offset,
        )
        if duplicate is not None:
            self._cache.set(cache_key, duplicate)
            return duplicate
        if self._remaining <= 0:
            return self._exhausted_message
        self._remaining -= 1

        read_text = getattr(artifacts_namespace, "read_text", None)
        if callable(read_text):
            rendered = await read_text(
                normalized_artifact_id,
                mode=normalized_mode,
                query=normalized_query or None,
                offset=normalized_offset,
                max_bytes=normalized_max_bytes,
                purpose=self._purpose,
            )
            self._cache.set(cache_key, rendered)
            self._record_successful_read(
                normalized_artifact_id,
                mode=normalized_mode,
                query=normalized_query,
                offset=normalized_offset,
                text=rendered,
            )
            return rendered

        data = await artifacts_namespace.read(
            normalized_artifact_id,
            mode=normalized_mode,
            query=normalized_query or None,
            offset=normalized_offset,
            max_bytes=normalized_max_bytes,
            purpose=self._purpose,
        )
        if isinstance(data, dict) and data.get("available") is False and self._on_unavailable:
            self._on_unavailable(dict(data))
        rendered = format_artifact_read_result(data if isinstance(data, dict) else {})
        self._cache.set(cache_key, rendered)
        self._record_successful_read(
            normalized_artifact_id,
            mode=normalized_mode,
            query=normalized_query,
            offset=normalized_offset,
            text=rendered,
        )
        return rendered

    def _semantic_duplicate_message(
        self,
        artifact_id: str,
        *,
        mode: str,
        query: str,
        offset: int,
    ) -> str | None:
        state = self._read_state.get(artifact_id)
        if not state:
            return None
        if mode == "chunks" and not query and offset == 0 and state.get("summary_read"):
            return (
                f"Artifact {artifact_id} summary was already provided in this step. "
                "Do not reread chunks from offset 0. Use mode=\"search\" with a "
                "specific missing claim, or continue from a Next offset returned by "
                "a previous chunks read."
            )
        chunk_offsets = state.get("chunk_offsets")
        if mode == "chunks" and isinstance(chunk_offsets, set) and offset in chunk_offsets:
            return (
                f"Artifact {artifact_id} chunk offset {offset} was already provided "
                "in this step. Continue from the latest Next offset or use "
                "mode=\"search\" for a specific missing detail."
            )
        searches = state.get("searches")
        if mode == "search" and query and isinstance(searches, set) and query in searches:
            return (
                f"Artifact {artifact_id} search query was already provided in this "
                "step. Use the prior excerpt or ask a more specific follow-up query."
            )
        return None

    def _record_successful_read(
        self,
        artifact_id: str,
        *,
        mode: str,
        query: str,
        offset: int,
        text: str,
    ) -> None:
        if not str(text or "").strip():
            return
        state = self._read_state.setdefault(
            artifact_id,
            {
                "summary_read": False,
                "chunk_offsets": set(),
                "searches": set(),
            },
        )
        if mode == "summary":
            state["summary_read"] = True
        elif mode == "chunks":
            chunk_offsets = state.setdefault("chunk_offsets", set())
            if isinstance(chunk_offsets, set):
                chunk_offsets.add(offset)
        elif mode == "search" and query:
            searches = state.setdefault("searches", set())
            if isinstance(searches, set):
                searches.add(query)


def normalize_artifact_id(value: Any) -> str:
    artifact_id = str(value or "").strip()
    prefix = "artifact://"
    if artifact_id.startswith(prefix):
        artifact_id = artifact_id[len(prefix):].strip()
    return artifact_id


def artifact_read_cache_key(
    artifact_id: Any,
    *,
    mode: str = "summary",
    query: str | None = None,
    offset: int = 0,
    max_bytes: int = 12000,
) -> tuple[Any, ...]:
    return (
        normalize_artifact_id(artifact_id),
        str(mode or "summary"),
        str(query or ""),
        int(offset or 0),
        int(max_bytes or 12000),
    )


def format_artifact_read_result(data: dict[str, Any]) -> str:
    """Render platform.artifacts.read output as prompt-safe text.

    Agents should not parse platform envelopes, base64 payloads, or chunk
    continuation metadata themselves. This function accepts current and legacy
    artifact-read shapes and returns a stable string for model context.
    """
    artifact_id = str(data.get("artifact_id") or "?")
    mode = str(data.get("mode") or "read")
    if data.get("available") is False:
        message = str(data.get("message") or data.get("error") or "not available")
        return f"Artifact {artifact_id} is unavailable: {message}"

    lines = [f"[artifact {artifact_id}] mode={mode}"]
    if summary := data.get("summary"):
        lines.append(f"Summary:\n{summary}")
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    if metadata:
        lines.append(
            "Metadata:\n"
            + json.dumps(metadata, ensure_ascii=False, indent=2)[:4000]
        )

    content = data.get("content")
    content_encoding = _artifact_content_encoding(content, metadata)
    content_type = _artifact_content_type(content, metadata)
    if isinstance(content, str) and content:
        rendered_text = decode_artifact_text(
            content,
            encoding=content_encoding,
            content_type=content_type,
        )
        if rendered_text:
            lines.append(f"Content:\n{rendered_text}")
    elif isinstance(content, dict) and content:
        text = decode_artifact_text(
            content.get("data") or content.get("text") or "",
            encoding=content_encoding,
            content_type=content_type,
        )
        if text:
            lines.append(f"Content:\n{text}")
    elif isinstance(content, list) and content:
        rendered: list[str] = []
        for idx, item in enumerate(content, start=1):
            if isinstance(item, dict):
                offset = item.get("offset")
                item_metadata = metadata | {
                    key: item[key]
                    for key in ("encoding", "content_type")
                    if key in item
                }
                text = decode_artifact_text(
                    item.get("text") or item.get("content") or item.get("data") or "",
                    encoding=_artifact_content_encoding(item, item_metadata),
                    content_type=_artifact_content_type(item, item_metadata),
                )
                prefix = f"{idx}. "
                if offset is not None:
                    prefix += f"offset={offset} "
                rendered.append(prefix + text)
            else:
                rendered.append(f"{idx}. {item}")
        lines.append("Content:\n" + "\n\n".join(rendered))

    if excerpts := data.get("excerpts"):
        rendered = []
        for idx, item in enumerate(excerpts, start=1):
            if isinstance(item, dict):
                offset = item.get("offset")
                text = decode_artifact_text(
                    item.get("excerpt") or item.get("text") or item.get("content") or "",
                    encoding=str(item.get("encoding")) if item.get("encoding") else None,
                    content_type=_artifact_content_type(item, metadata),
                )
                prefix = f"{idx}. "
                if offset is not None:
                    prefix += f"offset={offset} "
                rendered.append(prefix + text)
            else:
                rendered.append(f"{idx}. {item}")
        if rendered:
            lines.append("Excerpts:\n" + "\n\n".join(rendered))

    legacy_content = content if isinstance(content, dict) else {}
    next_offset = (
        data.get("next_offset")
        or metadata.get("next_offset")
        or legacy_content.get("next_offset")
    )
    if next_offset is not None:
        lines.append(f"Next offset: {next_offset}")
    return "\n\n".join(line for line in lines if line)


def decode_artifact_text(
    value: Any,
    *,
    encoding: str | None = None,
    content_type: str | None = None,
) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized_encoding = str(encoding or "").strip().lower()
    if normalized_encoding == "base64":
        try:
            decoded = base64.b64decode(text, validate=True)
            try:
                text = decoded.decode("utf-8")
            except UnicodeDecodeError:
                if _content_type_is_textual(content_type):
                    text = decoded.decode("utf-8", errors="ignore")
                else:
                    raise
        except (binascii.Error, UnicodeDecodeError, ValueError):
            return (
                "[base64 content omitted: the platform returned binary or "
                "non-UTF-8 content that is not useful as agent evidence]"
            )
    return render_artifact_text(text, content_type=content_type)


def render_artifact_text(text: str, *, content_type: str | None = None) -> str:
    content_type_norm = str(content_type or "").lower()
    stripped = text.strip()
    if not stripped:
        return ""
    if "json" not in content_type_norm and not stripped.startswith(("{", "[")):
        return stripped
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    return render_json_artifact(payload)


def render_json_artifact(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("final_markdown", "analysis", "markdown", "report", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:12000]
        nested = _extract_nested_artifact_payload(payload)
        if nested is not None:
            return render_json_artifact(nested)
        lines: list[str] = []
        if answer := payload.get("answer"):
            lines.append(f"Answer: {answer}")
        if summary := payload.get("summary"):
            lines.append(f"Summary: {summary}")
        results = payload.get("results")
        if isinstance(results, list) and results:
            rendered_results: list[str] = []
            for idx, item in enumerate(results[:12], start=1):
                if not isinstance(item, dict):
                    rendered_results.append(f"{idx}. {item}")
                    continue
                title = str(item.get("title") or f"Result {idx}").strip()
                url = str(item.get("url") or "").strip()
                content = str(
                    item.get("content") or item.get("snippet") or ""
                ).strip().replace("\n", " ")
                line = f"{idx}. {title}"
                if url:
                    line += f"\nURL: {url}"
                if content:
                    line += f"\nSnippet: {content[:900]}"
                rendered_results.append(line)
            lines.append("Results:\n" + "\n\n".join(rendered_results))
        if lines:
            return "\n\n".join(lines)
    return json.dumps(payload, ensure_ascii=False, indent=2)[:12000]


def _artifact_content_encoding(
    content: Any,
    metadata: dict[str, Any],
) -> str | None:
    if isinstance(content, dict) and content.get("encoding"):
        return str(content.get("encoding"))
    if metadata.get("encoding"):
        return str(metadata.get("encoding"))
    return None


def _artifact_content_type(
    content: Any,
    metadata: dict[str, Any],
) -> str | None:
    if isinstance(content, dict) and content.get("content_type"):
        return str(content.get("content_type"))
    if metadata.get("content_type"):
        return str(metadata.get("content_type"))
    return None


def _content_type_is_textual(content_type: str | None) -> bool:
    normalized = str(content_type or "").strip().lower()
    if not normalized:
        return False
    return (
        normalized.startswith("text/")
        or "json" in normalized
        or "xml" in normalized
        or "yaml" in normalized
        or "markdown" in normalized
    )


def _extract_nested_artifact_payload(payload: dict[str, Any]) -> Any | None:
    for key in (
        "structured_output",
        "final_payload",
        "payload",
        "data",
    ):
        nested = payload.get(key)
        if isinstance(nested, (dict, list)):
            return nested
    provides = payload.get("provides_artifacts")
    if isinstance(provides, dict):
        for value in provides.values():
            if isinstance(value, dict):
                structured = value.get("structured_output")
                if isinstance(structured, (dict, list)):
                    return structured
    return None


__all__ = [
    "ArtifactReadCache",
    "ArtifactReader",
    "artifact_read_cache_key",
    "decode_artifact_text",
    "format_artifact_read_result",
    "normalize_artifact_id",
    "render_artifact_text",
    "render_json_artifact",
]
