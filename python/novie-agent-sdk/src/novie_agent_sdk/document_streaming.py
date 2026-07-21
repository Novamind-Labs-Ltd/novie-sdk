"""Streaming utilities for document-style LangGraph/DeepAgents runtimes."""
from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from novie_protocol.agents import AgentStreamEvent

from .timeout_policy import DEFAULT_SDK_TIMEOUTS
from .workpad import workpad_checkpoint_event

DEFAULT_KEEPALIVE_INTERVAL_SECONDS = DEFAULT_SDK_TIMEOUTS.stream_keepalive_seconds
DEFAULT_SUBTASK_IDLE_TIMEOUT_SECONDS = DEFAULT_SDK_TIMEOUTS.subtask_idle_seconds
KEEPALIVE_DONE = object()
DEFAULT_MIN_SUBTASK_RESULT_CHARS = 900
DEFAULT_SUBTASK_EVIDENCE_MAX_CHARS = 1_000_000

_INCOMPLETE_MARKERS = (
    "budget exhausted",
    "model-call budget exhausted",
    "context budget exceeded",
    "step budget exhausted",
    "needs_more_budget",
    "status: incomplete",
    "status: needs_more_budget",
)

_EVIDENCE_MARKERS = (
    "## evidence",
    "evidence:",
    "source",
    "sources",
    "citation",
    "artifact://",
    "http://",
    "https://",
)


@dataclass(frozen=True)
class SubtaskResultAssessment:
    status: str
    result_chars: int
    reasons: tuple[str, ...] = ()
    has_evidence_signal: bool = False

    @property
    def complete(self) -> bool:
        return self.status == "complete"


class SubtaskIdleTimeoutError(TimeoutError):
    """Raised when a DeepAgents subtask stays active without stream progress."""

    def __init__(
        self,
        *,
        idle_seconds: float,
        timeout_seconds: float,
        subtask: dict[str, Any] | None = None,
    ) -> None:
        self.idle_seconds = float(idle_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self.subtask = dict(subtask or {})
        subtask_id = str(self.subtask.get("subtask_id") or "unknown")
        super().__init__(
            "subtask idle timeout: "
            f"subtask_id={subtask_id} idle_seconds={int(self.idle_seconds)} "
            f"timeout_seconds={int(self.timeout_seconds)}"
        )


def _result_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_result_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return "\n".join(_result_text(item) for item in value)
    return str(value or "")


def assess_subtask_result(
    result: Any,
    *,
    result_chars: int | None = None,
    min_result_chars: int = DEFAULT_MIN_SUBTASK_RESULT_CHARS,
) -> SubtaskResultAssessment:
    text = _result_text(result).strip()
    measured_chars = len(text)
    chars = measured_chars if result_chars is None else max(0, int(result_chars))
    lowered = text.lower()
    reasons: list[str] = []
    if chars <= 0:
        reasons.append("empty_result")
    elif chars < min_result_chars:
        reasons.append("too_short")
    if any(marker in lowered for marker in _INCOMPLETE_MARKERS):
        reasons.append("budget_or_incomplete_marker")
    has_evidence_signal = any(marker in lowered for marker in _EVIDENCE_MARKERS)
    if text and not has_evidence_signal:
        reasons.append("missing_evidence_signal")
    return SubtaskResultAssessment(
        status="incomplete" if reasons else "complete",
        result_chars=chars,
        reasons=tuple(dict.fromkeys(reasons)),
        has_evidence_signal=has_evidence_signal,
    )


def keepalive_interval_seconds(
    *,
    env_var: str = "NOVIE_AGENT_KEEPALIVE_INTERVAL_S",
    default: float = DEFAULT_KEEPALIVE_INTERVAL_SECONDS,
) -> float:
    raw = os.getenv(env_var, "") or os.getenv("NOVIE_AGENT_KEEPALIVE_INTERVAL_S", "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _resolve_subtask_idle_timeout_seconds(
    *,
    env_var: str = "NOVIE_AGENT_SUBTASK_IDLE_TIMEOUT_S",
    default: float = DEFAULT_SUBTASK_IDLE_TIMEOUT_SECONDS,
) -> float:
    raw = os.getenv(env_var, "") or os.getenv("NOVIE_AGENT_SUBTASK_IDLE_TIMEOUT_S", "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def subtask_idle_timeout_seconds(
    *,
    env_var: str = "NOVIE_AGENT_SUBTASK_IDLE_TIMEOUT_S",
    default: float = DEFAULT_SUBTASK_IDLE_TIMEOUT_SECONDS,
) -> float:
    """Return the active-subtask idle timeout.

    A value <= 0 disables the timeout while preserving keepalive emission.
    """
    return _resolve_subtask_idle_timeout_seconds(env_var=env_var, default=default)


def content_stream_closed_event(
    *,
    phase_metadata: Callable[..., dict[str, Any]],
    mode: str,
    phase: str,
    capability_id: str | None,
    next_phase: str,
    event_name: str = "content_stream_closed",
) -> AgentStreamEvent:
    return AgentStreamEvent(
        kind="trace",
        metadata={
            **phase_metadata(
                runtime_phase=next_phase,
                mode=mode,
                phase=phase,
                capability_id=capability_id,
            ),
            "event": event_name,
            "content_stream_closed": True,
            "status": "finalizing",
        },
    )


def keepalive_event(
    *,
    phase_metadata: Callable[..., dict[str, Any]],
    runtime_phase: str,
    mode: str,
    phase: str,
    capability_id: str | None,
    idle_seconds: float,
) -> AgentStreamEvent:
    return AgentStreamEvent(
        kind="trace",
        metadata={
            **phase_metadata(
                runtime_phase=runtime_phase,
                mode=mode,
                phase=phase,
                capability_id=capability_id,
            ),
            "event": "agent_keepalive",
            "idle_seconds": int(idle_seconds),
            "status": "still_running",
        },
    )


async def with_keepalive(
    source: AsyncIterator[Any],
    *,
    phase_metadata: Callable[..., dict[str, Any]],
    runtime_phase: str,
    mode: str,
    phase: str,
    capability_id: str | None,
    interval_seconds: float | None = None,
    env_var: str = "NOVIE_AGENT_KEEPALIVE_INTERVAL_S",
) -> AsyncIterator[Any]:
    interval = (
        keepalive_interval_seconds(env_var=env_var)
        if interval_seconds is None
        else interval_seconds
    )
    if interval <= 0:
        async for item in source:
            yield item
        return

    queue: asyncio.Queue[Any] = asyncio.Queue()

    async def _produce() -> None:
        try:
            async for item in source:
                await queue.put(item)
        except Exception as exc:
            await queue.put(exc)
        finally:
            await queue.put(KEEPALIVE_DONE)

    producer = asyncio.create_task(_produce())
    loop = asyncio.get_running_loop()
    last_event_at = loop.time()
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=interval)
            except TimeoutError:
                now = loop.time()
                yield keepalive_event(
                    phase_metadata=phase_metadata,
                    runtime_phase=runtime_phase,
                    mode=mode,
                    phase=phase,
                    capability_id=capability_id,
                    idle_seconds=now - last_event_at,
                )
                continue
            if item is KEEPALIVE_DONE:
                break
            if isinstance(item, Exception):
                raise item
            last_event_at = loop.time()
            yield item
    finally:
        if not producer.done():
            producer.cancel()


def text_blob(*parts: Any) -> str:
    flattened: list[str] = []
    for part in parts:
        if isinstance(part, str):
            flattened.append(part)
        elif isinstance(part, dict):
            flattened.append(text_blob(*part.values()))
        elif isinstance(part, (list, tuple)):
            flattened.append(text_blob(*part))
    return "".join(flattened)


def extract_chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", chunk)
    return text_blob(content)


def is_assistant_content_chunk(chunk: Any) -> bool:
    if getattr(chunk, "type", None) == "tool":
        return False
    if getattr(chunk, "role", None) == "tool":
        return False
    if not hasattr(chunk, "content"):
        return False
    if getattr(chunk, "tool_calls", None) or getattr(chunk, "tool_call_chunks", None):
        return False
    return bool(extract_chunk_text(chunk))


def normalize_langgraph_stream_item(item: Any) -> tuple[tuple[str, ...], str, Any]:
    if isinstance(item, tuple) and len(item) == 3:
        namespace, stream_mode, payload = item
        if isinstance(namespace, tuple):
            ns = tuple(str(part) for part in namespace)
        elif isinstance(namespace, list):
            ns = tuple(str(part) for part in namespace)
        elif namespace:
            ns = (str(namespace),)
        else:
            ns = ()
        return ns, str(stream_mode), payload
    if isinstance(item, tuple) and len(item) == 2:
        stream_mode, payload = item
        return (), str(stream_mode), payload
    raise ValueError(f"unsupported langgraph stream item shape: {type(item).__name__}")


def _first_nonempty_line(text: str) -> str:
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _bounded_label(text: str, *, limit: int = 120) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def subtask_evidence_max_chars(
    *,
    env_var: str = "NOVIE_AGENT_SUBTASK_EVIDENCE_MAX_CHARS",
    default: int = DEFAULT_SUBTASK_EVIDENCE_MAX_CHARS,
) -> int:
    raw = os.getenv(env_var, "") or os.getenv("NOVIE_AGENT_SUBTASK_EVIDENCE_MAX_CHARS", "")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(0, value)


def bounded_subtask_evidence_text(text: str, *, limit: int | None = None) -> tuple[str, bool]:
    value = str(text or "").strip()
    max_chars = subtask_evidence_max_chars() if limit is None else max(0, int(limit))
    if max_chars <= 0 or len(value) <= max_chars:
        return value, False
    return value[:max_chars].rstrip() + "\n\n[truncated]", True


class SubtaskEventMapper:
    """Map DeepAgents ``task`` tool calls into platform-visible events."""

    def __init__(self, *, base_metadata: dict[str, Any] | None = None) -> None:
        self._base_metadata = dict(base_metadata or {})
        self._active: dict[str, dict[str, Any]] = {}
        self._counter = 0

    def map_tool_event(self, event: AgentStreamEvent) -> list[AgentStreamEvent]:
        if event.kind == "tool_call" and event.tool_name == "task":
            return [self._started_event(event), event]
        if event.kind == "tool_result" and event.tool_name == "task":
            completed = self._completed_event(event)
            workpad = self._workpad_event(event, completed)
            if workpad is not None:
                return [event, workpad, completed]
            return [event, completed]
        return [event]

    def map_subgraph_tool_event(
        self,
        event: AgentStreamEvent,
        *,
        namespace: tuple[str, ...] = (),
    ) -> list[AgentStreamEvent]:
        if event.tool_name == "task":
            return self.map_tool_event(event)
        record = self._active_record()
        if record is None:
            return [event]
        subtask_id = str(record.get("subtask_id") or "")
        subagent_type = str(record.get("subagent_type") or "subtask")
        tool_name = str(event.tool_name or "tool")
        if event.kind == "tool_call":
            progress = f"{subagent_type}: calling {tool_name}"
            event_name = "subtask.tool_call"
            semantic_phase = "subtask_tool_call"
        elif event.kind == "tool_result":
            progress = f"{subagent_type}: {tool_name} returned"
            event_name = "subtask.tool_result"
            semantic_phase = "subtask_tool_result"
        else:
            return [event]
        trace = AgentStreamEvent(
            kind="trace",
            metadata={
                **self._base_metadata,
                "event": event_name,
                "runtime_phase": "subtask",
                "semantic_phase": semantic_phase,
                "progress_label": progress,
                "status": "running",
                "subtask": {**dict(record), "status": "running", "namespace": list(namespace)},
                "subtask_id": subtask_id,
                "subagent_type": subagent_type,
                "tool_name": tool_name,
                "tool_call_id": event.tool_call_id,
                "visibility": "summary",
            },
        )
        enriched_event = AgentStreamEvent(
            kind=event.kind,
            content=event.content,
            output=event.output,
            metadata={
                **dict(event.metadata or {}),
                "subtask_id": subtask_id,
                "subagent_type": subagent_type,
                "subtask_namespace": list(namespace),
                "event": event_name,
                "visibility": "internal",
                "tool_result_visibility": "internal",
            },
            tool_name=event.tool_name,
            tool_args=event.tool_args,
            tool_result=event.tool_result,
            tool_call_id=event.tool_call_id,
        )
        return [trace, enriched_event]

    def subgraph_content_event(
        self,
        text: str,
        *,
        namespace: tuple[str, ...] = (),
    ) -> AgentStreamEvent | None:
        stripped = str(text or "").strip()
        if not stripped:
            return None
        record = self._active_record()
        if record is None:
            return None
        subagent_type = str(record.get("subagent_type") or "subtask")
        preview = _bounded_label(stripped, limit=180)
        return AgentStreamEvent(
            kind="trace",
            metadata={
                **self._base_metadata,
                "event": "subtask.stream_content",
                "runtime_phase": "subtask",
                "semantic_phase": "subtask_stream_content",
                "progress_label": f"{subagent_type}: {preview}",
                "status": "running",
                "text_delta": preview,
                "subtask": {**dict(record), "status": "running", "namespace": list(namespace)},
                "subtask_id": record.get("subtask_id"),
                "subagent_type": subagent_type,
                "visibility": "summary",
            },
        )

    def active_keepalive_event(self, *, idle_seconds: float) -> AgentStreamEvent | None:
        record = self._active_record()
        if record is None:
            return None
        key = str(record.get("subtask_id") or "")
        subagent_type = str(record.get("subagent_type") or "subtask")
        title = str(record.get("title") or "subtask")
        return AgentStreamEvent(
            kind="trace",
            metadata={
                **self._base_metadata,
                "event": "subtask.running",
                "runtime_phase": "subtask",
                "semantic_phase": "subtask_running",
                "progress_label": f"{subagent_type}: {title} still running ({int(idle_seconds)}s)",
                "status": "still_running",
                "subtask": {**dict(record), "status": "running", "idle_seconds": int(idle_seconds)},
                "subtask_id": key,
                "subagent_type": subagent_type,
                "visibility": "summary",
            },
        )

    def active_timeout_event(
        self,
        *,
        idle_seconds: float,
        timeout_seconds: float,
    ) -> AgentStreamEvent | None:
        record = self._active_record()
        if record is None:
            return None
        key = str(record.get("subtask_id") or "")
        subagent_type = str(record.get("subagent_type") or "subtask")
        title = str(record.get("title") or "subtask")
        return AgentStreamEvent(
            kind="trace",
            metadata={
                **self._base_metadata,
                "event": "subtask.idle_timeout",
                "runtime_phase": "subtask",
                "semantic_phase": "subtask_timeout",
                "progress_label": (
                    f"{subagent_type}: {title} idle timeout "
                    f"({int(idle_seconds)}s)"
                ),
                "status": "timeout",
                "subtask": {
                    **dict(record),
                    "status": "timeout",
                    "idle_seconds": int(idle_seconds),
                    "timeout_seconds": int(timeout_seconds),
                },
                "subtask_id": key,
                "subagent_type": subagent_type,
                "idle_seconds": int(idle_seconds),
                "timeout_seconds": int(timeout_seconds),
                "visibility": "summary",
            },
        )

    def _active_record(self) -> dict[str, Any] | None:
        if not self._active:
            return None
        _key, record = next(reversed(self._active.items()))
        return dict(record)

    def _call_key(self, event: AgentStreamEvent) -> str:
        if event.tool_call_id:
            return str(event.tool_call_id)
        if event.kind == "tool_result" and len(self._active) == 1:
            return next(iter(self._active))
        self._counter += 1
        return f"task-{self._counter}"

    def _started_event(self, event: AgentStreamEvent) -> AgentStreamEvent:
        key = self._call_key(event)
        args = event.tool_args if isinstance(event.tool_args, dict) else {}
        description = str(args.get("description") or "")
        subagent_type = str(args.get("subagent_type") or "unknown")
        title = _bounded_label(_first_nonempty_line(description) or subagent_type)
        record = {
            "subtask_id": key,
            "subagent_type": subagent_type,
            "title": title,
            "description_chars": len(description),
        }
        self._active[key] = record
        return AgentStreamEvent(
            kind="trace",
            metadata={
                **self._base_metadata,
                "event": "subtask.started",
                "runtime_phase": "subtask",
                "semantic_phase": "subtask_running",
                "progress_label": f"{subagent_type}: {title}",
                "subtask": dict(record),
                "subtask_id": key,
                "subagent_type": subagent_type,
                "tool_call_id": event.tool_call_id,
                "visibility": "summary",
            },
        )

    def _workpad_event(
        self,
        event: AgentStreamEvent,
        completed: AgentStreamEvent,
    ) -> AgentStreamEvent | None:
        result_text = str(getattr(event, "tool_result", "") or "").strip()
        if not result_text:
            return None
        metadata = dict(completed.metadata or {})
        subtask = metadata.get("subtask") if isinstance(metadata.get("subtask"), dict) else {}
        content, truncated = bounded_subtask_evidence_text(result_text)
        title = str(subtask.get("title") or metadata.get("subagent_type") or "Subtask")
        return workpad_checkpoint_event(
            kind="subtask_evidence_card",
            title=title,
            content=content,
            base_metadata={
                **self._base_metadata,
                "runtime_phase": "subtask",
                "semantic_phase": metadata.get("semantic_phase") or "subtask_complete",
                "visibility": "internal",
            },
            metadata={
                "subtask_id": metadata.get("subtask_id"),
                "subagent_type": metadata.get("subagent_type"),
                "subtask_status": metadata.get("subtask_status"),
                "subtask_incomplete_reasons": list(metadata.get("subtask_incomplete_reasons") or []),
                "tool_result_chars": metadata.get("tool_result_chars"),
                "content_truncated": truncated,
            },
        )

    def _completed_event(self, event: AgentStreamEvent) -> AgentStreamEvent:
        key = self._call_key(event)
        record = self._active.pop(key, None) or {
            "subtask_id": key,
            "subagent_type": "unknown",
            "title": "subtask",
        }
        result_chars = 0
        if isinstance(event.metadata, dict):
            try:
                result_chars = int(event.metadata.get("tool_result_chars") or 0)
            except (TypeError, ValueError):
                result_chars = 0
        assessment = assess_subtask_result(
            getattr(event, "tool_result", None),
            result_chars=result_chars or None,
        )
        event_name = "subtask.completed" if assessment.complete else "subtask.incomplete"
        semantic_phase = "subtask_complete" if assessment.complete else "subtask_incomplete"
        status_label = "complete" if assessment.complete else "incomplete"
        return AgentStreamEvent(
            kind="trace",
            metadata={
                **self._base_metadata,
                "event": event_name,
                "runtime_phase": "subtask",
                "semantic_phase": semantic_phase,
                "progress_label": f"{record.get('subagent_type')}: {record.get('title')} {status_label}",
                "subtask": {
                    **dict(record),
                    "result_chars": assessment.result_chars,
                    "status": assessment.status,
                    "reasons": list(assessment.reasons),
                    "has_evidence_signal": assessment.has_evidence_signal,
                },
                "subtask_id": key,
                "subagent_type": record.get("subagent_type"),
                "tool_call_id": event.tool_call_id,
                "tool_result_chars": assessment.result_chars,
                "subtask_status": assessment.status,
                "subtask_incomplete_reasons": list(assessment.reasons),
                "visibility": "summary",
            },
        )


async def with_subtask_keepalive(
    source: AsyncIterator[Any],
    *,
    subtask_events: SubtaskEventMapper,
    phase_metadata: Callable[..., dict[str, Any]],
    runtime_phase: str,
    mode: str,
    phase: str,
    capability_id: str | None,
    interval_seconds: float | None = None,
    subtask_idle_timeout_seconds: float | None = None,
    env_var: str = "NOVIE_AGENT_KEEPALIVE_INTERVAL_S",
    subtask_idle_timeout_env_var: str = "NOVIE_AGENT_SUBTASK_IDLE_TIMEOUT_S",
) -> AsyncIterator[Any]:
    interval = (
        keepalive_interval_seconds(env_var=env_var)
        if interval_seconds is None
        else interval_seconds
    )
    idle_timeout = (
        _resolve_subtask_idle_timeout_seconds(
            env_var=subtask_idle_timeout_env_var,
        )
        if subtask_idle_timeout_seconds is None
        else float(subtask_idle_timeout_seconds)
    )
    if interval <= 0:
        async for item in source:
            yield item
        return

    queue: asyncio.Queue[Any] = asyncio.Queue()

    async def _produce() -> None:
        try:
            async for item in source:
                await queue.put(item)
        except Exception as exc:
            await queue.put(exc)
        finally:
            await queue.put(KEEPALIVE_DONE)

    producer = asyncio.create_task(_produce())
    loop = asyncio.get_running_loop()
    last_event_at = loop.time()
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=interval)
            except TimeoutError:
                now = loop.time()
                idle_seconds = now - last_event_at
                subtask_keepalive = subtask_events.active_keepalive_event(
                    idle_seconds=idle_seconds
                )
                if subtask_keepalive is not None:
                    if idle_timeout > 0 and idle_seconds >= idle_timeout:
                        timeout_event = subtask_events.active_timeout_event(
                            idle_seconds=idle_seconds,
                            timeout_seconds=idle_timeout,
                        )
                        if timeout_event is not None:
                            yield timeout_event
                            timeout_subtask = dict(
                                timeout_event.metadata.get("subtask") or {}
                            )
                        else:
                            timeout_subtask = {}
                        raise SubtaskIdleTimeoutError(
                            idle_seconds=idle_seconds,
                            timeout_seconds=idle_timeout,
                            subtask=timeout_subtask,
                        )
                    yield subtask_keepalive
                else:
                    yield keepalive_event(
                        phase_metadata=phase_metadata,
                        runtime_phase=runtime_phase,
                        mode=mode,
                        phase=phase,
                        capability_id=capability_id,
                        idle_seconds=idle_seconds,
                    )
                continue
            if item is KEEPALIVE_DONE:
                break
            if isinstance(item, Exception):
                raise item
            last_event_at = loop.time()
            yield item
    finally:
        if not producer.done():
            producer.cancel()


__all__ = [
    "DEFAULT_KEEPALIVE_INTERVAL_SECONDS",
    "DEFAULT_MIN_SUBTASK_RESULT_CHARS",
    "KEEPALIVE_DONE",
    "DEFAULT_SUBTASK_IDLE_TIMEOUT_SECONDS",
    "SubtaskIdleTimeoutError",
    "SubtaskEventMapper",
    "SubtaskResultAssessment",
    "assess_subtask_result",
    "content_stream_closed_event",
    "extract_chunk_text",
    "is_assistant_content_chunk",
    "keepalive_event",
    "keepalive_interval_seconds",
    "normalize_langgraph_stream_item",
    "subtask_idle_timeout_seconds",
    "text_blob",
    "with_keepalive",
    "with_subtask_keepalive",
]
