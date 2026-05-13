"""Agent-side observability helpers for LangChain and manual usage reporting.

The SDK owns the agent-facing API. Langfuse is an optional sink behind this
facade; platform/project usage management should consume the Novie usage event
shape rather than depending on Langfuse as the source of truth.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import UUID

try:  # LangChain is optional for non-Python/non-LangChain agents.
    from langchain_core.callbacks import AsyncCallbackHandler as _LangChainCallbackBase
except ImportError:  # pragma: no cover - exercised when optional extra is absent.
    class _LangChainCallbackBase:  # type: ignore[no-redef]
        pass


UsageEventEmitter = Callable[[dict[str, Any]], Awaitable[None]]


class ObservabilitySink(Protocol):
    """Sink for agent-side observability events."""

    async def record_usage(self, event: "UsageReport") -> None: ...


@dataclass(frozen=True, slots=True)
class UsageReport:
    """Agent-side standard LLM usage event.

    The platform enriches this with trusted org/project/user context from the
    invocation boundary before writing the authoritative usage ledger.
    """

    event_id: str
    recorded_at: str
    agent_id: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    task_id: str | None = None
    step_id: str | None = None
    session_id: str | None = None
    trace_id: str | None = None
    phase: str | None = None
    turn_id: str | None = None
    span_name: str | None = None
    idempotency_key: str | None = None
    raw_usage_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        agent_id: str,
        provider: str,
        model: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        latency_ms: float | None = None,
        task_id: str | None = None,
        step_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        phase: str | None = None,
        turn_id: str | None = None,
        span_name: str | None = None,
        idempotency_key: str | None = None,
        raw_usage_metadata: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "UsageReport":
        return cls(
            event_id=f"usg_evt_{uuid.uuid4().hex}",
            recorded_at=datetime.now(timezone.utc).isoformat(),
            agent_id=agent_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            task_id=task_id,
            step_id=step_id,
            session_id=session_id,
            trace_id=trace_id,
            phase=phase,
            turn_id=turn_id,
            span_name=span_name,
            idempotency_key=idempotency_key,
            raw_usage_metadata=dict(raw_usage_metadata or {}),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "recorded_at": self.recorded_at,
            "agent_id": self.agent_id,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "phase": self.phase,
            "turn_id": self.turn_id,
            "span_name": self.span_name,
            "idempotency_key": self.idempotency_key,
            "raw_usage_metadata": dict(self.raw_usage_metadata),
            "metadata": dict(self.metadata),
        }

    def to_platform_task_event(self) -> dict[str, Any]:
        """Return an A2A task event shape consumed by the platform activity."""
        usage = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        payload = {
            "agent_event_kind": "token_usage",
            "usage": usage,
            "provider": self.provider,
            "model": self.model,
            "phase": self.phase,
            "turn_id": self.turn_id,
            "span_name": self.span_name,
            "latency_ms": self.latency_ms,
            "trace_id": self.trace_id,
            "idempotency_key": self.idempotency_key,
            "raw_usage_metadata": dict(self.raw_usage_metadata),
            "metadata": dict(self.metadata),
        }
        return {
            "event_id": self.event_id,
            "task_id": self.task_id or "",
            "kind": "status_changed",
            "timestamp": self.recorded_at,
            "summary": _usage_summary(self),
            "payload": payload,
        }


class NoOpObservabilitySink:
    async def record_usage(self, event: UsageReport) -> None:
        return None


class LangfuseObservabilitySink:
    """Best-effort Langfuse writer hidden behind the SDK facade."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client or self._build_client()

    @staticmethod
    def _build_client() -> Any:
        from langfuse import Langfuse

        return Langfuse()

    async def record_usage(self, event: UsageReport) -> None:
        await asyncio.to_thread(self._record_usage_sync, event)

    def _record_usage_sync(self, event: UsageReport) -> None:
        usage = {
            "input": event.input_tokens,
            "output": event.output_tokens,
            "total": event.total_tokens,
        }
        metadata = event.to_dict()
        metadata.pop("raw_usage_metadata", None)
        if hasattr(self._client, "generation"):
            self._client.generation(
                name=event.span_name or "llm",
                trace_id=event.trace_id,
                model=event.model,
                metadata=metadata,
                usage=usage,
            )
        elif hasattr(self._client, "trace"):
            trace = self._client.trace(
                id=event.trace_id,
                name=f"agent:{event.agent_id}",
                session_id=event.session_id,
                metadata=metadata,
            )
            generation = getattr(trace, "generation", None)
            if generation is not None:
                generation(
                    name=event.span_name or "llm",
                    model=event.model,
                    usage=usage,
                    metadata=metadata,
                )
        flush = getattr(self._client, "flush", None)
        if flush is not None:
            flush()


def build_default_sinks() -> tuple[ObservabilitySink, ...]:
    """Build sinks from env without making Langfuse a hard dependency."""
    enabled = os.getenv("NOVIE_LANGFUSE_ENABLED", "").lower() in {"1", "true", "yes"}
    if not enabled:
        return ()
    try:
        return (LangfuseObservabilitySink(),)
    except Exception:
        return ()


@dataclass(slots=True)
class AgentObservability:
    """Scoped observability facade attached to SDK runtime contexts."""

    agent_id: str
    session_id: str = ""
    step_id: str = ""
    trace_id: str = ""
    task_id: str = ""
    sinks: tuple[ObservabilitySink, ...] = ()
    task_event_emitter: UsageEventEmitter | None = None

    def scoped(
        self,
        *,
        session_id: str = "",
        step_id: str = "",
        trace_id: str = "",
        task_id: str = "",
        task_event_emitter: UsageEventEmitter | None = None,
    ) -> "AgentObservability":
        return AgentObservability(
            agent_id=self.agent_id,
            session_id=session_id,
            step_id=step_id,
            trace_id=trace_id,
            task_id=task_id,
            sinks=self.sinks,
            task_event_emitter=task_event_emitter,
        )

    def langchain_callbacks(
        self,
        *,
        phase: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[Any]:
        return [
            NovieLangChainCallbackHandler(
                self,
                phase=phase,
                metadata=metadata,
            )
        ]

    def langchain_config(
        self,
        *,
        phase: str | None = None,
        metadata: dict[str, Any] | None = None,
        base: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = dict(base or {})
        callbacks = list(config.get("callbacks") or [])
        callbacks.extend(self.langchain_callbacks(phase=phase, metadata=metadata))
        config["callbacks"] = callbacks
        return config

    async def report_llm_usage(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        latency_ms: float | None = None,
        phase: str | None = None,
        turn_id: str | None = None,
        span_name: str | None = None,
        idempotency_key: str | None = None,
        raw_usage_metadata: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UsageReport:
        report = UsageReport.new(
            agent_id=self.agent_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            task_id=self.task_id or None,
            step_id=self.step_id or None,
            session_id=self.session_id or None,
            trace_id=self.trace_id or None,
            phase=phase,
            turn_id=turn_id,
            span_name=span_name,
            idempotency_key=idempotency_key,
            raw_usage_metadata=raw_usage_metadata,
            metadata=metadata,
        )
        for sink in self.sinks:
            try:
                await sink.record_usage(report)
            except Exception:
                pass
        if self.task_event_emitter is not None:
            await self.task_event_emitter(report.to_platform_task_event())
        return report


class NovieLangChainCallbackHandler(_LangChainCallbackBase):
    """LangChain callback that records LLM usage through ``AgentObservability``."""

    def __init__(
        self,
        observability: AgentObservability,
        *,
        phase: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._observability = observability
        self._phase = phase
        self._metadata = dict(metadata or {})
        self._start_times: dict[str, float] = {}
        self._tool_call_count = 0

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._start_times[str(run_id)] = time.perf_counter()

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._start_times[str(run_id)] = time.perf_counter()

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._tool_call_count += 1

    async def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        start = self._start_times.pop(str(run_id), None)
        latency_ms = (time.perf_counter() - start) * 1000 if start is not None else None
        usage = _extract_usage(response)
        provider, model = _resolve_provider_model(response, usage)
        await self._observability.report_llm_usage(
            provider=provider,
            model=model,
            input_tokens=_safe_int(
                usage.get("input_tokens") or usage.get("prompt_tokens")
            ),
            output_tokens=_safe_int(
                usage.get("output_tokens") or usage.get("completion_tokens")
            ),
            total_tokens=_safe_int(usage.get("total_tokens") or usage.get("total")),
            latency_ms=latency_ms,
            phase=self._phase,
            span_name="langchain.llm",
            raw_usage_metadata=usage,
            metadata={**self._metadata, "tool_call_count": self._tool_call_count},
        )
        self._tool_call_count = 0


def _extract_usage(response: Any) -> dict[str, Any]:
    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if usage:
        return dict(usage)
    generations = getattr(response, "generations", None) or []
    if generations:
        first = generations[0]
        if first:
            generation_info = getattr(first[0], "generation_info", None) or {}
            usage = (
                generation_info.get("usage_metadata")
                or generation_info.get("usage")
                or {}
            )
            if usage:
                return dict(usage)
    return {}


def _resolve_provider_model(response: Any, usage: dict[str, Any]) -> tuple[str, str]:
    llm_output = getattr(response, "llm_output", None) or {}
    model_name = (
        llm_output.get("model_name")
        or llm_output.get("model")
        or usage.get("model")
        or "unknown"
    )
    model_text = str(model_name)
    if "/" in model_text:
        provider, _, model = model_text.partition("/")
        return provider.lower(), model.lower()
    return str(usage.get("provider") or "unknown").lower(), model_text.lower()


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_summary(report: UsageReport) -> str:
    total = report.total_tokens
    if total is None:
        total = (report.input_tokens or 0) + (report.output_tokens or 0)
    return (
        f"token usage provider={report.provider} model={report.model} "
        f"in={report.input_tokens or 0} out={report.output_tokens or 0} total={total}"
    )
