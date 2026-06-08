"""Reusable runtime primitives for document-style agents."""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage
from novie_protocol.agents import AgentStreamEvent

from .context_budget import (
    budget_limit,
    budget_status,
    budget_summary,
    context_budget_from_inputs,
    default_context_budget,
    effective_phase_timeout_seconds,
    estimated_tokens,
    wall_clock_deadline,
)
from .document_streaming import (
    extract_chunk_text,
    is_assistant_content_chunk,
    normalize_langgraph_stream_item,
    with_keepalive,
)


@dataclass(frozen=True)
class DocumentGraphStreamResult:
    narrative: str
    last_messages: list[Any]
    content_chars: int = 0


PhaseMetadataBuilder = Callable[..., dict[str, Any]]


class DocumentAgentTemplate:
    """Shared runtime helpers for custom document-agent loops.

    This intentionally does not hide an agent's domain workflow. It centralizes
    cross-agent mechanics: budget math, keepalives, graph stream normalization,
    and stable progress events.
    """

    def __init__(
        self,
        *,
        owner_agent_id: str,
        phase_metadata: PhaseMetadataBuilder,
        keepalive_env_var: str = "NOVIE_AGENT_KEEPALIVE_INTERVAL_S",
        context_budget_source: str = "document_agent_default",
        context_budget_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.owner_agent_id = owner_agent_id
        self._phase_metadata = phase_metadata
        self._keepalive_env_var = keepalive_env_var
        self._default_context_budget = default_context_budget(
            source=context_budget_source,
            overrides=context_budget_overrides,
        )

    def context_budget(self, inputs: dict[str, Any] | None) -> dict[str, Any]:
        return context_budget_from_inputs(inputs, defaults=self._default_context_budget)

    def wall_clock_deadline(self, context_budget: dict[str, Any]) -> float | None:
        return wall_clock_deadline(context_budget)

    def effective_phase_timeout_seconds(
        self,
        context_budget: dict[str, Any],
        phase_name: str,
        default: float,
        deadline: float | None,
    ) -> float:
        return effective_phase_timeout_seconds(context_budget, phase_name, default, deadline)

    def budget_summary(
        self,
        *,
        context_budget: dict[str, Any],
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        degraded_codes: list[str] | None = None,
    ) -> dict[str, Any]:
        return budget_summary(
            context_budget=context_budget,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            degraded_codes=degraded_codes,
        )

    def estimated_tokens(self, value: Any) -> int:
        return estimated_tokens(value)

    def budget_limit(self, context_budget: dict[str, Any], key: str, default: int) -> int:
        return budget_limit(context_budget, key, default)

    def phase_metadata(
        self,
        *,
        runtime_phase: str,
        semantic_phase: str | None = None,
        mode: str,
        phase: str,
        capability_id: str | None,
    ) -> dict[str, Any]:
        return self._phase_metadata(
            runtime_phase=runtime_phase,
            semantic_phase=semantic_phase,
            mode=mode,
            phase=phase,
            capability_id=capability_id,
        )

    def budget_estimate_event(
        self,
        *,
        runtime_phase: str,
        semantic_phase: str,
        mode: str,
        phase: str,
        capability_id: str,
        context_budget: dict[str, Any],
        estimated_input_tokens: int,
        estimated_output_tokens: int,
    ) -> AgentStreamEvent:
        status = budget_status(
            context_budget=context_budget,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
        )
        return AgentStreamEvent(
            kind="trace",
            metadata={
                **self.phase_metadata(
                    runtime_phase=runtime_phase,
                    semantic_phase=semantic_phase,
                    mode=mode,
                    phase=phase,
                    capability_id=capability_id,
                ),
                "event": "llm_budget_estimate",
                "estimated_input_tokens": estimated_input_tokens,
                "estimated_output_tokens": estimated_output_tokens,
                "estimated_total_tokens": estimated_input_tokens + estimated_output_tokens,
                "max_input_tokens": budget_limit(context_budget, "max_input_tokens", 12000),
                "max_output_tokens": budget_limit(context_budget, "max_output_tokens", 8000),
                "max_total_tokens": budget_limit(context_budget, "max_total_tokens", 40000),
                "budget_status": status,
            },
        )

    def budget_degraded_event(
        self,
        *,
        runtime_phase: str,
        semantic_phase: str,
        mode: str,
        phase: str,
        capability_id: str,
        degradation_code: str,
        budget_status: str | None = None,
        timeout_seconds: float | None = None,
    ) -> AgentStreamEvent:
        metadata: dict[str, Any] = {
            **self.phase_metadata(
                runtime_phase=runtime_phase,
                semantic_phase=semantic_phase,
                mode=mode,
                phase=phase,
                capability_id=capability_id,
            ),
            "event": "degraded",
            "degradation_code": degradation_code,
            "retryability": "not_retryable_without_replan",
        }
        if budget_status:
            metadata["budget_status"] = budget_status
        if timeout_seconds is not None:
            metadata["timeout_seconds"] = timeout_seconds
        return AgentStreamEvent(kind="trace", metadata=metadata)

    def final_deliverable_progress_event(
        self,
        *,
        runtime_phase: str,
        semantic_phase: str = "finalizing_output",
        mode: str,
        phase: str,
        capability_id: str,
        summary: str,
        progress_label: str,
    ) -> AgentStreamEvent:
        return AgentStreamEvent(
            kind="trace",
            metadata={
                **self.phase_metadata(
                    runtime_phase=runtime_phase,
                    semantic_phase=semantic_phase,
                    mode=mode,
                    phase=phase,
                    capability_id=capability_id,
                ),
                "event": "final_deliverable_progress",
                "status": "finalizing",
                "summary": summary,
                "progress_label": progress_label,
            },
        )

    async def with_keepalive(
        self,
        source: AsyncIterator[Any],
        *,
        runtime_phase: str,
        semantic_phase: str,
        mode: str,
        phase: str,
        capability_id: str,
        interval_seconds: float | None = None,
    ) -> AsyncIterator[Any]:
        def _metadata(
            *,
            runtime_phase: str,
            mode: str,
            phase: str,
            capability_id: str | None,
        ) -> dict[str, Any]:
            return self.phase_metadata(
                runtime_phase=runtime_phase,
                semantic_phase=semantic_phase,
                mode=mode,
                phase=phase,
                capability_id=capability_id,
            )

        async for item in with_keepalive(
            source,
            phase_metadata=_metadata,
            runtime_phase=runtime_phase,
            mode=mode,
            phase=phase,
            capability_id=capability_id,
            interval_seconds=interval_seconds,
            env_var=self._keepalive_env_var,
        ):
            yield item

    async def stream_graph_run(
        self,
        *,
        graph: Any,
        prompt: str,
        callbacks: list[Any],
        runtime_phase: str,
        semantic_phase: str,
        mode: str,
        phase: str,
        capability_id: str,
        config: dict[str, Any] | None = None,
        content_metadata: dict[str, Any] | None = None,
        extract_values_text: Callable[[Any], str] | None = None,
    ) -> AsyncIterator[AgentStreamEvent | DocumentGraphStreamResult]:
        """Drive a LangGraph/DeepAgents stream into content events and result."""
        run_input = {"messages": [HumanMessage(content=prompt)]}
        narrative_parts: list[str] = []
        last_messages: list[Any] = []
        content_chars = 0

        async def _graph_stream() -> AsyncIterator[Any]:
            stream_config = dict(config or {})
            if callbacks:
                existing_callbacks = list(stream_config.get("callbacks") or [])
                stream_config["callbacks"] = [*existing_callbacks, *callbacks]
            try:
                stream = graph.astream(
                    run_input,
                    stream_mode=["messages", "values"],
                    config=stream_config or None,
                )
            except TypeError:
                stream = graph.astream(run_input)
            async for item in stream:
                yield item

        async for item in self.with_keepalive(
            _graph_stream(),
            runtime_phase=runtime_phase,
            semantic_phase=semantic_phase,
            mode=mode,
            phase=phase,
            capability_id=capability_id,
        ):
            if isinstance(item, AgentStreamEvent):
                yield item
                continue
            namespace, stream_mode, payload = normalize_langgraph_stream_item(item)
            _ = namespace
            if stream_mode == "messages":
                message_chunk = payload[0] if isinstance(payload, tuple) and payload else payload
                if is_assistant_content_chunk(message_chunk):
                    text = extract_chunk_text(message_chunk)
                    if text:
                        narrative_parts.append(text)
                        content_chars += len(text)
                        yield AgentStreamEvent(
                            kind="content",
                            content=text,
                            metadata=dict(content_metadata or {}),
                        )
                continue
            if stream_mode == "values":
                if isinstance(payload, dict):
                    last_messages = payload.get("messages") or last_messages
                value_text = extract_values_text(payload) if extract_values_text else ""
                if value_text:
                    narrative_parts.append(value_text)

        yield DocumentGraphStreamResult(
            narrative="\n".join(part for part in narrative_parts if part).strip(),
            last_messages=last_messages,
            content_chars=content_chars,
        )


__all__ = [
    "DocumentAgentTemplate",
    "DocumentGraphStreamResult",
]
