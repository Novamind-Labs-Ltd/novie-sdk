"""PlatformChatModel — LangChain ``BaseChatModel`` adapter for the platform LLM.

Wraps ``ctx.platform.llm.chat`` / ``ctx.platform.llm.structured`` as a
LangChain-compatible chat model so agents that use LangGraph / LangChain
pipelines can consume the platform LLM without holding an API key.

First-version capabilities:
- ``ainvoke(messages)`` — delegates to ``platform.llm.chat``.
- ``with_structured_output(schema)`` — delegates to ``platform.llm.structured``.
  Accepts a ``dict`` (JSON Schema) or a Pydantic ``BaseModel`` class.
- ``bind_tools(tools)`` — converts LangChain tools to OpenAI-compatible
  schemas and delegates tool-call generation to ``platform.llm.chat``.

Not supported in v1:
- Streaming (stream returns a single chunk containing the full response).
- ``batch`` / ``abatch``.

Usage::

    model = PlatformChatModel(platform_ns)
    resp = await model.ainvoke([HumanMessage(content="hello")])

    structured = model.with_structured_output(MySchema)
    result = await structured.ainvoke([HumanMessage(content="extract")])
"""
from __future__ import annotations

import logging
import json
from typing import Any, Iterator, Optional, Sequence, TYPE_CHECKING

from .llm_contract import (
    normalise_tool_call_chunks,
    normalise_tool_calls,
    sanitize_additional_kwargs,
)

if TYPE_CHECKING:
    from .platform_namespace import PlatformNamespace

try:
    from langchain_core.language_models.chat_models import BaseChatModel as _BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import ConfigDict, Field
except ImportError:  # pragma: no cover - exercised only without langchain-core
    _BaseChatModel = object  # type: ignore[assignment]
    ChatGeneration = None  # type: ignore[assignment]
    ChatResult = None  # type: ignore[assignment]
    ConfigDict = dict  # type: ignore[assignment]
    Field = lambda default=None, **_kwargs: default  # type: ignore[assignment]

_log = logging.getLogger(__name__)

def _require_langchain() -> None:
    try:
        import langchain_core  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "langchain-core is required for PlatformChatModel. "
            "Install it with: pip install langchain-core"
        ) from exc


class PlatformChatModel(_BaseChatModel):
    """Thin LangChain-compatible wrapper around ``ctx.platform.llm``.

    Implements enough of the Runnable chat-model protocol for use with
    LangGraph nodes and ``ainvoke`` / ``astream`` /
    ``with_structured_output`` workflows.
    """

    platform_ns: Any = Field(exclude=True)
    model: str | None = None
    temperature: float | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        platform_ns: "PlatformNamespace",
        *,
        model: str | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> None:
        _require_langchain()
        super().__init__(
            platform_ns=platform_ns,
            model=model,
            temperature=temperature,
            tools=list(tools or []),
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

    # ── LangChain BaseChatModel duck-typing ──────────────────────────────

    @property
    def _llm_type(self) -> str:
        return "novie_platform"

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError(
            "PlatformChatModel does not support synchronous invoke. Use ainvoke."
        )

    async def _agenerate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        msg = await self.ainvoke(messages, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _message_to_dict(self, message: Any) -> dict[str, Any]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        if isinstance(message, SystemMessage):
            return {"role": "system", "content": str(message.content)}
        if isinstance(message, AIMessage):
            payload: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            tool_calls = getattr(message, "tool_calls", None)
            additional_kwargs = getattr(message, "additional_kwargs", None)
            if not tool_calls and isinstance(additional_kwargs, dict):
                tool_calls = additional_kwargs.get("tool_calls")
            if tool_calls:
                normalised_tool_calls = normalise_tool_calls(tool_calls)
                if normalised_tool_calls:
                    payload["tool_calls"] = normalised_tool_calls
            invalid_tool_calls = getattr(message, "invalid_tool_calls", None)
            if invalid_tool_calls:
                payload["invalid_tool_calls"] = [dict(item) for item in invalid_tool_calls]
            if additional_kwargs:
                sanitized = sanitize_additional_kwargs(additional_kwargs)
                if sanitized:
                    payload["additional_kwargs"] = sanitized
            return payload
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": str(message.content)}
        if isinstance(message, ToolMessage):
            payload = {
                "role": "tool",
                "content": str(message.content),
                "tool_call_id": str(message.tool_call_id),
            }
            if getattr(message, "name", None):
                payload["name"] = str(message.name)
            return payload
        role = getattr(message, "type", "user")
        content = getattr(message, "content", "")
        return {"role": role, "content": content}

    async def ainvoke(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke the platform chat model.

        Accepts a list of LangChain messages or ``str``.
        Returns a LangChain ``AIMessage``.
        """
        from langchain_core.messages import AIMessage

        messages = self._normalise_input(input)
        result = await self.platform_ns.llm.chat(
            messages,
            model=self.model,
            temperature=self.temperature,
            tools=self.tools or None,
            tool_choice=self.tool_choice,
            parallel_tool_calls=self.parallel_tool_calls,
        )
        content = result.get("content") or ""
        usage_metadata = result.get("usage_metadata") or {}
        msg = AIMessage(
            content=content,
            tool_calls=normalise_tool_calls(result.get("tool_calls") or []),
            invalid_tool_calls=list(result.get("invalid_tool_calls") or []),
            additional_kwargs=sanitize_additional_kwargs(result.get("additional_kwargs") or {}),
        )
        msg.usage_metadata = usage_metadata  # type: ignore[assignment]
        return msg

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Synchronous wrapper — not supported; raises NotImplementedError."""
        raise NotImplementedError(
            "PlatformChatModel does not support synchronous invoke. Use ainvoke."
        )

    def bind_tools(self, tools: Any, **kwargs: Any) -> "PlatformChatModel":
        """Return a model variant with OpenAI-compatible tool schemas bound."""
        from langchain_core.utils.function_calling import convert_to_openai_tool

        if tools is None:
            tool_schemas: list[dict[str, Any]] = []
        else:
            tool_schemas = [convert_to_openai_tool(tool) for tool in tools]
        return PlatformChatModel(
            self.platform_ns,
            model=self.model,
            temperature=self.temperature,
            tools=tool_schemas,
            tool_choice=kwargs.get("tool_choice"),
            parallel_tool_calls=kwargs.get("parallel_tool_calls"),
        )

    def with_structured_output(
        self,
        schema: Any,
        *,
        method: str = "json_schema",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> "_StructuredPlatformModel":
        """Return a model variant that enforces structured JSON-schema output.

        ``schema`` can be a ``dict`` (JSON Schema) or a Pydantic
        ``BaseModel`` class.  In the latter case, the schema is derived
        with ``model.model_json_schema()``.
        """
        json_schema = _resolve_schema(schema)
        return _StructuredPlatformModel(
            platform_ns=self.platform_ns,
            output_schema=json_schema,
            pydantic_class=schema if _is_pydantic_class(schema) else None,
            model=self.model,
            temperature=self.temperature,
        )

    def _normalise_input(self, input: Any) -> list[dict[str, str]]:
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        if isinstance(input, list):
            return [self._message_to_dict(m) for m in input]
        return [self._message_to_dict(input)]

    # ── Streaming ─────────────────────────────────────────────────────────

    async def astream(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Stream platform chat chunks when the namespace supports it."""
        from langchain_core.messages import AIMessageChunk

        stream_chat = getattr(getattr(self.platform_ns, "llm", None), "stream_chat", None)
        if callable(stream_chat):
            tool_call_stream_state: dict[Any, Any] = {}
            async for event in stream_chat(
                self._normalise_input(input),
                model=self.model,
                temperature=self.temperature,
                tools=self.tools or None,
                tool_choice=self.tool_choice,
                parallel_tool_calls=self.parallel_tool_calls,
            ):
                event_type = str(event.get("type") or "")
                if event_type in {"accepted", "heartbeat", "progress"}:
                    continue
                if event_type == "chunk":
                    delta = event.get("delta") if isinstance(event.get("delta"), dict) else {}
                    yield AIMessageChunk(
                        content=delta.get("content") or "",
                        tool_call_chunks=normalise_tool_call_chunks(
                            delta.get("tool_call_chunks") or [],
                            tool_call_stream_state,
                        ),
                        response_metadata=dict(delta.get("response_metadata") or {}),
                    )
                    continue
                if event_type == "completed":
                    return
                if event_type in {"error", "cancelled"}:
                    raise RuntimeError(
                        "platform.llm.chat stream failed: "
                        f"{event.get('error_code') or event.get('reason') or event.get('explanation') or 'unknown'}"
                    )

        result = await self.ainvoke(input, config, **kwargs)
        tool_call_chunks = []
        for index, call in enumerate(getattr(result, "tool_calls", None) or []):
            tool_call_chunks.append(
                {
                    "name": call.get("name"),
                    "args": json.dumps(call.get("args") or {}),
                    "id": call.get("id"),
                    "index": index,
                }
            )
        yield AIMessageChunk(
            content=result.content or "",
            tool_call_chunks=tool_call_chunks,
            response_metadata=dict(getattr(result, "response_metadata", None) or {}),
        )


class _StructuredPlatformModel:
    """Returned by ``PlatformChatModel.with_structured_output``."""

    def __init__(
        self,
        platform_ns: "PlatformNamespace",
        output_schema: dict[str, Any],
        pydantic_class: Any | None,
        model: str | None,
        temperature: float | None,
    ) -> None:
        self._platform_ns = platform_ns
        self._output_schema = output_schema
        self._pydantic_class = pydantic_class
        self._model = model
        self._temperature = temperature

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
        elif isinstance(input, list):
            base = PlatformChatModel(self._platform_ns)
            messages = [base._message_to_dict(m) for m in input]
        else:
            base = PlatformChatModel(self._platform_ns)
            messages = [base._message_to_dict(input)]

        result = await self._platform_ns.llm.structured(
            messages,
            self._output_schema,
            model=self._model,
            temperature=self._temperature,
        )
        structured = result.get("structured") or result
        if self._pydantic_class is not None and isinstance(structured, dict):
            return self._pydantic_class.model_validate(structured)
        return structured

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "PlatformChatModel structured output does not support synchronous invoke."
        )


def _resolve_schema(schema: Any) -> dict[str, Any]:
    """Convert a Pydantic class or dict to a JSON Schema dict."""
    if isinstance(schema, dict):
        return schema
    if _is_pydantic_class(schema):
        return dict(schema.model_json_schema())
    raise TypeError(
        f"with_structured_output expects a dict (JSON Schema) or Pydantic BaseModel "
        f"class, got {type(schema).__name__!r}"
    )


def _is_pydantic_class(obj: Any) -> bool:
    try:
        from pydantic import BaseModel as _BM

        return isinstance(obj, type) and issubclass(obj, _BM)
    except ImportError:
        return False


__all__ = [
    "PlatformChatModel",
]
