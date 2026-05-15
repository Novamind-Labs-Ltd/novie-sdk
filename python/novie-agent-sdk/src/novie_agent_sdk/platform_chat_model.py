"""PlatformChatModel — LangChain ``BaseChatModel`` adapter for the platform LLM.

Wraps ``ctx.platform.llm.chat`` / ``ctx.platform.llm.structured`` as a
LangChain-compatible chat model so agents that use LangGraph / LangChain
pipelines can consume the platform LLM without holding an API key.

First-version capabilities:
- ``ainvoke(messages)`` — delegates to ``platform.llm.chat``.
- ``with_structured_output(schema)`` — delegates to ``platform.llm.structured``.
  Accepts a ``dict`` (JSON Schema) or a Pydantic ``BaseModel`` class.

Not supported in v1:
- Tool calling / function calling.
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
from typing import Any, Iterator, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .platform_namespace import PlatformNamespace

_log = logging.getLogger(__name__)

_TOOL_CALLING_NOT_SUPPORTED_MSG = (
    "PlatformChatModel v1 does not support tool calling. "
    "Use a BYOK ChatOpenAI model for tool-calling workflows."
)


def _require_langchain() -> None:
    try:
        import langchain_core  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "langchain-core is required for PlatformChatModel. "
            "Install it with: pip install langchain-core"
        ) from exc


class PlatformChatModel:
    """Thin LangChain-compatible wrapper around ``ctx.platform.llm``.

    Implements enough of the ``BaseChatModel`` protocol for use with
    basic LangGraph nodes and ``ainvoke`` / ``with_structured_output``
    workflows.  Full BaseChatModel inheritance is avoided to keep this
    module dependency-free when langchain_core is not installed.
    """

    def __init__(
        self,
        platform_ns: "PlatformNamespace",
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        _require_langchain()
        self._platform_ns = platform_ns
        self._model = model
        self._temperature = temperature
        self._output_schema: dict[str, Any] | None = None

    # ── LangChain BaseChatModel duck-typing ──────────────────────────────

    @property
    def _llm_type(self) -> str:
        return "novie_platform"

    def _message_to_dict(self, message: Any) -> dict[str, str]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        if isinstance(message, SystemMessage):
            return {"role": "system", "content": str(message.content)}
        if isinstance(message, AIMessage):
            return {"role": "assistant", "content": str(message.content)}
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": str(message.content)}
        role = getattr(message, "type", "user")
        content = str(getattr(message, "content", ""))
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

        if self._output_schema is not None:
            return await self._invoke_structured(input)

        messages = self._normalise_input(input)
        result = await self._platform_ns.llm.chat(
            messages,
            model=self._model,
            temperature=self._temperature,
        )
        content = result.get("content") or ""
        usage_metadata = result.get("usage_metadata") or {}
        msg = AIMessage(content=content)
        msg.usage_metadata = usage_metadata  # type: ignore[assignment]
        return msg

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Synchronous wrapper — not supported; raises NotImplementedError."""
        raise NotImplementedError(
            "PlatformChatModel does not support synchronous invoke. Use ainvoke."
        )

    def bind_tools(self, tools: Any, **kwargs: Any) -> "PlatformChatModel":
        """Tool binding is not supported in PlatformChatModel v1."""
        raise NotImplementedError(_TOOL_CALLING_NOT_SUPPORTED_MSG)

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
            platform_ns=self._platform_ns,
            output_schema=json_schema,
            pydantic_class=schema if _is_pydantic_class(schema) else None,
            model=self._model,
            temperature=self._temperature,
        )

    async def _invoke_structured(self, input: Any) -> Any:
        """Internal path when output_schema is bound."""
        assert self._output_schema is not None
        messages = self._normalise_input(input)
        result = await self._platform_ns.llm.structured(
            messages,
            self._output_schema,
            model=self._model,
            temperature=self._temperature,
        )
        return result.get("structured") or result

    def _normalise_input(self, input: Any) -> list[dict[str, str]]:
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        if isinstance(input, list):
            return [self._message_to_dict(m) for m in input]
        return [self._message_to_dict(input)]

    # ── Streaming stub (single-chunk) ─────────────────────────────────────

    async def astream(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Streaming — returns a single-chunk async generator (no true streaming)."""
        result = await self.ainvoke(input, config, **kwargs)
        yield result


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
