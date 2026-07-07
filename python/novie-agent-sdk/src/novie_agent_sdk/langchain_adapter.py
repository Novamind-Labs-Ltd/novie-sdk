"""LangChain adapters for SDK-managed platform LLM calls."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult
from pydantic import PrivateAttr


class PlatformStructuredChatModel(BaseChatModel):
    """Delegate chat to a LangChain model and structured output to ctx.llm."""

    _model: Any = PrivateAttr()
    _llm: Any = PrivateAttr()

    def __init__(self, *, model: Any, llm_facade: Any) -> None:
        super().__init__()
        self._model = model
        self._llm = llm_facade

    def _wrapped_model(self) -> Any:
        return object.__getattribute__(self, "__pydantic_private__")["_model"]

    def _wrapped_llm(self) -> Any:
        return object.__getattribute__(self, "__pydantic_private__")["_llm"]

    @property
    def _llm_type(self) -> str:
        return "novie-platform-structured-wrapper"

    def __getattr__(self, name: str) -> Any:
        if name == "_model":
            return self._wrapped_model()
        if name == "_llm":
            return self._wrapped_llm()
        return getattr(self._wrapped_model(), name)

    def bind_tools(self, *args: Any, **kwargs: Any) -> "PlatformStructuredChatModel":
        return PlatformStructuredChatModel(
            model=self._wrapped_model().bind_tools(*args, **kwargs),
            llm_facade=self._wrapped_llm(),
        )

    def bind(self, *args: Any, **kwargs: Any) -> "PlatformStructuredChatModel":
        return PlatformStructuredChatModel(
            model=self._wrapped_model().bind(*args, **kwargs),
            llm_facade=self._wrapped_llm(),
        )

    def with_config(self, *args: Any, **kwargs: Any) -> "PlatformStructuredChatModel":
        return PlatformStructuredChatModel(
            model=self._wrapped_model().with_config(*args, **kwargs),
            llm_facade=self._wrapped_llm(),
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        if not getattr(self._wrapped_llm(), "platform_available", False):
            return self._wrapped_model().with_structured_output(schema, **kwargs)
        return _PlatformStructuredRunnable(self._wrapped_llm(), schema, kwargs)

    def _generate(self, messages: list[Any], **kwargs: Any) -> ChatResult:
        return self._wrapped_model()._generate(messages, **kwargs)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrapped_model().invoke(*args, **kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return await self._wrapped_model().ainvoke(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrapped_model().stream(*args, **kwargs)

    async def astream(self, *args: Any, **kwargs: Any) -> Any:
        async for item in self._wrapped_model().astream(*args, **kwargs):
            yield item


class _PlatformStructuredRunnable:
    def __init__(self, llm_facade: Any, schema: Any, kwargs: Mapping[str, Any]) -> None:
        self._llm = llm_facade
        self._schema = schema
        self._kwargs = dict(kwargs)

    async def ainvoke(
        self,
        input: Any,
        config: Mapping[str, Any] | None = None,
        **_kwargs: Any,
    ) -> Any:
        result = await self._llm.structured(
            _messages_to_wire(input),
            _schema_to_json_schema(self._schema),
            method=self._kwargs.get("method"),
            strict=self._kwargs.get("strict"),
        )
        parsed = _parse_structured(self._schema, result.get("structured", {}))
        if self._kwargs.get("include_raw"):
            return {"raw": None, "parsed": parsed, "parsing_error": None}
        return parsed


def wrap_langchain_model_for_platform_structured_output(
    model: Any,
    llm_facade: Any | None,
) -> Any:
    """Use SDK structured calls while preserving a normal LangChain model."""
    if llm_facade is None or not getattr(llm_facade, "platform_available", False):
        return model
    return PlatformStructuredChatModel(model=model, llm_facade=llm_facade)


def _schema_to_json_schema(schema: Any) -> dict[str, Any]:
    if isinstance(schema, Mapping):
        return dict(schema)
    if hasattr(schema, "model_json_schema"):
        return dict(schema.model_json_schema())
    raise TypeError("structured schema must be a JSON schema mapping or Pydantic model")


def _parse_structured(schema: Any, value: Any) -> Any:
    if hasattr(schema, "model_validate"):
        return schema.model_validate(value)
    return value


def _messages_to_wire(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [{"role": "user", "content": str(value)}]
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list):
        value = [value]
    return [_message_to_wire(message) for message in value]


def _message_to_wire(message: Any) -> dict[str, Any]:
    if isinstance(message, tuple) and len(message) >= 2:
        return {"role": _role_from_type(str(message[0])), "content": str(message[1])}
    role = _role_from_type(str(getattr(message, "type", "user")))
    payload: dict[str, Any] = {"role": role, "content": getattr(message, "content", "")}
    if role == "tool" and getattr(message, "tool_call_id", None):
        payload["tool_call_id"] = str(message.tool_call_id)
    return payload


def _role_from_type(value: str) -> str:
    return {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
    }.get(value, value or "user")


__all__ = [
    "PlatformStructuredChatModel",
    "wrap_langchain_model_for_platform_structured_output",
]
