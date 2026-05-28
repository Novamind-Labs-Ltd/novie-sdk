from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from novie_agent_sdk.platform_chat_model import PlatformChatModel


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


class _LlmStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"messages": messages, **kwargs})
        return {
            "content": "",
            "tool_calls": [
                {
                    "name": "lookup",
                    "args": {"query": "hello"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
            "usage_metadata": {"total_tokens": 12},
        }


class _PlatformNamespaceStub:
    def __init__(self) -> None:
        self.llm = _LlmStub()


@tool
def lookup(query: str) -> str:
    """Lookup a query."""
    return f"result:{query}"


def test_bind_tools_forwards_openai_tool_schema_and_returns_tool_calls() -> None:
    platform = _PlatformNamespaceStub()
    model = PlatformChatModel(platform).bind_tools([lookup], tool_choice="auto")

    assert isinstance(model, BaseChatModel)
    message = _run(model.ainvoke([HumanMessage(content="hello")]))

    assert platform.llm.calls[0]["tools"][0]["function"]["name"] == "lookup"
    assert platform.llm.calls[0]["tool_choice"] == "auto"
    assert message.tool_calls[0]["name"] == "lookup"
    assert message.tool_calls[0]["args"] == {"query": "hello"}


def test_astream_preserves_tool_call_chunks() -> None:
    platform = _PlatformNamespaceStub()
    model = PlatformChatModel(platform).bind_tools([lookup])

    async def _collect() -> list[Any]:
        return [chunk async for chunk in model.astream([HumanMessage(content="hello")])]

    chunks = _run(_collect())

    assert len(chunks) == 1
    assert chunks[0].tool_call_chunks[0]["name"] == "lookup"
    assert chunks[0].tool_call_chunks[0]["args"] == '{"query": "hello"}'
