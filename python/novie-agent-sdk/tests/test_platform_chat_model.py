from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
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


class _StreamingLlmStub(_LlmStub):
    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any):  # type: ignore[no-untyped-def]
        self.calls.append({"messages": messages, **kwargs})
        yield {"type": "accepted"}
        yield {"type": "chunk", "delta": {"content": "hel"}}
        yield {"type": "chunk", "delta": {"content": "lo"}}
        yield {"type": "completed", "result": {"content": "hello"}}


class _ToolStreamingLlmStub(_LlmStub):
    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any):  # type: ignore[no-untyped-def]
        self.calls.append({"messages": messages, **kwargs})
        yield {
            "type": "chunk",
            "delta": {
                "content": "",
                "tool_call_chunks": [
                    {
                        "type": "function",
                        "id": "toolu_1",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"quer',
                        },
                    }
                ],
            },
        }
        yield {
            "type": "chunk",
            "delta": {
                "content": "",
                "tool_call_chunks": [
                    {
                        "type": "function",
                        "id": None,
                        "function": {
                            "name": None,
                            "arguments": 'y":"hello"}',
                        },
                    }
                ],
            },
        }
        yield {"type": "completed", "result": {"content": ""}}


class _ToolBoundStreamingLlmStub(_LlmStub):
    def __init__(self) -> None:
        super().__init__()
        self.stream_calls = 0

    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any):  # type: ignore[no-untyped-def]
        self.stream_calls += 1
        self.calls.append({"messages": messages, **kwargs, "stream": True})
        yield {
            "type": "chunk",
            "delta": {
                "content": "",
                "tool_call_chunks": [
                    {
                        "id": "toolu_1",
                        "name": "lookup",
                        "args": "{}",
                    },
                    {"id": None, "name": None, "args": 'ry":"hello"}'},
                ],
            },
        }


class _StreamingPlatformNamespaceStub:
    def __init__(self) -> None:
        self.llm = _StreamingLlmStub()


class _ToolStreamingPlatformNamespaceStub:
    def __init__(self) -> None:
        self.llm = _ToolStreamingLlmStub()


class _ToolBoundStreamingPlatformNamespaceStub:
    def __init__(self) -> None:
        self.llm = _ToolBoundStreamingLlmStub()


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


def test_astream_bound_tools_uses_terminal_result_not_tool_delta_stream() -> None:
    platform = _ToolBoundStreamingPlatformNamespaceStub()
    model = PlatformChatModel(platform).bind_tools([lookup])

    async def _collect() -> list[Any]:
        return [chunk async for chunk in model.astream([HumanMessage(content="hello")])]

    chunks = _run(_collect())

    assert platform.llm.stream_calls == 0
    assert platform.llm.calls[0]["tools"][0]["function"]["name"] == "lookup"
    assert chunks[0].tool_call_chunks == [
        {
            "name": "lookup",
            "args": '{"query": "hello"}',
            "id": "call-1",
            "index": 0,
            "type": "tool_call_chunk",
        }
    ]


def test_astream_uses_platform_stream_chat_chunks() -> None:
    platform = _StreamingPlatformNamespaceStub()
    model = PlatformChatModel(platform)

    async def _collect() -> list[Any]:
        return [chunk async for chunk in model.astream([HumanMessage(content="hello")])]

    chunks = _run(_collect())

    assert [chunk.content for chunk in chunks] == ["hel", "lo"]
    assert platform.llm.calls[0]["messages"] == [{"role": "user", "content": "hello"}]


def test_ai_message_raw_additional_tool_calls_are_canonicalised() -> None:
    platform = _PlatformNamespaceStub()
    model = PlatformChatModel(platform)

    _run(
        model.ainvoke(
            [
                AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "toolu_1",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"query":"hello"}',
                                },
                            },
                            {"id": "call_1", "name": "", "args": {}},
                        ],
                        "keep": "metadata",
                    },
                )
            ]
        )
    )

    assert platform.llm.calls[0]["messages"][0] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "toolu_1",
                "name": "lookup",
                "args": {"query": "hello"},
            }
        ],
        "additional_kwargs": {"keep": "metadata"},
    }


def test_astream_normalises_openai_function_tool_chunks() -> None:
    platform = _ToolStreamingPlatformNamespaceStub()
    model = PlatformChatModel(platform)

    async def _collect() -> list[Any]:
        return [chunk async for chunk in model.astream([HumanMessage(content="hello")])]

    chunks = _run(_collect())
    pieces = [
        piece
        for chunk in chunks
        for piece in getattr(chunk, "tool_call_chunks", None) or []
    ]

    assert [piece["id"] for piece in pieces] == ["toolu_1", "toolu_1"]
    assert [piece["index"] for piece in pieces] == [0, 0]
    assert pieces[0]["name"] == "lookup"
    assert pieces[0]["args"] == '{"quer'
    assert pieces[1]["args"] == 'y":"hello"}'
