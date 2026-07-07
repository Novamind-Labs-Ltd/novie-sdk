from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from novie_agent_sdk.langchain_adapter import wrap_langchain_model_for_platform_structured_output


class _Answer(BaseModel):
    answer: str


class _FakeModel:
    def __init__(self) -> None:
        self.ainvoke = AsyncMock(return_value="chat")

    def with_structured_output(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("platform mode should use ctx.llm.structured")


class _FakeLlm:
    platform_available = True

    def __init__(self) -> None:
        self.structured = AsyncMock(return_value={"structured": {"answer": "ok"}})


def test_platform_structured_output_uses_llm_facade() -> None:
    llm = _FakeLlm()
    model = wrap_langchain_model_for_platform_structured_output(_FakeModel(), llm)

    result = asyncio.run(
        model.with_structured_output(_Answer, method="json_schema", strict=True).ainvoke(
            [HumanMessage(content="go")]
        )
    )

    assert result == _Answer(answer="ok")
    kwargs = llm.structured.await_args.kwargs
    assert llm.structured.await_args.args[0] == [{"role": "user", "content": "go"}]
    assert kwargs["method"] == "json_schema"
    assert kwargs["strict"] is True


def test_platform_adapter_keeps_plain_chat_on_wrapped_model() -> None:
    base = _FakeModel()
    model = wrap_langchain_model_for_platform_structured_output(base, _FakeLlm())

    assert asyncio.run(model.ainvoke([HumanMessage(content="hi")])) == "chat"
    base.ainvoke.assert_awaited_once()
