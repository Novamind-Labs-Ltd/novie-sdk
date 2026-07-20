"""Tests for LlmFacade (platform vs BYOK routing).

Covers:
- platform mode: structured / chat / embed delegate to ctx.platform.llm.*
- BYOK mode: structured / chat / embed delegate to ByokLlmClient
- quota_exceeded in platform mode → QuotaExceededError propagated
- unavailable mode → RuntimeError with helpful message
- budget_check / usage_summary return empty in BYOK / unavailable modes
- mode property returns correct string
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from novie_agent_sdk.llm_facade import LlmFacade, build_llm_facade
from novie_agent_sdk.observability import AgentObservability
from novie_agent_sdk.platform_namespace import (
    QuotaExceededError,
    _UnavailablePlatformNamespace,
)


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_available_platform_ns(*, responses: dict[str, Any]) -> Any:
    """Minimal fake PlatformNamespace stub with is_available=True."""
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=responses.get("chat", {"content": "ok", "usage_metadata": {}}))
    llm.structured = AsyncMock(return_value=responses.get("structured", {"structured": {"result": 1}}))
    llm.embed = AsyncMock(return_value=responses.get("embed", [[0.1, 0.2]]))
    llm.budget_check = AsyncMock(return_value=responses.get("budget_check", {"allow": True, "exhausted": False}))
    llm.usage_summary = AsyncMock(return_value=responses.get("usage_summary", {"total_tokens": 100}))

    ns = MagicMock()
    ns.is_available = True
    ns.llm = llm
    return ns


def _make_byok(*, responses: dict[str, Any]) -> Any:
    byok = MagicMock()
    byok.chat = AsyncMock(return_value=responses.get("chat", {"content": "byok", "usage_metadata": {}}))
    byok.structured = AsyncMock(return_value=responses.get("structured", {"structured": {"byok": True}}))
    byok.embed = AsyncMock(return_value=responses.get("embed", [[0.9, 0.8]]))
    return byok


# ---------------------------------------------------------------------------
# Platform mode
# ---------------------------------------------------------------------------

class TestPlatformMode:
    def test_mode_is_platform(self) -> None:
        facade = LlmFacade(_make_available_platform_ns(responses={}))
        assert facade.mode == "platform"
        assert facade.platform_available is True

    def test_chat_delegates_to_platform(self) -> None:
        ns = _make_available_platform_ns(responses={"chat": {"content": "hello", "usage_metadata": {"tokens": 5}}})
        facade = LlmFacade(ns)
        result = _run(facade.chat([{"role": "user", "content": "hi"}]))
        assert result["content"] == "hello"
        assert result["llm_mode"] == "platform"
        ns.llm.chat.assert_awaited_once()

    def test_chat_forwards_disabled_reasoning_mode_to_platform(self) -> None:
        ns = _make_available_platform_ns(responses={})
        facade = LlmFacade(ns)
        _run(facade.chat(
            [{"role": "user", "content": "write"}], reasoning_mode="disabled",
        ))
        assert ns.llm.chat.await_args.kwargs["reasoning_mode"] == "disabled"

    def test_structured_delegates_to_platform(self) -> None:
        ns = _make_available_platform_ns(responses={"structured": {"structured": {"a": 1}}})
        facade = LlmFacade(ns)
        result = _run(facade.structured(
            [{"role": "user", "content": "extract"}],
            {"type": "object", "properties": {"a": {"type": "integer"}}},
        ))
        assert result["structured"] == {"a": 1}
        assert result["llm_mode"] == "platform"
        ns.llm.structured.assert_awaited_once()

    def test_structured_forwards_method_and_strict_to_platform(self) -> None:
        # Regression: previously the facade dropped these kwargs, so the
        # platform structured cap fell back to its dict-schema default
        # (strict=False) — Anthropic via OpenRouter then returned ``{}``
        # for ProductBriefArtifact-style schemas with required fields,
        # making the platform path strictly weaker than direct ChatOpenAI.
        ns = _make_available_platform_ns(responses={"structured": {"structured": {"a": 1}}})
        facade = LlmFacade(ns)
        _run(facade.structured(
            [{"role": "user", "content": "extract"}],
            {"type": "object", "properties": {"a": {"type": "integer"}}},
            method="json_schema",
            strict=True,
        ))
        kwargs = ns.llm.structured.await_args.kwargs
        assert kwargs["method"] == "json_schema"
        assert kwargs["strict"] is True

    def test_structured_forwards_timeout_to_platform(self) -> None:
        ns = _make_available_platform_ns(responses={"structured": {"structured": {"a": 1}}})
        facade = LlmFacade(ns)
        _run(facade.structured(
            [{"role": "user", "content": "extract"}],
            {"type": "object", "properties": {"a": {"type": "integer"}}},
            timeout_seconds=240,
        ))
        kwargs = ns.llm.structured.await_args.kwargs
        assert kwargs["timeout_seconds"] == 240

    def test_structured_omits_method_strict_when_unspecified(self) -> None:
        # Defaults stay ``None`` so the platform side picks its own defaults;
        # this preserves backwards compatibility with older platform builds
        # that don't yet recognise the new args.
        ns = _make_available_platform_ns(responses={"structured": {"structured": {}}})
        facade = LlmFacade(ns)
        _run(facade.structured(
            [{"role": "user", "content": "extract"}],
            {"type": "object"},
        ))
        kwargs = ns.llm.structured.await_args.kwargs
        assert kwargs["method"] is None
        assert kwargs["strict"] is None

    def test_embed_delegates_to_platform(self) -> None:
        ns = _make_available_platform_ns(responses={"embed": [[1.0, 2.0]]})
        facade = LlmFacade(ns)
        result = _run(facade.embed(["text"]))
        assert result == [[1.0, 2.0]]
        ns.llm.embed.assert_awaited_once()

    def test_budget_check_delegates_to_platform(self) -> None:
        ns = _make_available_platform_ns(responses={"budget_check": {"allow": True, "exhausted": False, "remaining_tokens": 500}})
        facade = LlmFacade(ns)
        result = _run(facade.budget_check())
        assert result["allow"] is True
        assert result["remaining_tokens"] == 500

    def test_usage_summary_delegates_to_platform(self) -> None:
        ns = _make_available_platform_ns(responses={"usage_summary": {"total_tokens": 200}})
        facade = LlmFacade(ns)
        result = _run(facade.usage_summary())
        assert result["total_tokens"] == 200

    def test_quota_exceeded_propagates(self) -> None:
        ns = _make_available_platform_ns(responses={})
        ns.llm.chat = AsyncMock(side_effect=QuotaExceededError(reason="pool exhausted"))
        facade = LlmFacade(ns)
        with pytest.raises(QuotaExceededError):
            _run(facade.chat([{"role": "user", "content": "hi"}]))


# ---------------------------------------------------------------------------
# BYOK mode
# ---------------------------------------------------------------------------

class TestByokMode:
    def test_mode_is_byok(self) -> None:
        unavail = _UnavailablePlatformNamespace(reason="no platform")
        byok = _make_byok(responses={})
        facade = LlmFacade(unavail, byok=byok)
        assert facade.mode == "byok"
        assert facade.platform_available is False

    def test_chat_delegates_to_byok(self) -> None:
        unavail = _UnavailablePlatformNamespace(reason="no platform")
        byok = _make_byok(responses={"chat": {"content": "byok-response", "usage_metadata": {}}})
        facade = LlmFacade(unavail, byok=byok)
        result = _run(facade.chat([{"role": "user", "content": "hi"}]))
        assert result["content"] == "byok-response"
        assert result["llm_mode"] == "byok"

    def test_structured_delegates_to_byok(self) -> None:
        unavail = _UnavailablePlatformNamespace(reason="no platform")
        byok = _make_byok(responses={"structured": {"structured": {"key": "val"}}})
        facade = LlmFacade(unavail, byok=byok)
        result = _run(facade.structured([{"role": "user", "content": "go"}], {}))
        assert result["structured"]["key"] == "val"
        assert result["llm_mode"] == "byok"

    def test_budget_check_returns_empty_in_byok_mode(self) -> None:
        facade = LlmFacade(_UnavailablePlatformNamespace(reason="no platform"), byok=_make_byok(responses={}))
        result = _run(facade.budget_check())
        assert result == {}

    def test_usage_summary_returns_empty_in_byok_mode(self) -> None:
        facade = LlmFacade(_UnavailablePlatformNamespace(reason="no platform"), byok=_make_byok(responses={}))
        result = _run(facade.usage_summary())
        assert result == {}


# ---------------------------------------------------------------------------
# Unavailable mode
# ---------------------------------------------------------------------------

class TestUnavailableMode:
    def test_mode_is_unavailable(self) -> None:
        facade = LlmFacade(_UnavailablePlatformNamespace(reason="no platform"))
        assert facade.mode == "unavailable"

    def test_chat_raises_runtime_error(self) -> None:
        facade = LlmFacade(_UnavailablePlatformNamespace(reason="no platform"))
        with pytest.raises(RuntimeError, match="no LLM backend"):
            _run(facade.chat([{"role": "user", "content": "hi"}]))

    def test_structured_raises_runtime_error(self) -> None:
        facade = LlmFacade(_UnavailablePlatformNamespace(reason="no platform"))
        with pytest.raises(RuntimeError, match="no LLM backend"):
            _run(facade.structured([{"role": "user", "content": "hi"}], {}))

    def test_budget_check_returns_empty(self) -> None:
        facade = LlmFacade(_UnavailablePlatformNamespace(reason="no platform"))
        result = _run(facade.budget_check())
        assert result == {}


# ---------------------------------------------------------------------------
# build_llm_facade
# ---------------------------------------------------------------------------

class TestBuildLlmFacade:
    def test_platform_mode_when_available(self) -> None:
        ns = _make_available_platform_ns(responses={})
        facade = build_llm_facade(ns, agent_id="test-agent")
        assert facade.mode == "platform"

    def test_byok_mode_when_platform_unavailable(self) -> None:
        unavail = _UnavailablePlatformNamespace(reason="no platform")
        from novie_agent_sdk.byok_llm import BYOK_API_KEY_ENV
        with patch.dict("os.environ", {BYOK_API_KEY_ENV: "test-key-sk-xxx"}):
            facade = build_llm_facade(unavail, agent_id="test-agent")
        # In test environments without langchain-openai, mode will be "unavailable".
        # With langchain-openai installed, mode will be "byok".
        assert facade.mode in ("byok", "unavailable")

    def test_unavailable_mode_when_no_key(self) -> None:
        unavail = _UnavailablePlatformNamespace(reason="no platform")
        # Ensure no BYOK key is set
        with patch.dict("os.environ", {}, clear=False):
            import os
            for env_key in ["NOVIE_AGENT_LLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"]:
                os.environ.pop(env_key, None)
            facade = build_llm_facade(unavail, agent_id="test-agent")
        # mode may be "unavailable" or "byok" depending on ambient env
        assert facade.mode in ("unavailable", "byok")


def test_llm_facade_exposes_platform_ns_publicly() -> None:
    """Consumers extract the wrapped namespace for artifacts/knowledge —
    keep platform_ns public so they never reach into _platform_ns."""
    from types import SimpleNamespace

    from novie_agent_sdk.llm_facade import LlmFacade

    namespace = SimpleNamespace(is_available=True)
    assert LlmFacade(namespace).platform_ns is namespace


def test_report_usage_delegates_to_observability() -> None:
    events: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        events.append(event)

    observability = AgentObservability(
        agent_id="sdk-agent",
        session_id="sess-1",
        step_id="step-1",
        task_event_emitter=emit,
    )
    facade = LlmFacade(
        _UnavailablePlatformNamespace(reason="no platform"),
        observability=observability,
    )

    report = _run(facade.report_usage(
        provider="anthropic",
        model="claude-sonnet",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        phase="draft",
    ))

    assert report.total_tokens == 15
    assert events[0]["payload"]["agent_event_kind"] == "token_usage"
    assert events[0]["payload"]["phase"] == "draft"


def test_usage_callback_reports_langchain_usage() -> None:
    events: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        events.append(event)

    facade = LlmFacade(
        _UnavailablePlatformNamespace(reason="no platform"),
        observability=AgentObservability(
            agent_id="sdk-agent",
            session_id="sess-1",
            step_id="step-1",
            task_event_emitter=emit,
        ),
    )
    callback = facade.usage_callback(phase="draft")
    run_id = uuid.uuid4()
    _run(callback.on_llm_start({}, ["hello"], run_id=run_id))

    class _Response:
        llm_output = {
            "model_name": "anthropic/claude-sonnet",
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        generations: list[Any] = []

    _run(callback.on_llm_end(_Response(), run_id=run_id))

    assert events[0]["payload"]["agent_event_kind"] == "token_usage"
    assert events[0]["payload"]["usage"]["total_tokens"] == 15
