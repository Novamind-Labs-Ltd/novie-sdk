"""Tests for LlmNamespace in platform_namespace.py.

Covers:
- platform.llm.chat success path.
- quota_exceeded in result dict → QuotaExceededError raised.
- quota_exceeded in envelope error_code → QuotaExceededError raised.
- budget_check and usage_summary degrade gracefully on error.
- _UnavailablePlatformNamespace.llm raises QuotaExceededError on quota_exceeded.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from novie_agent_sdk.platform_namespace import (
    CapabilityCallDiagnostics,
    LlmNamespace,
    PlatformLlmCallError,
    PlatformLlmTransportError,
    PlatformNamespace,
    QuotaExceededError,
    _UnavailablePlatformNamespace,
    _CapabilityCaller,
)


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def _make_ns(*, responses: dict[str, CapabilityCallDiagnostics]) -> PlatformNamespace:
    """Build a PlatformNamespace with a fake caller that returns canned responses."""
    caller = MagicMock(spec=_CapabilityCaller)

    async def _invoke(cap_id: str, args: Any) -> CapabilityCallDiagnostics:
        return responses.get(
            cap_id,
            CapabilityCallDiagnostics(
                ok=False, capability_id=cap_id, kind="transport_error", detail="no canned response",
            ),
        )

    caller.invoke_with_diagnostics = _invoke
    return PlatformNamespace(caller)


class TestLlmChat:
    def test_success(self) -> None:
        ns = _make_ns(responses={
            "platform.llm.chat": CapabilityCallDiagnostics(
                ok=True,
                capability_id="platform.llm.chat",
                result={"content": "Hi there!", "usage_metadata": {"total_tokens": 30}},
            ),
        })
        result = _run(ns.llm.chat([{"role": "user", "content": "Hello"}]))
        assert result["content"] == "Hi there!"

    def test_quota_exceeded_in_result_raises(self) -> None:
        ns = _make_ns(responses={
            "platform.llm.chat": CapabilityCallDiagnostics(
                ok=True,
                capability_id="platform.llm.chat",
                result={
                    "error": "quota_exceeded",
                    "reason": "org token pool exhausted",
                    "org_id": "org-1",
                    "remaining_tokens": 0,
                },
            ),
        })
        with pytest.raises(QuotaExceededError) as exc_info:
            _run(ns.llm.chat([{"role": "user", "content": "Hello"}]))
        assert exc_info.value.org_id == "org-1"
        assert exc_info.value.remaining_tokens == 0
        assert "exhausted" in exc_info.value.reason.lower()

    def test_quota_exceeded_in_error_code_raises(self) -> None:
        ns = _make_ns(responses={
            "platform.llm.chat": CapabilityCallDiagnostics(
                ok=False,
                capability_id="platform.llm.chat",
                error_code="quota_exceeded",
                kind="platform_unavailable",
            ),
        })
        with pytest.raises(QuotaExceededError):
            _run(ns.llm.chat([{"role": "user", "content": "Hello"}]))

    def test_transport_error_raises_platform_llm_transport_error(self) -> None:
        """Pre-0.3.3 SDKs returned ``{}`` here, which let the analyst's
        finalize chain interpret a transport failure as an LLM-returned
        empty object and surface as ``ProductBriefArtifact summary Field
        required``.  0.3.3 raises ``PlatformLlmTransportError`` so callers
        can decide whether to retry or surface a clear failure."""
        ns = _make_ns(responses={
            "platform.llm.chat": CapabilityCallDiagnostics(
                ok=False,
                capability_id="platform.llm.chat",
                kind="transport_error",
                detail="read timeout",
            ),
        })
        with pytest.raises(PlatformLlmTransportError) as excinfo:
            _run(ns.llm.chat([{"role": "user", "content": "Hello"}]))
        assert excinfo.value.capability_id == "platform.llm.chat"
        assert excinfo.value.is_transient is True
        assert "read timeout" in excinfo.value.detail

    def test_non_transport_failure_raises_platform_llm_call_error(self) -> None:
        """Other non-quota failures (binding denied, schema violation,
        platform unavailable) also raise — they're operational failures,
        not data the agent should silently absorb."""
        ns = _make_ns(responses={
            "platform.llm.chat": CapabilityCallDiagnostics(
                ok=False,
                capability_id="platform.llm.chat",
                kind="binding_denied",
                error_code="denied_by_binding",
                detail="no grant",
            ),
        })
        with pytest.raises(PlatformLlmCallError) as excinfo:
            _run(ns.llm.chat([{"role": "user", "content": "Hello"}]))
        assert excinfo.value.kind == "binding_denied"
        assert excinfo.value.is_transient is False
        # Quota exceptions are deliberately a separate class — make sure
        # this isn't accidentally a ``QuotaExceededError`` subclass.
        assert not isinstance(excinfo.value, QuotaExceededError)


class TestLlmEmbed:
    def test_success(self) -> None:
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        ns = _make_ns(responses={
            "platform.llm.embed": CapabilityCallDiagnostics(
                ok=True,
                capability_id="platform.llm.embed",
                result={"embeddings": vectors, "count": 2},
            ),
        })
        result = _run(ns.llm.embed(["hello", "world"]))
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]

    def test_quota_exceeded_raises(self) -> None:
        ns = _make_ns(responses={
            "platform.llm.embed": CapabilityCallDiagnostics(
                ok=True,
                capability_id="platform.llm.embed",
                result={"error": "quota_exceeded", "org_id": "org-2", "remaining_tokens": 0, "reason": "exhausted"},
            ),
        })
        with pytest.raises(QuotaExceededError) as exc_info:
            _run(ns.llm.embed(["hello"]))
        assert exc_info.value.org_id == "org-2"


class TestLlmBudgetCheck:
    def test_success(self) -> None:
        ns = _make_ns(responses={
            "platform.llm.budget_check": CapabilityCallDiagnostics(
                ok=True,
                capability_id="platform.llm.budget_check",
                result={"allow": True, "remaining_tokens": 500},
            ),
        })
        result = _run(ns.llm.budget_check())
        assert result["allow"] is True
        assert result["remaining_tokens"] == 500

    def test_returns_empty_on_error(self) -> None:
        ns = _make_ns(responses={
            "platform.llm.budget_check": CapabilityCallDiagnostics(
                ok=False,
                capability_id="platform.llm.budget_check",
                kind="transport_error",
            ),
        })
        result = _run(ns.llm.budget_check())
        assert result == {}


class TestLlmUsageSummary:
    def test_success(self) -> None:
        ns = _make_ns(responses={
            "platform.llm.usage_summary": CapabilityCallDiagnostics(
                ok=True,
                capability_id="platform.llm.usage_summary",
                result={"total_tokens": 1234, "scope": "session"},
            ),
        })
        result = _run(ns.llm.usage_summary(scope="session"))
        assert result["total_tokens"] == 1234

    def test_returns_empty_on_error(self) -> None:
        ns = _make_ns(responses={
            "platform.llm.usage_summary": CapabilityCallDiagnostics(
                ok=False,
                capability_id="platform.llm.usage_summary",
                kind="transport_error",
            ),
        })
        result = _run(ns.llm.usage_summary())
        assert result == {}


class TestUnavailableNamespaceHasLlm:
    """_UnavailablePlatformNamespace exposes .llm sub-namespace."""

    def test_llm_attribute_exists(self) -> None:
        ns = _UnavailablePlatformNamespace(reason="test")
        assert hasattr(ns, "llm")

    def test_llm_chat_raises_platform_llm_call_error(self) -> None:
        """Calling ``ns.llm.chat`` on the unavailable namespace is a
        programming error in normal flow (handlers should branch on
        ``LlmFacade.platform_available`` / fall back to BYOK first), but
        if it happens we surface it as a structured failure rather than
        silently returning ``{}`` — same reasoning as the transport_error
        case."""
        ns = _UnavailablePlatformNamespace(reason="test")
        with pytest.raises(PlatformLlmCallError) as excinfo:
            _run(ns.llm.chat([{"role": "user", "content": "hi"}]))
        # ``unconfigured`` is not in the transient set — agents shouldn't
        # retry against an unconfigured platform.
        assert excinfo.value.kind == "unconfigured"
        assert excinfo.value.is_transient is False


class TestQuotaExceededError:
    def test_attributes(self) -> None:
        exc = QuotaExceededError(org_id="org-x", remaining_tokens=5, reason="pool done")
        assert exc.org_id == "org-x"
        assert exc.remaining_tokens == 5
        assert "pool done" in str(exc)

    def test_default_message(self) -> None:
        exc = QuotaExceededError()
        assert "exhausted" in str(exc).lower()
