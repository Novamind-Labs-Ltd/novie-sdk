"""LLM Facade — unified platform-or-BYOK LLM surface for SDK agents.

Exposes a consistent ``chat / structured / embed`` interface that
automatically delegates to the platform when available and falls back to
the agent's own key (BYOK) otherwise.  Agent code never needs to branch
on ``platform.is_available`` to decide which LLM path to use.

Usage::

    ctx.llm.structured(
        messages=[{"role": "user", "content": "..."}],
        output_schema=MyModel.model_json_schema(),
    )

The facade is injected into every ``InvokeContext``, ``StreamContext``,
and ``TaskContext`` automatically by the SDK runtime; agent code should
not instantiate it directly.
"""
from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .byok_llm import ByokLlmClient
    from .platform_namespace import (
        LlmNamespace,
        PlatformNamespace,
        QuotaExceededError,
        _UnavailablePlatformNamespace,
    )

_log = logging.getLogger(__name__)

LlmMode = Literal["platform", "byok", "unavailable"]


class LlmFacade:
    """Unified LLM surface for SDK agent handlers.

    Attributes:
        platform_available: True when ``ctx.platform.is_available``.
        mode: ``"platform"`` when using the platform-managed LLM,
            ``"byok"`` when using the agent's own key, ``"unavailable"``
            when neither is configured.
    """

    def __init__(
        self,
        platform_ns: "PlatformNamespace | _UnavailablePlatformNamespace",
        *,
        byok: "ByokLlmClient | None" = None,
    ) -> None:
        self._platform_ns = platform_ns
        self._byok = byok

    @property
    def platform_ns(self) -> "PlatformNamespace | _UnavailablePlatformNamespace":
        """Wrapped PlatformNamespace, for consumers that need the non-LLM
        namespaces (artifacts/knowledge/workpad) alongside the facade."""
        return self._platform_ns

    @property
    def platform_available(self) -> bool:
        return getattr(self._platform_ns, "is_available", False)

    @property
    def mode(self) -> LlmMode:
        if self.platform_available:
            return "platform"
        if self._byok is not None:
            return "byok"
        return "unavailable"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> dict[str, Any]:
        """Send a chat request.

        Routes to the platform when connected, to BYOK otherwise.

        Returns:
            ``{"content": str, "usage_metadata": {...}, "llm_mode": str}``.

        Raises:
            QuotaExceededError: In platform mode when the org pool is
                exhausted.
            RuntimeError: In BYOK mode when no API key is configured.
        """
        if self.platform_available:
            result = await self._platform_ns.llm.chat(
                messages,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
            )
            return {**result, "llm_mode": "platform"}

        if self._byok is not None:
            if tools:
                raise RuntimeError(
                    "LlmFacade BYOK chat does not support tool-calling through "
                    "the facade. Use a LangChain ChatModel for tool workflows."
                )
            result = await self._byok.chat(
                messages,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            return {**result, "llm_mode": "byok"}

        raise RuntimeError(
            "LlmFacade: no LLM backend configured. "
            "Either connect to the Novie platform (set NOVIE_PLATFORM_BASE_URL) "
            f"or configure a BYOK key ({_byok_key_hint()})."
        )

    async def stream_text(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a chat response as platform-shaped text events.

        Platform mode delegates to ``llm.stream_chat`` when available. BYOK and
        legacy/non-streaming platform callers fall back to ``chat`` and emit the
        final text once as a chunk, followed by a completed event. Consumers can
        therefore build one visibility path without branching on backend mode.
        """
        if self.platform_available:
            stream_chat = getattr(getattr(self._platform_ns, "llm", None), "stream_chat", None)
            if callable(stream_chat):
                async for event in stream_chat(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                ):
                    yield event
                return

        result = await self.chat(
            messages,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        content = str(result.get("content") or "")
        if content:
            yield {"type": "chunk", "delta": {"content": content}}
        yield {"type": "completed", "result": result}

    async def structured(
        self,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
        *,
        model: str | None = None,
        temperature: float | None = None,
        method: str | None = None,
        strict: bool | None = None,
    ) -> dict[str, Any]:
        """Invoke the LLM with a JSON-schema structured output contract.

        Routes to the platform when connected, to BYOK otherwise.

        ``method`` and ``strict`` mirror ``langchain_openai.ChatOpenAI``'s
        ``with_structured_output`` kwargs.  Leaving them ``None`` preserves
        each backend's defaults: platform now defaults to
        ``method="json_schema", strict=True`` (matching what a direct call
        with a Pydantic class gets); BYOK defers to ``langchain_openai``.

        Returns:
            ``{"structured": {...}, "llm_mode": str}``.

        Raises:
            QuotaExceededError: In platform mode when the org pool is
                exhausted.
            RuntimeError: In BYOK mode when no API key is configured.
        """
        if self.platform_available:
            result = await self._platform_ns.llm.structured(
                messages,
                output_schema,
                model=model,
                temperature=temperature,
                method=method,
                strict=strict,
            )
            return {**result, "llm_mode": "platform"}

        if self._byok is not None:
            result = await self._byok.structured(
                messages,
                output_schema,
                model=model,
                temperature=temperature,
                method=method,
                strict=strict,
            )
            return {**result, "llm_mode": "byok"}

        raise RuntimeError(
            "LlmFacade: no LLM backend configured. "
            "Either connect to the Novie platform or configure a BYOK key."
        )

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings.

        Routes to the platform when connected, to BYOK otherwise.

        Returns:
            List of embedding vectors.

        Raises:
            QuotaExceededError: In platform mode when the org pool is
                exhausted.
        """
        if self.platform_available:
            return await self._platform_ns.llm.embed(texts, model=model)

        if self._byok is not None:
            return await self._byok.embed(texts, model=model)

        raise RuntimeError(
            "LlmFacade: no LLM backend configured for embeddings."
        )

    async def budget_check(self) -> dict[str, Any]:
        """Check the current budget status (platform mode only).

        Returns an empty dict in BYOK / unavailable modes — standalone
        agents are not subject to platform quota enforcement.
        """
        if self.platform_available:
            return await self._platform_ns.llm.budget_check()
        return {}

    async def usage_summary(self, *, scope: str = "session") -> dict[str, Any]:
        """Fetch cumulative LLM usage (platform mode only).

        Returns an empty dict in BYOK / unavailable modes.
        """
        if self.platform_available:
            return await self._platform_ns.llm.usage_summary(scope=scope)
        return {}


def build_llm_facade(
    platform_ns: Any,
    *,
    agent_id: str = "agent",
) -> LlmFacade:
    """Build an ``LlmFacade`` paired with a ``ByokLlmClient`` from env.

    When the platform is available, BYOK is set up as a safety net in
    case platform calls fail.  When the platform is unavailable, BYOK is
    the primary backend.

    ``ByokLlmClient`` construction is best-effort: if the env key is
    missing in standalone mode the facade is created in ``"unavailable"``
    mode (``mode == "unavailable"``).  The agent gets a clear error only
    when it actually attempts an LLM call.
    """
    byok: Any = None
    if not getattr(platform_ns, "is_available", False):
        try:
            from .byok_llm import ByokLlmClient

            byok = ByokLlmClient.from_env(agent_id=agent_id)
        except (RuntimeError, ImportError):
            byok = None
    return LlmFacade(platform_ns, byok=byok)


def _byok_key_hint() -> str:
    from .byok_llm import BYOK_API_KEY_ENV

    return BYOK_API_KEY_ENV


__all__ = [
    "LlmFacade",
    "LlmMode",
    "build_llm_facade",
]
