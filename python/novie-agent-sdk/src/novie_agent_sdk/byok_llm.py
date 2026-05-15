"""BYOK (Bring Your Own Key) LLM wrapper for standalone agent mode.

When an external agent runs **outside** the Novie platform (standalone mode),
it configures its own LLM provider key.  This module provides a thin wrapper
that:

1. Builds a LangChain ``BaseChatModel`` from standard environment variables
   or explicit arguments.
2. Optionally attaches a ``NovieLangChainCallbackHandler`` so that usage is
   reported back to the platform when the agent does reconnect later, or
   forwarded to any ``ObservabilitySink`` (e.g. Langfuse).
3. Exposes a ``chat / structured / embed`` interface that mirrors
   ``ctx.platform.llm.*`` so agent code can switch between standalone and
   platform-connected modes without changing call sites.

This module deliberately has **no** platform token pool enforcement — quota
control only applies when the agent is connected to the platform and uses
``ctx.platform.llm.*``.  Standalone mode is uncontrolled by design.

Requires ``langchain_openai`` (``pip install langchain-openai``).
"""
from __future__ import annotations

import logging
import os
from typing import Any

_log = logging.getLogger(__name__)

# ── Public env-var names (agent configures their own key) ─────────────────
BYOK_API_KEY_ENV = "NOVIE_AGENT_LLM_API_KEY"       # agent's own key
BYOK_BASE_URL_ENV = "NOVIE_AGENT_LLM_BASE_URL"      # e.g. OpenRouter endpoint
BYOK_MODEL_ENV = "NOVIE_AGENT_LLM_MODEL"            # e.g. "anthropic/claude-opus-4.6"
BYOK_EMBED_MODEL_ENV = "NOVIE_AGENT_LLM_EMBED_MODEL"


def _require_langchain_openai() -> None:
    try:
        import langchain_openai  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "langchain-openai is required for ByokLlmClient. "
            "Install it with: pip install langchain-openai"
        ) from exc


class ByokLlmClient:
    """Standalone LLM client for agents running outside the Novie platform.

    Usage::

        client = ByokLlmClient.from_env(agent_id="my-agent")
        response = await client.chat([{"role": "user", "content": "Hello"}])
        print(response["content"])

        # Or with structured output:
        result = await client.structured(
            [{"role": "user", "content": "Extract name"}],
            output_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )

    When ``NOVIE_PLATFORM_BASE_URL`` is set, automatically reports usage to
    the platform via ``ctx.platform`` so operators can see standalone agent
    costs in the UI even when the agent manages its own key.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str = "gpt-4o",
        embed_model: str = "text-embedding-3-small",
        agent_id: str = "standalone-agent",
        usage_sink: Any | None = None,
    ) -> None:
        _require_langchain_openai()
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._embed_model = embed_model
        self._agent_id = agent_id
        self._usage_sink = usage_sink

    @classmethod
    def from_env(
        cls,
        *,
        agent_id: str = "standalone-agent",
        usage_sink: Any | None = None,
    ) -> "ByokLlmClient":
        """Build from standard SDK env vars.

        Raises ``RuntimeError`` when ``NOVIE_AGENT_LLM_API_KEY`` is not set.
        """
        api_key = os.getenv(BYOK_API_KEY_ENV, "").strip()
        if not api_key:
            # Fall back to common keys so agent devs can run against vanilla
            # OpenAI without setting the SDK-specific env var.
            api_key = (
                os.getenv("OPENAI_API_KEY", "")
                or os.getenv("OPENROUTER_API_KEY", "")
            ).strip()
        if not api_key:
            raise RuntimeError(
                f"No LLM API key found. Set {BYOK_API_KEY_ENV} (or OPENAI_API_KEY) "
                "in your environment to enable standalone LLM calls."
            )
        return cls(
            api_key=api_key,
            base_url=os.getenv(BYOK_BASE_URL_ENV) or None,
            model=os.getenv(BYOK_MODEL_ENV, "gpt-4o"),
            embed_model=os.getenv(BYOK_EMBED_MODEL_ENV, "text-embedding-3-small"),
            agent_id=agent_id,
            usage_sink=usage_sink,
        )

    @property
    def model(self) -> str:
        return self._model

    # ── LLM call helpers ─────────────────────────────────────────────────

    def _build_chat_model(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> Any:
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "api_key": self._api_key,
            "stream_usage": True,  # enable token usage in streaming
        }
        if self._base_url:
            kwargs["base_url"] = self._base_url
        if temperature is not None:
            kwargs["temperature"] = temperature

        llm = ChatOpenAI(**kwargs)
        if self._usage_sink is not None:
            from .observability import NovieLangChainCallbackHandler

            cb = NovieLangChainCallbackHandler(
                agent_id=self._agent_id,
                usage_sink=self._usage_sink,
            )
            llm.callbacks = [cb]
        return llm

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Send a chat request using the agent's own key.

        Returns ``{"content": str, "usage_metadata": {...}}``.

        No quota enforcement is applied — this is standalone/BYOK mode.
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        def _to_msg(m: dict[str, str]) -> Any:
            role = str(m.get("role") or "user")
            content = str(m.get("content") or "")
            if role == "system":
                return SystemMessage(content=content)
            if role == "assistant":
                return AIMessage(content=content)
            return HumanMessage(content=content)

        llm = self._build_chat_model(model=model, temperature=temperature)
        response = await llm.ainvoke([_to_msg(m) for m in messages])
        content = getattr(response, "content", str(response))
        usage_metadata = dict(getattr(response, "usage_metadata", None) or {})
        return {"content": content, "usage_metadata": usage_metadata}

    async def structured(
        self,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Structured JSON-schema output using the agent's own key.

        Returns ``{"structured": {...}}``.
        """
        import json as _json

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        def _to_msg(m: dict[str, str]) -> Any:
            role = str(m.get("role") or "user")
            content = str(m.get("content") or "")
            if role == "system":
                return SystemMessage(content=content)
            if role == "assistant":
                return AIMessage(content=content)
            return HumanMessage(content=content)

        llm = self._build_chat_model(model=model, temperature=temperature)
        structured_llm = llm.with_structured_output(output_schema)
        response = await structured_llm.ainvoke([_to_msg(m) for m in messages])
        if hasattr(response, "model_dump"):
            value = response.model_dump()
        elif isinstance(response, dict):
            value = dict(response)
        else:
            value = _json.loads(str(response))
        return {"structured": value}

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings using the agent's own key.

        Returns a list of embedding vectors (one per input text).
        """
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict[str, Any] = {
            "model": model or self._embed_model,
            "api_key": self._api_key,
        }
        if self._base_url:
            kwargs["base_url"] = self._base_url
        embeddings = OpenAIEmbeddings(**kwargs)
        vectors = await embeddings.aembed_documents(texts)
        return [list(v) for v in vectors]


__all__ = [
    "BYOK_API_KEY_ENV",
    "BYOK_BASE_URL_ENV",
    "BYOK_MODEL_ENV",
    "BYOK_EMBED_MODEL_ENV",
    "ByokLlmClient",
]
