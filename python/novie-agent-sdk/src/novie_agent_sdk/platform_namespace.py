"""EXPERT_AGENT_SDK W4 — platform-callback namespace for SDK agents.

Moves analyst-specific platform-callback logic into a reusable SDK
surface so external agents stop copying ``novie_analyst/_services.py``.

Surface (locked by ``test_platform_namespace.py``):

    ns = build_platform_namespace(
        incoming_headers,
        agent_id="my-agent",
        base_url="http://platform.local",
    )
    hits = await ns.knowledge.search("widgets", top_k=5)
    rec = await ns.checkpoints.put(
        owner_agent_id="my-agent",
        thread_id="t-1",
        payload={"phase": "synthesis"},
    )
    diagnostics = ns.last_diagnostics()  # tuple of every non-OK call

Failure modes are surfaced as ``CapabilityCallDiagnostics`` rather than
exceptions so handlers can degrade predictably (acceptance bullet
"Callback failures degrade predictably and can be reported in final
metadata"). Five symbolic ``kind`` values:

- ``binding_denied`` — HTTP 403 / envelope ``error_code=denied_by_binding``
- ``transport_error`` — couldn't reach the platform (timeout / connect)
- ``platform_unavailable`` — non-OK envelope without a binding-specific code
- ``schema_violation`` — response parses as JSON but doesn't match the
  capability's contract (missing ``status`` / ``result`` / wrong type)
- ``no_results`` — call succeeded but returned an empty list (surfaced
  symbolically so synthesis can footnote it)

W3's ``ArtifactAgentApp`` wires this in by default: ``ctx.platform`` is
a live ``PlatformNamespace`` when ``NOVIE_PLATFORM_BASE_URL`` is set
and incoming headers contain tenant/project; otherwise it's an
``_UnavailablePlatformNamespace`` that returns ``platform_unavailable``
diagnostics on every call.

The SDK never imports from ``novie_platform``; only from
``novie_protocol`` (already a dep) and ``httpx`` (already a dep).
"""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

from .platform_callback import (
    build_platform_callback_headers,
    sign_platform_callback_headers,
)
from .runtime import RequestHeaders


_log = logging.getLogger(__name__)

DegradationKind = Literal[
    "binding_denied",
    "transport_error",
    "platform_unavailable",
    "schema_violation",
    "no_results",
    "unconfigured",
]


_DEFAULT_TIMEOUT_SECONDS = 8.0
# LLM capability invocations go through the platform → provider (OpenRouter,
# Anthropic, OpenAI) round-trip; ``platform.llm.structured`` against a
# Pydantic schema with required fields routinely takes 10–30 s and can spike
# higher under provider contention.  Reusing the 8 s default for short
# capabilities (knowledge / checkpoints) here used to silently turn every
# slow LLM call into ``httpx.ReadTimeout`` → ``_unwrap`` → ``{}`` → upstream
# ``ProductBriefArtifact summary Field required`` validation error, which
# read on the platform side as ``RemoteProtocolError`` once the analyst's
# stream collapsed.  Treat LLM as its own latency tier so callers don't have
# to second-guess the SDK default.
_DEFAULT_LLM_TIMEOUT_SECONDS = 120.0

_KNOWLEDGE_SEARCH_CAP = "platform.knowledge.search"
_CHECKPOINT_PUT_CAP = "platform.external_agent_checkpoint.put"
_CHECKPOINT_GET_CAP = "platform.external_agent_checkpoint.get"
_CHECKPOINT_LIST_CAP = "platform.external_agent_checkpoint.list"
_LLM_CHAT_CAP = "platform.llm.chat"
_LLM_STRUCTURED_CAP = "platform.llm.structured"
_LLM_EMBED_CAP = "platform.llm.embed"
_LLM_BUDGET_CHECK_CAP = "platform.llm.budget_check"
_LLM_USAGE_SUMMARY_CAP = "platform.llm.usage_summary"


# ── Diagnostics ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CapabilityCallDiagnostics:
    """Outcome of one capability call.

    ``ok`` is True iff a usable result came back. ``result`` is the parsed
    capability ``result`` block on success, ``None`` otherwise. ``kind``
    is set on every non-OK call (and on ``no_results`` for empty
    successes); handlers branch on it rather than parsing ``detail``.
    """

    ok: bool
    capability_id: str = ""
    result: dict[str, Any] | None = None
    kind: DegradationKind | None = None
    error_code: str = ""
    detail: str = ""

    def to_metadata_entry(self) -> dict[str, Any]:
        """Render as a small dict the handler can stuff into
        ``ArtifactResult.metadata`` so consumers see degradation
        without needing the full diagnostics object."""
        return {
            "capability_id": self.capability_id,
            "ok": self.ok,
            "kind": self.kind,
            "error_code": self.error_code,
        }


def classify_envelope_error(
    error_code: str | None, http_status: int | None,
) -> DegradationKind:
    """Map a platform capability HTTP error to a ``DegradationKind``.

    Mirrors ``novie_analyst.degradation.classify_envelope_error`` so
    agents that already read those flags interpret SDK-emitted ones the
    same way. ``error_code`` from the platform envelope wins when set;
    ``http_status`` is the surfaced HTTP status code (None for envelope
    failures that arrived as 200 with ``status != ok``).
    """
    code = (error_code or "").strip().lower()
    if code == "denied_by_binding" or http_status == 403:
        return "binding_denied"
    return "platform_unavailable"


# ── Internal HTTP caller ────────────────────────────────────────────────────


class _CapabilityCaller:
    """SDK-owned async caller for ``POST /capabilities/{id}/invoke``.

    Returns ``CapabilityCallDiagnostics`` instead of raising so the
    namespace layer can degrade predictably. Uses ``httpx`` (already a
    SDK dep) — analyst's ``urllib`` shim was an analyst-image-size
    optimization that the SDK doesn't need.
    """

    def __init__(
        self,
        base_url: str,
        forward_headers: Mapping[str, str],
        *,
        agent_id: str,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = dict(forward_headers)
        self._agent_id = agent_id
        self._timeout = timeout_seconds
        self._client = client

    async def invoke_with_diagnostics(
        self,
        capability_id: str,
        arguments: Mapping[str, Any],
    ) -> CapabilityCallDiagnostics:
        path = f"/capabilities/{capability_id}/invoke"
        headers = sign_platform_callback_headers(
            self._headers, method="POST", path=path,
        )
        body: dict[str, Any] = {
            "arguments": dict(arguments),
            "caller_type": "agent",
            "caller_id": f"agent:{self._agent_id}",
            "caller_mode": "execute",
            "mode": "execute",
        }
        try:
            if self._client is not None:
                response = await self._client.post(
                    path, json=body, headers=headers,
                )
            else:
                async with httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=self._timeout,
                ) as client:
                    response = await client.post(
                        path, json=body, headers=headers,
                    )
        except httpx.TransportError as exc:
            _log.warning(
                "platform capability transport error capability=%s reason=%r",
                capability_id, exc,
            )
            return CapabilityCallDiagnostics(
                ok=False,
                capability_id=capability_id,
                kind="transport_error",
                detail=str(exc),
            )

        if response.status_code >= 400:
            envelope_code = _extract_envelope_code(response)
            kind = classify_envelope_error(envelope_code, response.status_code)
            _log.warning(
                "platform capability call failed capability=%s status=%s code=%s kind=%s",
                capability_id, response.status_code, envelope_code, kind,
            )
            return CapabilityCallDiagnostics(
                ok=False,
                capability_id=capability_id,
                kind=kind,
                error_code=envelope_code or "",
                detail=_safe_text(response),
            )

        try:
            envelope = response.json()
        except (ValueError, httpx.DecodingError):
            _log.warning(
                "platform capability returned non-JSON capability=%s", capability_id,
            )
            return CapabilityCallDiagnostics(
                ok=False,
                capability_id=capability_id,
                kind="schema_violation",
                error_code="non_json_response",
            )
        if not isinstance(envelope, dict):
            return CapabilityCallDiagnostics(
                ok=False,
                capability_id=capability_id,
                kind="schema_violation",
                error_code="non_object_envelope",
            )
        if str(envelope.get("status")) != "ok":
            envelope_code = str(envelope.get("error_code") or "") or None
            kind = classify_envelope_error(envelope_code, http_status=None)
            return CapabilityCallDiagnostics(
                ok=False,
                capability_id=capability_id,
                kind=kind,
                error_code=envelope_code or "",
                detail=str(envelope.get("explanation") or ""),
            )
        result = envelope.get("result")
        if result is not None and not isinstance(result, dict):
            return CapabilityCallDiagnostics(
                ok=False,
                capability_id=capability_id,
                kind="schema_violation",
                error_code="non_object_result",
            )
        return CapabilityCallDiagnostics(
            ok=True,
            capability_id=capability_id,
            result=result if isinstance(result, dict) else None,
        )


def _extract_envelope_code(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except (ValueError, httpx.DecodingError):
        return None
    if not isinstance(body, dict):
        return None
    direct = body.get("error_code")
    if isinstance(direct, str) and direct:
        return direct
    detail = body.get("detail")
    if isinstance(detail, dict):
        nested = detail.get("error_code")
        if isinstance(nested, str) and nested:
            return nested
    return None


def _safe_text(response: httpx.Response) -> str:
    try:
        return response.text[:500]
    except Exception:  # noqa: BLE001 — defensive
        return ""


# ── Namespaces ──────────────────────────────────────────────────────────────


class QuotaExceededError(RuntimeError):
    """Raised when the organisation's token pool is exhausted.

    External agents should catch this at the task boundary and surface
    a user-visible "organisation token pool exhausted" message rather
    than retrying — retrying won't help until the pool is refilled.

    Attributes:
        org_id: Organisation whose pool was exhausted.
        remaining_tokens: Remaining tokens at the time of the check
            (will be 0 or very close to 0).
        reason: Human-readable explanation from the platform.
    """

    def __init__(
        self,
        *,
        org_id: str = "",
        remaining_tokens: int = 0,
        reason: str = "org token pool exhausted",
    ) -> None:
        super().__init__(reason)
        self.org_id = org_id
        self.remaining_tokens = remaining_tokens
        self.reason = reason


class PlatformLlmCallError(RuntimeError):
    """Raised when a ``platform.llm.*`` capability call cannot return a
    usable result (transport error, platform 5xx, schema violation,
    binding denied, platform unavailable).

    Distinct from ``QuotaExceededError`` (which represents a *known*
    business state that the agent should surface) — ``PlatformLlmCallError``
    represents an *operational* failure where pretending the call returned
    ``{}`` would silently corrupt downstream Pydantic / JSON-schema
    validation.

    Attributes:
        capability_id: Platform capability that failed (e.g.
            ``"platform.llm.structured"``).
        kind: One of the ``DegradationKind`` values, mirroring the SDK's
            ``CapabilityCallDiagnostics.kind``.  Use this to decide whether
            a retry is meaningful (``transport_error`` may be transient,
            ``binding_denied`` is not).
        error_code: Platform envelope ``error_code`` when the failure
            arrived as a non-OK envelope.  Empty string for transport-layer
            failures.
        detail: Human-readable explanation suitable for logs / metadata.
    """

    def __init__(
        self,
        *,
        capability_id: str,
        kind: DegradationKind | None,
        error_code: str = "",
        detail: str = "",
    ) -> None:
        message = (
            f"platform LLM capability {capability_id!r} failed: "
            f"kind={kind} error_code={error_code or '<none>'} detail={detail or '<none>'}"
        )
        super().__init__(message)
        self.capability_id = capability_id
        self.kind = kind
        self.error_code = error_code
        self.detail = detail

    @property
    def is_transient(self) -> bool:
        """True when the failure category is plausibly retryable.

        ``transport_error`` (httpx connect/read timeout, dropped TCP) and
        ``platform_unavailable`` (5xx envelope without a binding code) are
        treated as transient.  ``binding_denied`` and ``schema_violation``
        are not — retrying without changing the request will yield the
        same outcome.
        """
        return self.kind in {"transport_error", "platform_unavailable"}


class PlatformLlmTransportError(PlatformLlmCallError):
    """``PlatformLlmCallError`` subclass for transport-layer failures.

    Kept as a distinct class so handler code can still
    ``except PlatformLlmTransportError`` for the most common case
    (httpx ``ReadTimeout`` / ``ConnectError``) without having to inspect
    ``.kind``.
    """

    def __init__(self, *, capability_id: str, detail: str = "") -> None:
        super().__init__(
            capability_id=capability_id,
            kind="transport_error",
            error_code="",
            detail=detail,
        )


class PlatformNamespaceProtocol(Protocol):
    """The shape of ``ctx.platform`` exposed to handlers."""

    knowledge: "KnowledgeNamespace"
    checkpoints: "CheckpointsNamespace"
    llm: "LlmNamespace"

    async def invoke_capability(
        self, capability_id: str, arguments: Mapping[str, Any],
    ) -> CapabilityCallDiagnostics: ...

    async def invoke_llm_capability(
        self, capability_id: str, arguments: Mapping[str, Any],
    ) -> CapabilityCallDiagnostics: ...

    def last_diagnostics(self) -> tuple[CapabilityCallDiagnostics, ...]: ...


class KnowledgeNamespace:
    """``platform.knowledge.search`` — replaces the analyst's
    ``HttpWikiService.search`` for any artifact agent."""

    def __init__(self, parent: "PlatformNamespace") -> None:
        self._parent = parent

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run ``platform.knowledge.search``.

        Returns the list of hits on success, ``[]`` on degradation. The
        triggering ``CapabilityCallDiagnostics`` is recorded on the
        parent namespace so the handler can attach it to final
        metadata.
        """
        payload: dict[str, Any] = {"query": query, "top_k": int(top_k)}
        scope = (project_id or "").strip() or self._parent.default_project_id
        if scope:
            payload["project_id"] = str(scope)
        diagnostics = await self._parent.invoke_capability(
            _KNOWLEDGE_SEARCH_CAP, payload,
        )
        if not diagnostics.ok:
            return []
        result = diagnostics.result or {}
        results_raw = result.get("results")
        if not isinstance(results_raw, list):
            self._parent._record_diagnostics(  # noqa: SLF001
                CapabilityCallDiagnostics(
                    ok=False,
                    capability_id=_KNOWLEDGE_SEARCH_CAP,
                    kind="schema_violation",
                    error_code="missing_results_list",
                )
            )
            return []
        out = [item for item in results_raw if isinstance(item, dict)]
        if not out:
            self._parent._record_diagnostics(  # noqa: SLF001
                CapabilityCallDiagnostics(
                    ok=True,
                    capability_id=_KNOWLEDGE_SEARCH_CAP,
                    kind="no_results",
                )
            )
        return out


class CheckpointsNamespace:
    """``platform.checkpoints.put/get/list`` — wraps
    ``platform.external_agent_checkpoint.*`` capabilities so worker
    agents can persist phase boundaries without a hand-rolled HTTP
    client.

    Records are returned as plain dicts (not the analyst's
    ``ExternalAgentCheckpointRecord``) so the SDK stays decoupled from
    domain dataclasses; agents that need typed records can map the
    dict themselves.
    """

    def __init__(self, parent: "PlatformNamespace") -> None:
        self._parent = parent

    async def put(
        self,
        *,
        owner_agent_id: str,
        thread_id: str,
        payload: Mapping[str, Any],
        checkpoint_id: str | None = None,
        session_id: str | None = None,
        workflow_id: str | None = None,
        step_id: str | None = None,
        checkpoint_format: str = "langgraph",
        checkpoint_version: str = "1",
        summary: str | None = None,
        parent_checkpoint_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        args: dict[str, Any] = {
            "owner_agent_id": owner_agent_id,
            "thread_id": thread_id,
            "payload": dict(payload),
            "checkpoint_format": checkpoint_format,
            "checkpoint_version": checkpoint_version,
            "metadata": dict(metadata) if metadata else {},
        }
        for key, value in (
            ("checkpoint_id", checkpoint_id),
            ("session_id", session_id),
            ("workflow_id", workflow_id),
            ("step_id", step_id),
            ("summary", summary),
            ("parent_checkpoint_id", parent_checkpoint_id),
        ):
            if value:
                args[key] = value
        diagnostics = await self._parent.invoke_capability(
            _CHECKPOINT_PUT_CAP, args,
        )
        if not diagnostics.ok:
            return None
        result = diagnostics.result or {}
        block = result.get("checkpoint")
        return dict(block) if isinstance(block, dict) else None

    async def get(
        self,
        *,
        owner_agent_id: str,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> dict[str, Any] | None:
        args: dict[str, Any] = {
            "owner_agent_id": owner_agent_id,
            "thread_id": thread_id,
        }
        if checkpoint_id:
            args["checkpoint_id"] = checkpoint_id
        diagnostics = await self._parent.invoke_capability(
            _CHECKPOINT_GET_CAP, args,
        )
        if not diagnostics.ok:
            return None
        result = diagnostics.result or {}
        block = result.get("checkpoint")
        return dict(block) if isinstance(block, dict) else None

    async def list(
        self,
        *,
        owner_agent_id: str,
        thread_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        args = {
            "owner_agent_id": owner_agent_id,
            "thread_id": thread_id,
            "limit": int(limit),
        }
        diagnostics = await self._parent.invoke_capability(
            _CHECKPOINT_LIST_CAP, args,
        )
        if not diagnostics.ok:
            return []
        result = diagnostics.result or {}
        items = result.get("checkpoints") or result.get("items") or []
        if not isinstance(items, list):
            self._parent._record_diagnostics(  # noqa: SLF001
                CapabilityCallDiagnostics(
                    ok=False,
                    capability_id=_CHECKPOINT_LIST_CAP,
                    kind="schema_violation",
                    error_code="missing_checkpoints_list",
                )
            )
            return []
        return [dict(entry) for entry in items if isinstance(entry, dict)]


class LlmNamespace:
    """``platform.llm.*`` — platform-managed LLM calls for connected agents.

    When an external agent is connected to the Novie platform it should
    use these methods instead of calling an LLM provider directly.
    Benefits:
    - No provider key required in the agent environment.
    - Usage is metered against the org token pool.
    - Hard stop when the pool is exhausted (``QuotaExceededError``).
    - Full audit trail and cost reporting in the Novie UI.

    All methods raise ``QuotaExceededError`` when the platform returns
    ``error_code="quota_exceeded"`` so the agent can surface a clear
    message and stop processing, rather than receiving an opaque error.

    All methods raise ``PlatformLlmTransportError`` /
    ``PlatformLlmCallError`` (instead of returning ``{}``) when the
    platform call cannot complete; this prevents transport timeouts from
    being silently fed into Pydantic / JSON-schema validation downstream.
    """

    def __init__(self, parent: "PlatformNamespace") -> None:  # noqa: F821
        self._parent = parent

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Invoke the platform chat LLM.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            model: Optional model override (e.g. ``"anthropic/claude-opus-4.6"``).
                   Defaults to the platform-configured model.
            temperature: Sampling temperature override.

        Returns:
            ``{"content": str, "usage_metadata": {...}}`` on success.

        Raises:
            QuotaExceededError: Org token pool is exhausted.
        """
        args: dict[str, Any] = {"messages": messages}
        if model:
            args["model"] = model
        if temperature is not None:
            args["temperature"] = temperature
        diagnostics = await self._parent.invoke_llm_capability(_LLM_CHAT_CAP, args)
        return self._unwrap(diagnostics, _LLM_CHAT_CAP)

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
        """Invoke the platform chat LLM with a JSON-schema structured output.

        Args:
            messages: Conversation messages.
            output_schema: JSON Schema dict describing the expected output.
            model: Optional model override.
            temperature: Sampling temperature override.
            method: Structured-output method (``json_schema``, ``function_calling``,
                or ``json_mode``).  ``None`` lets the platform pick the default
                (``json_schema``).
            strict: Force strict-mode structured output.  ``None`` lets the
                platform default kick in (currently ``True``); explicit ``False``
                disables strict for callers whose schema isn't strict-compatible.

        Returns:
            ``{"structured": {...}}`` on success.

        Raises:
            QuotaExceededError: Org token pool is exhausted.
        """
        args: dict[str, Any] = {
            "messages": messages,
            "output_schema": output_schema,
        }
        if model:
            args["model"] = model
        if temperature is not None:
            args["temperature"] = temperature
        if method is not None:
            args["method"] = method
        if strict is not None:
            args["strict"] = strict
        diagnostics = await self._parent.invoke_llm_capability(_LLM_STRUCTURED_CAP, args)
        return self._unwrap(diagnostics, _LLM_STRUCTURED_CAP)

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings with the platform embedding model.

        Returns:
            List of embedding vectors (one per input text).  Returns
            ``[]`` on non-quota errors (degraded mode).

        Raises:
            QuotaExceededError: Org token pool is exhausted.
        """
        args: dict[str, Any] = {"texts": texts}
        if model:
            args["model"] = model
        diagnostics = await self._parent.invoke_llm_capability(_LLM_EMBED_CAP, args)
        result = self._unwrap(diagnostics, _LLM_EMBED_CAP)
        embeddings = result.get("embeddings") or []
        return [list(v) for v in embeddings if isinstance(v, (list, tuple))]

    async def budget_check(self) -> dict[str, Any]:
        """Check the current org token pool status without consuming tokens.

        Returns a dict with at minimum ``{"remaining_tokens": int,
        "total_tokens": int, "exhausted": bool}``.  Returns an empty
        dict on error (non-raising).
        """
        diagnostics = await self._parent.invoke_llm_capability(_LLM_BUDGET_CHECK_CAP, {})
        if not diagnostics.ok:
            return {}
        return dict(diagnostics.result or {})

    async def usage_summary(self, *, scope: str = "session") -> dict[str, Any]:
        """Fetch the cumulative LLM usage summary for the given scope.

        Args:
            scope: One of ``"session"``, ``"org"``, ``"project"``, ``"user"``.

        Returns:
            Summary dict on success, ``{}`` on error (non-raising).
        """
        diagnostics = await self._parent.invoke_llm_capability(
            _LLM_USAGE_SUMMARY_CAP, {"scope": scope},
        )
        if not diagnostics.ok:
            return {}
        return dict(diagnostics.result or {})

    def _unwrap(
        self, diagnostics: CapabilityCallDiagnostics, capability_id: str
    ) -> dict[str, Any]:
        if not diagnostics.ok:
            if diagnostics.error_code == "quota_exceeded":
                result = diagnostics.result or {}
                raise QuotaExceededError(
                    org_id=result.get("org_id", ""),
                    remaining_tokens=int(result.get("remaining_tokens", 0)),
                    reason=result.get("reason", "org token pool exhausted"),
                )
            # Pre-0.3.3 SDKs returned ``{}`` here, which let the analyst
            # finalize chain interpret a transport failure as a successful
            # but empty LLM response (``ProductBriefArtifact summary Field
            # required``).  Surface the failure instead so callers can
            # retry, fall back, or fail loud — never silently feed an empty
            # dict into a Pydantic schema.
            if diagnostics.kind == "transport_error":
                raise PlatformLlmTransportError(
                    capability_id=capability_id,
                    detail=diagnostics.detail,
                )
            raise PlatformLlmCallError(
                capability_id=capability_id,
                kind=diagnostics.kind,
                error_code=diagnostics.error_code,
                detail=diagnostics.detail,
            )
        result = diagnostics.result or {}
        if result.get("error") == "quota_exceeded":
            raise QuotaExceededError(
                org_id=result.get("org_id", ""),
                remaining_tokens=int(result.get("remaining_tokens", 0)),
                reason=result.get("reason", "org token pool exhausted"),
            )
        return result


# ── Top-level namespace ─────────────────────────────────────────────────────


class PlatformNamespace:
    """Live ``ctx.platform`` for an SDK agent.

    Owns two ``_CapabilityCaller`` instances — a "default" one (short
    timeout, used by knowledge / checkpoints / arbitrary capabilities)
    and an "llm" one (long timeout, used by ``LlmNamespace``).  Splitting
    them avoids the historical pitfall where a 8 s default turned every
    real LLM round-trip into ``httpx.ReadTimeout`` and made the SDK return
    ``{}`` to callers; LLM gets its own latency tier so callers don't
    have to second-guess.

    Records every non-OK call on ``_diagnostics`` so handlers can surface
    them via ``last_diagnostics()`` (acceptance bullet "Callback failures
    degrade predictably and can be reported in final metadata").
    """

    knowledge: KnowledgeNamespace
    checkpoints: CheckpointsNamespace
    llm: LlmNamespace

    def __init__(
        self,
        caller: _CapabilityCaller,
        *,
        default_project_id: str = "",
        llm_caller: _CapabilityCaller | None = None,
    ) -> None:
        self._caller = caller
        # Falls back to the default caller for backward compatibility with
        # callers that build a ``PlatformNamespace`` directly (mostly tests).
        # ``build_platform_namespace`` always provides a dedicated long-
        # timeout caller in production.
        self._llm_caller = llm_caller or caller
        self.default_project_id = default_project_id
        self._diagnostics: list[CapabilityCallDiagnostics] = []
        self.knowledge = KnowledgeNamespace(self)
        self.checkpoints = CheckpointsNamespace(self)
        self.llm = LlmNamespace(self)

    @property
    def is_available(self) -> bool:
        return True

    async def invoke_capability(
        self,
        capability_id: str,
        arguments: Mapping[str, Any],
    ) -> CapabilityCallDiagnostics:
        diagnostics = await self._caller.invoke_with_diagnostics(
            capability_id, arguments,
        )
        if not diagnostics.ok:
            self._diagnostics.append(diagnostics)
        return diagnostics

    async def invoke_llm_capability(
        self,
        capability_id: str,
        arguments: Mapping[str, Any],
    ) -> CapabilityCallDiagnostics:
        """Invoke a ``platform.llm.*`` capability with the long-timeout caller.

        Used by ``LlmNamespace`` so ``platform.llm.structured`` (which can
        legitimately take 30+ seconds against Anthropic / OpenAI) doesn't
        share the 8 s default that's appropriate for short capabilities
        like knowledge / checkpoints.
        """
        diagnostics = await self._llm_caller.invoke_with_diagnostics(
            capability_id, arguments,
        )
        if not diagnostics.ok:
            self._diagnostics.append(diagnostics)
        return diagnostics

    def last_diagnostics(self) -> tuple[CapabilityCallDiagnostics, ...]:
        return tuple(self._diagnostics)

    def _record_diagnostics(self, diagnostics: CapabilityCallDiagnostics) -> None:
        self._diagnostics.append(diagnostics)


class _UnavailablePlatformNamespace:
    """Stand-in when ``NOVIE_PLATFORM_BASE_URL`` is unset or required
    headers are missing.

    Returns ``platform_unavailable`` diagnostics on every call and
    empty/None results from sub-namespaces, so a handler that calls
    ``ctx.platform.knowledge.search(...)`` in a dev environment without
    a configured platform still progresses (just without callback
    enrichment). Acceptance bullet: "Tests cover missing platform URL".
    """

    knowledge: KnowledgeNamespace
    checkpoints: CheckpointsNamespace
    llm: LlmNamespace

    def __init__(self, *, reason: str) -> None:
        self._reason = reason
        self.default_project_id = ""
        self._diagnostics: list[CapabilityCallDiagnostics] = []
        self.knowledge = KnowledgeNamespace(self)  # type: ignore[arg-type]
        self.checkpoints = CheckpointsNamespace(self)  # type: ignore[arg-type]
        self.llm = LlmNamespace(self)  # type: ignore[arg-type]

    @property
    def is_available(self) -> bool:
        return False

    async def invoke_capability(
        self,
        capability_id: str,
        arguments: Mapping[str, Any],
    ) -> CapabilityCallDiagnostics:
        diagnostics = CapabilityCallDiagnostics(
            ok=False,
            capability_id=capability_id,
            kind="unconfigured",
            error_code="platform_unavailable",
            detail=self._reason,
        )
        self._diagnostics.append(diagnostics)
        return diagnostics

    async def invoke_llm_capability(
        self,
        capability_id: str,
        arguments: Mapping[str, Any],
    ) -> CapabilityCallDiagnostics:
        return await self.invoke_capability(capability_id, arguments)

    def last_diagnostics(self) -> tuple[CapabilityCallDiagnostics, ...]:
        return tuple(self._diagnostics)

    def _record_diagnostics(self, diagnostics: CapabilityCallDiagnostics) -> None:
        self._diagnostics.append(diagnostics)


# ── Factory ─────────────────────────────────────────────────────────────────


def build_platform_namespace(
    incoming_headers: RequestHeaders | Mapping[str, str],
    *,
    agent_id: str,
    base_url: str | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    llm_timeout_seconds: float = _DEFAULT_LLM_TIMEOUT_SECONDS,
    client: httpx.AsyncClient | None = None,
) -> PlatformNamespace | _UnavailablePlatformNamespace:
    """Construct a per-request ``ctx.platform`` from incoming A2A
    headers.

    ``base_url`` falls back to ``NOVIE_PLATFORM_BASE_URL`` if not given.
    Returns ``_UnavailablePlatformNamespace`` (instead of raising) when
    base URL is missing or required tenant/project headers are absent;
    handlers can read ``platform.is_available`` to branch.

    ``timeout_seconds`` controls the short-capability caller (knowledge,
    checkpoints, etc.); ``llm_timeout_seconds`` controls the dedicated
    LLM caller.  The LLM tier defaults to 120 s because
    ``platform.llm.structured`` against a Pydantic schema with required
    fields routinely takes 10–30 s and can spike higher under provider
    contention; sharing the 8 s default would silently turn slow LLM
    calls into ``httpx.ReadTimeout`` → ``_unwrap`` → ``{}`` → upstream
    Pydantic validation errors.

    ``client`` lets tests inject an ``httpx.AsyncClient`` (e.g. one
    bound to an ASGI transport) without booting a real HTTP server.
    The same client is reused for both callers when supplied.
    """
    base = (base_url or os.getenv("NOVIE_PLATFORM_BASE_URL", "") or "").strip()
    if not base:
        return _UnavailablePlatformNamespace(
            reason="NOVIE_PLATFORM_BASE_URL is not set",
        )
    forward_headers = build_platform_callback_headers(
        incoming_headers, agent_id=agent_id,
    )
    if not forward_headers["x-novie-org-id"] or not forward_headers["x-novie-project-id"]:
        return _UnavailablePlatformNamespace(
            reason=(
                "incoming A2A headers missing tenant/project — "
                "platform callbacks would 400"
            ),
        )
    caller = _CapabilityCaller(
        base,
        forward_headers,
        agent_id=agent_id,
        timeout_seconds=timeout_seconds,
        client=client,
    )
    llm_caller = _CapabilityCaller(
        base,
        forward_headers,
        agent_id=agent_id,
        timeout_seconds=llm_timeout_seconds,
        client=client,
    )
    return PlatformNamespace(
        caller,
        default_project_id=forward_headers["x-novie-project-id"],
        llm_caller=llm_caller,
    )


__all__ = [
    "CapabilityCallDiagnostics",
    "CheckpointsNamespace",
    "DegradationKind",
    "KnowledgeNamespace",
    "LlmNamespace",
    "PlatformLlmCallError",
    "PlatformLlmTransportError",
    "PlatformNamespace",
    "PlatformNamespaceProtocol",
    "QuotaExceededError",
    "build_platform_namespace",
    "classify_envelope_error",
]
