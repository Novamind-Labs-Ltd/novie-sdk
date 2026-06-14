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

import json
import logging
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx

from .artifact_text import (
    ArtifactReadCache,
    format_artifact_read_result,
    normalize_artifact_id,
)
from .platform_callback import (
    build_platform_callback_headers,
    sign_platform_callback_headers,
)
from .llm_contract import normalise_llm_result, normalise_stream_event
from .runtime import RequestHeaders


_log = logging.getLogger(__name__)

DegradationKind = Literal[
    "binding_denied",
    "transport_error",
    "timeout",
    "provider_error",
    "schema_error",
    "quota_error",
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
_DEFAULT_LLM_HEARTBEAT_TIMEOUT_SECONDS = 60.0

_KNOWLEDGE_SEARCH_CAP = "platform.knowledge.search"
_WEB_SEARCH_CAP = "platform.web.search"
_ARTIFACT_CREATE_CAP = "platform.artifacts.create"
_ARTIFACT_READ_CAP = "platform.artifacts.read"
_ARTIFACT_SEARCH_CAP = "platform.artifacts.search"
_WORKPAD_SNAPSHOT_CAP = "platform.workpads.snapshot"
_WORKPAD_RECORD_ENTRY_CAP = "platform.workpads.record_entry"
_WORKPAD_SET_FINAL_DELIVERABLE_CAP = "platform.workpads.set_final_deliverable"
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
    error_envelope: dict[str, Any] | None = None
    retryable: bool | None = None
    timeout_seconds: float | None = None
    provider_status_code: int | None = None
    raw_detail: Any | None = None

    def to_metadata_entry(self) -> dict[str, Any]:
        """Render as a small dict the handler can stuff into
        ``ArtifactResult.metadata`` so consumers see degradation
        without needing the full diagnostics object."""
        entry = {
            "capability_id": self.capability_id,
            "ok": self.ok,
            "kind": self.kind,
            "error_code": self.error_code,
        }
        if self.error_envelope:
            entry["error_envelope"] = dict(self.error_envelope)
        if self.retryable is not None:
            entry["retryable"] = self.retryable
        if self.timeout_seconds is not None:
            entry["timeout_seconds"] = self.timeout_seconds
        if self.provider_status_code is not None:
            entry["provider_status_code"] = self.provider_status_code
        return entry


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
        *,
        timeout_seconds: float | None = None,
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
        call_timeout = _effective_timeout(timeout_seconds, self._timeout)
        try:
            if self._client is not None:
                response = await self._client.post(
                    path, json=body, headers=headers, timeout=call_timeout,
                )
            else:
                async with httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=call_timeout,
                ) as client:
                    response = await client.post(
                        path, json=body, headers=headers,
                    )
        except httpx.TimeoutException as exc:
            envelope = _timeout_error_envelope(
                capability_id,
                timeout_seconds=call_timeout,
                detail=str(exc),
            )
            _log.warning(
                "platform capability timeout capability=%s timeout_seconds=%s reason=%r",
                capability_id, call_timeout, exc,
            )
            return _diagnostics_from_error_envelope(
                capability_id=capability_id,
                envelope=envelope,
                default_kind="timeout",
                detail=str(exc),
            )
        except httpx.TransportError as exc:
            _log.warning(
                "platform capability transport error capability=%s reason=%r",
                capability_id, exc,
            )
            envelope = {
                "kind": "transport_error",
                "capability_id": capability_id,
                "retryable": True,
                "reason_code": "platform_llm_transport_error"
                if capability_id.startswith("platform.llm.")
                else "platform_transport_error",
                "raw_detail": str(exc),
            }
            return CapabilityCallDiagnostics(
                ok=False,
                capability_id=capability_id,
                kind="transport_error",
                error_code=str(envelope["reason_code"]),
                detail=str(exc),
                error_envelope=envelope,
                retryable=True,
                raw_detail=str(exc),
            )

        if response.status_code >= 400:
            error_envelope = _extract_error_envelope(response)
            envelope_code = _error_envelope_reason_code(error_envelope) or _extract_envelope_code(response)
            kind = _error_envelope_kind(error_envelope) or classify_envelope_error(envelope_code, response.status_code)
            _log.warning(
                "platform capability call failed capability=%s status=%s code=%s kind=%s",
                capability_id, response.status_code, envelope_code, kind,
            )
            return _diagnostics_from_error_envelope(
                capability_id=capability_id,
                envelope=error_envelope,
                default_kind=kind,
                default_error_code=envelope_code or "",
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
            error_envelope = _extract_error_envelope_from_mapping(envelope)
            envelope_code = _error_envelope_reason_code(error_envelope) or str(envelope.get("error_code") or "") or None
            kind = _error_envelope_kind(error_envelope) or classify_envelope_error(envelope_code, http_status=None)
            return _diagnostics_from_error_envelope(
                capability_id=capability_id,
                envelope=error_envelope,
                default_kind=kind,
                default_error_code=envelope_code or "",
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

    async def invoke_stream_with_diagnostics(
        self,
        capability_id: str,
        arguments: Mapping[str, Any],
        *,
        heartbeat_timeout_seconds: float = _DEFAULT_LLM_HEARTBEAT_TIMEOUT_SECONDS,
    ) -> CapabilityCallDiagnostics:
        path = f"/capabilities/{capability_id}/invoke-stream"
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

        async def consume(client: httpx.AsyncClient) -> CapabilityCallDiagnostics:
            async with client.stream("POST", path, json=body, headers=headers) as response:
                if response.status_code == 404:
                    await response.aread()
                    return CapabilityCallDiagnostics(
                        ok=False,
                        capability_id=capability_id,
                        kind="platform_unavailable",
                        error_code="stream_endpoint_not_found",
                        detail=_safe_text(response),
                    )
                if response.status_code >= 400:
                    await response.aread()
                    envelope_code = _extract_envelope_code(response)
                    return CapabilityCallDiagnostics(
                        ok=False,
                        capability_id=capability_id,
                        kind=classify_envelope_error(envelope_code, response.status_code),
                        error_code=envelope_code or "",
                        detail=_safe_text(response),
                    )

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        return CapabilityCallDiagnostics(
                            ok=False,
                            capability_id=capability_id,
                            kind="schema_violation",
                            error_code="non_json_stream_event",
                            detail=line[:200],
                        )
                    if not isinstance(event, dict):
                        return CapabilityCallDiagnostics(
                            ok=False,
                            capability_id=capability_id,
                            kind="schema_violation",
                            error_code="non_object_stream_event",
                        )

                    event_type = str(event.get("type") or "")
                    if not event_type and "status" in event:
                        if str(event.get("status")) == "ok":
                            result = event.get("result")
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
                        envelope_code = str(event.get("error_code") or "") or None
                        return CapabilityCallDiagnostics(
                            ok=False,
                            capability_id=capability_id,
                            kind=classify_envelope_error(envelope_code, http_status=None),
                            error_code=envelope_code or "",
                            detail=str(event.get("explanation") or ""),
                        )

                    if event_type in {"accepted", "heartbeat", "progress"}:
                        continue
                    if event_type == "chunk":
                        # ``llm.chat`` uses the streaming endpoint for long-running
                        # calls but still returns one final ChatResult. Intermediate
                        # token/tool deltas belong to ``llm.stream_chat`` callers;
                        # wait for the terminal ``completed`` event here.
                        continue
                    if event_type == "completed":
                        result = event.get("result")
                        if result is not None and not isinstance(result, dict):
                            return CapabilityCallDiagnostics(
                                ok=False,
                                capability_id=capability_id,
                                kind="schema_violation",
                                error_code="non_object_stream_result",
                            )
                        return CapabilityCallDiagnostics(
                            ok=True,
                            capability_id=capability_id,
                            result=result if isinstance(result, dict) else None,
                        )
                    if event_type in {"error", "cancelled"}:
                        envelope = event.get("envelope")
                        envelope_code = str(event.get("error_code") or "") or None
                        if isinstance(envelope, dict) and not envelope_code:
                            envelope_code = str(envelope.get("error_code") or "") or None
                        return CapabilityCallDiagnostics(
                            ok=False,
                            capability_id=capability_id,
                            kind=classify_envelope_error(envelope_code, http_status=None),
                            error_code=envelope_code or "",
                            detail=str(
                                event.get("explanation")
                                or event.get("reason")
                                or ""
                            ),
                        )
                    return CapabilityCallDiagnostics(
                        ok=False,
                        capability_id=capability_id,
                        kind="schema_violation",
                        error_code="unknown_stream_event",
                        detail=event_type,
                    )

                return CapabilityCallDiagnostics(
                    ok=False,
                    capability_id=capability_id,
                    kind="transport_error",
                    error_code="stream_closed_without_completion",
                )

        try:
            heartbeat_timeout = max(float(heartbeat_timeout_seconds), 1.0)
            if self._client is not None:
                return await consume(self._client)
            timeout = httpx.Timeout(
                timeout=float(self._timeout),
                read=heartbeat_timeout,
            )
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=timeout,
            ) as client:
                return await consume(client)
        except TimeoutError as exc:
            _log.warning(
                "platform capability stream timed out capability=%s reason=%r",
                capability_id,
                exc,
            )
            return CapabilityCallDiagnostics(
                ok=False,
                capability_id=capability_id,
                kind="transport_error",
                error_code="stream_heartbeat_timeout",
                detail=str(exc),
            )
        except httpx.TransportError as exc:
            _log.warning(
                "platform capability stream transport error capability=%s reason=%r",
                capability_id,
                exc,
            )
            return CapabilityCallDiagnostics(
                ok=False,
                capability_id=capability_id,
                kind="transport_error",
                detail=str(exc),
            )

    async def invoke_event_stream(
        self,
        capability_id: str,
        arguments: Mapping[str, Any],
        *,
        heartbeat_timeout_seconds: float = _DEFAULT_LLM_HEARTBEAT_TIMEOUT_SECONDS,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield raw NDJSON capability stream events.

        This is used by platform LLM chat streaming where intermediate
        ``chunk`` events are meaningful to LangChain's ``astream`` contract.
        """
        path = f"/capabilities/{capability_id}/invoke-stream"
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

        async def consume(client: httpx.AsyncClient) -> AsyncIterator[dict[str, Any]]:
            async with client.stream("POST", path, json=body, headers=headers) as response:
                if response.status_code >= 400:
                    await response.aread()
                    envelope_code = _extract_envelope_code(response)
                    if response.status_code == 404 and not envelope_code:
                        envelope_code = "stream_endpoint_not_found"
                    yield {
                        "type": "error",
                        "error_code": envelope_code or "",
                        "explanation": _safe_text(response),
                    }
                    return
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        yield {
                            "type": "error",
                            "error_code": "non_json_stream_event",
                            "explanation": line[:200],
                        }
                        return
                    if not isinstance(event, dict):
                        yield {
                            "type": "error",
                            "error_code": "non_object_stream_event",
                        }
                        return
                    yield event

        try:
            heartbeat_timeout = max(float(heartbeat_timeout_seconds), 1.0)
            stream_state: dict[Any, Any] = {}
            if self._client is not None:
                async for event in consume(self._client):
                    yield normalise_stream_event(event, stream_state)
                return
            timeout = httpx.Timeout(
                timeout=float(self._timeout),
                read=heartbeat_timeout,
            )
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=timeout,
            ) as client:
                async for event in consume(client):
                    yield normalise_stream_event(event, stream_state)
        except TimeoutError as exc:
            yield {
                "type": "error",
                "error_code": "stream_heartbeat_timeout",
                "explanation": str(exc),
            }
        except httpx.TransportError as exc:
            yield {
                "type": "error",
                "error_code": "transport_error",
                "explanation": str(exc),
            }


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


def _is_high_risk_inline_capability(capability_id: str) -> bool:
    value = capability_id.strip().lower()
    if not value:
        return False
    read_suffixes = (
        ".get",
        ".list",
        ".search",
        ".status",
        ".summary",
        ".usage_summary",
    )
    if any(value.endswith(suffix) for suffix in read_suffixes):
        return False
    write_markers = (
        ".create",
        ".update",
        ".delete",
        ".move",
        ".recover",
        ".retry",
        ".invoke",
        ".put",
        ".comment",
    )
    return any(marker in value for marker in write_markers)


def _safe_text(response: httpx.Response) -> str:
    try:
        return response.text[:500]
    except Exception:  # noqa: BLE001 — defensive
        return ""


def _effective_timeout(value: float | None, default: float) -> float:
    try:
        timeout = float(value) if value is not None else float(default)
    except (TypeError, ValueError):
        return float(default)
    return timeout if timeout > 0 else float(default)


def _structured_timeout_reason_code(capability_id: str) -> str:
    if capability_id == _LLM_STRUCTURED_CAP:
        return "platform_llm_structured_timeout"
    return f"{capability_id.replace('.', '_')}_timeout"


def _timeout_error_envelope(
    capability_id: str,
    *,
    timeout_seconds: float,
    detail: str,
    model: str | None = None,
    phase: str = "transport",
) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "kind": "timeout",
        "capability_id": capability_id,
        "phase": phase,
        "timeout_seconds": timeout_seconds,
        "retryable": True,
        "reason_code": _structured_timeout_reason_code(capability_id),
        "raw_detail": detail,
    }
    if model:
        envelope["model"] = model
    return envelope


def _extract_error_envelope(response: httpx.Response) -> dict[str, Any] | None:
    try:
        payload = response.json()
    except (ValueError, httpx.DecodingError):
        return None
    if isinstance(payload, dict):
        return _extract_error_envelope_from_mapping(payload)
    return None


def _extract_error_envelope_from_mapping(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        envelope = metadata.get("error_envelope")
        if isinstance(envelope, dict):
            return dict(envelope)
    detail = payload.get("detail")
    if isinstance(detail, dict):
        envelope = detail.get("error_envelope")
        if isinstance(envelope, dict):
            return dict(envelope)
        if "kind" in detail or "reason_code" in detail:
            return dict(detail)
    error = payload.get("error")
    if isinstance(error, dict):
        return dict(error)
    if "kind" in payload or "reason_code" in payload:
        return dict(payload)
    return None


def _error_envelope_kind(envelope: Mapping[str, Any] | None) -> DegradationKind | None:
    if not envelope:
        return None
    kind = str(envelope.get("kind") or "").strip()
    if kind in {
        "binding_denied",
        "transport_error",
        "timeout",
        "provider_error",
        "schema_error",
        "quota_error",
        "platform_unavailable",
        "schema_violation",
        "no_results",
        "unconfigured",
    }:
        return kind  # type: ignore[return-value]
    return None


def _error_envelope_reason_code(envelope: Mapping[str, Any] | None) -> str:
    if not envelope:
        return ""
    return str(envelope.get("reason_code") or envelope.get("error_code") or "").strip()


def _diagnostics_from_error_envelope(
    *,
    capability_id: str,
    envelope: Mapping[str, Any] | None,
    default_kind: DegradationKind | None,
    default_error_code: str = "",
    detail: str = "",
    provider_status_code: int | None = None,
) -> CapabilityCallDiagnostics:
    copied = dict(envelope or {})
    kind = _error_envelope_kind(copied) or default_kind
    error_code = _error_envelope_reason_code(copied) or default_error_code
    retryable = copied.get("retryable")
    timeout_seconds_raw = copied.get("timeout_seconds")
    timeout_seconds: float | None = None
    if timeout_seconds_raw is not None:
        try:
            timeout_seconds = float(timeout_seconds_raw)
        except (TypeError, ValueError):
            timeout_seconds = None
    provider_status = provider_status_code
    if provider_status is None and copied.get("provider_status_code") is not None:
        try:
            provider_status = int(copied["provider_status_code"])
        except (TypeError, ValueError):
            provider_status = None
    raw_detail = copied.get("raw_detail", detail)
    return CapabilityCallDiagnostics(
        ok=False,
        capability_id=capability_id,
        kind=kind,
        error_code=error_code,
        detail=detail or str(raw_detail or ""),
        error_envelope=copied or None,
        retryable=bool(retryable) if retryable is not None else None,
        timeout_seconds=timeout_seconds,
        provider_status_code=provider_status,
        raw_detail=raw_detail,
    )


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
        error_envelope: Mapping[str, Any] | None = None,
        retryable: bool | None = None,
        timeout_seconds: float | None = None,
        provider_status_code: int | None = None,
        raw_detail: Any | None = None,
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
        self.error_envelope = dict(error_envelope or {})
        self.reason_code = str(
            self.error_envelope.get("reason_code")
            or self.error_envelope.get("error_code")
            or error_code
            or ""
        )
        self.retryable = retryable
        self.timeout_seconds = timeout_seconds
        self.provider_status_code = provider_status_code
        self.raw_detail = raw_detail

    @property
    def is_transient(self) -> bool:
        """True when the failure category is plausibly retryable.

        ``transport_error`` (httpx connect/read timeout, dropped TCP) and
        ``platform_unavailable`` (5xx envelope without a binding code) are
        treated as transient.  ``binding_denied`` and ``schema_violation``
        are not — retrying without changing the request will yield the
        same outcome.
        """
        if self.retryable is not None:
            return bool(self.retryable)
        return self.kind in {"transport_error", "timeout", "platform_unavailable"}


class PlatformLlmTransportError(PlatformLlmCallError):
    """``PlatformLlmCallError`` subclass for transport-layer failures.

    Kept as a distinct class so handler code can still
    ``except PlatformLlmTransportError`` for the most common case
    (httpx ``ReadTimeout`` / ``ConnectError``) without having to inspect
    ``.kind``.
    """

    def __init__(
        self,
        *,
        capability_id: str,
        detail: str = "",
        error_code: str = "",
        error_envelope: Mapping[str, Any] | None = None,
        retryable: bool | None = True,
        timeout_seconds: float | None = None,
        provider_status_code: int | None = None,
        raw_detail: Any | None = None,
    ) -> None:
        super().__init__(
            capability_id=capability_id,
            kind="transport_error",
            error_code=error_code,
            detail=detail,
            error_envelope=error_envelope,
            retryable=retryable,
            timeout_seconds=timeout_seconds,
            provider_status_code=provider_status_code,
            raw_detail=raw_detail,
        )


class PlatformLlmTimeoutError(PlatformLlmTransportError):
    """``platform.llm.*`` call exceeded the per-call or platform timeout."""

    def __init__(
        self,
        *,
        capability_id: str,
        detail: str = "",
        error_code: str = "",
        error_envelope: Mapping[str, Any] | None = None,
        retryable: bool | None = True,
        timeout_seconds: float | None = None,
        provider_status_code: int | None = None,
        raw_detail: Any | None = None,
    ) -> None:
        PlatformLlmCallError.__init__(
            self,
            capability_id=capability_id,
            kind="timeout",
            error_code=error_code,
            detail=detail,
            error_envelope=error_envelope,
            retryable=retryable,
            timeout_seconds=timeout_seconds,
            provider_status_code=provider_status_code,
            raw_detail=raw_detail,
        )


class PlatformLlmProviderError(PlatformLlmCallError):
    """Provider returned a non-timeout LLM capability error."""


class PlatformLlmSchemaError(PlatformLlmCallError):
    """Platform or provider returned an invalid structured-output contract."""


class PlatformLlmQuotaError(PlatformLlmCallError):
    """Platform reported an LLM quota error in capability-error form."""


class PlatformNamespaceProtocol(Protocol):
    """The shape of ``ctx.platform`` exposed to handlers."""

    knowledge: "KnowledgeNamespace"
    web: "WebNamespace"
    artifacts: "ArtifactsNamespace"
    checkpoints: "CheckpointsNamespace"
    llm: "LlmNamespace"

    async def invoke_capability(
        self, capability_id: str, arguments: Mapping[str, Any],
    ) -> CapabilityCallDiagnostics: ...

    async def invoke_llm_capability(
        self,
        capability_id: str,
        arguments: Mapping[str, Any],
        *,
        timeout_seconds: float | None = None,
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


class WebNamespace:
    """``platform.web.search`` — platform-managed public web search.

    Agents should use this namespace instead of depending on provider-specific
    environment variables (for example ``TAVILY_API_KEY``). The platform owns
    provider credentials, audit, policy, and future provider routing.
    """

    def __init__(self, parent: "PlatformNamespace") -> None:
        self._parent = parent

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_answer: bool = True,
        include_raw_content: bool = False,
        topic: str = "general",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "max_results": int(max_results),
            "search_depth": search_depth,
            "include_answer": bool(include_answer),
            "include_raw_content": bool(include_raw_content),
            "topic": topic,
        }
        diagnostics = await self._parent.invoke_capability(
            _WEB_SEARCH_CAP, payload,
        )
        if not diagnostics.ok:
            return {
                "available": False,
                "provider": "",
                "error": diagnostics.error_code or diagnostics.kind or "web_search_failed",
                "message": diagnostics.detail,
                "results": [],
                "count": 0,
            }
        result = diagnostics.result or {}
        results_raw = result.get("results")
        if not isinstance(results_raw, list):
            self._parent._record_diagnostics(  # noqa: SLF001
                CapabilityCallDiagnostics(
                    ok=False,
                    capability_id=_WEB_SEARCH_CAP,
                    kind="schema_violation",
                    error_code="missing_results_list",
                )
            )
            return {
                "available": False,
                "provider": str(result.get("provider") or ""),
                "error": "missing_results_list",
                "results": [],
                "count": 0,
            }
        if not results_raw:
            self._parent._record_diagnostics(  # noqa: SLF001
                CapabilityCallDiagnostics(
                    ok=True,
                    capability_id=_WEB_SEARCH_CAP,
                    kind="no_results",
                    error_code=str(result.get("error") or ""),
                )
            )
        return dict(result)


class ArtifactsNamespace:
    """Budgeted artifact retrieval for connected agents.

    Full artifact payloads may be downloadable through platform HTTP APIs for
    frontends/offline tools, but SDK agents should use this namespace so every
    read carries a purpose and byte budget. The platform can then return a
    stored summary, bounded search excerpts, or bounded chunks instead of
    blindly injecting a large blob into model context.
    """

    def __init__(self, parent: "PlatformNamespace") -> None:
        self._parent = parent
        self._text_cache = ArtifactReadCache()

    async def create(
        self,
        *,
        artifact_type: str,
        content: Any,
        content_type: str = "text/markdown",
        summary: str = "",
        workflow_id: str | None = None,
        thread_id: str | None = None,
        step_id: str | None = None,
        agent_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_type": artifact_type,
            "content": content,
            "content_type": content_type,
            "summary": summary,
            "metadata": dict(metadata or {}),
        }
        for key, value in (
            ("workflow_id", workflow_id),
            ("thread_id", thread_id),
            ("step_id", step_id),
            ("agent_id", agent_id),
        ):
            if value:
                payload[key] = value
        diagnostics = await self._parent.invoke_capability(
            _ARTIFACT_CREATE_CAP, payload,
        )
        if not diagnostics.ok:
            return {
                "available": False,
                "error": diagnostics.error_code or diagnostics.kind or "artifact_create_failed",
                "message": diagnostics.detail,
            }
        return dict(diagnostics.result or {})

    async def describe(self, artifact_id: str, *, purpose: str = "") -> dict[str, Any]:
        return await self.read(
            artifact_id,
            mode="describe",
            purpose=purpose or "inspect artifact metadata",
        )

    async def summarize(self, artifact_id: str, *, purpose: str = "") -> dict[str, Any]:
        return await self.read(
            artifact_id,
            mode="summary",
            purpose=purpose or "inspect artifact summary",
        )

    async def search(
        self,
        artifact_id: str,
        query: str,
        *,
        purpose: str = "",
        max_bytes: int = 12000,
    ) -> dict[str, Any]:
        return await self.read(
            artifact_id,
            mode="search",
            query=query,
            purpose=purpose or "find relevant artifact excerpts",
            max_bytes=max_bytes,
        )

    async def read_chunks(
        self,
        artifact_id: str,
        *,
        purpose: str = "",
        offset: int = 0,
        max_bytes: int = 12000,
    ) -> dict[str, Any]:
        return await self.read(
            artifact_id,
            mode="chunks",
            purpose=purpose or "read bounded artifact chunk",
            offset=offset,
            max_bytes=max_bytes,
        )

    async def read(
        self,
        artifact_id: str,
        *,
        mode: str = "summary",
        purpose: str = "",
        query: str | None = None,
        offset: int = 0,
        max_bytes: int = 12000,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_id": artifact_id,
            "mode": mode,
            "purpose": purpose,
            "offset": int(offset),
            "max_bytes": int(max_bytes),
            # SDK intentionally does not expose allow_full=true. Agents should
            # use summary/search/chunks and let prompt assembly enforce budget.
            "allow_full": False,
        }
        if query is not None:
            payload["query"] = query
        diagnostics = await self._parent.invoke_capability(
            _ARTIFACT_READ_CAP, payload,
        )
        if not diagnostics.ok:
            return {
                "available": False,
                "artifact_id": artifact_id,
                "error": diagnostics.error_code or diagnostics.kind or "artifact_read_failed",
                "message": diagnostics.detail,
            }
        result = diagnostics.result or {}
        if result.get("available") is False:
            self._parent._record_diagnostics(  # noqa: SLF001
                CapabilityCallDiagnostics(
                    ok=True,
                    capability_id=_ARTIFACT_READ_CAP,
                    kind="no_results",
                    error_code=str(result.get("error") or ""),
                )
            )
        return dict(result)

    async def read_text(
        self,
        artifact_id: str,
        *,
        mode: str = "summary",
        purpose: str = "agent evidence retrieval",
        query: str | None = None,
        offset: int = 0,
        max_bytes: int = 12000,
    ) -> str:
        """Read an artifact and render the response as prompt-safe text."""
        normalized_artifact_id = normalize_artifact_id(artifact_id)
        cache_key = (
            normalized_artifact_id,
            str(mode or "summary"),
            str(query or ""),
            int(offset or 0),
            int(max_bytes or 12000),
            str(purpose or ""),
        )
        cached = self._text_cache.get(cache_key)
        if cached is not None:
            return cached
        data = await self.read(
            normalized_artifact_id,
            mode=mode,
            purpose=purpose,
            query=query,
            offset=offset,
            max_bytes=max_bytes,
        )
        rendered = format_artifact_read_result(data)
        self._text_cache.set(cache_key, rendered)
        return rendered

    async def search_index(
        self,
        *,
        thread_id: str | None = None,
        workflow_id: str | None = None,
        artifact_type_prefix: str | None = None,
        summary_contains: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "thread_id": thread_id,
            "workflow_id": workflow_id,
            "artifact_type_prefix": artifact_type_prefix,
            "summary_contains": summary_contains,
            "limit": int(limit),
        }
        diagnostics = await self._parent.invoke_capability(
            _ARTIFACT_SEARCH_CAP,
            {key: value for key, value in payload.items() if value not in (None, "")},
        )
        if not diagnostics.ok:
            return []
        result = diagnostics.result or {}
        items = result.get("items")
        if not isinstance(items, list):
            self._parent._record_diagnostics(  # noqa: SLF001
                CapabilityCallDiagnostics(
                    ok=False,
                    capability_id=_ARTIFACT_SEARCH_CAP,
                    kind="schema_violation",
                    error_code="missing_items_list",
                )
            )
            return []
        return [item for item in items if isinstance(item, dict)]


class WorkpadsNamespace:
    """Compact execution-workpad ledger access for connected agents."""

    def __init__(self, parent: "PlatformNamespace") -> None:
        self._parent = parent

    async def snapshot(
        self,
        *,
        workflow_id: str | None = None,
        limit: int = 24,
    ) -> dict[str, Any]:
        payload = {
            "workflow_id": workflow_id,
            "limit": int(limit),
        }
        diagnostics = await self._parent.invoke_capability(
            _WORKPAD_SNAPSHOT_CAP,
            {key: value for key, value in payload.items() if value not in (None, "")},
        )
        if not diagnostics.ok:
            return {
                "available": False,
                "error": diagnostics.error_code or diagnostics.kind or "workpad_snapshot_failed",
                "message": diagnostics.detail,
            }
        return dict(diagnostics.result or {})

    async def record_entry(
        self,
        *,
        kind: str,
        title: str = "",
        workflow_id: str | None = None,
        step_id: str | None = None,
        agent_id: str | None = None,
        capability_id: str | None = None,
        content_ref: str = "",
        content_preview: str = "",
        artifact_refs: list[dict[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": kind,
            "title": title,
            "content_ref": content_ref,
            "content_preview": content_preview,
            "artifact_refs": list(artifact_refs or []),
            "metadata": dict(metadata or {}),
        }
        for key, value in (
            ("workflow_id", workflow_id),
            ("step_id", step_id),
            ("agent_id", agent_id),
            ("capability_id", capability_id),
        ):
            if value:
                payload[key] = value
        diagnostics = await self._parent.invoke_capability(
            _WORKPAD_RECORD_ENTRY_CAP,
            payload,
        )
        if not diagnostics.ok:
            return {
                "available": False,
                "error": diagnostics.error_code or diagnostics.kind or "workpad_record_failed",
                "message": diagnostics.detail,
            }
        return dict(diagnostics.result or {})

    async def set_final_deliverable(
        self,
        artifact_ref: str,
        *,
        workflow_id: str | None = None,
        step_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_ref": artifact_ref,
            "metadata": dict(metadata or {}),
        }
        if workflow_id:
            payload["workflow_id"] = workflow_id
        if step_id:
            payload["step_id"] = step_id
        diagnostics = await self._parent.invoke_capability(
            _WORKPAD_SET_FINAL_DELIVERABLE_CAP,
            payload,
        )
        if not diagnostics.ok:
            return {
                "available": False,
                "error": diagnostics.error_code or diagnostics.kind or "workpad_final_failed",
                "message": diagnostics.detail,
            }
        return dict(diagnostics.result or {})


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
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> dict[str, Any]:
        """Invoke the platform chat LLM.

        Args:
            messages: List of OpenAI-compatible chat message dicts.
            model: Optional model override (e.g. ``"anthropic/claude-opus-4.6"``).
                   Defaults to the platform-configured model.
            temperature: Sampling temperature override.
            tools: Optional OpenAI-compatible tool definitions.
            tool_choice: Optional provider tool-choice directive.
            parallel_tool_calls: Optional provider parallel-tool-calling toggle.

        Returns:
            ``{"content": str, "tool_calls": [...], "usage_metadata": {...}}`` on success.

        Raises:
            QuotaExceededError: Org token pool is exhausted.
        """
        args: dict[str, Any] = {"messages": messages}
        if model:
            args["model"] = model
        if temperature is not None:
            args["temperature"] = temperature
        if max_output_tokens is not None:
            args["max_output_tokens"] = int(max_output_tokens)
        if tools:
            args["tools"] = tools
        if tool_choice is not None:
            args["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            args["parallel_tool_calls"] = parallel_tool_calls
        diagnostics = await self._parent.invoke_llm_capability(_LLM_CHAT_CAP, args)
        return normalise_llm_result(self._unwrap(diagnostics, _LLM_CHAT_CAP))

    async def stream_chat(
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
        """Stream the platform chat LLM as raw platform events."""
        args: dict[str, Any] = {"messages": messages}
        if model:
            args["model"] = model
        if temperature is not None:
            args["temperature"] = temperature
        if max_output_tokens is not None:
            args["max_output_tokens"] = int(max_output_tokens)
        if tools:
            args["tools"] = tools
        if tool_choice is not None:
            args["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            args["parallel_tool_calls"] = parallel_tool_calls

        stream = getattr(self._parent._llm_caller, "invoke_event_stream", None)
        if not callable(stream):
            diagnostics = await self._parent.invoke_llm_capability(_LLM_CHAT_CAP, args)
            yield {"type": "completed", "result": self._unwrap(diagnostics, _LLM_CHAT_CAP)}
            return
        async for event in stream(_LLM_CHAT_CAP, args):
            event_type = str(event.get("type") or "")
            if event_type in {"error", "cancelled"}:
                error_code = str(event.get("error_code") or "")
                raise PlatformLlmCallError(
                    capability_id=_LLM_CHAT_CAP,
                    kind=classify_envelope_error(error_code or None, http_status=None),
                    error_code=error_code,
                    detail=str(
                        event.get("explanation")
                        or event.get("reason")
                        or "platform LLM stream failed"
                    ),
                )
            yield event

    async def structured(
        self,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        method: str | None = None,
        strict: bool | None = None,
        timeout_seconds: float | None = None,
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
        if max_output_tokens is not None:
            args["max_output_tokens"] = int(max_output_tokens)
        if method is not None:
            args["method"] = method
        if strict is not None:
            args["strict"] = strict
        if timeout_seconds is not None:
            args["timeout_seconds"] = float(timeout_seconds)
        diagnostics = await self._parent.invoke_llm_capability(
            _LLM_STRUCTURED_CAP,
            args,
            timeout_seconds=timeout_seconds,
        )
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
                    error_code=diagnostics.error_code,
                    error_envelope=diagnostics.error_envelope,
                    retryable=diagnostics.retryable,
                    timeout_seconds=diagnostics.timeout_seconds,
                    provider_status_code=diagnostics.provider_status_code,
                    raw_detail=diagnostics.raw_detail,
                )
            if diagnostics.kind == "timeout" or "timeout" in diagnostics.error_code:
                raise PlatformLlmTimeoutError(
                    capability_id=capability_id,
                    detail=diagnostics.detail,
                    error_code=diagnostics.error_code,
                    error_envelope=diagnostics.error_envelope,
                    retryable=diagnostics.retryable,
                    timeout_seconds=diagnostics.timeout_seconds,
                    provider_status_code=diagnostics.provider_status_code,
                    raw_detail=diagnostics.raw_detail,
                )
            if diagnostics.kind == "schema_error" or diagnostics.error_code in {
                "invalid_args",
                "schema_violation",
                "structured_output_schema_error",
            }:
                raise PlatformLlmSchemaError(
                    capability_id=capability_id,
                    kind=diagnostics.kind,
                    error_code=diagnostics.error_code,
                    detail=diagnostics.detail,
                    error_envelope=diagnostics.error_envelope,
                    retryable=diagnostics.retryable,
                    timeout_seconds=diagnostics.timeout_seconds,
                    provider_status_code=diagnostics.provider_status_code,
                    raw_detail=diagnostics.raw_detail,
                )
            if diagnostics.kind == "quota_error" or diagnostics.error_code in {
                "quota_exceeded",
                "rate_limited",
                "llm_quota_exceeded",
            }:
                raise PlatformLlmQuotaError(
                    capability_id=capability_id,
                    kind=diagnostics.kind,
                    error_code=diagnostics.error_code,
                    detail=diagnostics.detail,
                    error_envelope=diagnostics.error_envelope,
                    retryable=diagnostics.retryable,
                    timeout_seconds=diagnostics.timeout_seconds,
                    provider_status_code=diagnostics.provider_status_code,
                    raw_detail=diagnostics.raw_detail,
                )
            if diagnostics.kind == "provider_error" or diagnostics.provider_status_code:
                raise PlatformLlmProviderError(
                    capability_id=capability_id,
                    kind=diagnostics.kind,
                    error_code=diagnostics.error_code,
                    detail=diagnostics.detail,
                    error_envelope=diagnostics.error_envelope,
                    retryable=diagnostics.retryable,
                    timeout_seconds=diagnostics.timeout_seconds,
                    provider_status_code=diagnostics.provider_status_code,
                    raw_detail=diagnostics.raw_detail,
                )
            raise PlatformLlmCallError(
                capability_id=capability_id,
                kind=diagnostics.kind,
                error_code=diagnostics.error_code,
                detail=diagnostics.detail,
                error_envelope=diagnostics.error_envelope,
                retryable=diagnostics.retryable,
                timeout_seconds=diagnostics.timeout_seconds,
                provider_status_code=diagnostics.provider_status_code,
                raw_detail=diagnostics.raw_detail,
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
    web: WebNamespace
    artifacts: ArtifactsNamespace
    workpads: WorkpadsNamespace
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
        self._stream_llm_chat = llm_caller is not None
        self.default_project_id = default_project_id
        self._diagnostics: list[CapabilityCallDiagnostics] = []
        self._mid_run_ask_active = False
        self.knowledge = KnowledgeNamespace(self)
        self.web = WebNamespace(self)
        self.artifacts = ArtifactsNamespace(self)
        self.workpads = WorkpadsNamespace(self)
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
        if self._mid_run_ask_active and _is_high_risk_inline_capability(capability_id):
            diagnostics = CapabilityCallDiagnostics(
                ok=False,
                capability_id=capability_id,
                kind="binding_denied",
                error_code="mid_run_ask_inline_write_denied",
                detail=(
                    "write/high-risk platform capability calls are refused while "
                    "a mid-run ask is active"
                ),
            )
            self._diagnostics.append(diagnostics)
            return diagnostics
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
        *,
        timeout_seconds: float | None = None,
    ) -> CapabilityCallDiagnostics:
        """Invoke a ``platform.llm.*`` capability with the long-timeout caller.

        Used by ``LlmNamespace`` so ``platform.llm.structured`` (which can
        legitimately take 30+ seconds against Anthropic / OpenAI) doesn't
        share the 8 s default that's appropriate for short capabilities
        like knowledge / checkpoints.
        """
        invoke_stream = getattr(self._llm_caller, "invoke_stream_with_diagnostics", None)
        if (
            self._stream_llm_chat
            and capability_id == _LLM_CHAT_CAP
            and callable(invoke_stream)
        ):
            diagnostics = await invoke_stream(capability_id, arguments)
            if not diagnostics.ok:
                self._diagnostics.append(diagnostics)
            return diagnostics

        diagnostics = await self._llm_caller.invoke_with_diagnostics(
            capability_id,
            arguments,
            timeout_seconds=timeout_seconds,
        )
        if not diagnostics.ok:
            self._diagnostics.append(diagnostics)
        return diagnostics

    def last_diagnostics(self) -> tuple[CapabilityCallDiagnostics, ...]:
        return tuple(self._diagnostics)

    def _record_diagnostics(self, diagnostics: CapabilityCallDiagnostics) -> None:
        self._diagnostics.append(diagnostics)

    def set_mid_run_ask_active(self, active: bool) -> None:
        self._mid_run_ask_active = bool(active)


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
    web: WebNamespace
    artifacts: ArtifactsNamespace
    workpads: WorkpadsNamespace
    checkpoints: CheckpointsNamespace
    llm: LlmNamespace

    def __init__(self, *, reason: str) -> None:
        self._reason = reason
        self.default_project_id = ""
        self._diagnostics: list[CapabilityCallDiagnostics] = []
        self.knowledge = KnowledgeNamespace(self)  # type: ignore[arg-type]
        self.web = WebNamespace(self)  # type: ignore[arg-type]
        self.artifacts = ArtifactsNamespace(self)  # type: ignore[arg-type]
        self.workpads = WorkpadsNamespace(self)  # type: ignore[arg-type]
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
        *,
        timeout_seconds: float | None = None,
    ) -> CapabilityCallDiagnostics:
        del timeout_seconds
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
    "ArtifactsNamespace",
    "CapabilityCallDiagnostics",
    "CheckpointsNamespace",
    "DegradationKind",
    "KnowledgeNamespace",
    "LlmNamespace",
    "PlatformLlmCallError",
    "PlatformLlmProviderError",
    "PlatformLlmQuotaError",
    "PlatformLlmSchemaError",
    "PlatformLlmTimeoutError",
    "PlatformLlmTransportError",
    "PlatformNamespace",
    "PlatformNamespaceProtocol",
    "QuotaExceededError",
    "build_platform_namespace",
    "classify_envelope_error",
    "WorkpadsNamespace",
]
