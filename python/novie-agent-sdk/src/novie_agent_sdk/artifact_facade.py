"""EXPERT_AGENT_SDK W3 — ``artifact_agent`` SDK facade.

Author-side surface for ``artifact_agent`` types: a single decorated
handler that returns an artifact, while the SDK owns HTTP routes,
stream envelopes, platform headers, and output projection.

Target API (pinned by ``test_artifact_facade.py``):

    from novie_agent_sdk import artifact_agent

    app = artifact_agent(manifest=".well-known/agent.json")

    @app.handle
    async def handle(ctx):
        await ctx.progress("Reading context")
        result = await run_model(ctx.input_text, ctx.project, ctx.attachments)
        return ctx.artifact(
            artifact_type="market_report",
            summary="Market report complete",
            content=result,
            metadata={"confidence": "medium"},
        )

The facade wraps the existing ``Agent`` runtime so authors no longer
write FastAPI routes, reach into ``InvokeContext.input``, or hand-roll
stream NDJSON event shapes.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .platform_namespace import build_platform_namespace
from .runtime import (
    Agent,
    InvokeContext,
    RequestHeaders,
    StreamContext,
)


# ── Result envelope ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ArtifactResult:
    """Author-returned artifact envelope.

    Constructed via ``ctx.artifact(...)``; the facade projects it onto
    ``/invoke`` and ``/stream`` outputs.
    """

    artifact_type: str
    summary: str
    content: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_invoke_output(self) -> dict[str, Any]:
        """Shape returned from ``/invoke``: the platform's
        ``invoke_endpoint`` wraps this in ``{"output": ..., "status":
        "completed"}``, so the artifact lands at ``response.output.*``.
        """
        return {
            "kind": "artifact",
            "artifact_type": self.artifact_type,
            "summary": self.summary,
            "content": self.content,
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

    def to_stream_event(self) -> dict[str, Any]:
        """Shape yielded as the final stream event before the SDK
        appends ``{"kind": "done", "output": {}}``."""
        return self.to_invoke_output()


@dataclass(frozen=True, slots=True)
class NeedsConfirmationResult:
    """One-shot result that asks the platform/user for confirmation.

    This is intentionally terminal for ``simple`` / ``stream`` requests:
    agents must not hold the HTTP request open while waiting for a human.
    Durable waits belong in ``tasks`` mode via ``ctx.wait_for_human(...)``.
    """

    prompt: str
    allowed_actions: Sequence[str] = ("approve", "request_changes", "reject")
    confirmation_id: str = ""
    resume_reference: Mapping[str, Any] = field(default_factory=dict)
    timeout_policy: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_confirmation_payload(self) -> dict[str, Any]:
        payload = {
            "confirmation_id": self.confirmation_id,
            "prompt": self.prompt,
            "allowed_actions": list(self.allowed_actions),
            "resume_reference": dict(self.resume_reference),
            "timeout_policy": dict(self.timeout_policy),
            "metadata": dict(self.metadata),
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload

    def to_invoke_response(self) -> dict[str, Any]:
        confirmation = self.to_confirmation_payload()
        return {
            "status": "needs_confirmation",
            "output": {
                "kind": "needs_confirmation",
                **confirmation,
            },
            "confirmation": confirmation,
            "events": [],
            "artifacts": [],
            "usage": [],
            "diagnostics": {},
        }

    def to_stream_event(self) -> dict[str, Any]:
        confirmation = self.to_confirmation_payload()
        return {
            "kind": "needs_confirmation",
            "status": "needs_confirmation",
            "confirmation": confirmation,
        }


# ── Platform namespace (W4: real knowledge / checkpoints client) ─────────────
#
# ``ctx.platform`` is a ``PlatformNamespace`` (live HTTP client) when
# ``NOVIE_PLATFORM_BASE_URL`` is set and incoming headers carry tenant +
# project; otherwise it's an ``_UnavailablePlatformNamespace`` that
# returns ``platform_unavailable`` diagnostics so handlers degrade
# gracefully. Either way ``ctx.platform.is_available`` lets handlers
# branch without exception-handling boilerplate.


# ── Context ──────────────────────────────────────────────────────────────────


_HandlerOutcome = ArtifactResult | NeedsConfirmationResult | dict[str, Any] | None
_CoercedOutcome = ArtifactResult | NeedsConfirmationResult


_PROGRESS_EVENT_KIND = "progress"


@dataclass
class ArtifactAgentContext:
    """Author-facing context for ``artifact_agent`` handlers.

    Wraps the underlying ``InvokeContext`` / ``StreamContext`` and
    projects the platform's request shape into named fields the
    author can rely on without touching ``input.get(...)``.

    ``progress(...)`` is mode-aware: in stream mode events are pushed
    to the NDJSON stream as they happen; in invoke mode they're
    buffered and surfaced via ``progress_log`` (so tests can assert
    on them without forcing the author to change handlers).
    """

    input_text: str
    inputs: Mapping[str, Any]
    project: Mapping[str, Any]
    member: Mapping[str, Any]
    runtime_context: Mapping[str, Any]
    attachments: Sequence[Mapping[str, Any]]
    capability_id: str
    headers: RequestHeaders
    platform: Any
    progress_log: list[dict[str, Any]] = field(default_factory=list)
    _progress_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None = None

    async def progress(
        self,
        message: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Surface a progress event.

        In stream mode the event lands in the NDJSON stream so the
        platform's session timeline mirrors it. In invoke mode the
        event is appended to ``progress_log`` (buffered) so the
        author's handler stays mode-agnostic.
        """
        event = {"kind": _PROGRESS_EVENT_KIND, "text": message}
        if metadata:
            event["metadata"] = dict(metadata)
        self.progress_log.append(event)
        if self._progress_emitter is not None:
            await self._progress_emitter(event)

    def artifact(
        self,
        *,
        artifact_type: str,
        summary: str,
        content: Any = None,
        metadata: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> ArtifactResult:
        """Build the ``ArtifactResult`` returned by the handler."""
        return ArtifactResult(
            artifact_type=artifact_type,
            summary=summary,
            content=content,
            metadata=dict(metadata) if metadata is not None else {},
            provenance=dict(provenance) if provenance is not None else {},
        )

    def needs_confirmation(
        self,
        *,
        prompt: str,
        allowed_actions: Sequence[str] = ("approve", "request_changes", "reject"),
        confirmation_id: str = "",
        resume_reference: Mapping[str, Any] | None = None,
        timeout_policy: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        reason: str = "",
    ) -> NeedsConfirmationResult:
        """Build a terminal one-shot confirmation result."""
        return NeedsConfirmationResult(
            prompt=prompt,
            allowed_actions=tuple(allowed_actions),
            confirmation_id=confirmation_id,
            resume_reference=(
                dict(resume_reference) if resume_reference is not None else {}
            ),
            timeout_policy=dict(timeout_policy) if timeout_policy is not None else {},
            metadata=dict(metadata) if metadata is not None else {},
            reason=reason,
        )


# ── Input projection ──────────────────────────────────────────────────────────


def _resolve_capability_id(inputs: Mapping[str, Any]) -> str:
    direct = inputs.get("capability_id")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    grants = inputs.get("capability_grants")
    if isinstance(grants, list):
        for item in grants:
            if not isinstance(item, dict):
                continue
            capability_id = item.get("capability_id")
            if isinstance(capability_id, str) and capability_id.strip():
                return capability_id.strip()
    return ""


_INPUT_TEXT_KEYS: tuple[str, ...] = (
    "input_text",
    "text",
    "query",
    "prompt",
    "message",
)


def _resolve_input_text(
    inputs: Mapping[str, Any], brief: Mapping[str, Any],
) -> str:
    for key in _INPUT_TEXT_KEYS:
        value = inputs.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("user_goal", "summary", "title"):
        value = brief.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _project_member(
    runtime_context_data: Mapping[str, Any], headers: RequestHeaders,
) -> dict[str, Any]:
    identity = runtime_context_data.get("identity") or {}
    if not isinstance(identity, Mapping):
        identity = {}
    return {
        "principal_id": identity.get("principal_id") or headers.user_id,
        "principal_type": identity.get("principal_type") or "user",
        "roles": tuple(identity.get("roles") or ()),
        "service_principal": headers.service_principal,
    }


def _project_project(
    runtime_context_data: Mapping[str, Any], headers: RequestHeaders,
) -> dict[str, Any]:
    tenant = runtime_context_data.get("tenant") or {}
    if not isinstance(tenant, Mapping):
        tenant = {}
    return {
        "tenant_id": tenant.get("tenant_id") or headers.tenant_id,
        "workspace_id": tenant.get("workspace_id") or headers.workspace_id,
        "project_id": headers.project_id,
        "session_id": runtime_context_data.get("session_id") or headers.session_id,
        "thread_id": runtime_context_data.get("thread_id") or "",
        "request_id": runtime_context_data.get("request_id") or headers.request_id,
    }


def _project_attachments(
    inputs: Mapping[str, Any], brief: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[Any] = []
    for source in (inputs, brief):
        attachments = source.get("attachments")
        if isinstance(attachments, list):
            candidates = attachments
            break
    return [item for item in candidates if isinstance(item, Mapping)]


def _build_context(
    *,
    input_payload: Mapping[str, Any],
    headers: RequestHeaders,
    platform: Any,
    progress_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
) -> ArtifactAgentContext:
    """Project the raw ``/invoke`` or ``/stream`` body onto an
    ``ArtifactAgentContext``. Pure projection — no I/O, no logging."""
    inputs_raw = input_payload.get("inputs", input_payload)
    if not isinstance(inputs_raw, Mapping):
        inputs_raw = {}
    runtime_ctx_raw = input_payload.get("context") or {}
    if not isinstance(runtime_ctx_raw, Mapping):
        runtime_ctx_raw = {}
    brief_raw = input_payload.get("brief") or inputs_raw.get("brief") or {}
    if not isinstance(brief_raw, Mapping):
        brief_raw = {}
    return ArtifactAgentContext(
        input_text=_resolve_input_text(inputs_raw, brief_raw),
        inputs=dict(inputs_raw),
        project=_project_project(runtime_ctx_raw, headers),
        member=_project_member(runtime_ctx_raw, headers),
        runtime_context=dict(runtime_ctx_raw),
        attachments=_project_attachments(inputs_raw, brief_raw),
        capability_id=_resolve_capability_id(inputs_raw),
        headers=headers,
        platform=platform,
        _progress_emitter=progress_emitter,
    )


# ── Handler outcome → wire shape ─────────────────────────────────────────────


def _coerce_outcome(outcome: _HandlerOutcome) -> _CoercedOutcome:
    if isinstance(outcome, (ArtifactResult, NeedsConfirmationResult)):
        return outcome
    if outcome is None:
        raise RuntimeError(
            "artifact_agent handler returned None — call ``ctx.artifact(...)`` "
            "and return the result."
        )
    if isinstance(outcome, Mapping):
        artifact_type = outcome.get("artifact_type")
        summary = outcome.get("summary")
        if not isinstance(artifact_type, str) or not artifact_type:
            raise RuntimeError(
                "artifact_agent handler returned a dict without "
                "``artifact_type``; either return the dataclass from "
                "``ctx.artifact(...)`` or include ``artifact_type``."
            )
        if not isinstance(summary, str):
            summary = ""
        return ArtifactResult(
            artifact_type=artifact_type,
            summary=summary,
            content=outcome.get("content"),
            metadata=dict(outcome.get("metadata") or {}),
            provenance=dict(outcome.get("provenance") or {}),
        )
    raise RuntimeError(
        "artifact_agent handler must return ``ctx.artifact(...)``, "
        "``ctx.needs_confirmation(...)``, or a dict; got "
        f"{type(outcome).__name__}."
    )


# ── Facade ───────────────────────────────────────────────────────────────────


HandlerFn = Callable[[ArtifactAgentContext], Awaitable[_HandlerOutcome]]


class ArtifactAgentApp:
    """Thin wrapper around ``Agent`` that exposes the ``artifact_agent``
    authoring surface.

    Owns the ``Agent`` instance under the hood; ``serve()`` /
    ``build_app()`` / ``configure_registration()`` proxy to it so the
    facade is a drop-in replacement for the lower-level surface.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        platform: Any | None = None,
        platform_base_url: str | None = None,
    ) -> None:
        self._agent = agent
        self._platform_override = platform
        self._platform_base_url = platform_base_url
        self._handler: HandlerFn | None = None

    def _resolve_platform(self, headers: RequestHeaders) -> Any:
        """Pick the ``ctx.platform`` for one request.

        Precedence:

        1. ``platform=`` injected at construction (tests / W4+ wiring)
        2. Live ``PlatformNamespace`` built from incoming headers +
           ``platform_base_url`` (or ``NOVIE_PLATFORM_BASE_URL``)
        3. ``_UnavailablePlatformNamespace`` if neither is configured
        """
        if self._platform_override is not None:
            return self._platform_override
        return build_platform_namespace(
            headers,
            agent_id=self._agent._manifest.agent_id,  # noqa: SLF001
            base_url=self._platform_base_url,
        )

    @property
    def agent(self) -> Agent:
        """Underlying runtime agent — escape hatch for advanced
        callers that need ``Agent`` features the facade doesn't
        expose yet."""
        return self._agent

    def handle(self, fn: HandlerFn) -> HandlerFn:
        """Register ``fn`` as the artifact handler. Both ``/invoke``
        and ``/stream`` endpoints route through it."""
        if not asyncio.iscoroutinefunction(fn):
            raise TypeError(
                "@app.handle expects an async function; got a sync "
                f"callable {fn!r}."
            )
        self._handler = fn
        self._wire_invoke()
        self._wire_stream()
        return fn

    def _wire_invoke(self) -> None:
        async def _invoke_adapter(ctx: InvokeContext) -> dict[str, Any]:
            assert self._handler is not None  # registered before wire
            facade_ctx = _build_context(
                input_payload=ctx.input,
                headers=ctx.headers,
                platform=self._resolve_platform(ctx.headers),
                progress_emitter=None,  # buffered; surface via progress_log
            )
            outcome = await self._handler(facade_ctx)
            result = _coerce_outcome(outcome)
            if isinstance(result, NeedsConfirmationResult):
                return result.to_invoke_response()
            return result.to_invoke_output()

        self._agent.invoke(_invoke_adapter)

    def _wire_stream(self) -> None:
        async def _stream_adapter(
            ctx: StreamContext,
        ) -> AsyncIterator[dict[str, Any]]:
            assert self._handler is not None
            event_queue: asyncio.Queue[
                tuple[str, dict[str, Any] | _CoercedOutcome | BaseException | None]
            ] = asyncio.Queue()

            async def _push(event: dict[str, Any]) -> None:
                await event_queue.put(("progress", event))

            facade_ctx = _build_context(
                input_payload=ctx.input,
                headers=ctx.headers,
                platform=self._resolve_platform(ctx.headers),
                progress_emitter=_push,
            )

            async def _runner() -> None:
                try:
                    outcome = await self._handler(facade_ctx)  # type: ignore[misc]
                    await event_queue.put(("artifact", _coerce_outcome(outcome)))
                except BaseException as exc:  # noqa: BLE001
                    await event_queue.put(("error", exc))
                finally:
                    await event_queue.put(("done", None))

            runner_task = asyncio.create_task(_runner())
            try:
                while True:
                    kind, payload = await event_queue.get()
                    if kind == "done":
                        break
                    if kind == "error":
                        raise payload  # type: ignore[misc]
                    if kind == "progress":
                        yield payload  # type: ignore[misc]
                        continue
                    if kind == "artifact":
                        assert isinstance(
                            payload, (ArtifactResult, NeedsConfirmationResult),
                        )
                        yield payload.to_stream_event()
                        continue
            finally:
                if not runner_task.done():
                    runner_task.cancel()
                    try:
                        await runner_task
                    except (asyncio.CancelledError, Exception):
                        pass

        self._agent.stream(_stream_adapter)

    def configure_registration(self, *args: Any, **kwargs: Any) -> Any:
        return self._agent.configure_registration(*args, **kwargs)

    def build_app(self, *args: Any, **kwargs: Any) -> Any:
        return self._agent.build_app(*args, **kwargs)

    async def serve(self, *args: Any, **kwargs: Any) -> Any:
        return await self._agent.serve(*args, **kwargs)


def artifact_agent(
    *,
    manifest: str | Path | dict[str, Any],
    platform: Any | None = None,
    platform_base_url: str | None = None,
) -> ArtifactAgentApp:
    """Construct an ``artifact_agent`` SDK facade.

    ``manifest`` accepts the same inputs as ``Agent.from_manifest`` —
    a path string, ``Path``, or pre-loaded dict.

    ``platform`` lets tests / advanced callers inject a fixed
    ``ctx.platform`` object (must implement
    ``PlatformNamespaceProtocol``). When omitted, ``ctx.platform`` is
    built per-request from the incoming A2A headers and
    ``platform_base_url`` (which falls back to
    ``NOVIE_PLATFORM_BASE_URL``); requests without those land on
    ``_UnavailablePlatformNamespace`` so handlers degrade gracefully.
    """
    if isinstance(manifest, Mapping):
        agent = Agent.from_manifest_dict(dict(manifest))
    else:
        agent = Agent.from_manifest(manifest)
    return ArtifactAgentApp(
        agent, platform=platform, platform_base_url=platform_base_url,
    )


__all__ = [
    "ArtifactAgentApp",
    "ArtifactAgentContext",
    "ArtifactResult",
    "NeedsConfirmationResult",
    "artifact_agent",
]
