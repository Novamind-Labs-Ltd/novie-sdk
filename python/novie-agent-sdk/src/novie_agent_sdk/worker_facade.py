"""EXPERT_AGENT_SDK W5 — ``worker_agent`` SDK facade.

Author-side surface for ``worker_agent`` types: a single decorated
async handler runs as a background task while the SDK owns the
``tasks`` protocol routes (``POST /tasks``, ``GET /tasks/{id}``,
``GET /tasks/{id}/events``, ``GET /tasks/{id}/result``,
``POST /tasks/{id}/cancel``) and lifecycle state machine.

Target API (pinned by ``test_worker_facade.py``):

    from novie_agent_sdk import worker_agent

    app = worker_agent(manifest=".well-known/agent.json")

    @app.task
    async def run(ctx):
        await ctx.progress("Preparing workspace")
        if ctx.is_cancelled:
            return ctx.fail("cancelled before start")
        output = await execute_work(ctx.task, ctx.repo, ctx.upstream)
        return ctx.result(
            summary="Task completed",
            output=output,
            artifacts=[],
        )

The facade builds on ``Agent`` (which already owns the tasks-protocol
endpoints + cancel + idempotency replay) — it just projects the
platform's request shape onto named fields and gives the author a
typed result envelope (``WorkerResult``) and a clean failure path
(``ctx.fail(...)`` raising ``WorkerFailure``).
"""
from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .platform_namespace import build_platform_namespace
from .public_errors import PublicAgentError
from .runtime import (
    Agent,
    InMemoryTaskStore,
    RequestHeaders,
    SqliteTaskStore,
    TaskContext,
    TaskStore,
)


_SAFE_AGENT_ID_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


# ── Failure path ─────────────────────────────────────────────────────────────


class WorkerFailure(PublicAgentError):
    """Raised by ``ctx.fail(...)`` to terminate a task with an explicit
    failure reason.

    The SDK's ``_run_task`` wrapper catches any ``Exception`` from the
    handler and calls ``set_task_error`` (status → ``failed``);
    ``WorkerFailure`` is just a typed exception that carries the
    reason verbatim so the platform's ``GET /tasks/{id}`` returns the
    author's message instead of a generic stringified exception.
    """

    def __init__(self, reason: str, *, metadata: Mapping[str, Any] | None = None) -> None:
        super().__init__(error_code="worker_failure", public_message=reason)
        self.reason = reason
        self.metadata = dict(metadata) if metadata else {}


# ── Result envelope ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """Author-returned task result.

    Constructed via ``ctx.result(...)``; the facade projects it onto
    the dict shape that ``Agent._run_task`` writes to the task store
    (which then surfaces as ``GET /tasks/{id}/result``'s ``output``).
    """

    summary: str
    output: Any = None
    artifacts: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_task_output(self) -> dict[str, Any]:
        """Shape stored by the SDK and returned from
        ``GET /tasks/{id}/result``'s ``output`` field."""
        return {
            "kind": "worker_result",
            "summary": self.summary,
            "output": self.output,
            "artifacts": [dict(item) for item in self.artifacts],
            "metadata": dict(self.metadata),
        }


# ── Context ──────────────────────────────────────────────────────────────────


_PROGRESS_EVENT_KIND = "progress"
_ARTIFACT_EVENT_KIND = "artifact"

_VALID_WAIT_STATUSES = frozenset({"waiting_for_input", "waiting_for_human"})


@dataclass(frozen=True, slots=True)
class HumanWaitRequest:
    """Structured HITL wait payload emitted by ``ctx.wait_for_human``.

    This is intentionally JSON-shape friendly so platform timelines,
    gates, and future timeout escalation all read the same fields.
    """

    gate_id: str = ""
    prompt: str = ""
    allowed_actions: Sequence[str] = ("approve", "request_changes", "reject")
    resume_reference: Mapping[str, Any] = field(default_factory=dict)
    timeout_policy: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "wait_kind": "waiting_for_human",
            "gate_id": self.gate_id,
            "prompt": self.prompt,
            "allowed_actions": list(self.allowed_actions),
            "resume_reference": dict(self.resume_reference),
            "timeout_policy": dict(self.timeout_policy),
            "metadata": dict(self.metadata),
        }


@dataclass
class WorkerTaskContext:
    """Author-facing context for ``worker_agent`` task handlers.

    Wraps the underlying ``TaskContext`` and projects the platform's
    request shape onto named fields (``task`` / ``repo`` /
    ``target_branch`` / ``upstream`` / ``metadata``) so the author
    doesn't reach into ``ctx.input.get(...)``. Convenience methods
    delegate to the underlying ``TaskContext.emit_event`` /
    ``set_status`` so authors keep one mental model.
    """

    task_id: str
    task: Any
    repo: Mapping[str, Any]
    target_branch: str
    upstream: Mapping[str, Any]
    metadata: Mapping[str, Any]
    inputs: Mapping[str, Any]
    headers: RequestHeaders
    platform: Any
    _task_ctx: TaskContext = field(repr=False)

    @property
    def is_cancelled(self) -> bool:
        """True if the platform has called ``POST /tasks/{id}/cancel``.
        Cooperative — handlers should check periodically and return
        ``ctx.fail(...)`` or simply return early."""
        return self._task_ctx.is_cancelled

    async def progress(
        self,
        message: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Emit a progress event onto the task event stream.

        Surfaces via ``GET /tasks/{id}/events`` so platform-side
        timeline mirrors handler progress in real time.
        """
        payload: dict[str, Any] = {"text": message}
        if metadata:
            payload["metadata"] = dict(metadata)
        await self._task_ctx.emit_event(
            _PROGRESS_EVENT_KIND, payload, summary=message[:200],
        )

    async def artifact(
        self,
        *,
        artifact_type: str,
        summary: str,
        content: Any = None,
        metadata: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Emit an artifact event during task execution.

        Distinct from the final result: workers can stream multiple
        intermediate artifacts (file diffs, test runs, etc.) and then
        return ``ctx.result(...)`` summarizing the run.
        """
        await self._task_ctx.emit_event(
            _ARTIFACT_EVENT_KIND,
            {
                "artifact_type": artifact_type,
                "summary": summary,
                "content": content,
                "metadata": dict(metadata) if metadata else {},
                "provenance": dict(provenance) if provenance else {},
            },
            summary=summary[:200],
        )

    def result(
        self,
        *,
        summary: str,
        output: Any = None,
        artifacts: Sequence[Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> WorkerResult:
        """Build the ``WorkerResult`` returned from the handler."""
        return WorkerResult(
            summary=summary,
            output=output,
            artifacts=tuple(dict(item) for item in artifacts),
            metadata=dict(metadata) if metadata else {},
        )

    def fail(
        self,
        reason: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "WorkerFailure":
        """Build a ``WorkerFailure`` carrying ``reason``.

        Authors typically write ``raise ctx.fail("...")`` so the
        statement reads naturally; returning is also accepted (the
        adapter detects either path).
        """
        return WorkerFailure(reason, metadata=metadata)

    async def wait_for_input(
        self,
        *,
        prompt: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Surface a ``waiting_for_input`` status transition.

        Platform polling sees the status change and can prompt the
        user. The handler must transition back to ``running``
        explicitly via ``resume_running()`` before continuing work.
        """
        await self._task_ctx.set_status("waiting_for_input")
        if prompt or metadata:
            await self._task_ctx.emit_event(
                "wait_prompt",
                {
                    "wait_kind": "waiting_for_input",
                    "prompt": prompt or "",
                    "metadata": dict(metadata) if metadata else {},
                },
                summary=(prompt or "waiting_for_input")[:200],
            )

    async def wait_for_human(
        self,
        *,
        gate_id: str | None = None,
        prompt: str | None = None,
        allowed_actions: Sequence[str] | None = None,
        resume_reference: Mapping[str, Any] | None = None,
        timeout_policy: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Surface a structured ``waiting_for_human`` HITL gate.

        The payload mirrors the production HTTP contract: gate id,
        operator prompt, allowed actions, resume reference, timeout
        policy, and arbitrary metadata.
        """
        await self._task_ctx.set_status("waiting_for_human")
        payload = HumanWaitRequest(
            gate_id=gate_id or "",
            prompt=prompt or "",
            allowed_actions=tuple(allowed_actions or (
                "approve", "request_changes", "reject",
            )),
            resume_reference=dict(resume_reference) if resume_reference else {},
            timeout_policy=dict(timeout_policy) if timeout_policy else {},
            metadata=dict(metadata) if metadata else {},
        )
        await self._task_ctx.emit_event(
            "wait_prompt",
            payload.to_event_payload(),
            summary=(
                prompt
                or f"waiting_for_human gate={gate_id or '<none>'}"
            )[:200],
        )

    async def resume_running(self) -> None:
        """Transition back from a ``waiting_*`` status to ``running``."""
        await self._task_ctx.set_status("running")


# ── Input projection ─────────────────────────────────────────────────────────


def _project_field(inputs: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in inputs:
            return inputs[key]
    return None


def _build_context(
    *,
    task_ctx: TaskContext,
    platform: Any,
) -> WorkerTaskContext:
    inputs_raw = task_ctx.input.get("inputs", task_ctx.input)
    if not isinstance(inputs_raw, Mapping):
        inputs_raw = {}
    repo = _project_field(inputs_raw, "repo", "repository")
    if not isinstance(repo, Mapping):
        repo = {}
    upstream = _project_field(inputs_raw, "upstream", "upstream_context")
    if not isinstance(upstream, Mapping):
        upstream = {}
    metadata = _project_field(inputs_raw, "metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    target_branch = _project_field(inputs_raw, "target_branch", "branch")
    if not isinstance(target_branch, str):
        target_branch = ""
    return WorkerTaskContext(
        task_id=task_ctx.task_id,
        task=_project_field(inputs_raw, "task", "instruction", "prompt"),
        repo=dict(repo),
        target_branch=target_branch,
        upstream=dict(upstream),
        metadata=dict(metadata),
        inputs=dict(inputs_raw),
        headers=task_ctx.headers,
        platform=platform,
        _task_ctx=task_ctx,
    )


# ── Outcome → wire shape ─────────────────────────────────────────────────────


_HandlerOutcome = WorkerResult | WorkerFailure | dict[str, Any] | None


def _coerce_outcome(outcome: _HandlerOutcome) -> WorkerResult:
    if isinstance(outcome, WorkerResult):
        return outcome
    if isinstance(outcome, WorkerFailure):
        # ``ctx.fail(...)`` returned (not raised) — re-raise so the
        # SDK's task runner sets the task error.
        raise outcome
    if outcome is None:
        raise RuntimeError(
            "worker_agent handler returned None — call "
            "``ctx.result(...)`` and return the result, or "
            "``raise ctx.fail(\"reason\")`` to terminate."
        )
    if isinstance(outcome, Mapping):
        summary = outcome.get("summary")
        if not isinstance(summary, str):
            summary = ""
        return WorkerResult(
            summary=summary,
            output=outcome.get("output"),
            artifacts=tuple(
                dict(item) for item in outcome.get("artifacts") or ()
                if isinstance(item, Mapping)
            ),
            metadata=dict(outcome.get("metadata") or {}),
        )
    raise RuntimeError(
        "worker_agent handler must return ``ctx.result(...)`` or a "
        f"dict; got {type(outcome).__name__}."
    )


# ── Facade ───────────────────────────────────────────────────────────────────


HandlerFn = Callable[[WorkerTaskContext], Awaitable[_HandlerOutcome]]


class WorkerAgentApp:
    """Thin wrapper around ``Agent`` for ``worker_agent`` types.

    ``@app.task`` registers a single async handler; the SDK owns the
    full ``tasks`` protocol surface plus state machine
    (queued / running / waiting_* / completed / failed / cancelled).
    Cooperative cancellation is exposed as ``ctx.is_cancelled``.
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

    @property
    def agent(self) -> Agent:
        return self._agent

    def _resolve_platform(self, headers: RequestHeaders) -> Any:
        if self._platform_override is not None:
            return self._platform_override
        return build_platform_namespace(
            headers,
            agent_id=self._agent._manifest.agent_id,  # noqa: SLF001
            base_url=self._platform_base_url,
        )

    def task(self, fn: HandlerFn) -> HandlerFn:
        """Register ``fn`` as the worker task handler."""
        if not asyncio.iscoroutinefunction(fn):
            raise TypeError(
                "@app.task expects an async function; got a sync "
                f"callable {fn!r}."
            )
        self._handler = fn
        self._wire_task()
        return fn

    def _wire_task(self) -> None:
        async def _task_adapter(task_ctx: TaskContext) -> dict[str, Any]:
            assert self._handler is not None
            facade_ctx = _build_context(
                task_ctx=task_ctx,
                platform=self._resolve_platform(task_ctx.headers),
            )
            outcome = await self._handler(facade_ctx)
            result = _coerce_outcome(outcome)
            return result.to_task_output()

        self._agent.task(_task_adapter)

    def configure_registration(self, *args: Any, **kwargs: Any) -> Any:
        return self._agent.configure_registration(*args, **kwargs)

    def build_app(self, *args: Any, **kwargs: Any) -> Any:
        return self._agent.build_app(*args, **kwargs)

    async def serve(self, *args: Any, **kwargs: Any) -> Any:
        return await self._agent.serve(*args, **kwargs)


def worker_agent(
    *,
    manifest: str | Path | dict[str, Any],
    platform: Any | None = None,
    platform_base_url: str | None = None,
    task_store: TaskStore | None = None,
    sqlite_path: str | Path | None = None,
) -> WorkerAgentApp:
    """Construct a ``worker_agent`` SDK facade.

    ``manifest`` accepts the same inputs as ``Agent.from_manifest``.
    ``task_store`` overrides the default in-memory store; pass
    ``sqlite_path="..."`` for a persistence-backed store without
    constructing one yourself. If neither is provided, manifests with
    ``execution.durability=task_store`` and production-mode workers
    automatically use ``SqliteTaskStore`` at
    ``NOVIE_AGENT_TASK_STORE_PATH`` or
    ``NOVIE_AGENT_STATE_DIR/{agent_id}.tasks.sqlite3``.
    ``platform`` / ``platform_base_url`` mirror the W3 facade so
    handlers can hit ``ctx.platform.knowledge.search(...)`` /
    ``ctx.platform.checkpoints.put(...)`` exactly the same way.
    """
    if isinstance(manifest, Mapping):
        agent = _agent_from_dict(dict(manifest), task_store=task_store, sqlite_path=sqlite_path)
    else:
        agent = _agent_from_path(manifest, task_store=task_store, sqlite_path=sqlite_path)
    return WorkerAgentApp(
        agent, platform=platform, platform_base_url=platform_base_url,
    )


def _resolve_store(
    *,
    manifest: Any,
    task_store: TaskStore | None,
    sqlite_path: str | Path | None,
) -> TaskStore:
    if task_store is not None:
        return task_store
    if sqlite_path is not None:
        return SqliteTaskStore(str(sqlite_path))
    durability = getattr(getattr(manifest, "execution", None), "durability", "none")
    production = os.getenv("NOVIE_ENV", "").strip().lower() == "production"
    if durability == "task_store" or production:
        return SqliteTaskStore(str(_default_sqlite_path(str(manifest.agent_id))))
    return InMemoryTaskStore()


def _default_sqlite_path(agent_id: str) -> Path:
    explicit = os.getenv("NOVIE_AGENT_TASK_STORE_PATH", "").strip()
    if explicit:
        path = Path(explicit)
    else:
        base = Path(os.getenv("NOVIE_AGENT_STATE_DIR", "").strip() or ".novie")
        safe_agent_id = _SAFE_AGENT_ID_CHARS.sub("-", agent_id).strip(".-")
        path = base / f"{safe_agent_id or 'agent'}.tasks.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _agent_from_path(
    path: str | Path,
    *,
    task_store: TaskStore | None,
    sqlite_path: str | Path | None,
) -> Agent:
    from novie_protocol.contracts.agent_sdk_v2 import AgentManifestV2

    manifest = AgentManifestV2.from_file(path)
    errors = manifest.validate()
    if errors:
        raise ValueError(f"Invalid manifest at {path}: {errors}")
    return Agent(manifest, task_store=_resolve_store(
        manifest=manifest, task_store=task_store, sqlite_path=sqlite_path,
    ))


def _agent_from_dict(
    d: dict[str, Any],
    *,
    task_store: TaskStore | None,
    sqlite_path: str | Path | None,
) -> Agent:
    from novie_protocol.contracts.agent_sdk_v2 import AgentManifestV2

    manifest = AgentManifestV2.from_dict(d)
    errors = manifest.validate()
    if errors:
        raise ValueError(f"Invalid manifest dict: {errors}")
    return Agent(manifest, task_store=_resolve_store(
        manifest=manifest, task_store=task_store, sqlite_path=sqlite_path,
    ))


__all__ = [
    "HandlerFn",
    "HumanWaitRequest",
    "WorkerAgentApp",
    "WorkerFailure",
    "WorkerResult",
    "WorkerTaskContext",
    "worker_agent",
]
