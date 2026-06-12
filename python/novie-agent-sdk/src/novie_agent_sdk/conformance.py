"""EXPERT_AGENT_SDK W8 — official conformance test suite + compatibility matrix.

Backs ``novie agents conformance`` (CLI) and is exposed as a public SDK
API so external agent repos can run the same probes in their own CI
without copying assertions out of the platform tree.

Surface (locked by ``test_conformance.py``):

- ``CompatibilityMatrix`` — frozen dc carrying the SDK version,
  manifest schema version, platform protocol version, and the
  agent ``kind`` values currently supported by the platform.
- ``current_compatibility()`` — the platform's known-good matrix.
  Bumped intentionally when any of the four pillars changes.
- ``verify_compatibility(matrix, manifest_dict)`` — runs the
  pre-runtime gate the platform applies before accepting a
  registration.
- ``ConformanceProbe`` / ``ConformanceReport`` — structured per-probe
  envelope (status ∈ ``{"pass", "fail", "skip"}`` + detail + hint)
  so CI can branch programmatically and so failures point at the
  next concrete remediation.
- ``run_conformance(base_url, *, agent_type, ...)`` — async runner
  that probes a *running* agent server. Probes:
  1. ``healthcheck`` — ``GET /healthz`` returns 200.
  2. ``manifest_serving`` — ``GET /.well-known/agent.json`` returns
     a JSON body that round-trips through ``AgentManifestV2.from_dict``
     and passes ``.validate()``.
  3. ``manifest_compatibility`` — manifest values match the
     compatibility matrix (acceptance bullet "Platform can reject
     incompatible agents before runtime execution").
  4. ``stream_artifact_lifecycle`` (artifact agents) — ``POST
     /stream`` returns NDJSON terminating in ``kind=done``.
  5. ``task_worker_lifecycle`` (worker agents) — ``POST /tasks`` →
     poll → ``GET /tasks/{id}/result`` round-trip.
  6. ``progress_events`` — at least one ``progress`` event observed
     in the lifecycle (so platform timelines don't stall silently).
  7. ``error_envelope`` — malformed input returns a 4xx with a
     structured error so platform diagnostic tooling can branch
     programmatically.
- ``restart_hook`` — optional async callback used by durability
  probes to restart the target agent between first call and replay.
  Without it, restart probes are explicit skips with remediation hints.

Compatibility matrix bumps:
- SDK version comes from ``novie-agent-sdk`` ``pyproject.toml`` —
  bump together when shipping a breaking change.
- ``MANIFEST_SCHEMA_VERSION`` follows ``AgentManifestV2`` — bump
  when the contract's ``from_dict``/``validate`` rules change.
- ``PLATFORM_PROTOCOL_VERSION`` is the wire version the platform
  speaks (currently ``v2`` matching ``agent_sdk_v2``).
- ``SUPPORTED_AGENT_TYPES`` is the kinds the planner can route to.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from novie_protocol.contracts.agent_sdk_v2 import AgentManifestV2


_log = logging.getLogger(__name__)


# ── Compatibility matrix ────────────────────────────────────────────────────


SDK_VERSION = "0.3.15"
MANIFEST_SCHEMA_VERSION = "v2"
PLATFORM_PROTOCOL_VERSION = "v2"
SUPPORTED_AGENT_KINDS: tuple[str, ...] = (
    "expert_basic",
    "expert_complex",
)
SUPPORTED_PROTOCOL_MODES: tuple[str, ...] = (
    "simple",
    "stream",
    "tasks",
)


@dataclass(frozen=True, slots=True)
class CompatibilityMatrix:
    """Snapshot of the four pillars the platform checks before
    accepting an agent registration."""

    sdk_version: str
    manifest_schema_version: str
    platform_protocol_version: str
    supported_agent_kinds: tuple[str, ...]
    supported_protocol_modes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sdk_version": self.sdk_version,
            "manifest_schema_version": self.manifest_schema_version,
            "platform_protocol_version": self.platform_protocol_version,
            "supported_agent_kinds": list(self.supported_agent_kinds),
            "supported_protocol_modes": list(self.supported_protocol_modes),
        }


def current_compatibility() -> CompatibilityMatrix:
    """The platform's currently-known-good matrix.

    Bumped intentionally when any of the four pillars changes. Tests
    pin this so a drift fails CI loudly before reaching the platform.
    """
    return CompatibilityMatrix(
        sdk_version=SDK_VERSION,
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        platform_protocol_version=PLATFORM_PROTOCOL_VERSION,
        supported_agent_kinds=SUPPORTED_AGENT_KINDS,
        supported_protocol_modes=SUPPORTED_PROTOCOL_MODES,
    )


def verify_compatibility(
    matrix: CompatibilityMatrix, manifest_dict: dict[str, Any],
) -> list[str]:
    """Pre-runtime gate. Returns a list of human-readable error
    strings (empty = compatible).

    Acceptance bullet: "Compatibility errors are clear enough for
    non-platform engineers." Each error includes both the offending
    field and the supported set so the operator sees the next concrete
    fix.
    """
    errors: list[str] = []
    kind = manifest_dict.get("kind")
    if kind not in matrix.supported_agent_kinds:
        errors.append(
            f"manifest.kind={kind!r} is not supported. The current "
            f"platform accepts only: {sorted(matrix.supported_agent_kinds)}."
        )
    protocol_mode = manifest_dict.get("protocol_mode")
    if protocol_mode not in matrix.supported_protocol_modes:
        errors.append(
            f"manifest.protocol_mode={protocol_mode!r} is not supported. "
            f"The current platform accepts only: "
            f"{sorted(matrix.supported_protocol_modes)}."
        )
    runtime = manifest_dict.get("runtime")
    if runtime != "external_a2a":
        errors.append(
            f"manifest.runtime={runtime!r} must be 'external_a2a' for "
            "external agents registered via this SDK."
        )
    if not manifest_dict.get("agent_id"):
        errors.append("manifest.agent_id must be a non-empty string.")
    if not manifest_dict.get("version"):
        errors.append("manifest.version must be a non-empty string.")
    return errors


# ── Conformance probes ──────────────────────────────────────────────────────


ProbeStatus = Literal["pass", "fail", "skip"]
RestartHook = Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ConformanceProbe:
    """One probe inside a conformance run."""

    name: str
    status: ProbeStatus
    detail: str = ""
    hint: str = ""
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "hint": self.hint,
            "url": self.url,
        }


@dataclass(frozen=True, slots=True)
class ConformanceReport:
    """Aggregate of a full conformance run."""

    base_url: str
    matrix: CompatibilityMatrix
    probes: tuple[ConformanceProbe, ...] = field(default_factory=tuple)
    manifest: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return not any(p.status == "fail" for p in self.probes)

    @property
    def failures(self) -> tuple[ConformanceProbe, ...]:
        return tuple(p for p in self.probes if p.status == "fail")

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "ok": self.ok,
            "matrix": self.matrix.to_dict(),
            "probes": [p.to_dict() for p in self.probes],
            "manifest": self.manifest,
        }


def _passed(name: str, *, detail: str = "", url: str = "") -> ConformanceProbe:
    return ConformanceProbe(name=name, status="pass", detail=detail, url=url)


def _failed(
    name: str, detail: str, *, hint: str = "", url: str = "",
) -> ConformanceProbe:
    return ConformanceProbe(
        name=name, status="fail", detail=detail, hint=hint, url=url,
    )


def _skipped(name: str, detail: str, *, hint: str = "") -> ConformanceProbe:
    return ConformanceProbe(
        name=name, status="skip", detail=detail, hint=hint,
    )


# ── Individual probe implementations ────────────────────────────────────────


async def _probe_healthcheck(
    client: httpx.AsyncClient, base_url: str,
) -> ConformanceProbe:
    url = f"{base_url}/healthz"
    try:
        resp = await client.get("/healthz")
    except httpx.TransportError as exc:
        return _failed(
            "healthcheck",
            f"transport error: {exc}",
            hint="Boot the agent (``novie agents dev .``) and retry.",
            url=url,
        )
    if resp.status_code != 200:
        return _failed(
            "healthcheck",
            f"unexpected HTTP {resp.status_code}",
            hint=(
                "Healthcheck must return 200. The SDK runtime mounts "
                "/healthz automatically — check that the FastAPI app "
                "isn't being shadowed."
            ),
            url=url,
        )
    return _passed("healthcheck", url=url)


async def _probe_manifest_serving(
    client: httpx.AsyncClient, base_url: str,
) -> tuple[ConformanceProbe, dict[str, Any] | None]:
    url = f"{base_url}/.well-known/agent.json"
    try:
        resp = await client.get("/.well-known/agent.json")
    except httpx.TransportError as exc:
        return (
            _failed(
                "manifest_serving",
                f"transport error: {exc}",
                hint="Make sure the dev server is running.",
                url=url,
            ),
            None,
        )
    if resp.status_code != 200:
        return (
            _failed(
                "manifest_serving",
                f"unexpected HTTP {resp.status_code}",
                hint="The SDK runtime serves the manifest from .well-known/agent.json automatically.",
                url=url,
            ),
            None,
        )
    try:
        data = resp.json()
    except ValueError as exc:
        return _failed("manifest_serving", f"non-JSON body: {exc}", url=url), None
    if not isinstance(data, dict):
        return (
            _failed(
                "manifest_serving",
                f"manifest must be a JSON object; got {type(data).__name__}",
                url=url,
            ),
            None,
        )
    try:
        manifest = AgentManifestV2.from_dict(data)
    except Exception as exc:  # noqa: BLE001
        return (
            _failed(
                "manifest_serving",
                f"manifest does not parse as AgentManifestV2: {exc!r}",
                hint=(
                    "Run ``novie agents validate agent.yaml`` and "
                    "``novie agents generate-manifest agent.yaml`` to "
                    "regenerate before retrying."
                ),
                url=url,
            ),
            None,
        )
    validate_errors = manifest.validate()
    if validate_errors:
        return (
            _failed(
                "manifest_serving",
                f"manifest validate() returned: {validate_errors}",
                hint="Run ``novie agents validate agent.yaml`` to fix author-side issues.",
                url=url,
            ),
            data,
        )
    return _passed("manifest_serving", detail=f"agent_id={data.get('agent_id')}", url=url), data


def _probe_manifest_compatibility(
    matrix: CompatibilityMatrix, manifest: dict[str, Any] | None,
) -> ConformanceProbe:
    if manifest is None:
        return _skipped(
            "manifest_compatibility",
            "manifest_serving did not produce a parseable manifest",
        )
    errors = verify_compatibility(matrix, manifest)
    if errors:
        return _failed(
            "manifest_compatibility",
            "; ".join(errors),
            hint=(
                "The platform rejects incompatible manifests at registration "
                "time. Bump the SDK version or pick a supported kind / "
                "protocol_mode and regenerate the manifest."
            ),
        )
    return _passed(
        "manifest_compatibility",
        detail=(
            f"sdk={matrix.sdk_version} schema={matrix.manifest_schema_version} "
            f"protocol={matrix.platform_protocol_version}"
        ),
    )


async def _probe_error_envelope(
    client: httpx.AsyncClient, base_url: str, *, has_invoke: bool,
) -> ConformanceProbe:
    """Verify the agent returns a 4xx with a structured body when the
    platform sends a malformed payload. Skipped when the agent doesn't
    expose ``/invoke`` (worker-only agents — error envelope on
    ``/tasks`` is similar but covered separately).

    Acceptance bullet: "Platform diagnostic tooling can branch on the
    error envelope" — surfaces shape so the agent never returns
    HTML on bad input.
    """
    if not has_invoke:
        return _skipped(
            "error_envelope",
            "no /invoke endpoint exposed — error envelope check is artifact-agent only",
        )
    url = f"{base_url}/invoke"
    try:
        resp = await client.post("/invoke", content=b"{not json")
    except httpx.TransportError as exc:
        return _failed(
            "error_envelope",
            f"transport error: {exc}",
            url=url,
        )
    if resp.status_code < 400 or resp.status_code >= 500:
        return _failed(
            "error_envelope",
            f"expected HTTP 4xx for malformed body; got {resp.status_code}",
            hint=(
                "Malformed input must be rejected with a 4xx, not 5xx. "
                "Check that the SDK's ``_parse_json`` helper is wired."
            ),
            url=url,
        )
    try:
        body = resp.json()
    except ValueError:
        return _failed(
            "error_envelope",
            "error response was not JSON; platform diagnostic tooling can't branch on it",
            hint="Return ``{detail: {error: <code>, ...}}`` from FastAPI HTTPException.",
            url=url,
        )
    if not isinstance(body, dict):
        return _failed(
            "error_envelope",
            f"error body must be a JSON object; got {type(body).__name__}",
            url=url,
        )
    return _passed(
        "error_envelope",
        detail=f"HTTP {resp.status_code}, structured body present",
        url=url,
    )


async def _probe_stream_artifact_lifecycle(
    client: httpx.AsyncClient, base_url: str,
) -> tuple[ConformanceProbe, list[dict[str, Any]]]:
    url = f"{base_url}/stream"
    try:
        resp = await client.post(
            "/stream", json={"input": {"inputs": {"text": "conformance"}}},
        )
    except httpx.TransportError as exc:
        return _failed(
            "stream_artifact_lifecycle",
            f"transport error: {exc}", url=url,
        ), []
    if resp.status_code != 200:
        return _failed(
            "stream_artifact_lifecycle",
            f"unexpected HTTP {resp.status_code}",
            url=url,
        ), []
    content_type = resp.headers.get("content-type", "")
    if not content_type.startswith("application/x-ndjson"):
        return _failed(
            "stream_artifact_lifecycle",
            f"expected NDJSON content-type; got {content_type!r}",
            hint="The SDK stream wrapper sets application/x-ndjson automatically.",
            url=url,
        ), []
    events: list[dict[str, Any]] = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            return _failed(
                "stream_artifact_lifecycle",
                f"line is not valid JSON: {line!r}: {exc}",
                url=url,
            ), events
    if not events:
        return _failed(
            "stream_artifact_lifecycle",
            "stream returned no events",
            url=url,
        ), events
    kinds = {e.get("kind") for e in events}
    if not (kinds & {"final", "done", "artifact", "terminal_error"}):
        return _failed(
            "stream_artifact_lifecycle",
            f"stream did not emit a terminal event (final/done/artifact/terminal_error); kinds={sorted(k for k in kinds if k)}",
            hint="The SDK auto-appends ``{kind: 'done'}`` and converts handler exceptions to ``terminal_error``.",
            url=url,
        ), events
    return _passed(
        "stream_artifact_lifecycle",
        detail=f"{len(events)} events, kinds={sorted(k for k in kinds if k)}",
        url=url,
    ), events


async def _probe_task_worker_lifecycle(
    client: httpx.AsyncClient,
    base_url: str,
    *,
    poll_interval_s: float,
    max_polls: int,
) -> tuple[ConformanceProbe, list[dict[str, Any]]]:
    url = f"{base_url}/tasks"
    try:
        create = await client.post(
            "/tasks", json={"input": {"inputs": {"task": "conformance"}}},
        )
    except httpx.TransportError as exc:
        return _failed(
            "task_worker_lifecycle",
            f"create transport error: {exc}",
            url=url,
        ), []
    if create.status_code != 202:
        return _failed(
            "task_worker_lifecycle",
            f"create returned HTTP {create.status_code}; expected 202",
            url=url,
        ), []
    try:
        task_id = create.json()["task_id"]
    except (ValueError, KeyError) as exc:
        return _failed(
            "task_worker_lifecycle",
            f"create response missing task_id: {exc}",
            url=url,
        ), []

    final_status = "?"
    for _ in range(max_polls):
        await asyncio.sleep(poll_interval_s)
        try:
            poll = await client.get(f"/tasks/{task_id}")
        except httpx.TransportError as exc:
            return _failed(
                "task_worker_lifecycle",
                f"poll transport error: {exc}",
                url=url,
            ), []
        if poll.status_code != 200:
            return _failed(
                "task_worker_lifecycle",
                f"poll returned HTTP {poll.status_code}",
                url=url,
            ), []
        try:
            final_status = poll.json()["status"]
        except (ValueError, KeyError):
            return _failed(
                "task_worker_lifecycle",
                "poll body missing status",
                url=url,
            ), []
        if final_status in {"completed", "failed", "cancelled"}:
            break

    if final_status != "completed":
        return _failed(
            "task_worker_lifecycle",
            f"task did not complete; final status={final_status}",
            hint="Check that the worker handler returns ``ctx.result(...)``.",
            url=url,
        ), []

    try:
        events_resp = await client.get(f"/tasks/{task_id}/events")
    except httpx.TransportError as exc:
        return _failed(
            "task_worker_lifecycle",
            f"events transport error: {exc}",
            url=url,
        ), []
    events: list[dict[str, Any]] = []
    if events_resp.status_code == 200:
        try:
            raw_events = events_resp.json().get("events", []) or []
        except ValueError:
            raw_events = []
        # The lifecycle probe only consumes well-formed list shapes —
        # the dedicated ``task_event_stream_semantics`` probe surfaces
        # malformed shapes as a separate failure.
        if isinstance(raw_events, list):
            events = [e for e in raw_events if isinstance(e, dict)]

    try:
        result_resp = await client.get(f"/tasks/{task_id}/result")
    except httpx.TransportError as exc:
        return _failed(
            "task_worker_lifecycle",
            f"result transport error: {exc}",
            url=url,
        ), events
    if result_resp.status_code != 200:
        return _failed(
            "task_worker_lifecycle",
            f"result returned HTTP {result_resp.status_code}",
            url=url,
        ), events
    return _passed(
        "task_worker_lifecycle",
        detail=f"task_id={task_id} terminal=completed",
        url=url,
    ), events


def _probe_progress_events(
    events: list[dict[str, Any]],
) -> ConformanceProbe:
    """The lifecycle probe captures events; this probe turns them into
    a separate pass/fail so a working lifecycle without progress
    events is treated as a contract gap.
    """
    if not events:
        return _skipped(
            "progress_events",
            "lifecycle probe did not capture any events",
            hint=(
                "If the lifecycle passes but no events were captured, "
                "the agent is using the simple ``/invoke`` path. "
                "Switch to ``/stream`` or ``/tasks`` to surface progress."
            ),
        )
    progress = [e for e in events if e.get("kind") == "progress"]
    if not progress:
        return _failed(
            "progress_events",
            f"no progress events in lifecycle ({len(events)} events captured)",
            hint=(
                "Call ``ctx.progress(message)`` from your handler so the "
                "platform timeline mirrors progress."
            ),
        )
    return _passed(
        "progress_events", detail=f"{len(progress)} progress events captured",
    )


# ── EXTERNAL_AGENT_HTTP W7 — idempotency + cancellation + event stream ──────


async def _probe_invoke_idempotency(
    client: httpx.AsyncClient, base_url: str,
) -> ConformanceProbe:
    """Acceptance bullet (W7): "one-shot idempotency and retry
    classification". Two ``POST /invoke`` calls with the same
    ``Idempotency-Key`` must return byte-identical responses.
    """
    url = f"{base_url}/invoke"
    idem = f"conformance-invoke-idem-{uuid.uuid4().hex}"
    headers = {"Idempotency-Key": idem}
    payload = {"input": {"inputs": {"text": "idem-conformance"}}}
    try:
        first = await client.post("/invoke", json=payload, headers=headers)
        second = await client.post("/invoke", json=payload, headers=headers)
    except httpx.TransportError as exc:
        return _failed("invoke_idempotency", f"transport error: {exc}", url=url)
    if first.status_code != 200 or second.status_code != 200:
        return _failed(
            "invoke_idempotency",
            f"unexpected HTTP statuses {first.status_code}/{second.status_code}",
            hint="Both replays must succeed with HTTP 200.",
            url=url,
        )
    if first.content != second.content:
        return _failed(
            "invoke_idempotency",
            "duplicate Idempotency-Key returned different bodies",
            hint=(
                "Persist the terminal /invoke response keyed by "
                "Idempotency-Key so retries replay without re-running "
                "side effects."
            ),
            url=url,
        )
    return _passed(
        "invoke_idempotency",
        detail=f"replay matched ({len(first.content)} bytes)",
        url=url,
    )


async def _probe_stream_idempotency(
    client: httpx.AsyncClient, base_url: str,
) -> ConformanceProbe:
    """Acceptance bullet (W7): the streamed event sequence must replay
    identically when the same ``Idempotency-Key`` is sent twice."""
    url = f"{base_url}/stream"
    idem = f"conformance-stream-idem-{uuid.uuid4().hex}"
    headers = {"Idempotency-Key": idem}
    payload = {"input": {"inputs": {"text": "idem-conformance"}}}
    try:
        first = await client.post("/stream", json=payload, headers=headers)
        second = await client.post("/stream", json=payload, headers=headers)
    except httpx.TransportError as exc:
        return _failed("stream_idempotency", f"transport error: {exc}", url=url)
    if first.status_code != 200 or second.status_code != 200:
        return _failed(
            "stream_idempotency",
            f"unexpected HTTP statuses {first.status_code}/{second.status_code}",
            url=url,
        )
    first_lines = [line for line in first.text.splitlines() if line.strip()]
    second_lines = [line for line in second.text.splitlines() if line.strip()]
    if first_lines != second_lines:
        return _failed(
            "stream_idempotency",
            (
                f"replay event sequence differed (first={len(first_lines)} "
                f"lines, second={len(second_lines)} lines)"
            ),
            hint=(
                "Persist the full NDJSON event sequence per "
                "Idempotency-Key so duplicate /stream calls return the "
                "exact same wire transcript."
            ),
            url=url,
        )
    return _passed(
        "stream_idempotency",
        detail=f"replay matched ({len(first_lines)} events)",
        url=url,
    )


async def _probe_tasks_idempotency(
    client: httpx.AsyncClient, base_url: str,
) -> ConformanceProbe:
    """Acceptance bullet (W7): "idempotent task creation". Two
    ``POST /tasks`` calls with the same ``Idempotency-Key`` must
    return the same ``task_id``.
    """
    url = f"{base_url}/tasks"
    idem = f"conformance-tasks-idem-{uuid.uuid4().hex}"
    headers = {"Idempotency-Key": idem}
    payload = {"input": {"inputs": {"task": "idem-conformance"}}}
    try:
        first = await client.post("/tasks", json=payload, headers=headers)
        second = await client.post("/tasks", json=payload, headers=headers)
    except httpx.TransportError as exc:
        return _failed("tasks_idempotency", f"transport error: {exc}", url=url)
    if first.status_code not in (200, 201, 202):
        return _failed(
            "tasks_idempotency",
            f"first create returned HTTP {first.status_code}",
            url=url,
        )
    if second.status_code not in (200, 201, 202):
        return _failed(
            "tasks_idempotency",
            f"replay returned HTTP {second.status_code}",
            url=url,
        )
    try:
        first_id = first.json()["task_id"]
        second_id = second.json()["task_id"]
    except (ValueError, KeyError) as exc:
        return _failed(
            "tasks_idempotency",
            f"missing task_id in response body: {exc}",
            url=url,
        )
    if first_id != second_id:
        return _failed(
            "tasks_idempotency",
            f"different task_ids on replay: {first_id!r} vs {second_id!r}",
            hint=(
                "Persist Idempotency-Key → task_id mapping so duplicate "
                "POST /tasks calls return the original task without "
                "creating a second worker job."
            ),
            url=url,
        )
    if not second.json().get("status"):
        return _failed(
            "tasks_idempotency",
            "duplicate /tasks response missing status field",
            url=url,
        )
    return _passed(
        "tasks_idempotency",
        detail=f"task_id={first_id} replay matched",
        url=url,
    )


async def _probe_task_event_stream_semantics(
    client: httpx.AsyncClient, base_url: str,
) -> ConformanceProbe:
    """Acceptance bullet (W7): "event stream/list semantics". The
    ``GET /tasks/{id}/events`` endpoint must return ``{task_id,
    events: list[...]}`` even before the task completes — operator
    tooling polls events while tasks are mid-flight.
    """
    url = f"{base_url}/tasks"
    payload = {"input": {"inputs": {"task": "events-shape-conformance"}}}
    try:
        create = await client.post("/tasks", json=payload)
    except httpx.TransportError as exc:
        return _failed(
            "task_event_stream_semantics",
            f"transport error on create: {exc}",
            url=url,
        )
    if create.status_code not in (200, 201, 202):
        return _failed(
            "task_event_stream_semantics",
            f"create returned HTTP {create.status_code}",
            url=url,
        )
    try:
        task_id = create.json()["task_id"]
    except (ValueError, KeyError) as exc:
        return _failed(
            "task_event_stream_semantics",
            f"create response missing task_id: {exc}",
            url=url,
        )
    events_url = f"{base_url}/tasks/{task_id}/events"
    try:
        events_resp = await client.get(f"/tasks/{task_id}/events")
    except httpx.TransportError as exc:
        return _failed(
            "task_event_stream_semantics",
            f"transport error on /events: {exc}",
            url=events_url,
        )
    if events_resp.status_code != 200:
        return _failed(
            "task_event_stream_semantics",
            f"/events returned HTTP {events_resp.status_code}",
            url=events_url,
        )
    try:
        body = events_resp.json()
    except ValueError:
        return _failed(
            "task_event_stream_semantics",
            "/events body is not JSON",
            url=events_url,
        )
    if not isinstance(body, dict):
        return _failed(
            "task_event_stream_semantics",
            f"/events body must be a JSON object; got {type(body).__name__}",
            url=events_url,
        )
    if body.get("task_id") != task_id:
        return _failed(
            "task_event_stream_semantics",
            f"/events body task_id={body.get('task_id')!r} != created task_id={task_id!r}",
            url=events_url,
        )
    events_field = body.get("events")
    if not isinstance(events_field, list):
        return _failed(
            "task_event_stream_semantics",
            f"events field must be a list; got {type(events_field).__name__}",
            hint=(
                "Always return ``events: []`` even when the task hasn't "
                "emitted anything yet. Operator polling expects the list "
                "shape from the first call."
            ),
            url=events_url,
        )
    return _passed(
        "task_event_stream_semantics",
        detail=f"task_id={task_id} events={len(events_field)}",
        url=events_url,
    )


async def _probe_task_cancellation(
    client: httpx.AsyncClient,
    base_url: str,
    *,
    manifest: dict[str, Any] | None,
) -> ConformanceProbe:
    """Acceptance bullet (W7): "cancellation". For agents that declare
    ``execution.supports_cancel``, a ``POST /tasks/{id}/cancel`` must
    succeed (HTTP 202 or already-terminal 409); skipped otherwise so
    workers without cooperative cancel aren't penalized."""
    supports_cancel = False
    if isinstance(manifest, dict):
        execution = manifest.get("execution")
        if isinstance(execution, dict):
            supports_cancel = bool(execution.get("supports_cancel"))
    if not supports_cancel:
        return _skipped(
            "task_cancellation",
            "manifest.execution.supports_cancel is false",
            hint=(
                "Set ``execution.supports_cancel=true`` in agent.yaml for "
                "long-running workers so the platform can cancel runaway "
                "tasks."
            ),
        )
    url = f"{base_url}/tasks"
    payload = {"input": {"inputs": {"task": "cancel-conformance"}}}
    try:
        create = await client.post("/tasks", json=payload)
    except httpx.TransportError as exc:
        return _failed(
            "task_cancellation", f"create transport error: {exc}", url=url,
        )
    if create.status_code not in (200, 201, 202):
        return _failed(
            "task_cancellation",
            f"create returned HTTP {create.status_code}",
            url=url,
        )
    try:
        task_id = create.json()["task_id"]
    except (ValueError, KeyError) as exc:
        return _failed(
            "task_cancellation",
            f"create response missing task_id: {exc}",
            url=url,
        )
    cancel_url = f"{base_url}/tasks/{task_id}/cancel"
    try:
        cancel = await client.post(f"/tasks/{task_id}/cancel")
    except httpx.TransportError as exc:
        return _failed(
            "task_cancellation",
            f"transport error on cancel: {exc}",
            url=cancel_url,
        )
    if cancel.status_code not in (200, 202, 409):
        return _failed(
            "task_cancellation",
            (
                f"cancel returned HTTP {cancel.status_code}; "
                "expected 202 (cancelled) or 409 (already terminal)"
            ),
            hint=(
                "Wire up cooperative cancellation in your worker so "
                "POST /tasks/{id}/cancel returns 202 while running."
            ),
            url=cancel_url,
        )
    return _passed(
        "task_cancellation",
        detail=f"task_id={task_id} cancel HTTP {cancel.status_code}",
        url=cancel_url,
    )


# ── Type derivation ─────────────────────────────────────────────────────────


def _derive_protocol_probe(
    *, agent_type_hint: str | None, manifest: dict[str, Any] | None,
) -> Literal["invoke", "stream", "tasks", "skip"]:
    """Resolve the protocol-specific probe from an explicit hint or
    the agent's manifest.

    Accepts both EXPERT_AGENT_SDK type names (``artifact-agent`` /
    ``worker-agent`` / ``tool``) and EXTERNAL_AGENT_HTTP W7 type names
    (``oneshot`` / ``stream`` / ``worker``) so the same CLI can serve
    both audiences.
    """
    if agent_type_hint:
        hint = agent_type_hint.strip().lower()
        # EXTERNAL_AGENT_HTTP W7 type names
        if hint == "oneshot":
            return "invoke"
        if hint == "stream":
            return "stream"
        if hint == "worker":
            return "tasks"
        # EXPERT_AGENT_SDK type names
        if hint in {"artifact", "artifact-agent", "artifact_agent"}:
            return "stream"
        if hint in {"worker-agent", "worker_agent"}:
            return "tasks"
        if hint == "tool":
            return "invoke"
    if manifest is not None:
        protocol = str(manifest.get("protocol_mode") or "").strip().lower()
        if protocol == "tasks":
            return "tasks"
        if protocol == "stream":
            return "stream"
        if protocol == "simple":
            return "invoke"
    return "skip"


def _manifest_durability(manifest: dict[str, Any] | None) -> str:
    if manifest is None:
        return "none"
    execution = manifest.get("execution")
    if isinstance(execution, dict):
        durability = execution.get("durability")
        if isinstance(durability, str):
            return durability
    metadata = manifest.get("metadata")
    if isinstance(metadata, dict):
        durability = metadata.get("durability")
        if isinstance(durability, str):
            return durability
    return "none"


async def _probe_one_shot_restart_replay(
    client: httpx.AsyncClient,
    base_url: str,
    *,
    protocol: Literal["invoke", "stream", "tasks", "skip"],
    manifest: dict[str, Any] | None,
    restart_hook: RestartHook | None,
) -> ConformanceProbe | None:
    if _manifest_durability(manifest) != "result_cache":
        return None
    if protocol not in {"invoke", "stream"}:
        return _skipped(
            "oneshot_restart_replay",
            "result_cache durability is only checked for simple/stream agents",
        )
    if restart_hook is None:
        return _skipped(
            "oneshot_restart_replay",
            "restart_hook not provided",
            hint=(
                "Pass a restart hook from CI/operator tooling to verify "
                "that completed invoke/stream results replay after restart."
            ),
        )

    idem = f"conformance-restart-{uuid.uuid4().hex}"
    headers = {"Idempotency-Key": idem}
    path = "/invoke" if protocol == "invoke" else "/stream"
    url = f"{base_url}{path}"
    payload = {"input": {"inputs": {"text": "restart conformance"}}}
    try:
        first = await client.post(path, json=payload, headers=headers)
    except httpx.TransportError as exc:
        return _failed("oneshot_restart_replay", f"first call transport error: {exc}", url=url)
    if first.status_code != 200:
        return _failed(
            "oneshot_restart_replay",
            f"first call returned HTTP {first.status_code}",
            url=url,
        )

    try:
        await restart_hook()
    except Exception as exc:  # noqa: BLE001
        return _failed(
            "oneshot_restart_replay",
            f"restart_hook failed: {exc!r}",
            hint="Make the restart hook block until the agent is ready again.",
            url=url,
        )

    try:
        second = await client.post(path, json=payload, headers=headers)
    except httpx.TransportError as exc:
        return _failed("oneshot_restart_replay", f"replay transport error: {exc}", url=url)
    if second.status_code != 200:
        return _failed(
            "oneshot_restart_replay",
            f"replay returned HTTP {second.status_code}",
            hint="Completed result_cache invocations must replay after restart.",
            url=url,
        )
    if first.content != second.content:
        return _failed(
            "oneshot_restart_replay",
            "replay body differed from the pre-restart response",
            hint="Persist the terminal invoke response or full stream event sequence.",
            url=url,
        )
    return _passed(
        "oneshot_restart_replay",
        detail=f"{protocol} replayed after restart",
        url=url,
    )


async def _probe_task_restart_persistence(
    client: httpx.AsyncClient,
    base_url: str,
    *,
    protocol: Literal["invoke", "stream", "tasks", "skip"],
    manifest: dict[str, Any] | None,
    restart_hook: RestartHook | None,
    poll_interval_s: float,
    max_polls: int,
) -> ConformanceProbe | None:
    if _manifest_durability(manifest) != "task_store":
        return None
    if protocol != "tasks":
        return _skipped(
            "task_restart_persistence",
            "task_store durability is only checked for tasks agents",
        )
    if restart_hook is None:
        return _skipped(
            "task_restart_persistence",
            "restart_hook not provided",
            hint=(
                "Pass a restart hook from CI/operator tooling to verify "
                "task status/event/result persistence after restart."
            ),
        )

    url = f"{base_url}/tasks"
    try:
        create = await client.post(
            "/tasks",
            json={"input": {"inputs": {"task": "restart conformance"}}},
            headers={"Idempotency-Key": f"conformance-task-{uuid.uuid4().hex}"},
        )
    except httpx.TransportError as exc:
        return _failed("task_restart_persistence", f"create transport error: {exc}", url=url)
    if create.status_code != 202:
        return _failed(
            "task_restart_persistence",
            f"create returned HTTP {create.status_code}; expected 202",
            url=url,
        )
    try:
        task_id = create.json()["task_id"]
    except (ValueError, KeyError) as exc:
        return _failed("task_restart_persistence", f"create missing task_id: {exc}", url=url)

    try:
        await restart_hook()
    except Exception as exc:  # noqa: BLE001
        return _failed(
            "task_restart_persistence",
            f"restart_hook failed: {exc!r}",
            hint="Make the restart hook block until the agent is ready again.",
            url=url,
        )

    status = "?"
    for _ in range(max_polls):
        await asyncio.sleep(poll_interval_s)
        try:
            poll = await client.get(f"/tasks/{task_id}")
        except httpx.TransportError as exc:
            return _failed("task_restart_persistence", f"poll transport error: {exc}", url=url)
        if poll.status_code == 404:
            return _failed(
                "task_restart_persistence",
                "task returned 404 after restart",
                hint="Persist accepted task records before returning from POST /tasks.",
                url=url,
            )
        if poll.status_code != 200:
            return _failed(
                "task_restart_persistence",
                f"poll returned HTTP {poll.status_code}",
                url=url,
            )
        try:
            status = str(poll.json()["status"])
        except (ValueError, KeyError):
            return _failed("task_restart_persistence", "poll body missing status", url=url)
        if status in {"completed", "failed", "cancelled"}:
            break

    if status != "completed":
        return _failed(
            "task_restart_persistence",
            f"task did not complete after restart; final status={status}",
            hint="A task_store worker must restore accepted work and expose terminal state.",
            url=url,
        )
    result = await client.get(f"/tasks/{task_id}/result")
    if result.status_code != 200:
        return _failed(
            "task_restart_persistence",
            f"terminal result returned HTTP {result.status_code} after restart",
            hint="Persist terminal task results with the task record.",
            url=url,
        )
    return _passed(
        "task_restart_persistence",
        detail=f"task_id={task_id} completed/result readable after restart",
        url=url,
    )


# ── Top-level runner ─────────────────────────────────────────────────────────


async def run_conformance(
    base_url: str,
    *,
    agent_type: str | None = None,
    matrix: CompatibilityMatrix | None = None,
    timeout_s: float = 5.0,
    poll_interval_s: float = 0.05,
    max_polls: int = 80,
    client: httpx.AsyncClient | None = None,
    restart_hook: RestartHook | None = None,
) -> ConformanceReport:
    """Run the official conformance suite against a *running* agent.

    ``client`` lets tests inject ``httpx.MockTransport`` so the suite
    runs without a real server. ``matrix`` defaults to
    ``current_compatibility()`` so external agent CI pins drift loudly.
    """
    base_url = base_url.rstrip("/") or "http://localhost:8010"
    matrix = matrix or current_compatibility()
    own_client = client is None

    async def _do(_client: httpx.AsyncClient) -> ConformanceReport:
        probes: list[ConformanceProbe] = []

        probes.append(await _probe_healthcheck(_client, base_url))

        manifest_probe, manifest = await _probe_manifest_serving(_client, base_url)
        probes.append(manifest_probe)
        probes.append(_probe_manifest_compatibility(matrix, manifest))

        protocol = _derive_protocol_probe(
            agent_type_hint=agent_type, manifest=manifest,
        )
        captured_events: list[dict[str, Any]] = []
        if protocol == "stream":
            probe, events = await _probe_stream_artifact_lifecycle(_client, base_url)
            probes.append(probe)
            captured_events = events
            # W7: stream-mode idempotency replay.
            probes.append(await _probe_stream_idempotency(_client, base_url))
        elif protocol == "tasks":
            probe, events = await _probe_task_worker_lifecycle(
                _client,
                base_url,
                poll_interval_s=poll_interval_s,
                max_polls=max_polls,
            )
            probes.append(probe)
            captured_events = events
            # W7: task creation idempotency, event-stream shape, cancel.
            probes.append(await _probe_tasks_idempotency(_client, base_url))
            probes.append(
                await _probe_task_event_stream_semantics(_client, base_url)
            )
            probes.append(
                await _probe_task_cancellation(
                    _client, base_url, manifest=manifest,
                )
            )
        elif protocol == "invoke":
            probes.append(
                _skipped(
                    "stream_artifact_lifecycle",
                    "agent runs in simple/invoke mode — covered by error_envelope",
                )
            )
            # W7: oneshot idempotency replay.
            probes.append(await _probe_invoke_idempotency(_client, base_url))
        else:
            probes.append(
                _skipped(
                    "stream_artifact_lifecycle",
                    "could not resolve protocol — pass --type or fix manifest",
                    hint="Pass ``--type oneshot``, ``--type stream``, or ``--type worker``.",
                )
            )

        restart_probe = await _probe_one_shot_restart_replay(
            _client,
            base_url,
            protocol=protocol,
            manifest=manifest,
            restart_hook=restart_hook,
        )
        if restart_probe is not None:
            probes.append(restart_probe)
        restart_probe = await _probe_task_restart_persistence(
            _client,
            base_url,
            protocol=protocol,
            manifest=manifest,
            restart_hook=restart_hook,
            poll_interval_s=poll_interval_s,
            max_polls=max_polls,
        )
        if restart_probe is not None:
            probes.append(restart_probe)

        probes.append(_probe_progress_events(captured_events))
        probes.append(
            await _probe_error_envelope(
                _client,
                base_url,
                has_invoke=(
                    protocol in {"invoke", "stream"}
                    or (manifest is not None and manifest.get("protocol_mode") in {"simple", "stream"})
                ),
            )
        )

        return ConformanceReport(
            base_url=base_url,
            matrix=matrix,
            probes=tuple(probes),
            manifest=manifest,
        )

    if own_client:
        async with httpx.AsyncClient(
            base_url=base_url, timeout=timeout_s,
        ) as fresh:
            return await _do(fresh)
    return await _do(client)


__all__ = [
    "CompatibilityMatrix",
    "ConformanceProbe",
    "ConformanceReport",
    "MANIFEST_SCHEMA_VERSION",
    "PLATFORM_PROTOCOL_VERSION",
    "ProbeStatus",
    "RestartHook",
    "SDK_VERSION",
    "SUPPORTED_AGENT_KINDS",
    "SUPPORTED_PROTOCOL_MODES",
    "current_compatibility",
    "run_conformance",
    "verify_compatibility",
]
