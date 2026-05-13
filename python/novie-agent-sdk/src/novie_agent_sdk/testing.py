"""Reusable test helpers for agent-side HTTP contracts.

These helpers are intentionally small and dependency-light so official and
external agents can share the same protocol assertions without copying the
same checks into every test module.
"""
from __future__ import annotations

import json
from typing import Any

import httpx


async def assert_http_json_healthcheck(
    client: httpx.AsyncClient,
    *,
    health_path: str = "/healthz",
) -> dict[str, Any]:
    """Assert the agent health endpoint follows the minimal A2A expectation."""
    response = await client.get(health_path)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    assert payload.get("status") == "ok"
    return payload


async def assert_http_json_agent_card(
    client: httpx.AsyncClient,
    *,
    expected_agent_id: str,
    card_path: str = "/.well-known/agent.json",
) -> dict[str, Any]:
    """Assert the static agent card endpoint is present and coherent."""
    response = await client.get(card_path)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    assert payload.get("agent_id") == expected_agent_id
    assert payload.get("runtime") == "external_a2a"
    return payload


async def assert_http_json_invoke_contract(
    client: httpx.AsyncClient,
    *,
    payload: dict[str, Any],
    invoke_path: str = "/invoke",
    expected_artifact_type: str | None = None,
) -> dict[str, Any]:
    """Assert `/invoke` returns the A2A simple transport envelope.

    A2A runtime v2 envelope: ``{"output": {...}, "status": "completed"}``.
    """
    response = await client.post(invoke_path, json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, dict), f"Expected JSON object from /invoke; got: {body!r}"
    assert body.get("status") == "completed" and "output" in body, (
        f"/invoke response must have {{status: completed, output: {{...}}}}; "
        f"got: {body!r}"
    )
    output = body.get("output") or {}
    assert isinstance(output, dict), f"output must be a dict; got: {output!r}"
    if expected_artifact_type is not None:
        assert output.get("artifact_type") == expected_artifact_type, (
            f"Expected artifact_type={expected_artifact_type!r}; got {output.get('artifact_type')!r}"
        )
    return body


async def assert_http_json_invoke_idempotency_replay(
    client: httpx.AsyncClient,
    *,
    payload: dict[str, Any],
    idempotency_key: str,
    invoke_path: str = "/invoke",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assert duplicate `POST /invoke` calls replay the first terminal result."""
    headers = {"Idempotency-Key": idempotency_key}
    first = await client.post(invoke_path, json=payload, headers=headers)
    assert first.status_code == 200, first.text
    second = await client.post(invoke_path, json=payload, headers=headers)
    assert second.status_code == 200, second.text
    first_body = first.json()
    second_body = second.json()
    assert first_body == second_body, (
        "duplicate /invoke with the same Idempotency-Key must replay the "
        f"original response; first={first_body!r} second={second_body!r}"
    )
    assert first_body.get("status") == "completed"
    assert "output" in first_body
    return first_body, second_body


async def assert_http_json_stream_contract(
    client: httpx.AsyncClient,
    *,
    payload: dict[str, Any],
    stream_path: str = "/stream",
) -> list[dict[str, Any]]:
    """Assert `/stream` returns NDJSON events.

    A2A runtime v2 terminates the stream with a ``kind=done`` sentinel appended
    by the SDK runtime layer; the payload ``kind=final`` event from the agent
    handler may appear before the sentinel. Both shapes are accepted.
    """
    response = await client.post(stream_path, json=payload)
    assert response.status_code == 200, response.text
    assert response.headers.get("content-type", "").startswith("application/x-ndjson"), (
        f"Expected application/x-ndjson content-type; got: {response.headers.get('content-type')}"
    )
    events = _parse_stream_events(response.text)
    kinds = {e.get("kind") for e in events}
    assert kinds & {"final", "done"}, (
        f"stream must produce at least one final or done event; got kinds: {kinds}"
    )
    return events


async def assert_http_json_stream_idempotency_replay(
    client: httpx.AsyncClient,
    *,
    payload: dict[str, Any],
    idempotency_key: str,
    stream_path: str = "/stream",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assert duplicate `POST /stream` calls replay the same NDJSON transcript."""
    headers = {"Idempotency-Key": idempotency_key}
    first = await _post_stream_events(client, stream_path, payload, headers=headers)
    second = await _post_stream_events(client, stream_path, payload, headers=headers)
    assert first == second, (
        "duplicate /stream with the same Idempotency-Key must replay the "
        f"original event sequence; first={first!r} second={second!r}"
    )
    return first, second


async def assert_http_json_tasks_idempotency_replay(
    client: httpx.AsyncClient,
    *,
    payload: dict[str, Any],
    idempotency_key: str,
    tasks_path: str = "/tasks",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assert duplicate `POST /tasks` calls return the same external task id."""
    headers = {"Idempotency-Key": idempotency_key}
    first = await client.post(tasks_path, json=payload, headers=headers)
    assert first.status_code in (200, 201, 202), first.text
    second = await client.post(tasks_path, json=payload, headers=headers)
    assert second.status_code in (200, 201, 202), second.text
    first_body = first.json()
    second_body = second.json()
    assert first_body.get("task_id") == second_body.get("task_id"), (
        "duplicate /tasks with the same Idempotency-Key must return the "
        f"original task_id; first={first_body!r} second={second_body!r}"
    )
    assert second_body.get("status"), (
        f"duplicate /tasks response must include status: {second_body!r}"
    )
    return first_body, second_body


async def _post_stream_events(
    client: httpx.AsyncClient,
    stream_path: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    response = await client.post(stream_path, json=payload, headers=headers)
    assert response.status_code == 200, response.text
    return _parse_stream_events(response.text)


def _parse_stream_events(text: str) -> list[dict[str, Any]]:
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines, "stream endpoint returned no NDJSON events"
    events: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        events.append(json.loads(line))
    return events
