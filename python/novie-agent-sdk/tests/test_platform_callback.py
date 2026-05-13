from __future__ import annotations

import json

import httpx
import pytest
from novie_agent_sdk.platform_callback import (
    PlatformCallbackClient,
    build_platform_callback_headers,
    sign_platform_callback_headers,
)
from novie_agent_sdk.runtime import RequestHeaders


def test_build_callback_headers_forward_user_identity() -> None:
    headers = build_platform_callback_headers(
        RequestHeaders(
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            project_id="project-1",
            user_id="user-1",
            session_id="session-1",
            trace_id="trace-1",
        ),
        agent_id="analyst",
    )

    assert headers["x-novie-org-id"] == "tenant-1"
    assert headers["x-novie-workspace-id"] == "workspace-1"
    assert headers["x-novie-project-id"] == "project-1"
    assert headers["x-novie-user-id"] == "user-1"
    assert "x-novie-service-principal" not in headers
    assert headers["x-novie-request-id"] == "trace-1"


def test_build_callback_headers_falls_back_to_service_principal() -> None:
    headers = build_platform_callback_headers(
        {"x-novie-tenant-id": "tenant-1", "x-novie-project-id": "project-1"},
        agent_id="analyst",
    )

    assert headers["x-novie-service-principal"] == "agent:analyst"
    assert "x-novie-user-id" not in headers


def test_sign_callback_headers_matches_gateway_canonical_shape() -> None:
    headers = sign_platform_callback_headers(
        {
            "x-novie-org-id": "tenant-1",
            "x-novie-project-id": "project-1",
            "x-novie-user-id": "user-1",
            "x-novie-session-id": "session-1",
            "x-novie-request-id": "request-1",
        },
        method="POST",
        path="/capabilities/platform.knowledge.search/invoke",
        secret="secret",
        timestamp="100",
    )

    assert headers["x-novie-timestamp"] == "100"
    assert headers["x-novie-sig"].startswith("sha256=")


@pytest.mark.asyncio
async def test_platform_callback_client_invokes_capability_with_signed_headers(monkeypatch) -> None:
    monkeypatch.setenv("NOVIE_TRUSTED_HEADER_SECRET", "secret")
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"ok": True, "result": {"count": 1}})

    client = httpx.AsyncClient(
        base_url="http://platform.test",
        transport=httpx.MockTransport(handler),
    )
    callback = PlatformCallbackClient(
        "http://platform.test",
        RequestHeaders(
            tenant_id="tenant-1",
            project_id="project-1",
            user_id="user-1",
            session_id="session-1",
            request_id="request-1",
        ),
        agent_id="analyst",
        client=client,
    )

    try:
        result = await callback.invoke_capability(
            "platform.knowledge.search",
            {"query": "architecture"},
        )
    finally:
        await client.aclose()

    assert result["result"] == {"count": 1}
    assert seen["path"] == "/capabilities/platform.knowledge.search/invoke"
    headers = seen["headers"]
    assert isinstance(headers, dict)
    assert headers["x-novie-user-id"] == "user-1"
    assert headers["x-novie-sig"].startswith("sha256=")
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["caller_id"] == "agent:analyst"
