"""Helpers for agent-to-platform callback calls.

External agents receive platform-signed A2A headers on ``/invoke``, ``/stream``,
or ``/tasks``. When they call back into platform capabilities, they should
forward the same tenant/session/principal boundary instead of inventing local
headers. This module centralizes that mapping for all Python agents.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .platform_security import sign_agent_platform_headers
from .runtime import RequestHeaders


def build_platform_callback_headers(
    incoming: RequestHeaders | Mapping[str, str],
    *,
    agent_id: str,
    auth_source: str = "agent_callback",
) -> dict[str, str]:
    """Build unsigned trusted headers for agent -> platform callbacks."""
    source = _IncomingHeaders(incoming)
    org_id = source.pick("x-novie-org-id", "x-novie-tenant-id") or os.getenv(
        "NOVIE_ORG_ID", ""
    ).strip()
    workspace_id = source.pick("x-novie-workspace-id") or org_id
    project_id = (
        source.pick("x-novie-project-id")
        or os.getenv("NOVIE_PROJECT_ID", "").strip()
        or workspace_id
        or org_id
    )
    user_id = source.pick("x-novie-user-id") or os.getenv("NOVIE_USER_ID", "").strip()
    service_principal = source.pick("x-novie-service-principal")
    if not user_id and not service_principal:
        service_principal = f"agent:{agent_id}"
    request_id = (
        source.pick("x-novie-request-id")
        or source.pick("x-novie-trace-id")
        or source.pick("x-novie-step-id")
    )

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-novie-org-id": org_id,
        "x-novie-project-id": project_id,
        "x-novie-session-id": source.pick("x-novie-session-id"),
        "x-novie-request-id": request_id,
        "x-novie-auth-source": auth_source,
    }
    if workspace_id:
        headers["x-novie-workspace-id"] = workspace_id
    if user_id:
        headers["x-novie-user-id"] = user_id
    else:
        headers["x-novie-service-principal"] = service_principal
    on_behalf_of_user_id = source.pick("x-novie-on-behalf-of-user-id")
    if on_behalf_of_user_id:
        headers["x-novie-on-behalf-of-user-id"] = on_behalf_of_user_id
    for incoming_name, outgoing_name in (
        ("x-novie-workflow-id", "x-novie-workflow-id"),
        ("x-novie-thread-id", "x-novie-thread-id"),
        ("x-novie-step-id", "x-novie-step-id"),
    ):
        value = source.pick(incoming_name)
        if value:
            headers[outgoing_name] = value
    return headers


def sign_platform_callback_headers(
    headers: Mapping[str, str],
    *,
    method: str,
    path: str,
    secret: str | None = None,
    timestamp: str | None = None,
) -> dict[str, str]:
    """Return headers with ``x-novie-timestamp`` and ``x-novie-sig``.

    The canonical string matches the Novie agent-platform signed envelope.
    """
    return sign_agent_platform_headers(
        headers,
        method=method,
        path=path,
        secret=secret,
        timestamp=timestamp,
    )


class PlatformCallbackClient:
    """Small async client for ``POST /invocations`` callbacks.

    Returns the raw platform envelope dict (not a
    ``CapabilityCallDiagnostics``). The envelope's success payload key is
    ``output`` (was ``result`` on the legacy ``/capabilities/{id}/invoke``
    route) — any consumer reading the return value must use
    ``envelope["output"]``, not ``envelope["result"]``.
    """

    def __init__(
        self,
        base_url: str,
        incoming_headers: RequestHeaders | Mapping[str, str],
        *,
        agent_id: str,
        timeout_seconds: float = 30.0,
        client: Any | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = build_platform_callback_headers(
            incoming_headers,
            agent_id=agent_id,
        )
        self._agent_id = agent_id
        self._timeout_seconds = timeout_seconds
        self._client = client

    async def invoke_capability(
        self,
        capability_id: str,
        arguments: dict[str, Any],
        *,
        caller_mode: str = "execute",
    ) -> dict[str, Any]:
        import httpx

        # `caller_mode` is kept for signature stability only: legacy values
        # (interactive/preview/delegated) don't map onto /invocations' mode
        # vocabulary, and nothing calls this with a non-default value.
        del caller_mode
        path = "/invocations"
        headers = sign_platform_callback_headers(
            self._headers,
            method="POST",
            path=path,
        )
        body = {
            "capability_id": capability_id,
            "provider_id": capability_id.rsplit(".", 1)[0],
            "mode": "execute",
            "inputs": arguments,
        }
        if self._client is not None:
            response = await self._client.post(path, json=body, headers=headers)
        else:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
            ) as client:
                response = await client.post(path, json=body, headers=headers)
        response.raise_for_status()
        parsed = response.json()
        return parsed if isinstance(parsed, dict) else {"output": parsed}


class _IncomingHeaders:
    def __init__(self, source: RequestHeaders | Mapping[str, str]) -> None:
        if isinstance(source, RequestHeaders):
            self._items = {
                "x-novie-tenant-id": source.tenant_id,
                "x-novie-workspace-id": source.workspace_id,
                "x-novie-project-id": source.project_id,
                "x-novie-user-id": source.user_id,
                "x-novie-service-principal": source.service_principal,
                "x-novie-session-id": source.session_id,
                "x-novie-request-id": source.request_id,
                "x-novie-trace-id": source.trace_id,
                "x-novie-step-id": source.step_id,
            }
            self._items.update(
                {str(k).lower(): str(v) for k, v in source.raw.items()}
            )
        else:
            self._items = {str(k).lower(): str(v) for k, v in source.items()}

    def pick(self, *keys: str) -> str:
        for key in keys:
            value = str(self._items.get(key.lower()) or "").strip()
            if value:
                return value
        return ""
