"""Helpers for agent-to-platform callback calls.

External agents receive platform-signed A2A headers on ``/invoke``, ``/stream``,
or ``/tasks``. When they call back into platform capabilities, they should
forward the same tenant/session/principal boundary instead of inventing local
headers. This module centralizes that mapping for all Python agents.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from collections.abc import Mapping
from typing import Any

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
    workspace_id = source.pick("x-novie-workspace-id")
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

    The canonical string matches ``novie_platform.gateway.api.deps``.
    """
    signing_secret = (
        secret if secret is not None else os.getenv("NOVIE_TRUSTED_HEADER_SECRET", "")
    ).strip()
    out = dict(headers)
    if not signing_secret:
        return out
    ts = timestamp or str(int(time.time()))
    out["x-novie-timestamp"] = ts
    canonical = "\n".join(
        [
            method.upper(),
            _normalize_path(path),
            out.get("x-novie-org-id", ""),
            out.get("x-novie-project-id", ""),
            out.get("x-novie-user-id", ""),
            out.get("x-novie-service-principal", ""),
            out.get("x-novie-session-id", ""),
            out.get("x-novie-request-id", ""),
            ts,
        ]
    )
    signature = hmac.new(
        signing_secret.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    out["x-novie-sig"] = f"sha256={signature}"
    return out


class PlatformCallbackClient:
    """Small async client for ``POST /capabilities/{id}/invoke`` callbacks."""

    def __init__(
        self,
        base_url: str,
        incoming_headers: RequestHeaders | Mapping[str, str],
        *,
        agent_id: str,
        timeout_seconds: float = 8.0,
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

        path = f"/capabilities/{capability_id}/invoke"
        headers = sign_platform_callback_headers(
            self._headers,
            method="POST",
            path=path,
        )
        body = {
            "arguments": arguments,
            "caller_type": "agent",
            "caller_id": f"agent:{self._agent_id}",
            "caller_mode": caller_mode,
            "mode": caller_mode,
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
        return parsed if isinstance(parsed, dict) else {"result": parsed}


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
        else:
            self._items = {str(k).lower(): str(v) for k, v in source.items()}

    def pick(self, *keys: str) -> str:
        for key in keys:
            value = str(self._items.get(key.lower()) or "").strip()
            if value:
                return value
        return ""


def _normalize_path(path: str) -> str:
    if not path:
        return "/"
    if path.startswith("http://") or path.startswith("https://"):
        from urllib.parse import urlsplit

        parsed = urlsplit(path)
        return parsed.path or "/"
    return path if path.startswith("/") else f"/{path}"
