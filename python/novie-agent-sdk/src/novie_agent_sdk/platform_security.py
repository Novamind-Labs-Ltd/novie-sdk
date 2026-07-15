"""Shared agent-platform signing helpers."""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from collections.abc import Mapping
from urllib.parse import urlsplit

DEV_AGENT_PLATFORM_SHARED_SECRET = "novie-dev-agent-platform-shared-secret"


class AgentPlatformSignatureError(PermissionError):
    """Raised when an agent-platform signed envelope is missing or invalid."""

    def __init__(self, code: str, reason: str = "") -> None:
        super().__init__(reason or code)
        self.code = code
        self.reason = reason or code


def agent_platform_shared_secret(explicit: str | None = None) -> str:
    if explicit is not None and explicit.strip():
        return explicit.strip()
    configured = os.getenv("NOVIE_AGENT_PLATFORM_SHARED_SECRET", "").strip()
    if configured:
        return configured
    if _runtime_mode() == "production":
        raise RuntimeError("NOVIE_AGENT_PLATFORM_SHARED_SECRET is not configured")
    return DEV_AGENT_PLATFORM_SHARED_SECRET


def sign_agent_platform_headers(
    headers: Mapping[str, str],
    *,
    method: str,
    path: str,
    secret: str | None = None,
    timestamp: str | None = None,
) -> dict[str, str]:
    out = dict(headers)
    ts = timestamp or str(int(time.time()))
    out["x-novie-timestamp"] = ts
    signature = agent_platform_signature(
        out,
        method=method,
        path=path,
        secret=agent_platform_shared_secret(secret),
        timestamp=ts,
    )
    out["x-novie-sig"] = f"sha256={signature}"
    return out


def verify_agent_platform_headers(
    headers: Mapping[str, str],
    *,
    method: str,
    path: str,
    secret: str | None = None,
    ttl_seconds: int = 300,
) -> None:
    values = _lower_headers(headers)
    timestamp = values.get("x-novie-timestamp", "")
    provided = values.get("x-novie-sig", "").removeprefix("sha256=").strip()
    if not timestamp or not provided:
        raise AgentPlatformSignatureError(
            "agent_platform_signature_required",
            "x-novie-timestamp and x-novie-sig are required",
        )
    try:
        issued_at = int(timestamp)
    except ValueError as exc:
        raise AgentPlatformSignatureError(
            "invalid_agent_platform_signature_timestamp",
        ) from exc
    if abs(int(time.time()) - issued_at) > max(1, ttl_seconds):
        raise AgentPlatformSignatureError("stale_agent_platform_signature")
    expected = agent_platform_signature(
        values,
        method=method,
        path=path,
        secret=agent_platform_shared_secret(secret),
        timestamp=timestamp,
    )
    if not hmac.compare_digest(expected, provided):
        raise AgentPlatformSignatureError("invalid_agent_platform_signature")


def agent_platform_signature(
    headers: Mapping[str, str],
    *,
    method: str,
    path: str,
    secret: str,
    timestamp: str,
) -> str:
    values = _lower_headers(headers)
    canonical_parts = [
        method.upper(),
        _normalize_path(path),
        values.get("x-novie-org-id") or values.get("x-novie-tenant-id", ""),
        values.get("x-novie-project-id", ""),
        values.get("x-novie-workspace-id", ""),
        values.get("x-novie-user-id", ""),
        values.get("x-novie-service-principal", ""),
        values.get("x-novie-session-id", ""),
        values.get("x-novie-request-id", ""),
        timestamp,
    ]
    on_behalf_of_user_id = values.get("x-novie-on-behalf-of-user-id", "")
    if on_behalf_of_user_id:
        canonical_parts.append(on_behalf_of_user_id)
    canonical = "\n".join(canonical_parts)
    return hmac.new(
        secret.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _runtime_mode() -> str:
    mode = os.getenv("NOVIE_RUNTIME_MODE", "").strip().lower()
    if mode:
        return mode
    if os.getenv("NOVIE_ENV", "").strip().lower() == "production":
        return "production"
    return "development"


def _lower_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _normalize_path(path: str) -> str:
    if not path:
        return "/"
    if path.startswith("http://") or path.startswith("https://"):
        parsed = urlsplit(path)
        return parsed.path or "/"
    return path if path.startswith("/") else f"/{path}"
