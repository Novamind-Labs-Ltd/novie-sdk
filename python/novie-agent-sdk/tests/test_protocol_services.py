from __future__ import annotations

import httpx
import pytest

from novie_agent_sdk import (
    ExternalAgentCheckpointPutError,
    build_gateway_client,
    build_http_platform_services,
)
from novie_agent_sdk.protocol_services import HttpExternalAgentCheckpointService
from novie_protocol.contracts import ExecutionContext, IdentityContext, TenantScope

# Import from the sibling test module directly by name (not a relative
# "from .test_platform_namespace import ..." -- this package has no
# tests/__init__.py, so pytest's default "prepend" import mode treats every
# test file as a standalone top-level module; a relative import fails with
# "attempted relative import with no known parent package". pytest still
# puts tests/ on sys.path in this mode, so the bare module name works.
from test_platform_namespace import _build_with_responder


def test_build_http_platform_services_returns_none_without_platform_url(monkeypatch) -> None:
    monkeypatch.delenv("NOVIE_PLATFORM_BASE_URL", raising=False)

    services = build_http_platform_services({}, agent_id="demo")

    assert services is None


def test_build_gateway_client_returns_none_without_platform_url(monkeypatch) -> None:
    monkeypatch.delenv("NOVIE_PLATFORM_BASE_URL", raising=False)

    gateway = build_gateway_client({}, agent_id="demo")

    assert gateway is None


def _ctx() -> ExecutionContext:
    return ExecutionContext(
        request_id="req-1",
        session_id="sess-1",
        thread_id="thread-1",
        tenant=TenantScope(tenant_id="tenant-1", workspace_id="workspace-1"),
        identity=IdentityContext(principal_id="agent:demo", principal_type="service"),
    )


@pytest.mark.asyncio
async def test_protocol_checkpoint_service_put_propagates_raise_on_binding_denied() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error_code": "denied_by_binding"})

    namespace = _build_with_responder(responder)
    service = HttpExternalAgentCheckpointService(namespace)

    with pytest.raises(ExternalAgentCheckpointPutError) as excinfo:
        await service.put(
            _ctx(),
            owner_agent_id="demo",
            thread_id="thread-1",
            payload={"phase": "x"},
        )

    assert excinfo.value.kind == "binding_denied"


@pytest.mark.asyncio
async def test_protocol_checkpoint_service_put_returns_record_on_success() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "output": {"checkpoint": {"checkpoint_id": "ck-1", "thread_id": "thread-1"}},
            },
        )

    namespace = _build_with_responder(responder)
    service = HttpExternalAgentCheckpointService(namespace)

    record = await service.put(
        _ctx(),
        owner_agent_id="demo",
        thread_id="thread-1",
        payload={"phase": "x"},
    )

    assert record.checkpoint_id == "ck-1"
