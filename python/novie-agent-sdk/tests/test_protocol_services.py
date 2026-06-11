from __future__ import annotations

from novie_agent_sdk import build_gateway_client, build_http_platform_services


def test_build_http_platform_services_returns_none_without_platform_url(monkeypatch) -> None:
    monkeypatch.delenv("NOVIE_PLATFORM_BASE_URL", raising=False)

    services = build_http_platform_services({}, agent_id="demo")

    assert services is None


def test_build_gateway_client_returns_none_without_platform_url(monkeypatch) -> None:
    monkeypatch.delenv("NOVIE_PLATFORM_BASE_URL", raising=False)

    gateway = build_gateway_client({}, agent_id="demo")

    assert gateway is None
