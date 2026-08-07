"""A heartbeat that stops must say so, and must come back.

The accident: on a dev cluster an agent's heartbeat loop stopped at 14:43 and
never resumed. The process stayed up, kept answering /healthz, and kept looking
registered — the only evidence anywhere was a timestamp in the platform's
database that had stopped advancing. Nothing was in the agent's logs, because
every heartbeat failure was `_log.debug` and the loop's exit was not recorded
at all.

So these tests are about what is *said* and what *recovers*, not about the
happy path.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from novie_agent_sdk.runtime import RegistrationClient
from novie_protocol.contracts.agent_sdk_v2 import AgentManifestV2


def _manifest(agent_id: str = "beater") -> AgentManifestV2:
    return AgentManifestV2(
        agent_id=agent_id,
        name="Test",
        version="0.1.0",
        kind="expert_basic",
        runtime="external_a2a",
        capabilities=(),
        declared_gates=(),
        protocol_mode="simple",
        endpoint="http://localhost:8000",
    )


def _client(**kwargs) -> RegistrationClient:
    return RegistrationClient(
        "http://platform", _manifest(), heartbeat_interval=0.01, **kwargs
    )


# ── failures are audible ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_failing_heartbeat_is_logged_at_warning(monkeypatch, caplog):
    """It was `debug`, which in production is the same as silence."""
    client = _client()

    class _Boom:
        async def __aenter__(self):
            raise ConnectionError("platform unreachable")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr("httpx.AsyncClient", lambda **_kw: _Boom())

    with caplog.at_level(logging.WARNING, logger="novie_agent_sdk.runtime"):
        task = asyncio.create_task(client._heartbeat_loop())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "a heartbeat failure produced no warning"
    assert "heartbeat failed" in warnings[0].getMessage()
    assert "beater" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_repeated_failures_report_how_many(monkeypatch, caplog):
    """One blip is noise; a run of them is the agent going dark. The count is
    what distinguishes them in a log."""
    client = _client()

    class _Boom:
        async def __aenter__(self):
            raise ConnectionError("still down")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr("httpx.AsyncClient", lambda **_kw: _Boom())

    with caplog.at_level(logging.WARNING, logger="novie_agent_sdk.runtime"):
        task = asyncio.create_task(client._heartbeat_loop())
        await asyncio.sleep(0.08)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("consecutive_failures=2" in m for m in messages), messages[:3]


@pytest.mark.asyncio
async def test_a_deliberate_stop_is_not_reported_as_a_fault(caplog):
    """Cancellation is shutdown, not death — it must not cry wolf."""
    client = _client()
    with caplog.at_level(logging.INFO, logger="novie_agent_sdk.runtime"):
        await client.start_heartbeat()
        await asyncio.sleep(0.02)
        await client.stop_heartbeat()

    messages = [r.getMessage() for r in caplog.records]
    assert any("cancelled" in m for m in messages)
    assert not any("exited unexpectedly" in m for m in messages)


# ── and the loop comes back ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_loop_that_dies_is_restarted(monkeypatch, caplog):
    """Logging the exit is necessary but not sufficient: nothing else in the
    process notices, and the platform marks the agent stale within minutes."""
    client = _client()
    runs = 0

    async def _die_once() -> None:
        nonlocal runs
        runs += 1
        if runs == 1:
            raise RuntimeError("loop died")
        await asyncio.sleep(3600)

    monkeypatch.setattr(client, "_heartbeat_loop", _die_once)

    with caplog.at_level(logging.WARNING, logger="novie_agent_sdk.runtime"):
        await client.start_heartbeat()
        await asyncio.sleep(0.05)

    assert runs >= 2, "the heartbeat loop was not restarted after it died"
    assert any("restarting it" in r.getMessage() for r in caplog.records)
    await client.stop_heartbeat()


@pytest.mark.asyncio
async def test_a_stopped_client_is_not_resurrected():
    """The watchdog must not fight an intentional shutdown."""
    client = _client()

    async def _sleep_forever() -> None:
        await asyncio.sleep(3600)

    monkeypatch_target = client
    monkeypatch_target._heartbeat_loop = _sleep_forever  # type: ignore[method-assign]

    await client.start_heartbeat()
    await asyncio.sleep(0.02)
    await client.stop_heartbeat()
    await asyncio.sleep(0.05)

    assert client._heartbeat_task is not None
    assert client._heartbeat_task.cancelled()
