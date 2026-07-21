from __future__ import annotations

from novie_agent_sdk.timeout_policy import (
    DEFAULT_SDK_TIMEOUTS,
    invocation_lease_renewal_seconds,
)


def test_sdk_timeout_profile_separates_business_requests_from_liveness() -> None:
    assert DEFAULT_SDK_TIMEOUTS.capability_request_seconds == 8
    assert DEFAULT_SDK_TIMEOUTS.llm_request_seconds == 120
    assert DEFAULT_SDK_TIMEOUTS.llm_stream_read_idle_seconds == 60
    assert DEFAULT_SDK_TIMEOUTS.stream_keepalive_seconds == 25
    assert DEFAULT_SDK_TIMEOUTS.invocation_lease_seconds == 300


def test_invocation_lease_renewal_is_time_based_and_bounded_by_lease() -> None:
    assert invocation_lease_renewal_seconds(300) == 60
    assert invocation_lease_renewal_seconds(30) == 10
