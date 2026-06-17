from __future__ import annotations

from typing import Any

import httpx
import pytest

from novie_agent_sdk.github_operations import (
    GitHubOperationError,
    build_github_operations_client,
)


@pytest.mark.asyncio
async def test_merge_pull_request_calls_agent_operation_gateway() -> None:
    captured: dict[str, Any] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("authorization")
        captured["json"] = request.read()
        return httpx.Response(
            200,
            json={
                "data": {
                    "merged": True,
                    "repositoryFullName": "org/repo",
                    "pullRequestNumber": 42,
                }
            },
        )

    transport = httpx.MockTransport(responder)
    async with httpx.AsyncClient(transport=transport, base_url="http://platform.test") as http:
        client = build_github_operations_client(
            {"Authorization": "Bearer runtime-token"},
            agent_run_id="run-1",
            base_url="http://platform.test",
            client=http,
        )
        result = await client.merge_pull_request(
            "org/repo",
            42,
            merge_method="squash",
        )

    assert result["merged"] is True
    assert captured["path"] == "/agent-runs/run-1/github/operations/merge-pull-request"
    assert captured["authorization"] == "Bearer runtime-token"
    assert b'"repositoryFullName":"org/repo"' in captured["json"]
    assert b'"mergeMethod":"squash"' in captured["json"]


@pytest.mark.asyncio
async def test_client_rejects_missing_runtime_token() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        base_url="http://platform.test",
    ) as http:
        client = build_github_operations_client(
            {},
            agent_run_id="run-1",
            base_url="http://platform.test",
            client=http,
        )

        with pytest.raises(GitHubOperationError) as exc:
            await client.read_file("org/repo", "README.md")

    assert exc.value.error_code == "github_operation_token_required"


@pytest.mark.asyncio
async def test_pull_request_calls_agent_operation_gateway() -> None:
    captured: dict[str, Any] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("authorization")
        captured["json"] = request.read()
        return httpx.Response(
            200,
            json={
                "data": {
                    "number": 42,
                    "repositoryFullName": "org/repo",
                    "state": "open",
                    "baseRefName": "main",
                }
            },
        )

    transport = httpx.MockTransport(responder)
    async with httpx.AsyncClient(transport=transport, base_url="http://platform.test") as http:
        client = build_github_operations_client(
            {"Authorization": "Bearer runtime-token"},
            agent_run_id="run-1",
            base_url="http://platform.test",
            client=http,
        )
        result = await client.pull_request("org/repo", 42)

    assert result["baseRefName"] == "main"
    assert captured["path"] == "/agent-runs/run-1/github/operations/pull-request"
    assert captured["authorization"] == "Bearer runtime-token"
    assert b'"repositoryFullName":"org/repo"' in captured["json"]
    assert b'"number":42' in captured["json"]


@pytest.mark.asyncio
async def test_ref_calls_agent_operation_gateway() -> None:
    captured: dict[str, Any] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["json"] = request.read()
        return httpx.Response(200, json={"data": {"ref": "refs/heads/main", "treeSha": "tree-sha"}})

    transport = httpx.MockTransport(responder)
    async with httpx.AsyncClient(transport=transport, base_url="http://platform.test") as http:
        client = build_github_operations_client(
            {"Authorization": "Bearer runtime-token"},
            agent_run_id="run-1",
            base_url="http://platform.test",
            client=http,
        )
        result = await client.ref("org/repo", "heads/main")

    assert result["treeSha"] == "tree-sha"
    assert captured["path"] == "/agent-runs/run-1/github/operations/ref"
    assert b'"ref":"heads/main"' in captured["json"]


@pytest.mark.asyncio
async def test_update_pull_request_branch_calls_agent_operation_gateway() -> None:
    captured: dict[str, Any] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["json"] = request.read()
        return httpx.Response(200, json={"data": {"number": 42, "headSha": "new-head"}})

    transport = httpx.MockTransport(responder)
    async with httpx.AsyncClient(transport=transport, base_url="http://platform.test") as http:
        client = build_github_operations_client(
            {"Authorization": "Bearer runtime-token"},
            agent_run_id="run-1",
            base_url="http://platform.test",
            client=http,
        )
        result = await client.update_pull_request_branch(
            "org/repo",
            42,
            expected_head_sha="old-head",
        )

    assert result["headSha"] == "new-head"
    assert captured["path"] == "/agent-runs/run-1/github/operations/update-pull-request-branch"
    assert b'"expectedHeadSha":"old-head"' in captured["json"]


@pytest.mark.asyncio
async def test_gateway_error_maps_to_github_operation_error() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "detail": {
                    "error": "github_repository_not_granted",
                    "message": "denied",
                }
            },
        )

    transport = httpx.MockTransport(responder)
    async with httpx.AsyncClient(transport=transport, base_url="http://platform.test") as http:
        client = build_github_operations_client(
            {"Authorization": "Bearer runtime-token"},
            agent_run_id="run-1",
            base_url="http://platform.test",
            client=http,
        )
        with pytest.raises(GitHubOperationError) as exc:
            await client.read_file("org/other", "README.md")

    assert exc.value.status_code == 403
    assert exc.value.error_code == "github_repository_not_granted"
