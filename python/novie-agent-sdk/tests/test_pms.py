from __future__ import annotations

from typing import Any

import httpx
import pytest

from novie_agent_sdk.pms import (
    PmsApiError,
    build_pms_issue_client,
    normalize_pms_automation_action,
)


@pytest.mark.asyncio
async def test_list_candidate_issues_calls_pms_api_boundary() -> None:
    captured: dict[str, Any] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("authorization")
        captured["json"] = request.read()
        return httpx.Response(
            200,
            json={
                "data": {
                    "issues": [
                        {
                            "id": "issue-1",
                            "identifier": "NOV-1",
                            "title": "Implement feature",
                            "status": {
                                "id": "status-review",
                                "title": "QA Gate",
                                "stage": "InProgress",
                                "automationAction": "Review",
                            },
                            "organizationId": "tenant-1",
                            "workspaceId": "workspace-1",
                            "projectId": "project-1",
                            "linkedPullRequestUrls": ["https://github.com/org/repo/pull/12"],
                            "agenticOrchestrationValues": {"rework": {"instructionSource": "comment"}},
                        }
                    ]
                }
            },
        )

    transport = httpx.MockTransport(responder)
    async with httpx.AsyncClient(transport=transport, base_url="http://platform.test") as http:
        client = build_pms_issue_client(
            {"Authorization": "Bearer runtime-token"},
            base_url="http://platform.test",
            client=http,
        )
        issues = await client.list_candidate_issues(
            project_ids=["project-1"],
            automation_actions=["Review", "Rework"],
            include_human_review=True,
            organization_id="tenant-1",
            workspace_id="workspace-1",
        )

    assert captured["path"] == "/pms/issues/candidates"
    assert captured["authorization"] == "Bearer runtime-token"
    assert b'"projectIds":["project-1"]' in captured["json"]
    assert b'"automationActions":["Review","Rework"]' in captured["json"]
    assert b'"includeHumanReview":true' in captured["json"]
    assert issues[0].id == "issue-1"
    assert issues[0].automation_action == "Review"
    assert issues[0].status_title == "QA Gate"
    assert issues[0].linked_pr_urls == ("https://github.com/org/repo/pull/12",)
    assert issues[0].agentic_orchestration_values["rework"]["instructionSource"] == "comment"


@pytest.mark.asyncio
async def test_update_agentic_orchestration_values_uses_durable_contract() -> None:
    captured: dict[str, Any] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["json"] = request.read()
        return httpx.Response(
            200,
            json={
                "data": {
                    "agenticOrchestrationValues": {
                        "rework": {"lastDeniedReason": "rework_required_missing_human_input"}
                    }
                }
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(responder),
        base_url="http://platform.test",
    ) as http:
        client = build_pms_issue_client(
            {"Authorization": "Bearer runtime-token"},
            base_url="http://platform.test",
            client=http,
        )
        values = await client.update_agentic_orchestration_values(
            "issue-1",
            patch={"rework": {"lastDeniedReason": "rework_required_missing_human_input"}},
            actor_user_id="agent-user",
            organization_id="tenant-1",
            workspace_id="workspace-1",
        )

    assert captured["path"] == "/pms/issues/update-agentic-orchestration-values"
    assert b'"issueId":"issue-1"' in captured["json"]
    assert b'"actorUserId":"agent-user"' in captured["json"]
    assert values["rework"]["lastDeniedReason"] == "rework_required_missing_human_input"


@pytest.mark.asyncio
async def test_list_comments_maps_created_at_and_author() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "comments": [
                        {
                            "id": "comment-1",
                            "content": "Please rework",
                            "createdAt": "2026-06-19T01:02:03Z",
                            "author": {"id": "human-1", "name": "Human"},
                        }
                    ]
                }
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(responder),
        base_url="http://platform.test",
    ) as http:
        client = build_pms_issue_client(
            {"Authorization": "Bearer runtime-token"},
            base_url="http://platform.test",
            client=http,
        )
        comments = await client.list_comments("issue-1", first=20)

    assert comments[0].id == "comment-1"
    assert comments[0].author_id == "human-1"
    assert comments[0].created_at == "2026-06-19T01:02:03Z"


@pytest.mark.asyncio
async def test_client_rejects_missing_runtime_token() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        base_url="http://platform.test",
    ) as http:
        client = build_pms_issue_client(
            {},
            base_url="http://platform.test",
            client=http,
        )

        with pytest.raises(PmsApiError) as exc:
            await client.get_issue("issue-1")

    assert exc.value.error_code == "pms_api_token_required"


def test_normalize_pms_automation_action_accepts_wire_spellings() -> None:
    assert normalize_pms_automation_action("EPIC_MERGE") == "EpicMerge"
    assert normalize_pms_automation_action("re-work") == "Rework"
