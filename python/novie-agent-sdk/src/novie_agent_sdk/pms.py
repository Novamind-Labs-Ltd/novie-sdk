"""Typed PMS issue API client for automation dispatch.

This module is the SDK boundary required by ADR-071. It talks to the
platform-owned PMS API surface; callers must not depend on PMS internal REST
paths or GraphQL documents directly.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from .runtime import RequestHeaders


class PmsApiError(RuntimeError):
    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class PmsStatus:
    id: str
    title: str
    stage: str
    automation_action: str


@dataclass(frozen=True, slots=True)
class PmsComment:
    id: str
    content: str
    author_id: str
    author_name: str = ""
    created_at: str = ""


@dataclass(frozen=True, slots=True)
class PmsIssue:
    id: str
    identifier: str = ""
    title: str = ""
    description: str = ""
    status_id: str = ""
    status_title: str = ""
    status_stage: str = ""
    automation_action: str = ""
    tenant_id: str = ""
    organization_id: str = ""
    workspace_id: str = ""
    project_id: str = ""
    pms_issue_id: str = ""
    linked_pr_urls: tuple[str, ...] = ()
    parent_id: str = ""
    parent_identifier: str = ""
    agentic_orchestration_values: Mapping[str, Any] = field(default_factory=dict)
    comments: tuple[PmsComment, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)


class PmsIssueClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        runtime_token: str | None = None,
        incoming_headers: RequestHeaders | Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = (base_url or os.getenv("NOVIE_PLATFORM_BASE_URL") or "").rstrip("/")
        self._runtime_token = (runtime_token or _runtime_token_from_headers(incoming_headers) or "").strip()
        self._timeout_seconds = timeout_seconds
        self._client = client

    async def list_candidate_issues(
        self,
        *,
        project_ids: Sequence[str],
        automation_actions: Sequence[str],
        include_human_review: bool = False,
        organization_id: str | None = None,
        workspace_id: str | None = None,
    ) -> tuple[PmsIssue, ...]:
        payload = await self._post(
            "/pms/issues/candidates",
            {
                "projectIds": [value for value in project_ids if value],
                "automationActions": [value for value in automation_actions if value],
                "includeHumanReview": include_human_review,
                "organizationId": organization_id,
                "workspaceId": workspace_id,
            },
        )
        rows = _list_field(payload, "issues", "nodes", "items")
        return tuple(pms_issue_from_mapping(row) for row in rows)

    async def get_issue(
        self,
        issue_id: str,
        *,
        organization_id: str | None = None,
        workspace_id: str | None = None,
    ) -> PmsIssue:
        payload = await self._post(
            "/pms/issues/get",
            {
                "issueId": issue_id,
                "organizationId": organization_id,
                "workspaceId": workspace_id,
            },
        )
        row = _mapping_field(payload, "issue") or payload
        return pms_issue_from_mapping(row)

    async def transition_issue_status(
        self,
        issue_id: str,
        *,
        target_status_id: str | None = None,
        automation_action: str | None = None,
        title: str | None = None,
        actor_user_id: str | None = None,
        reason: str | None = None,
        organization_id: str | None = None,
        workspace_id: str | None = None,
    ) -> PmsIssue:
        payload = await self._post(
            "/pms/issues/transition-status",
            {
                "issueId": issue_id,
                "targetStatusId": target_status_id,
                "automationAction": automation_action,
                "title": title,
                "actorUserId": actor_user_id,
                "reason": reason,
                "organizationId": organization_id,
                "workspaceId": workspace_id,
            },
        )
        row = _mapping_field(payload, "issue") or payload
        return pms_issue_from_mapping(row)

    async def update_agentic_orchestration_values(
        self,
        issue_id: str,
        *,
        patch: Mapping[str, Any],
        actor_user_id: str | None = None,
        organization_id: str | None = None,
        workspace_id: str | None = None,
    ) -> Mapping[str, Any]:
        payload = await self._post(
            "/pms/issues/update-agentic-orchestration-values",
            {
                "issueId": issue_id,
                "patch": dict(patch),
                "actorUserId": actor_user_id,
                "organizationId": organization_id,
                "workspaceId": workspace_id,
            },
        )
        return _mapping_field(payload, "agenticOrchestrationValues") or payload

    async def add_comment(
        self,
        issue_id: str,
        content: str,
        *,
        author_id: str,
        organization_id: str | None = None,
        workspace_id: str | None = None,
    ) -> PmsComment:
        payload = await self._post(
            "/pms/issues/add-comment",
            {
                "issueId": issue_id,
                "content": content,
                "authorId": author_id,
                "organizationId": organization_id,
                "workspaceId": workspace_id,
            },
        )
        row = _mapping_field(payload, "comment") or payload
        return pms_comment_from_mapping(row)

    async def upsert_workpad_comment(
        self,
        issue_id: str,
        *,
        marker: str,
        content: str,
        author_id: str,
        organization_id: str | None = None,
        workspace_id: str | None = None,
    ) -> PmsComment:
        payload = await self._post(
            "/pms/issues/upsert-workpad-comment",
            {
                "issueId": issue_id,
                "marker": marker,
                "content": content,
                "authorId": author_id,
                "organizationId": organization_id,
                "workspaceId": workspace_id,
            },
        )
        row = _mapping_field(payload, "comment") or payload
        return pms_comment_from_mapping(row)

    async def list_comments(
        self,
        issue_id: str,
        *,
        first: int = 20,
        organization_id: str | None = None,
        workspace_id: str | None = None,
    ) -> tuple[PmsComment, ...]:
        payload = await self._post(
            "/pms/issues/comments",
            {
                "issueId": issue_id,
                "first": first,
                "organizationId": organization_id,
                "workspaceId": workspace_id,
            },
        )
        rows = _list_field(payload, "comments", "nodes", "items")
        return tuple(pms_comment_from_mapping(row) for row in rows)

    async def _post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self._base_url and self._client is None:
            raise PmsApiError("pms_api_unconfigured", "NOVIE_PLATFORM_BASE_URL is not set.")
        if not self._runtime_token:
            raise PmsApiError("pms_api_token_required", "A runtime token is required for PMS API calls.")
        body = {key: value for key, value in payload.items() if value is not None}
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._runtime_token}",
        }
        try:
            if self._client is not None:
                response = await self._client.post(
                    path,
                    json=body,
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
            else:
                async with httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=self._timeout_seconds,
                ) as client:
                    response = await client.post(path, json=body, headers=headers)
        except httpx.TransportError as exc:
            raise PmsApiError("pms_api_transport_error", str(exc)) from exc
        return _parse_response(response)


def build_pms_issue_client(
    incoming_headers: RequestHeaders | Mapping[str, str],
    *,
    base_url: str | None = None,
    timeout_seconds: float = 30.0,
    client: httpx.AsyncClient | None = None,
) -> PmsIssueClient:
    return PmsIssueClient(
        incoming_headers=incoming_headers,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        client=client,
    )


def pms_issue_from_mapping(data: Mapping[str, Any]) -> PmsIssue:
    status = _mapping_field(data, "status") or {}
    comments = _list_field(data, "comments", "recentComments")
    return PmsIssue(
        id=_str_field(data, "id"),
        identifier=_str_field(data, "identifier", "issueNumber", "issue_number"),
        title=_str_field(data, "title"),
        description=_str_field(data, "description"),
        status_id=_str_field(data, "statusId", "status_id") or _str_field(status, "id"),
        status_title=_str_field(data, "statusTitle", "status_title") or _str_field(status, "title"),
        status_stage=_str_field(data, "statusStage", "status_stage") or _str_field(status, "stage"),
        automation_action=normalize_pms_automation_action(
            _str_field(data, "automationAction", "automation_action")
            or _str_field(status, "automationAction", "automation_action")
        ),
        tenant_id=_str_field(data, "tenantId", "tenant_id", "organizationId", "organization_id"),
        organization_id=_str_field(data, "organizationId", "organization_id", "tenantId", "tenant_id"),
        workspace_id=_str_field(data, "workspaceId", "workspace_id"),
        project_id=_str_field(data, "projectId", "project_id") or _str_field(_mapping_field(data, "project") or {}, "id"),
        pms_issue_id=_str_field(data, "pmsIssueId", "pms_issue_id", "id"),
        linked_pr_urls=tuple(_string_list_field(data, "linkedPrUrls", "linkedPRUrls", "linkedPullRequestUrls", "linked_pr_urls")),
        parent_id=_str_field(data, "parentId", "parent_id") or _str_field(_mapping_field(data, "parent") or {}, "id"),
        parent_identifier=_str_field(data, "parentIdentifier", "parent_identifier")
        or _str_field(_mapping_field(data, "parent") or {}, "identifier"),
        agentic_orchestration_values=_mapping_field(
            data,
            "agenticOrchestrationValues",
            "agentic_orchestration_values",
        )
        or {},
        comments=tuple(pms_comment_from_mapping(row) for row in comments),
        raw=dict(data),
    )


def pms_comment_from_mapping(data: Mapping[str, Any]) -> PmsComment:
    author = _mapping_field(data, "author") or {}
    return PmsComment(
        id=_str_field(data, "id", "commentId", "comment_id"),
        content=_str_field(data, "content", "body"),
        author_id=_str_field(data, "authorId", "author_id") or _str_field(author, "id"),
        author_name=_str_field(data, "authorName", "author_name") or _str_field(author, "name"),
        created_at=_str_field(data, "createdAt", "created_at"),
    )


def normalize_pms_automation_action(value: str | None) -> str:
    normalized = (value or "").replace("_", "").replace("-", "").replace(" ", "").lower()
    return {
        "none": "None",
        "execute": "Execute",
        "review": "Review",
        "rework": "Rework",
        "merge": "Merge",
        "epicmerge": "EpicMerge",
    }.get(normalized, value or "")


def _parse_response(response: httpx.Response) -> Mapping[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise PmsApiError(
            "pms_api_invalid_response",
            "PMS API returned non-JSON.",
            status_code=response.status_code,
        ) from exc
    if response.status_code >= 400:
        detail = payload.get("detail") if isinstance(payload, Mapping) else {}
        if isinstance(detail, Mapping):
            error_code = str(detail.get("error") or "pms_api_failed")
            message = str(detail.get("message") or error_code)
        else:
            error_code = "pms_api_failed"
            message = str(detail or error_code)
        raise PmsApiError(error_code, message, status_code=response.status_code)
    if not isinstance(payload, Mapping):
        raise PmsApiError(
            "pms_api_invalid_response",
            "PMS API response must be a JSON object.",
            status_code=response.status_code,
        )
    data = payload.get("data")
    if isinstance(data, Mapping):
        return data
    return payload


def _runtime_token_from_headers(
    incoming_headers: RequestHeaders | Mapping[str, str] | None,
) -> str:
    if incoming_headers is None:
        return ""
    headers = (
        incoming_headers.as_dict()
        if isinstance(incoming_headers, RequestHeaders)
        else dict(incoming_headers)
    )
    lower = {str(key).lower(): str(value) for key, value in headers.items()}
    authorization = lower.get("authorization", "").strip()
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return ""


def _mapping_field(data: Mapping[str, Any], *names: str) -> Mapping[str, Any] | None:
    for name in names:
        value = data.get(name)
        if isinstance(value, Mapping):
            return value
    return None


def _list_field(data: Mapping[str, Any], *names: str) -> tuple[Mapping[str, Any], ...]:
    for name in names:
        value = data.get(name)
        if isinstance(value, list):
            return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _string_list_field(data: Mapping[str, Any], *names: str) -> tuple[str, ...]:
    for name in names:
        value = data.get(name)
        if isinstance(value, list):
            return tuple(str(item) for item in value if str(item).strip())
    return ()


def _str_field(data: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = data.get(name)
        if value is not None:
            return str(value)
    return ""


__all__ = [
    "PmsApiError",
    "PmsComment",
    "PmsIssue",
    "PmsIssueClient",
    "PmsStatus",
    "build_pms_issue_client",
    "normalize_pms_automation_action",
    "pms_comment_from_mapping",
    "pms_issue_from_mapping",
]
