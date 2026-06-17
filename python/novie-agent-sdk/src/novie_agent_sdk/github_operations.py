"""GitHub operations client for Novie SDK agents.

The SDK talks to the agent-operation gateway. GitHub tokens and CCTs stay inside
the platform/GHIS boundary.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import httpx

from .runtime import RequestHeaders


class GitHubOperationError(RuntimeError):
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


class GitHubOperationsClient:
    def __init__(
        self,
        *,
        agent_run_id: str,
        base_url: str | None = None,
        runtime_token: str | None = None,
        incoming_headers: RequestHeaders | Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = (base_url or os.getenv("NOVIE_PLATFORM_BASE_URL") or "").rstrip("/")
        self._agent_run_id = agent_run_id.strip()
        self._runtime_token = (runtime_token or _runtime_token_from_headers(incoming_headers) or "").strip()
        self._timeout_seconds = timeout_seconds
        self._client = client

    async def read_file(
        self,
        repository_full_name: str,
        path: str,
        *,
        ref: str | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            "read-file",
            {
                "repositoryFullName": repository_full_name,
                "path": path,
                "ref": ref,
                "maxBytes": max_bytes,
            },
        )

    async def search_code(
        self,
        repository_full_name: str,
        query: str,
        *,
        ref: str | None = None,
        max_results: int | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            "search-code",
            {
                "repositoryFullName": repository_full_name,
                "query": query,
                "ref": ref,
                "maxResults": max_results,
            },
        )

    async def ref(
        self,
        repository_full_name: str,
        ref: str,
    ) -> dict[str, Any]:
        return await self._post(
            "ref",
            {
                "repositoryFullName": repository_full_name,
                "ref": ref,
            },
        )

    async def tree(
        self,
        repository_full_name: str,
        tree_sha: str,
        *,
        recursive: bool = True,
    ) -> dict[str, Any]:
        return await self._post(
            "tree",
            {
                "repositoryFullName": repository_full_name,
                "treeSha": tree_sha,
                "recursive": recursive,
            },
        )

    async def blob(
        self,
        repository_full_name: str,
        sha: str,
        *,
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            "blob",
            {
                "repositoryFullName": repository_full_name,
                "sha": sha,
                "maxBytes": max_bytes,
            },
        )

    async def create_branch(
        self,
        repository_full_name: str,
        name: str,
        from_ref: str,
    ) -> dict[str, Any]:
        return await self._post(
            "create-branch",
            {
                "repositoryFullName": repository_full_name,
                "name": name,
                "fromRef": from_ref,
            },
        )

    async def commit_files(
        self,
        repository_full_name: str,
        branch: str,
        message: str,
        files: list[dict[str, str]],
        *,
        expected_head_sha: str | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            "commit-files",
            {
                "repositoryFullName": repository_full_name,
                "branch": branch,
                "message": message,
                "files": files,
                "expectedHeadSha": expected_head_sha,
            },
        )

    async def create_pull_request(
        self,
        repository_full_name: str,
        title: str,
        *,
        head: str,
        base: str,
        body: str | None = None,
        draft: bool = False,
    ) -> dict[str, Any]:
        return await self._post(
            "create-pull-request",
            {
                "repositoryFullName": repository_full_name,
                "title": title,
                "body": body,
                "head": head,
                "base": base,
                "draft": draft,
            },
        )

    async def comment_pull_request(
        self,
        repository_full_name: str,
        number: int,
        body: str,
    ) -> dict[str, Any]:
        return await self._post(
            "comment-pull-request",
            {
                "repositoryFullName": repository_full_name,
                "number": number,
                "body": body,
            },
        )

    async def pull_request(
        self,
        repository_full_name: str,
        number: int,
    ) -> dict[str, Any]:
        return await self._post(
            "pull-request",
            {
                "repositoryFullName": repository_full_name,
                "number": number,
            },
        )

    async def list_pull_requests(
        self,
        repository_full_name: str,
        *,
        head: str | None = None,
        base: str | None = None,
        state: str | None = None,
        first: int | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            "list-pull-requests",
            {
                "repositoryFullName": repository_full_name,
                "head": head,
                "base": base,
                "state": state,
                "first": first,
            },
        )

    async def pull_request_checks(
        self,
        repository_full_name: str,
        number: int,
    ) -> dict[str, Any]:
        return await self._post(
            "pull-request-checks",
            {
                "repositoryFullName": repository_full_name,
                "number": number,
            },
        )

    async def update_pull_request(
        self,
        repository_full_name: str,
        number: int,
        *,
        title: str | None = None,
        body: str | None = None,
        base: str | None = None,
        draft: bool | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            "update-pull-request",
            {
                "repositoryFullName": repository_full_name,
                "number": number,
                "title": title,
                "body": body,
                "base": base,
                "draft": draft,
            },
        )

    async def reopen_pull_request(
        self,
        repository_full_name: str,
        number: int,
    ) -> dict[str, Any]:
        return await self._post(
            "reopen-pull-request",
            {
                "repositoryFullName": repository_full_name,
                "number": number,
            },
        )

    async def add_pull_request_labels(
        self,
        repository_full_name: str,
        number: int,
        labels: list[str],
    ) -> dict[str, Any]:
        return await self._post(
            "add-pull-request-labels",
            {
                "repositoryFullName": repository_full_name,
                "number": number,
                "labels": labels,
            },
        )

    async def update_pull_request_branch(
        self,
        repository_full_name: str,
        number: int,
        *,
        expected_head_sha: str | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            "update-pull-request-branch",
            {
                "repositoryFullName": repository_full_name,
                "number": number,
                "expectedHeadSha": expected_head_sha,
            },
        )

    async def merge_pull_request(
        self,
        repository_full_name: str,
        number: int,
        *,
        merge_method: str | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            "merge-pull-request",
            {
                "repositoryFullName": repository_full_name,
                "number": number,
                "mergeMethod": merge_method,
            },
        )

    async def dispatch_workflow(
        self,
        repository_full_name: str,
        workflow_id: str,
        ref: str,
        *,
        inputs: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return await self._post(
            "dispatch-workflow",
            {
                "repositoryFullName": repository_full_name,
                "workflowId": workflow_id,
                "ref": ref,
                "inputs": inputs,
            },
        )

    async def _post(self, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not self._base_url:
            raise GitHubOperationError(
                "github_operation_unconfigured",
                "NOVIE_PLATFORM_BASE_URL is not set.",
            )
        if not self._agent_run_id:
            raise GitHubOperationError(
                "github_operation_agent_run_required",
                "An agent_run_id is required for GitHub operations.",
            )
        if not self._runtime_token:
            raise GitHubOperationError(
                "github_operation_token_required",
                "An agent runtime token is required for GitHub operations.",
            )
        path = f"/agent-runs/{self._agent_run_id}/github/operations/{operation}"
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
            raise GitHubOperationError(
                "github_operation_transport_error",
                str(exc),
            ) from exc
        return _parse_response(response)


def build_github_operations_client(
    incoming_headers: RequestHeaders | Mapping[str, str],
    *,
    agent_run_id: str,
    base_url: str | None = None,
    timeout_seconds: float = 30.0,
    client: httpx.AsyncClient | None = None,
) -> GitHubOperationsClient:
    return GitHubOperationsClient(
        agent_run_id=agent_run_id,
        incoming_headers=incoming_headers,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        client=client,
    )


def _parse_response(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise GitHubOperationError(
            "github_operation_invalid_response",
            "GitHub operation gateway returned non-JSON.",
            status_code=response.status_code,
        ) from exc
    if response.status_code >= 400:
        detail = payload.get("detail") if isinstance(payload, dict) else {}
        if isinstance(detail, Mapping):
            error_code = str(detail.get("error") or "github_operation_failed")
            message = str(detail.get("message") or error_code)
        else:
            error_code = "github_operation_failed"
            message = str(detail or error_code)
        raise GitHubOperationError(error_code, message, status_code=response.status_code)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise GitHubOperationError(
            "github_operation_invalid_response",
            "GitHub operation gateway response must include a data object.",
            status_code=response.status_code,
        )
    return dict(payload["data"])


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


__all__ = [
    "GitHubOperationError",
    "GitHubOperationsClient",
    "build_github_operations_client",
]
