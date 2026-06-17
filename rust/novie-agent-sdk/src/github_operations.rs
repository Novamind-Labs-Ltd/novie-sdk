//! GitHub operations client for Novie SDK agents.
//!
//! The SDK calls the run-scoped agent-operation gateway. GitHub tokens and CCTs
//! stay inside the platform/GHIS boundary.

use crate::error::{Error, Result};
use serde_json::{Map, Value, json};

#[derive(Debug)]
pub struct GitHubOperationsClient {
    base_url: String,
    runtime_token: String,
    agent_run_id: String,
    http: reqwest::Client,
}

impl GitHubOperationsClient {
    pub fn new(
        base_url: impl Into<String>,
        runtime_token: impl Into<String>,
        agent_run_id: impl Into<String>,
    ) -> Result<Self> {
        let base_url = base_url.into();
        let runtime_token = runtime_token.into();
        let agent_run_id = agent_run_id.into();
        if base_url.trim().is_empty() {
            return Err(Error::InvalidArgument("base_url is required".into()));
        }
        if runtime_token.trim().is_empty() {
            return Err(Error::InvalidArgument(
                "agent runtime token is required".into(),
            ));
        }
        if agent_run_id.trim().is_empty() {
            return Err(Error::InvalidArgument("agent_run_id is required".into()));
        }
        let http = reqwest::Client::builder().build().map_err(Error::from)?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            runtime_token: runtime_token.trim().to_owned(),
            agent_run_id: agent_run_id.trim().to_owned(),
            http,
        })
    }

    pub fn from_env(
        runtime_token: impl Into<String>,
        agent_run_id: impl Into<String>,
    ) -> Result<Self> {
        Self::new(
            std::env::var("NOVIE_PLATFORM_BASE_URL").unwrap_or_default(),
            runtime_token,
            agent_run_id,
        )
    }

    pub async fn read_file(
        &self,
        repository_full_name: &str,
        path: &str,
        ref_name: Option<&str>,
        max_bytes: Option<u32>,
    ) -> Result<Map<String, Value>> {
        self.post(
            "read-file",
            json!({
                "repositoryFullName": repository_full_name,
                "path": path,
                "ref": ref_name,
                "maxBytes": max_bytes,
            }),
        )
        .await
    }

    pub async fn search_code(
        &self,
        repository_full_name: &str,
        query: &str,
        ref_name: Option<&str>,
        max_results: Option<u32>,
    ) -> Result<Map<String, Value>> {
        self.post(
            "search-code",
            json!({
                "repositoryFullName": repository_full_name,
                "query": query,
                "ref": ref_name,
                "maxResults": max_results,
            }),
        )
        .await
    }

    pub async fn ref_(
        &self,
        repository_full_name: &str,
        ref_name: &str,
    ) -> Result<Map<String, Value>> {
        self.post(
            "ref",
            json!({
                "repositoryFullName": repository_full_name,
                "ref": ref_name,
            }),
        )
        .await
    }

    pub async fn tree(
        &self,
        repository_full_name: &str,
        tree_sha: &str,
        recursive: bool,
    ) -> Result<Map<String, Value>> {
        self.post(
            "tree",
            json!({
                "repositoryFullName": repository_full_name,
                "treeSha": tree_sha,
                "recursive": recursive,
            }),
        )
        .await
    }

    pub async fn blob(
        &self,
        repository_full_name: &str,
        sha: &str,
        max_bytes: Option<u32>,
    ) -> Result<Map<String, Value>> {
        self.post(
            "blob",
            json!({
                "repositoryFullName": repository_full_name,
                "sha": sha,
                "maxBytes": max_bytes,
            }),
        )
        .await
    }

    pub async fn create_branch(
        &self,
        repository_full_name: &str,
        name: &str,
        from_ref: &str,
    ) -> Result<Map<String, Value>> {
        self.post(
            "create-branch",
            json!({
                "repositoryFullName": repository_full_name,
                "name": name,
                "fromRef": from_ref,
            }),
        )
        .await
    }

    pub async fn commit_files(
        &self,
        repository_full_name: &str,
        branch: &str,
        message: &str,
        files: Vec<Value>,
        expected_head_sha: Option<&str>,
    ) -> Result<Map<String, Value>> {
        self.post(
            "commit-files",
            json!({
                "repositoryFullName": repository_full_name,
                "branch": branch,
                "message": message,
                "files": files,
                "expectedHeadSha": expected_head_sha,
            }),
        )
        .await
    }

    pub async fn create_pull_request(
        &self,
        repository_full_name: &str,
        title: &str,
        head: &str,
        base: &str,
        body: Option<&str>,
        draft: bool,
    ) -> Result<Map<String, Value>> {
        self.post(
            "create-pull-request",
            json!({
                "repositoryFullName": repository_full_name,
                "title": title,
                "body": body,
                "head": head,
                "base": base,
                "draft": draft,
            }),
        )
        .await
    }

    pub async fn comment_pull_request(
        &self,
        repository_full_name: &str,
        number: u64,
        body: &str,
    ) -> Result<Map<String, Value>> {
        self.post(
            "comment-pull-request",
            json!({
                "repositoryFullName": repository_full_name,
                "number": number,
                "body": body,
            }),
        )
        .await
    }

    pub async fn pull_request(
        &self,
        repository_full_name: &str,
        number: u64,
    ) -> Result<Map<String, Value>> {
        self.post(
            "pull-request",
            json!({
                "repositoryFullName": repository_full_name,
                "number": number,
            }),
        )
        .await
    }

    pub async fn list_pull_requests(
        &self,
        repository_full_name: &str,
        head: Option<&str>,
        base: Option<&str>,
        state: Option<&str>,
        first: Option<u32>,
    ) -> Result<Map<String, Value>> {
        self.post(
            "list-pull-requests",
            json!({
                "repositoryFullName": repository_full_name,
                "head": head,
                "base": base,
                "state": state,
                "first": first,
            }),
        )
        .await
    }

    pub async fn pull_request_checks(
        &self,
        repository_full_name: &str,
        number: u64,
    ) -> Result<Map<String, Value>> {
        self.post(
            "pull-request-checks",
            json!({
                "repositoryFullName": repository_full_name,
                "number": number,
            }),
        )
        .await
    }

    pub async fn update_pull_request(
        &self,
        repository_full_name: &str,
        number: u64,
        title: Option<&str>,
        body: Option<&str>,
        base: Option<&str>,
        draft: Option<bool>,
    ) -> Result<Map<String, Value>> {
        self.post(
            "update-pull-request",
            json!({
                "repositoryFullName": repository_full_name,
                "number": number,
                "title": title,
                "body": body,
                "base": base,
                "draft": draft,
            }),
        )
        .await
    }

    pub async fn reopen_pull_request(
        &self,
        repository_full_name: &str,
        number: u64,
    ) -> Result<Map<String, Value>> {
        self.post(
            "reopen-pull-request",
            json!({
                "repositoryFullName": repository_full_name,
                "number": number,
            }),
        )
        .await
    }

    pub async fn add_pull_request_labels(
        &self,
        repository_full_name: &str,
        number: u64,
        labels: Vec<String>,
    ) -> Result<Map<String, Value>> {
        self.post(
            "add-pull-request-labels",
            json!({
                "repositoryFullName": repository_full_name,
                "number": number,
                "labels": labels,
            }),
        )
        .await
    }

    pub async fn update_pull_request_branch(
        &self,
        repository_full_name: &str,
        number: u64,
        expected_head_sha: Option<&str>,
    ) -> Result<Map<String, Value>> {
        self.post(
            "update-pull-request-branch",
            json!({
                "repositoryFullName": repository_full_name,
                "number": number,
                "expectedHeadSha": expected_head_sha,
            }),
        )
        .await
    }

    pub async fn merge_pull_request(
        &self,
        repository_full_name: &str,
        number: u64,
        merge_method: Option<&str>,
    ) -> Result<Map<String, Value>> {
        self.post(
            "merge-pull-request",
            json!({
                "repositoryFullName": repository_full_name,
                "number": number,
                "mergeMethod": merge_method,
            }),
        )
        .await
    }

    pub async fn dispatch_workflow(
        &self,
        repository_full_name: &str,
        workflow_id: &str,
        ref_name: &str,
        inputs: Option<Map<String, Value>>,
    ) -> Result<Map<String, Value>> {
        self.post(
            "dispatch-workflow",
            json!({
                "repositoryFullName": repository_full_name,
                "workflowId": workflow_id,
                "ref": ref_name,
                "inputs": inputs,
            }),
        )
        .await
    }

    async fn post(&self, operation: &str, payload: Value) -> Result<Map<String, Value>> {
        let path = format!(
            "/agent-runs/{}/github/operations/{}",
            self.agent_run_id, operation
        );
        let url = format!("{}{}", self.base_url, path);
        let response = self
            .http
            .post(url)
            .bearer_auth(&self.runtime_token)
            .header(reqwest::header::ACCEPT, "application/json")
            .json(&compact(payload))
            .send()
            .await?;
        parse_gateway_response(response).await
    }
}

async fn parse_gateway_response(response: reqwest::Response) -> Result<Map<String, Value>> {
    let status = response.status().as_u16();
    let envelope: Value = response.json().await.map_err(|err| Error::Protocol {
        message: format!("GitHub operation gateway returned non-JSON (status={status}): {err}"),
        code: Some("github_operation_invalid_response".into()),
        http_status: Some(status),
        callback_id: None,
    })?;
    if status >= 400 {
        return Err(map_gateway_error(status, &envelope));
    }
    envelope
        .get("data")
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| Error::Protocol {
            message: "GitHub operation gateway returned no data object".into(),
            code: Some("github_operation_invalid_response".into()),
            http_status: Some(status),
            callback_id: None,
        })
}

fn map_gateway_error(status: u16, envelope: &Value) -> Error {
    let detail = envelope.get("detail").unwrap_or(envelope);
    let code = detail
        .get("error")
        .and_then(Value::as_str)
        .unwrap_or("github_operation_failed")
        .to_owned();
    let message = detail
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or(&code)
        .to_owned();
    match status {
        401 | 403 => Error::Auth {
            message,
            code: Some(code),
            http_status: Some(status),
            callback_id: None,
        },
        400 | 404 | 422 => Error::Protocol {
            message,
            code: Some(code),
            http_status: Some(status),
            callback_id: None,
        },
        503 => Error::Unavailable {
            message,
            code: Some(code),
            retry_after_ms: None,
            http_status: Some(status),
            callback_id: None,
        },
        _ => Error::Callback {
            message,
            code: Some(code),
            http_status: Some(status),
            callback_id: None,
        },
    }
}

fn compact(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .filter_map(|(key, value)| {
                    if value.is_null() {
                        None
                    } else {
                        Some((key, value))
                    }
                })
                .collect(),
        ),
        other => other,
    }
}
