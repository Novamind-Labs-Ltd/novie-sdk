//! Typed PMS issue API client for automation dispatch.
//!
//! ADR-071 requires agents and Cortex to use a PMS SDK/API boundary instead of
//! PMS internal REST endpoints. This client targets the platform-owned PMS API
//! surface and keeps the backend transport shape out of consumers.

use crate::error::{Error, Result};
use serde_json::{Map, Value, json};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PmsStatus {
    pub id: String,
    pub title: String,
    pub stage: String,
    pub automation_action: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PmsComment {
    pub id: String,
    pub content: String,
    pub author_id: String,
    pub author_name: String,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PmsIssue {
    pub id: String,
    pub identifier: String,
    pub title: String,
    pub description: String,
    pub status_id: String,
    pub status_title: String,
    pub status_stage: String,
    pub automation_action: String,
    pub tenant_id: String,
    pub organization_id: String,
    pub workspace_id: String,
    pub project_id: String,
    pub pms_issue_id: String,
    pub linked_pr_urls: Vec<String>,
    pub parent_id: String,
    pub parent_identifier: String,
    pub agentic_orchestration_values: Map<String, Value>,
    pub comments: Vec<PmsComment>,
    pub raw: Map<String, Value>,
}

#[derive(Debug)]
pub struct PmsIssueClient {
    base_url: String,
    runtime_token: String,
    http: reqwest::Client,
}

impl PmsIssueClient {
    pub fn new(base_url: impl Into<String>, runtime_token: impl Into<String>) -> Result<Self> {
        let base_url = base_url.into();
        let runtime_token = runtime_token.into();
        if base_url.trim().is_empty() {
            return Err(Error::InvalidArgument("base_url is required".into()));
        }
        if runtime_token.trim().is_empty() {
            return Err(Error::InvalidArgument("runtime token is required".into()));
        }
        let http = reqwest::Client::builder().build().map_err(Error::from)?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            runtime_token: runtime_token.trim().to_owned(),
            http,
        })
    }

    pub fn from_env(runtime_token: impl Into<String>) -> Result<Self> {
        Self::new(
            std::env::var("NOVIE_PLATFORM_BASE_URL").unwrap_or_default(),
            runtime_token,
        )
    }

    pub async fn list_candidate_issues(
        &self,
        project_ids: Vec<String>,
        automation_actions: Vec<String>,
        include_human_review: bool,
        organization_id: Option<&str>,
        workspace_id: Option<&str>,
    ) -> Result<Vec<PmsIssue>> {
        let payload = self
            .post(
                "/pms/issues/candidates",
                json!({
                    "projectIds": project_ids,
                    "automationActions": automation_actions,
                    "includeHumanReview": include_human_review,
                    "organizationId": organization_id,
                    "workspaceId": workspace_id,
                }),
            )
            .await?;
        Ok(list_field(&payload, &["issues", "nodes", "items"])
            .into_iter()
            .map(pms_issue_from_value)
            .collect())
    }

    pub async fn get_issue(
        &self,
        issue_id: &str,
        organization_id: Option<&str>,
        workspace_id: Option<&str>,
    ) -> Result<PmsIssue> {
        let payload = self
            .post(
                "/pms/issues/get",
                json!({
                    "issueId": issue_id,
                    "organizationId": organization_id,
                    "workspaceId": workspace_id,
                }),
            )
            .await?;
        Ok(pms_issue_from_value(
            mapping_field(&payload, &["issue"]).unwrap_or(&payload),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn transition_issue_status(
        &self,
        issue_id: &str,
        target_status_id: Option<&str>,
        automation_action: Option<&str>,
        title: Option<&str>,
        actor_user_id: Option<&str>,
        reason: Option<&str>,
        organization_id: Option<&str>,
        workspace_id: Option<&str>,
    ) -> Result<PmsIssue> {
        let payload = self
            .post(
                "/pms/issues/transition-status",
                json!({
                    "issueId": issue_id,
                    "targetStatusId": target_status_id,
                    "automationAction": automation_action,
                    "title": title,
                    "actorUserId": actor_user_id,
                    "reason": reason,
                    "organizationId": organization_id,
                    "workspaceId": workspace_id,
                }),
            )
            .await?;
        Ok(pms_issue_from_value(
            mapping_field(&payload, &["issue"]).unwrap_or(&payload),
        ))
    }

    pub async fn update_agentic_orchestration_values(
        &self,
        issue_id: &str,
        patch: Value,
        actor_user_id: Option<&str>,
        organization_id: Option<&str>,
        workspace_id: Option<&str>,
    ) -> Result<Map<String, Value>> {
        let payload = self
            .post(
                "/pms/issues/update-agentic-orchestration-values",
                json!({
                    "issueId": issue_id,
                    "patch": patch,
                    "actorUserId": actor_user_id,
                    "organizationId": organization_id,
                    "workspaceId": workspace_id,
                }),
            )
            .await?;
        if let Some(values) = mapping_field(&payload, &["agenticOrchestrationValues"]) {
            Ok(values.clone())
        } else {
            Ok(payload)
        }
    }

    pub async fn add_comment(
        &self,
        issue_id: &str,
        content: &str,
        author_id: &str,
        organization_id: Option<&str>,
        workspace_id: Option<&str>,
    ) -> Result<PmsComment> {
        let payload = self
            .post(
                "/pms/issues/add-comment",
                json!({
                    "issueId": issue_id,
                    "content": content,
                    "authorId": author_id,
                    "organizationId": organization_id,
                    "workspaceId": workspace_id,
                }),
            )
            .await?;
        Ok(pms_comment_from_value(
            mapping_field(&payload, &["comment"]).unwrap_or(&payload),
        ))
    }

    pub async fn upsert_workpad_comment(
        &self,
        issue_id: &str,
        marker: &str,
        content: &str,
        author_id: &str,
        organization_id: Option<&str>,
        workspace_id: Option<&str>,
    ) -> Result<PmsComment> {
        let payload = self
            .post(
                "/pms/issues/upsert-workpad-comment",
                json!({
                    "issueId": issue_id,
                    "marker": marker,
                    "content": content,
                    "authorId": author_id,
                    "organizationId": organization_id,
                    "workspaceId": workspace_id,
                }),
            )
            .await?;
        Ok(pms_comment_from_value(
            mapping_field(&payload, &["comment"]).unwrap_or(&payload),
        ))
    }

    pub async fn list_comments(
        &self,
        issue_id: &str,
        first: Option<u32>,
        organization_id: Option<&str>,
        workspace_id: Option<&str>,
    ) -> Result<Vec<PmsComment>> {
        let payload = self
            .post(
                "/pms/issues/comments",
                json!({
                    "issueId": issue_id,
                    "first": first,
                    "organizationId": organization_id,
                    "workspaceId": workspace_id,
                }),
            )
            .await?;
        Ok(list_field(&payload, &["comments", "nodes", "items"])
            .into_iter()
            .map(pms_comment_from_value)
            .collect())
    }

    async fn post(&self, path: &str, payload: Value) -> Result<Map<String, Value>> {
        let url = format!("{}{}", self.base_url, path);
        let response = self
            .http
            .post(url)
            .bearer_auth(&self.runtime_token)
            .header(reqwest::header::ACCEPT, "application/json")
            .json(&compact(payload))
            .send()
            .await?;
        parse_pms_response(response).await
    }
}

pub fn normalize_pms_automation_action(value: &str) -> String {
    let normalized = value
        .chars()
        .filter(|ch| *ch != '_' && *ch != '-' && !ch.is_whitespace())
        .collect::<String>()
        .to_lowercase();
    match normalized.as_str() {
        "none" => "None",
        "execute" => "Execute",
        "review" => "Review",
        "rework" => "Rework",
        "merge" => "Merge",
        "epicmerge" => "EpicMerge",
        _ => value,
    }
    .to_owned()
}

pub fn pms_issue_from_value(data: &Map<String, Value>) -> PmsIssue {
    let empty = Map::new();
    let status = mapping_field(data, &["status"]).unwrap_or(&empty);
    let project = mapping_field(data, &["project"]).unwrap_or(&empty);
    let parent = mapping_field(data, &["parent", "parentIssue"]).unwrap_or(&empty);
    let comments = list_field(data, &["comments", "recentComments"])
        .into_iter()
        .map(pms_comment_from_value)
        .collect::<Vec<_>>();
    PmsIssue {
        id: str_field(data, &["id"]),
        identifier: str_field(data, &["identifier", "issueNumber", "issue_number"]),
        title: str_field(data, &["title"]),
        description: str_field(data, &["description"]),
        status_id: non_empty_or(str_field(data, &["statusId", "status_id"]), || {
            str_field(status, &["id"])
        }),
        status_title: str_field(data, &["statusTitle", "status_title"])
            .pipe_non_empty_or(|| str_field(status, &["title"])),
        status_stage: str_field(data, &["statusStage", "status_stage"])
            .pipe_non_empty_or(|| str_field(status, &["stage"])),
        automation_action: normalize_pms_automation_action(&non_empty_or(
            str_field(data, &["automationAction", "automation_action"]),
            || str_field(status, &["automationAction", "automation_action"]),
        )),
        tenant_id: str_field(
            data,
            &["tenantId", "tenant_id", "organizationId", "organization_id"],
        ),
        organization_id: str_field(
            data,
            &["organizationId", "organization_id", "tenantId", "tenant_id"],
        ),
        workspace_id: str_field(data, &["workspaceId", "workspace_id"]),
        project_id: str_field(data, &["projectId", "project_id"])
            .pipe_non_empty_or(|| str_field(project, &["id"])),
        pms_issue_id: str_field(data, &["pmsIssueId", "pms_issue_id"])
            .pipe_non_empty_or(|| str_field(data, &["id"])),
        linked_pr_urls: string_list_field(
            data,
            &[
                "linkedPrUrls",
                "linkedPRUrls",
                "linkedPullRequestUrls",
                "linked_pr_urls",
            ],
        ),
        parent_id: str_field(data, &["parentId", "parent_id"])
            .pipe_non_empty_or(|| str_field(parent, &["id"])),
        parent_identifier: str_field(data, &["parentIdentifier", "parent_identifier"])
            .pipe_non_empty_or(|| str_field(parent, &["identifier"])),
        agentic_orchestration_values: mapping_field(
            data,
            &["agenticOrchestrationValues", "agentic_orchestration_values"],
        )
        .cloned()
        .unwrap_or_default(),
        comments,
        raw: data.clone(),
    }
}

pub fn pms_comment_from_value(data: &Map<String, Value>) -> PmsComment {
    let empty = Map::new();
    let author = mapping_field(data, &["author"]).unwrap_or(&empty);
    PmsComment {
        id: str_field(data, &["id", "commentId", "comment_id"]),
        content: str_field(data, &["content", "body"]),
        author_id: str_field(data, &["authorId", "author_id"])
            .pipe_non_empty_or(|| str_field(author, &["id"])),
        author_name: str_field(data, &["authorName", "author_name"])
            .pipe_non_empty_or(|| str_field(author, &["name"])),
        created_at: str_field(data, &["createdAt", "created_at"]),
    }
}

async fn parse_pms_response(response: reqwest::Response) -> Result<Map<String, Value>> {
    let status = response.status().as_u16();
    let envelope: Value = response.json().await.map_err(|err| Error::Protocol {
        message: format!("PMS API returned non-JSON (status={status}): {err}"),
        code: Some("pms_api_invalid_response".into()),
        http_status: Some(status),
        callback_id: None,
    })?;
    if status >= 400 {
        return Err(map_pms_error(status, &envelope));
    }
    if let Some(data) = envelope.get("data").and_then(Value::as_object) {
        return Ok(data.clone());
    }
    envelope
        .as_object()
        .cloned()
        .ok_or_else(|| Error::Protocol {
            message: "PMS API response must be a JSON object".into(),
            code: Some("pms_api_invalid_response".into()),
            http_status: Some(status),
            callback_id: None,
        })
}

fn map_pms_error(status: u16, envelope: &Value) -> Error {
    let detail = envelope.get("detail").unwrap_or(envelope);
    let code = detail
        .get("error")
        .and_then(Value::as_str)
        .unwrap_or("pms_api_failed")
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

fn mapping_field<'a>(
    data: &'a Map<String, Value>,
    names: &[&str],
) -> Option<&'a Map<String, Value>> {
    names
        .iter()
        .find_map(|name| data.get(*name).and_then(Value::as_object))
}

fn list_field<'a>(data: &'a Map<String, Value>, names: &[&str]) -> Vec<&'a Map<String, Value>> {
    names
        .iter()
        .find_map(|name| data.get(*name).and_then(Value::as_array))
        .map(|rows| rows.iter().filter_map(Value::as_object).collect())
        .unwrap_or_default()
}

fn string_list_field(data: &Map<String, Value>, names: &[&str]) -> Vec<String> {
    names
        .iter()
        .find_map(|name| data.get(*name).and_then(Value::as_array))
        .map(|rows| {
            rows.iter()
                .filter_map(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn str_field(data: &Map<String, Value>, names: &[&str]) -> String {
    names
        .iter()
        .find_map(|name| data.get(*name))
        .map(value_to_string)
        .unwrap_or_default()
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn non_empty_or(value: String, fallback: impl FnOnce() -> String) -> String {
    if value.trim().is_empty() {
        fallback()
    } else {
        value
    }
}

trait NonEmptyStringFallback {
    fn pipe_non_empty_or(self, fallback: impl FnOnce() -> String) -> String;
}

impl NonEmptyStringFallback for String {
    fn pipe_non_empty_or(self, fallback: impl FnOnce() -> String) -> String {
        non_empty_or(self, fallback)
    }
}
