//! Platform-managed LLM client for Rust agents.
//!
//! This mirrors the Python SDK's `ctx.platform.llm.*` surface. Connected agents
//! should prefer this client over direct provider calls so Novie can meter usage
//! and enforce org token pools. Standalone/BYOK agents can keep using their own
//! provider clients and report usage separately.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::error::{Error, Result};
use crate::headers::RequestHeaders;

use hmac::{Hmac, Mac};
use sha2::Sha256;

const DEFAULT_TIMEOUT_SECS: u64 = 30;
type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_owned(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_owned(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_owned(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChatOptions<'a> {
    pub model: Option<&'a str>,
    pub temperature: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct StructuredOptions<'a> {
    pub model: Option<&'a str>,
    pub temperature: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct EmbedOptions<'a> {
    pub model: Option<&'a str>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlatformLlmIdentity {
    pub org_id: String,
    pub project_id: String,
    pub workspace_id: String,
    pub user_id: String,
    pub service_principal: String,
    pub session_id: String,
    pub request_id: String,
    pub auth_source: String,
}

impl PlatformLlmIdentity {
    pub fn from_request_headers(headers: &RequestHeaders, agent_id: &str) -> Self {
        let service_principal =
            if headers.user_id.is_empty() && headers.service_principal.is_empty() {
                format!("agent:{agent_id}")
            } else {
                headers.service_principal.clone()
            };
        Self {
            org_id: first_non_empty(&[&headers.tenant_id]),
            project_id: first_non_empty(&[&headers.project_id, &headers.workspace_id]),
            workspace_id: headers.workspace_id.clone(),
            user_id: headers.user_id.clone(),
            service_principal,
            session_id: headers.session_id.clone(),
            request_id: first_non_empty(&[
                &headers.request_id,
                &headers.trace_id,
                &headers.step_id,
            ]),
            auth_source: "agent_callback".to_owned(),
        }
    }

    pub fn from_env(agent_id: &str) -> Self {
        let user_id = std::env::var("NOVIE_USER_ID").unwrap_or_default();
        let service_principal = if user_id.trim().is_empty() {
            format!("agent:{agent_id}")
        } else {
            String::new()
        };
        let workspace_id = std::env::var("NOVIE_WORKSPACE_ID").unwrap_or_default();
        let org_id = std::env::var("NOVIE_ORG_ID").unwrap_or_default();
        let project_id = std::env::var("NOVIE_PROJECT_ID")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| first_non_empty(&[&workspace_id, &org_id]));
        Self {
            org_id,
            project_id,
            workspace_id,
            user_id,
            service_principal,
            session_id: std::env::var("NOVIE_SESSION_ID").unwrap_or_default(),
            request_id: std::env::var("NOVIE_REQUEST_ID").unwrap_or_default(),
            auth_source: "agent_callback".to_owned(),
        }
    }
}

/// Client for `/capabilities/platform.llm.*/invoke`.
#[derive(Debug, Clone)]
pub struct PlatformLlmClient {
    base_url: String,
    token: String,
    agent_id: String,
    identity: PlatformLlmIdentity,
    http: reqwest::Client,
}

impl PlatformLlmClient {
    pub fn new(
        base_url: impl Into<String>,
        token: impl Into<String>,
        agent_id: impl Into<String>,
    ) -> Result<Self> {
        let base_url = base_url.into();
        let token = token.into();
        let agent_id = agent_id.into();
        if base_url.trim().is_empty() {
            return Err(Error::InvalidArgument("base_url is required".into()));
        }
        if token.trim().is_empty() {
            return Err(Error::InvalidArgument("token is required".into()));
        }
        if agent_id.trim().is_empty() {
            return Err(Error::InvalidArgument("agent_id is required".into()));
        }
        let identity = PlatformLlmIdentity::from_env(&agent_id);
        Self::with_identity(base_url, token, agent_id, identity)
    }

    pub fn with_identity(
        base_url: impl Into<String>,
        token: impl Into<String>,
        agent_id: impl Into<String>,
        identity: PlatformLlmIdentity,
    ) -> Result<Self> {
        let base_url = base_url.into();
        let token = token.into();
        let agent_id = agent_id.into();
        if base_url.trim().is_empty() {
            return Err(Error::InvalidArgument("base_url is required".into()));
        }
        if token.trim().is_empty() {
            return Err(Error::InvalidArgument("token is required".into()));
        }
        if agent_id.trim().is_empty() {
            return Err(Error::InvalidArgument("agent_id is required".into()));
        }
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .user_agent(format!(
                "novie-agent-sdk-rust/{}",
                env!("CARGO_PKG_VERSION")
            ))
            .build()
            .map_err(|e| Error::InvalidArgument(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            token,
            agent_id,
            identity,
            http,
        })
    }

    pub fn from_request_headers(
        base_url: impl Into<String>,
        token: impl Into<String>,
        agent_id: impl Into<String>,
        headers: &RequestHeaders,
    ) -> Result<Self> {
        let agent_id = agent_id.into();
        let identity = PlatformLlmIdentity::from_request_headers(headers, &agent_id);
        Self::with_identity(base_url, token, agent_id, identity)
    }

    pub fn from_env(agent_id: impl Into<String>) -> Result<Self> {
        let base_url = std::env::var("NOVIE_PLATFORM_BASE_URL").unwrap_or_default();
        let token = std::env::var("NOVIE_PLATFORM_TOKEN")
            .or_else(|_| std::env::var("NOVIE_PLATFORM_CALLBACK_TOKEN"))
            .unwrap_or_default();
        Self::new(base_url, token, agent_id)
    }

    pub async fn chat(
        &self,
        messages: Vec<ChatMessage>,
        opts: ChatOptions<'_>,
    ) -> Result<Map<String, Value>> {
        let mut args = json!({ "messages": messages });
        if let Some(model) = opts.model {
            args["model"] = Value::String(model.to_owned());
        }
        if let Some(temperature) = opts.temperature {
            args["temperature"] = json!(temperature);
        }
        self.invoke("platform.llm.chat", args).await
    }

    pub async fn structured(
        &self,
        messages: Vec<ChatMessage>,
        output_schema: Value,
        opts: StructuredOptions<'_>,
    ) -> Result<Map<String, Value>> {
        let mut args = json!({
            "messages": messages,
            "schema": output_schema,
        });
        if let Some(model) = opts.model {
            args["model"] = Value::String(model.to_owned());
        }
        if let Some(temperature) = opts.temperature {
            args["temperature"] = json!(temperature);
        }
        self.invoke("platform.llm.structured", args).await
    }

    pub async fn embed(&self, texts: Vec<String>, opts: EmbedOptions<'_>) -> Result<Vec<Vec<f64>>> {
        let mut args = json!({ "texts": texts });
        if let Some(model) = opts.model {
            args["model"] = Value::String(model.to_owned());
        }
        let result = self.invoke("platform.llm.embed", args).await?;
        let embeddings = result
            .get("embeddings")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Protocol {
                message: "platform.llm.embed returned no embeddings array".into(),
                code: Some("schema_violation".into()),
                http_status: Some(200),
                callback_id: None,
            })?;
        embeddings
            .iter()
            .map(|row| {
                row.as_array()
                    .ok_or_else(|| Error::Protocol {
                        message: "platform.llm.embed returned a non-array embedding row".into(),
                        code: Some("schema_violation".into()),
                        http_status: Some(200),
                        callback_id: None,
                    })?
                    .iter()
                    .map(|v| {
                        v.as_f64().ok_or_else(|| Error::Protocol {
                            message: "platform.llm.embed returned a non-number embedding value"
                                .into(),
                            code: Some("schema_violation".into()),
                            http_status: Some(200),
                            callback_id: None,
                        })
                    })
                    .collect()
            })
            .collect()
    }

    pub async fn budget_check(&self) -> Result<Map<String, Value>> {
        self.invoke("platform.llm.budget_check", json!({})).await
    }

    pub async fn usage_summary(&self, scope: &str) -> Result<Map<String, Value>> {
        self.invoke("platform.llm.usage_summary", json!({ "scope": scope }))
            .await
    }

    async fn invoke(&self, capability_id: &str, arguments: Value) -> Result<Map<String, Value>> {
        let path = format!("/capabilities/{}/invoke", capability_id);
        let url = format!("{}{}", self.base_url, path);
        let body = json!({
            "arguments": arguments,
            "caller_type": "agent",
            "caller_id": format!("agent:{}", self.agent_id),
            "caller_mode": "execute",
            "mode": "execute",
        });
        let response = self
            .http
            .post(url)
            .bearer_auth(&self.token)
            .headers(self.signed_headers("POST", &path)?)
            .json(&body)
            .send()
            .await?;
        let status = response.status().as_u16();
        let envelope: Value = response.json().await.map_err(|e| Error::Protocol {
            message: format!("platform returned non-JSON response (status={status}): {e}"),
            code: None,
            http_status: Some(status),
            callback_id: None,
        })?;

        if status >= 400 {
            return Err(map_capability_error(status, &envelope));
        }
        if envelope.get("status").and_then(Value::as_str) != Some("ok") {
            return Err(map_capability_error(status, &envelope));
        }
        let result = envelope.get("result").cloned().unwrap_or(Value::Null);
        result.as_object().cloned().ok_or_else(|| Error::Protocol {
            message: format!("{capability_id} returned non-object result"),
            code: Some("schema_violation".into()),
            http_status: Some(status),
            callback_id: None,
        })
    }

    fn signed_headers(&self, method: &str, path: &str) -> Result<reqwest::header::HeaderMap> {
        use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};

        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        insert_header(&mut headers, "x-novie-org-id", &self.identity.org_id)?;
        insert_header(
            &mut headers,
            "x-novie-project-id",
            &self.identity.project_id,
        )?;
        insert_header(
            &mut headers,
            "x-novie-workspace-id",
            &self.identity.workspace_id,
        )?;
        insert_header(&mut headers, "x-novie-user-id", &self.identity.user_id)?;
        insert_header(
            &mut headers,
            "x-novie-service-principal",
            &self.identity.service_principal,
        )?;
        insert_header(
            &mut headers,
            "x-novie-session-id",
            &self.identity.session_id,
        )?;
        insert_header(
            &mut headers,
            "x-novie-request-id",
            &self.identity.request_id,
        )?;
        insert_header(
            &mut headers,
            "x-novie-auth-source",
            &self.identity.auth_source,
        )?;

        let secret = std::env::var("NOVIE_TRUSTED_HEADER_SECRET").unwrap_or_default();
        if !secret.trim().is_empty() {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|e| Error::InvalidArgument(format!("system clock before epoch: {e}")))?
                .as_secs()
                .to_string();
            insert_header(&mut headers, "x-novie-timestamp", &timestamp)?;
            let signature =
                trusted_header_signature(method, path, &self.identity, &timestamp, &secret);
            insert_header(&mut headers, "x-novie-sig", &format!("sha256={signature}"))?;
        }

        // Compile-time sanity: HeaderName import remains used when optional
        // insertions are optimized.
        let _ = HeaderName::from_static("accept");
        Ok(headers)
    }
}

fn first_non_empty(values: &[&str]) -> String {
    values
        .iter()
        .map(|v| v.trim())
        .find(|v| !v.is_empty())
        .unwrap_or("")
        .to_owned()
}

fn insert_header(
    headers: &mut reqwest::header::HeaderMap,
    name: &'static str,
    value: &str,
) -> Result<()> {
    if value.trim().is_empty() {
        return Ok(());
    }
    let value = reqwest::header::HeaderValue::from_str(value)
        .map_err(|e| Error::InvalidArgument(format!("invalid header {name}: {e}")))?;
    headers.insert(reqwest::header::HeaderName::from_static(name), value);
    Ok(())
}

fn trusted_header_signature(
    method: &str,
    path: &str,
    identity: &PlatformLlmIdentity,
    timestamp: &str,
    secret: &str,
) -> String {
    let canonical = [
        method.to_uppercase(),
        path.to_owned(),
        identity.org_id.clone(),
        identity.project_id.clone(),
        identity.workspace_id.clone(),
        identity.user_id.clone(),
        identity.service_principal.clone(),
        identity.session_id.clone(),
        identity.request_id.clone(),
        timestamp.to_owned(),
    ]
    .join("\n");
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC accepts any key size");
    mac.update(canonical.as_bytes());
    bytes_to_lower_hex(&mac.finalize().into_bytes())
}

fn bytes_to_lower_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

fn map_capability_error(status: u16, envelope: &Value) -> Error {
    let detail = envelope.get("detail").unwrap_or(envelope);
    let code = detail
        .get("error_code")
        .or_else(|| envelope.get("error_code"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let message = detail
        .get("explanation")
        .or_else(|| detail.get("reason"))
        .or_else(|| envelope.get("explanation"))
        .and_then(Value::as_str)
        .unwrap_or("platform capability invocation failed")
        .to_owned();
    if code.as_deref() == Some("quota_exceeded") {
        let quota = detail
            .get("metadata")
            .and_then(|m| m.get("quota"))
            .unwrap_or(detail);
        return Error::QuotaExceeded {
            message,
            org_id: quota
                .get("org_id")
                .and_then(Value::as_str)
                .map(str::to_owned),
            remaining_tokens: quota.get("remaining_tokens").and_then(Value::as_u64),
        };
    }
    Error::Protocol {
        message,
        code,
        http_status: Some(status),
        callback_id: None,
    }
}
