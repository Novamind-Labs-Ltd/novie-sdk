//! HTTP transport with built-in retry policy.
//!
//! Mirrors the `_call` / `_call_once` retry loop in
//! `HttpCallbackPlatformServices`. The server controls backoff via
//! `retry_after_ms`; otherwise we fall back to exponential 200ms * 2^n.

use std::time::Duration;

use serde_json::Value;

use crate::error::{Error, Result};
use crate::headers::{RequestHeaders, agent_platform_shared_secret, agent_platform_signature};

const DEFAULT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_MAX_RETRIES: u32 = 2;
const DEFAULT_BACKOFF_MS: u64 = 200;

/// Builder-style settings for [`CallbackTransport`]. All fields have safe
/// defaults that match the Python SDK.
#[derive(Debug, Clone)]
pub struct TransportConfig {
    pub timeout: Duration,
    pub max_retries: u32,
    pub user_agent: String,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            max_retries: DEFAULT_MAX_RETRIES,
            user_agent: format!("novie-agent-sdk-rust/{}", env!("CARGO_PKG_VERSION")),
        }
    }
}

/// Shared HTTP transport reused by `PlatformServicesClient`,
/// `AgentStatusClient`, and the `notify_long_task_complete` helper.
///
/// Holds a single `reqwest::Client` (connection pool friendly) and the
/// platform's bearer token + base URL.
#[derive(Debug, Clone)]
pub struct CallbackTransport {
    base_url: String,
    auth: TransportAuth,
    http: reqwest::Client,
    max_retries: u32,
}

#[derive(Debug, Clone)]
enum TransportAuth {
    Bearer(String),
    Signed(RequestHeaders),
}

impl CallbackTransport {
    pub fn new(base_url: impl Into<String>, token: impl Into<String>) -> Result<Self> {
        Self::with_config(base_url, token, TransportConfig::default())
    }

    pub fn with_config(
        base_url: impl Into<String>,
        token: impl Into<String>,
        cfg: TransportConfig,
    ) -> Result<Self> {
        let base_url = base_url.into();
        let token = token.into();
        if base_url.is_empty() {
            return Err(Error::InvalidArgument("base_url is required".into()));
        }
        if token.is_empty() {
            return Err(Error::InvalidArgument("token is required".into()));
        }
        let http = build_http_client(&cfg)?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            auth: TransportAuth::Bearer(token),
            http,
            max_retries: cfg.max_retries,
        })
    }

    pub fn with_signed_headers(
        base_url: impl Into<String>,
        headers: RequestHeaders,
        cfg: TransportConfig,
    ) -> Result<Self> {
        let base_url = base_url.into();
        if base_url.is_empty() {
            return Err(Error::InvalidArgument("base_url is required".into()));
        }
        let http = build_http_client(&cfg)?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            auth: TransportAuth::Signed(headers),
            http,
            max_retries: cfg.max_retries,
        })
    }

    /// Construct from an existing `reqwest::Client`. Useful for tests that
    /// inject a custom executor or share a global pool.
    pub fn with_client(
        base_url: impl Into<String>,
        token: impl Into<String>,
        http: reqwest::Client,
        max_retries: u32,
    ) -> Result<Self> {
        let base_url = base_url.into();
        let token = token.into();
        if base_url.is_empty() {
            return Err(Error::InvalidArgument("base_url is required".into()));
        }
        if token.is_empty() {
            return Err(Error::InvalidArgument("token is required".into()));
        }
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            auth: TransportAuth::Bearer(token),
            http,
            max_retries,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn token(&self) -> Option<&str> {
        match &self.auth {
            TransportAuth::Bearer(token) => Some(token),
            TransportAuth::Signed(_) => None,
        }
    }

    /// Call a `service/method` RPC with the standard `{"kwargs": ...}` body
    /// envelope. Retries automatically on `Error::Unavailable`.
    pub async fn call(&self, service: &str, method: &str, kwargs: Value) -> Result<Value> {
        let url = format!("{}/{}/{}", self.base_url, service, method);
        let path = format!("/{service}/{method}");
        let body = serde_json::json!({ "kwargs": kwargs });
        let attempts = self.max_retries.saturating_add(1);

        let mut last: Option<Error> = None;
        for attempt in 0..attempts {
            match self.call_once(&url, &path, &body).await {
                Ok(v) => return Ok(v),
                Err(e) if e.is_retryable() && attempt + 1 < attempts => {
                    let wait_ms = e.retry_after_ms().unwrap_or_else(|| backoff_ms(attempt));
                    tracing::debug!(
                        service,
                        method,
                        attempt = attempt + 1,
                        max = attempts - 1,
                        wait_ms,
                        "callback retry on transient error"
                    );
                    last = Some(e);
                    tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last.expect("loop above always assigns last on non-terminal error"))
    }

    async fn call_once(&self, url: &str, path: &str, body: &Value) -> Result<Value> {
        let resp = self
            .apply_auth(self.http.post(url), "POST", path)?
            .json(body)
            .send()
            .await?;

        let status = resp.status();
        let body: Value = resp.json().await.map_err(|e| Error::Protocol {
            message: format!(
                "callback returned non-JSON body (status={}): {e}",
                status.as_u16()
            ),
            code: None,
            http_status: Some(status.as_u16()),
            callback_id: None,
        })?;

        if status == reqwest::StatusCode::OK
            && body.get("ok").and_then(|v| v.as_bool()) == Some(true)
        {
            return Ok(body.get("result").cloned().unwrap_or(Value::Null));
        }

        Err(map_error_envelope(status.as_u16(), &body))
    }

    /// Send a bare-JSON push payload (for `agent-status`, `long-task-complete`).
    /// Caller specifies the absolute path-suffix (e.g. `"/agent-status"`),
    /// the body is serialised as-is. Retries on transient failures.
    pub async fn push(&self, path_or_url: &str, body: &Value) -> Result<reqwest::Response> {
        let url = if path_or_url.starts_with("http://") || path_or_url.starts_with("https://") {
            path_or_url.to_string()
        } else {
            format!("{}{}", self.base_url, path_or_url)
        };
        let attempts = self.max_retries.saturating_add(1);

        let mut last: Option<Error> = None;
        for attempt in 0..attempts {
            match self.push_once(&url, body).await {
                Ok(resp) => return Ok(resp),
                Err(e) if e.is_retryable() && attempt + 1 < attempts => {
                    let wait_ms = e.retry_after_ms().unwrap_or_else(|| backoff_ms(attempt));
                    tracing::debug!(url, attempt = attempt + 1, wait_ms, "push retry");
                    last = Some(e);
                    tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last.expect("loop above always assigns last"))
    }

    async fn push_once(&self, url: &str, body: &Value) -> Result<reqwest::Response> {
        let path = url_path(url);
        let resp = self
            .apply_auth(self.http.post(url), "POST", &path)?
            .json(body)
            .send()
            .await?;
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        // Try to parse `{"error": {...}}` envelope for richer diagnostics.
        let parsed: Value = resp.json().await.unwrap_or(Value::Null);
        Err(map_error_envelope(status.as_u16(), &parsed))
    }

    fn apply_auth(
        &self,
        request: reqwest::RequestBuilder,
        method: &str,
        path: &str,
    ) -> Result<reqwest::RequestBuilder> {
        match &self.auth {
            TransportAuth::Bearer(token) => Ok(request.bearer_auth(token)),
            TransportAuth::Signed(headers) => {
                let timestamp = epoch_seconds();
                let secret = agent_platform_shared_secret().map_err(|err| Error::Auth {
                    message: "NOVIE_AGENT_PLATFORM_SHARED_SECRET is not configured".into(),
                    code: Some(err.code().into()),
                    http_status: None,
                    callback_id: None,
                })?;
                let signature =
                    agent_platform_signature(headers, method, path, &secret, &timestamp);
                let mut req = request
                    .header("x-novie-timestamp", timestamp)
                    .header("x-novie-sig", format!("sha256={signature}"));
                for (key, value) in signed_forward_headers(headers) {
                    if !value.is_empty() {
                        req = req.header(key, value);
                    }
                }
                Ok(req)
            }
        }
    }
}

fn build_http_client(cfg: &TransportConfig) -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(cfg.timeout)
        .user_agent(cfg.user_agent.clone())
        .build()
        .map_err(|e| Error::InvalidArgument(format!("failed to build HTTP client: {e}")))
}

fn epoch_seconds() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}

fn signed_forward_headers(headers: &RequestHeaders) -> [(&'static str, String); 8] {
    [
        ("x-novie-org-id", headers.tenant_id.clone()),
        ("x-novie-project-id", headers.project_id.clone()),
        ("x-novie-workspace-id", headers.workspace_id.clone()),
        ("x-novie-user-id", headers.user_id.clone()),
        (
            "x-novie-service-principal",
            headers.service_principal.clone(),
        ),
        ("x-novie-session-id", headers.session_id.clone()),
        ("x-novie-request-id", headers.request_id.clone()),
        ("x-novie-auth-source", "agent_callback".to_owned()),
    ]
}

fn url_path(url: &str) -> String {
    if let Some(scheme_idx) = url.find("://") {
        let rest = &url[(scheme_idx + 3)..];
        return rest
            .find('/')
            .map(|slash| rest[slash..].to_owned())
            .unwrap_or_else(|| "/".to_owned());
    }
    url.to_owned()
}

fn map_error_envelope(status: u16, body: &Value) -> Error {
    let err = body.get("error");
    let code = err
        .and_then(|e| e.get("code"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let message = err
        .and_then(|e| e.get("message"))
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| body.to_string());
    let retry_after_ms = err
        .and_then(|e| e.get("retry_after_ms"))
        .and_then(|v| v.as_u64());
    let callback_id = body
        .get("metadata")
        .and_then(|m| m.get("callback_id"))
        .and_then(|v| v.as_str())
        .map(String::from);

    match status {
        401 | 403 => Error::Auth {
            message,
            code,
            http_status: Some(status),
            callback_id,
        },
        400 | 404 | 422 => Error::Protocol {
            message,
            code,
            http_status: Some(status),
            callback_id,
        },
        503 => Error::Unavailable {
            message,
            code,
            retry_after_ms,
            http_status: Some(status),
            callback_id,
        },
        s if s >= 500 => Error::Unavailable {
            message,
            code,
            retry_after_ms,
            http_status: Some(status),
            callback_id,
        },
        _ => Error::Callback {
            message,
            code,
            http_status: Some(status),
            callback_id,
        },
    }
}

fn backoff_ms(attempt: u32) -> u64 {
    DEFAULT_BACKOFF_MS * 2u64.pow(attempt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headers::RequestHeaders;

    fn headers() -> RequestHeaders {
        RequestHeaders {
            tenant_id: "tenant-1".into(),
            workspace_id: "workspace-1".into(),
            project_id: "project-1".into(),
            user_id: "user-1".into(),
            session_id: "session-1".into(),
            request_id: "request-1".into(),
            ..Default::default()
        }
    }

    #[test]
    fn signed_transport_uses_novie_envelope_not_bearer() {
        let transport = CallbackTransport::with_signed_headers(
            "http://platform.test",
            headers(),
            TransportConfig::default(),
        )
        .unwrap();

        let req = transport
            .apply_auth(
                transport.http.post("http://platform.test/wiki/search"),
                "POST",
                "/wiki/search",
            )
            .unwrap()
            .build()
            .unwrap();

        assert!(req.headers().get("authorization").is_none());
        assert!(req.headers().get("x-novie-timestamp").is_some());
        assert!(req.headers().get("x-novie-sig").is_some());
        assert_eq!(req.headers()["x-novie-org-id"], "tenant-1");
    }

    #[test]
    fn bearer_transport_still_supports_scoped_push_tokens() {
        let transport = CallbackTransport::with_config(
            "http://platform.test",
            "scoped-token",
            TransportConfig::default(),
        )
        .unwrap();

        let req = transport
            .apply_auth(
                transport
                    .http
                    .post("http://platform.test/internal/callbacks/agent-status"),
                "POST",
                "/internal/callbacks/agent-status",
            )
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(req.headers()["authorization"], "Bearer scoped-token");
        assert!(req.headers().get("x-novie-sig").is_none());
    }
}
