//! Typed A2A request headers and platform signature verification.

use axum::http::HeaderMap;
use hmac::{Hmac, Mac};
use serde::Serialize;
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;
pub const DEV_AGENT_PLATFORM_SHARED_SECRET: &str = "novie-dev-agent-platform-shared-secret";

/// A2A identity and tracing headers injected by the Novie Platform.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RequestHeaders {
    pub tenant_id: String,
    pub session_id: String,
    pub step_id: String,
    pub trace_id: String,
    pub workspace_id: String,
    pub project_id: String,
    pub user_id: String,
    pub service_principal: String,
    pub auth_source: String,
    pub request_id: String,
    pub timestamp: String,
    pub signature: String,
    pub idempotency_key: String,
    pub auth_token: String,
    pub raw: Vec<(String, String)>,
}

impl RequestHeaders {
    /// Extract standard Novie headers from an HTTP header map.
    pub fn from_header_map(headers: &HeaderMap) -> Self {
        let raw: Vec<(String, String)> = headers
            .iter()
            .filter_map(|(key, value)| {
                Some((key.as_str().to_lowercase(), value.to_str().ok()?.to_owned()))
            })
            .collect();

        let value = |name: &str| {
            raw.iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
                .unwrap_or_default()
        };

        let authorization = value("authorization");
        let org_header = value("x-novie-org-id");
        let tenant_header = value("x-novie-tenant-id");
        let workspace_header = value("x-novie-workspace-id");
        let org_id = first_non_empty(&[org_header.as_str(), tenant_header.as_str()]);
        let workspace_id = first_non_empty(&[workspace_header.as_str(), org_id.as_str()]);

        Self {
            tenant_id: org_id,
            session_id: value("x-novie-session-id"),
            step_id: value("x-novie-step-id"),
            trace_id: value("x-novie-trace-id"),
            workspace_id,
            project_id: value("x-novie-project-id"),
            user_id: value("x-novie-user-id"),
            service_principal: value("x-novie-service-principal"),
            auth_source: value("x-novie-auth-source"),
            request_id: value("x-novie-request-id"),
            timestamp: value("x-novie-timestamp"),
            signature: value("x-novie-sig"),
            idempotency_key: value("idempotency-key"),
            auth_token: authorization
                .strip_prefix("Bearer ")
                .unwrap_or(&authorization)
                .trim()
                .to_owned(),
            raw,
        }
    }

    /// Verify Python-compatible platform-signed A2A identity headers.
    pub fn verify_signature(
        &self,
        secret: &str,
        method: &str,
        path: &str,
        ttl_seconds: i64,
    ) -> Result<(), HeaderVerificationError> {
        if self.timestamp.is_empty() || self.signature.is_empty() {
            return Err(HeaderVerificationError::SignatureRequired);
        }

        let issued_at = self
            .timestamp
            .parse::<i64>()
            .map_err(|_| HeaderVerificationError::InvalidTimestamp)?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| HeaderVerificationError::InvalidTimestamp)?
            .as_secs() as i64;
        if (now - issued_at).abs() > ttl_seconds.max(1) {
            return Err(HeaderVerificationError::StaleSignature);
        }

        let expected = agent_platform_signature(self, method, path, secret, &self.timestamp);
        let provided = self
            .signature
            .strip_prefix("sha256=")
            .unwrap_or(&self.signature);
        if !constant_time_eq(expected.as_bytes(), provided.trim().as_bytes()) {
            return Err(HeaderVerificationError::InvalidSignature);
        }

        Ok(())
    }
}

/// Stable error codes returned when signed A2A headers are invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum HeaderVerificationError {
    SignatureRequired,
    MissingSharedSecret,
    InvalidTimestamp,
    StaleSignature,
    InvalidSignature,
}

impl HeaderVerificationError {
    pub fn code(self) -> &'static str {
        match self {
            HeaderVerificationError::SignatureRequired => "agent_platform_signature_required",
            HeaderVerificationError::MissingSharedSecret => "agent_platform_signature_required",
            HeaderVerificationError::InvalidTimestamp => {
                "invalid_agent_platform_signature_timestamp"
            }
            HeaderVerificationError::StaleSignature => "stale_agent_platform_signature",
            HeaderVerificationError::InvalidSignature => "invalid_agent_platform_signature",
        }
    }
}

/// Whether the current runtime mode requires signed platform headers.
pub fn requires_signed_agent_headers() -> bool {
    if std::env::var("NOVIE_AGENT_REQUIRE_SIGNED_HEADERS").as_deref() == Ok("1") {
        return true;
    }
    if env_lower("NOVIE_RUNTIME_MODE") == "production" {
        return true;
    }
    env_lower("NOVIE_ENV") == "production"
}

/// Verify headers using environment-controlled production policy.
pub fn verify_agent_request_headers(
    headers: &RequestHeaders,
    method: &str,
    path: &str,
) -> Result<(), HeaderVerificationError> {
    if !requires_signed_agent_headers() {
        return Ok(());
    }

    let secret = agent_platform_shared_secret()?;

    let ttl = std::env::var("NOVIE_AGENT_PLATFORM_SIGNATURE_TTL_SECONDS")
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(300);
    headers.verify_signature(&secret, method, path, ttl)
}

pub fn agent_platform_shared_secret() -> Result<String, HeaderVerificationError> {
    let configured = std::env::var("NOVIE_AGENT_PLATFORM_SHARED_SECRET")
        .unwrap_or_default()
        .trim()
        .to_owned();
    if !configured.is_empty() {
        return Ok(configured);
    }
    if env_lower("NOVIE_RUNTIME_MODE") == "production" || env_lower("NOVIE_ENV") == "production" {
        return Err(HeaderVerificationError::MissingSharedSecret);
    }
    Ok(DEV_AGENT_PLATFORM_SHARED_SECRET.to_owned())
}

/// Python-compatible agent-platform canonical HMAC-SHA256 signature.
pub fn agent_platform_signature(
    headers: &RequestHeaders,
    method: &str,
    path: &str,
    secret: &str,
    timestamp: &str,
) -> String {
    let canonical = [
        method.to_uppercase(),
        normalize_path(path),
        headers.tenant_id.clone(),
        headers.project_id.clone(),
        headers.workspace_id.clone(),
        headers.user_id.clone(),
        headers.service_principal.clone(),
        headers.session_id.clone(),
        headers.request_id.clone(),
        timestamp.to_owned(),
    ]
    .join("\n");

    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC accepts any key size");
    mac.update(canonical.as_bytes());
    bytes_to_lower_hex(&mac.finalize().into_bytes())
}

fn env_lower(name: &str) -> String {
    std::env::var(name)
        .unwrap_or_default()
        .trim()
        .to_lowercase()
}

fn first_non_empty(values: &[&str]) -> String {
    values
        .iter()
        .map(|value| value.trim())
        .find(|value| !value.is_empty())
        .unwrap_or("")
        .to_owned()
}

fn normalize_path(path: &str) -> String {
    if path.trim().is_empty() {
        return "/".to_owned();
    }
    if let Some(scheme_idx) = path.find("://") {
        let rest = &path[(scheme_idx + 3)..];
        return rest
            .find('/')
            .map(|slash| rest[slash..].to_owned())
            .unwrap_or_else(|| "/".to_owned());
    }
    if path.starts_with('/') {
        path.to_owned()
    } else {
        format!("/{path}")
    }
}

fn bytes_to_lower_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0_u8, |acc, (l, r)| acc | (l ^ r))
        == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderMap;

    fn signed_headers() -> RequestHeaders {
        RequestHeaders {
            tenant_id: "tenant-1".to_owned(),
            workspace_id: "workspace-1".to_owned(),
            project_id: "project-1".to_owned(),
            user_id: "user-1".to_owned(),
            service_principal: "platform".to_owned(),
            session_id: "session-1".to_owned(),
            step_id: "step-1".to_owned(),
            idempotency_key: "idem-1".to_owned(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn extracts_headers_case_insensitively() {
        let mut map = HeaderMap::new();
        map.insert("X-Novie-Org-Id", "tenant-1".parse().unwrap());
        map.insert("Idempotency-Key", "idem-1".parse().unwrap());
        map.insert("Authorization", "Bearer token-1".parse().unwrap());

        let headers = RequestHeaders::from_header_map(&map);

        assert_eq!(headers.tenant_id, "tenant-1");
        assert_eq!(headers.idempotency_key, "idem-1");
        assert_eq!(headers.auth_token, "token-1");
    }

    #[test]
    fn verifies_python_compatible_signature() {
        let mut headers = signed_headers();
        headers.signature = agent_platform_signature(
            &headers,
            "POST",
            "/invoke",
            "shared-secret",
            &headers.timestamp,
        );

        assert!(
            headers
                .verify_signature("shared-secret", "POST", "/invoke", 300)
                .is_ok()
        );
    }

    #[test]
    fn accepts_sha256_prefixed_signature() {
        let mut headers = signed_headers();
        headers.signature = format!(
            "sha256={}",
            agent_platform_signature(
                &headers,
                "POST",
                "/invoke",
                "shared-secret",
                &headers.timestamp
            )
        );

        assert!(
            headers
                .verify_signature("shared-secret", "POST", "/invoke", 300)
                .is_ok()
        );
    }

    #[test]
    fn rejects_invalid_signature() {
        let mut headers = signed_headers();
        headers.signature = "bad".to_owned();

        assert_eq!(
            headers
                .verify_signature("shared-secret", "POST", "/invoke", 300)
                .unwrap_err(),
            HeaderVerificationError::InvalidSignature
        );
    }
}
