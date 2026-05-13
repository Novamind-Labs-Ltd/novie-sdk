//! Typed A2A request headers and platform signature verification.

use axum::http::HeaderMap;
use hmac::{Hmac, Mac};
use serde::Serialize;
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

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

        Self {
            tenant_id: value("x-novie-tenant-id"),
            session_id: value("x-novie-session-id"),
            step_id: value("x-novie-step-id"),
            trace_id: value("x-novie-trace-id"),
            workspace_id: value("x-novie-workspace-id"),
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

        let expected = a2a_header_signature(self, secret);
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
            HeaderVerificationError::SignatureRequired => "a2a_signature_required",
            HeaderVerificationError::MissingSharedSecret => "a2a_signature_required",
            HeaderVerificationError::InvalidTimestamp => "invalid_a2a_signature_timestamp",
            HeaderVerificationError::StaleSignature => "stale_a2a_signature",
            HeaderVerificationError::InvalidSignature => "invalid_a2a_signature",
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
) -> Result<(), HeaderVerificationError> {
    if !requires_signed_agent_headers() {
        return Ok(());
    }

    let secret = std::env::var("NOVIE_A2A_SHARED_SECRET")
        .unwrap_or_default()
        .trim()
        .to_owned();
    if secret.is_empty() {
        return Err(HeaderVerificationError::MissingSharedSecret);
    }

    let ttl = std::env::var("NOVIE_A2A_SIGNATURE_TTL_SECONDS")
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(300);
    headers.verify_signature(&secret, ttl)
}

/// Python-compatible canonical HMAC-SHA256 signature.
pub fn a2a_header_signature(headers: &RequestHeaders, secret: &str) -> String {
    let canonical = [
        headers.tenant_id.as_str(),
        headers.workspace_id.as_str(),
        headers.project_id.as_str(),
        headers.user_id.as_str(),
        headers.service_principal.as_str(),
        headers.session_id.as_str(),
        headers.step_id.as_str(),
        headers.idempotency_key.as_str(),
        headers.timestamp.as_str(),
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
        map.insert("X-Novie-Tenant-Id", "tenant-1".parse().unwrap());
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
        headers.signature = a2a_header_signature(&headers, "shared-secret");

        assert!(headers.verify_signature("shared-secret", 300).is_ok());
    }

    #[test]
    fn accepts_sha256_prefixed_signature() {
        let mut headers = signed_headers();
        headers.signature = format!("sha256={}", a2a_header_signature(&headers, "shared-secret"));

        assert!(headers.verify_signature("shared-secret", 300).is_ok());
    }

    #[test]
    fn rejects_invalid_signature() {
        let mut headers = signed_headers();
        headers.signature = "bad".to_owned();

        assert_eq!(
            headers.verify_signature("shared-secret", 300).unwrap_err(),
            HeaderVerificationError::InvalidSignature
        );
    }
}
