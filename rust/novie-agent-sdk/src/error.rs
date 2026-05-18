//! Error types for `novie-agent-sdk`.
//!
//! Mirrors `novie_agent_sdk.errors` 1:1 so cross-language behaviour
//! against the same `PLATFORM_CALLBACK_SPEC` is identical.

use thiserror::Error;

/// Convenience alias used throughout this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// All errors raised by this SDK.
///
/// The variants line up with the four failure modes documented in
/// `PLATFORM_CALLBACK_SPEC §5`. Token-related variants additionally cover
/// failures from `mint_callback_token` / `verify_callback_token`.
#[derive(Debug, Error)]
pub enum Error {
    /// 401 / 403 — token invalid or revoked. Not retriable.
    #[error("auth error: {message} (code={code:?}, http_status={http_status:?})")]
    Auth {
        message: String,
        code: Option<String>,
        http_status: Option<u16>,
        callback_id: Option<String>,
    },

    /// 400 / 404 / 422 — malformed request. Not retriable.
    #[error("protocol error: {message} (code={code:?}, http_status={http_status:?})")]
    Protocol {
        message: String,
        code: Option<String>,
        http_status: Option<u16>,
        callback_id: Option<String>,
    },

    /// 503 or transport error. Retriable; honours `retry_after_ms` from the server.
    #[error("platform unavailable: {message} (code={code:?}, retry_after_ms={retry_after_ms:?})")]
    Unavailable {
        message: String,
        code: Option<String>,
        retry_after_ms: Option<u64>,
        http_status: Option<u16>,
        callback_id: Option<String>,
    },

    /// Other 4xx/5xx that doesn't map to one of the above.
    #[error("callback error: {message} (code={code:?}, http_status={http_status:?})")]
    Callback {
        message: String,
        code: Option<String>,
        http_status: Option<u16>,
        callback_id: Option<String>,
    },

    /// Platform-managed LLM call was denied because the org token pool is exhausted.
    #[error("quota exceeded: {message} (org_id={org_id:?}, remaining_tokens={remaining_tokens:?})")]
    QuotaExceeded {
        message: String,
        org_id: Option<String>,
        remaining_tokens: Option<u64>,
    },

    /// LLM call was denied because the task-level budget pre-check failed.
    #[error("budget exceeded: {message} (task_id={task_id:?})")]
    BudgetExceeded {
        message: String,
        task_id: Option<String>,
    },

    /// The governance / quota service is not configured or unavailable.
    ///
    /// This is a soft error in preflight: tasks can choose to proceed
    /// without budget enforcement when governance is unavailable, or fail-safe
    /// depending on their policy.
    #[error("governance unavailable: {message}")]
    GovernanceUnavailable {
        message: String,
    },

    /// `verify_callback_token` rejected because exp ≤ now.
    #[error("callback token expired")]
    TokenExpired,

    /// `verify_callback_token` rejected because of bad header / signature / payload.
    #[error("callback token malformed: {0}")]
    TokenMalformed(String),

    /// Local programmer error — ill-formed payload, missing required fields, etc.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// JSON encode/decode failure on a payload we control.
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
}

impl Error {
    /// True for variants the caller is *expected* to retry transparently.
    ///
    /// Built-in retry inside [`crate::transport::CallbackTransport`] uses this
    /// flag implicitly via the `Unavailable` variant; callers building their
    /// own loops can use it for higher-level retries (e.g. tool retries).
    pub fn is_retryable(&self) -> bool {
        matches!(self, Error::Unavailable { .. })
    }

    /// Returns the platform-suggested backoff (ms) if the server provided one.
    pub fn retry_after_ms(&self) -> Option<u64> {
        match self {
            Error::Unavailable { retry_after_ms, .. } => *retry_after_ms,
            _ => None,
        }
    }

    /// Status code carried by the error if it originated from a real HTTP response.
    pub fn http_status(&self) -> Option<u16> {
        match self {
            Error::Auth { http_status, .. }
            | Error::Protocol { http_status, .. }
            | Error::Unavailable { http_status, .. }
            | Error::Callback { http_status, .. } => *http_status,
            Error::QuotaExceeded { .. } => Some(403),
            Error::BudgetExceeded { .. } => Some(429),
            Error::GovernanceUnavailable { .. } => Some(503),
            _ => None,
        }
    }

    /// Stable error code reported by the platform (e.g. `AUTH_EXPIRED_TOKEN`),
    /// when present.
    pub fn code(&self) -> Option<&str> {
        match self {
            Error::Auth { code, .. }
            | Error::Protocol { code, .. }
            | Error::Unavailable { code, .. }
            | Error::Callback { code, .. } => code.as_deref(),
            Error::QuotaExceeded { .. } => Some("quota_exceeded"),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for Error {
    fn from(err: reqwest::Error) -> Self {
        // reqwest transport failures (DNS, connect, timeout) map to Unavailable
        // so the built-in retry policy kicks in.
        Error::Unavailable {
            message: format!("transport error: {err}"),
            code: None,
            retry_after_ms: None,
            http_status: err.status().map(|s| s.as_u16()),
            callback_id: None,
        }
    }
}
