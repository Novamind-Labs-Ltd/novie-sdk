//! `AgentStatusClient` — push channel for `POST /internal/callbacks/agent-status`.
//!
//! Differences from the RPC `PlatformServicesClient`:
//!
//! - dedicated token (`trh.scope = "agent-status"`), only valid against this URL;
//! - bare `AgentStatusEvent` body (no `{kwargs}` envelope);
//! - server returns `202 Accepted` and dedupes by `event_id`.

use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde_json::{Map, Value};

use crate::agent_status::{AgentStatusEvent, AgentStatusKind};
use crate::error::{Error, Result};
use crate::payload::{AgentInvokePayload, AgentStatusCallbackConfig};
use crate::transport::{CallbackTransport, TransportConfig};

const DEFAULT_PUSH_TIMEOUT_SECS: u64 = 5;
/// One transparent retry by default — `event_id` makes the server idempotent.
const DEFAULT_PUSH_MAX_RETRIES: u32 = 1;

/// Client for the dedicated agent-status push endpoint.
///
/// Built around an [`Arc<CallbackTransport>`] so multiple clones share the
/// same connection pool. Holds the agent-status URL + scoped token (distinct
/// from the RPC token).
#[derive(Debug, Clone)]
pub struct AgentStatusClient {
    config: AgentStatusCallbackConfig,
    agent_id: String,
    transport: Arc<CallbackTransport>,
}

impl AgentStatusClient {
    /// Build a client from a parsed config + caller-supplied `agent_id`.
    pub fn new(config: AgentStatusCallbackConfig, agent_id: impl Into<String>) -> Result<Self> {
        Self::with_transport_config(config, agent_id, default_transport_cfg())
    }

    /// Build with a custom [`TransportConfig`] (e.g. tighter timeout for tests).
    pub fn with_transport_config(
        config: AgentStatusCallbackConfig,
        agent_id: impl Into<String>,
        transport_cfg: TransportConfig,
    ) -> Result<Self> {
        let agent_id = agent_id.into();
        if agent_id.is_empty() {
            return Err(Error::InvalidArgument("agent_id is required".into()));
        }
        // base_url is unused for absolute push URLs but must be non-empty;
        // we copy the push URL itself as the placeholder.
        let transport = CallbackTransport::with_config(
            config.url.clone(),
            config.token.clone(),
            transport_cfg,
        )?;
        Ok(Self {
            config,
            agent_id,
            transport: Arc::new(transport),
        })
    }

    /// Convenience: pull `agent_status_callback` straight out of an invoke
    /// payload. Returns `Err(Error::InvalidArgument)` if the platform did not
    /// enable the durable push channel.
    pub fn from_invoke_payload(
        payload: &AgentInvokePayload,
        agent_id_override: Option<&str>,
    ) -> Result<Self> {
        let cfg = payload
            .agent_status_callback
            .as_ref()
            .ok_or_else(|| {
                Error::InvalidArgument(
                    "invoke payload missing 'agent_status_callback'; \
                     platform did not enable durable agent status push"
                        .into(),
                )
            })?
            .clone();
        let agent_id = agent_id_override
            .map(str::to_string)
            .or_else(|| payload.agent_id().map(str::to_string))
            .ok_or_else(|| {
                Error::InvalidArgument(
                    "agent_id missing; pass it explicitly or include top-level \
                     'agent_id' in the invoke payload"
                        .into(),
                )
            })?;
        Self::new(cfg, agent_id)
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    pub fn session_id(&self) -> Option<&str> {
        self.config.session_id.as_deref()
    }

    pub fn thread_id(&self) -> Option<&str> {
        self.config.thread_id.as_deref()
    }

    pub fn plan_id(&self) -> Option<&str> {
        self.config.plan_id.as_deref()
    }

    /// Build and push an `AgentStatusEvent` in one call. Returns the
    /// `event_id` the platform acknowledged — which equals the one we minted.
    pub async fn report(&self, opts: ReportOptions<'_>) -> Result<String> {
        let event = AgentStatusEvent {
            event_id: opts
                .event_id
                .map(str::to_string)
                .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string()),
            occurred_at: opts.occurred_at.unwrap_or_else(Utc::now),
            kind: opts.kind,
            agent_id: self.agent_id.clone(),
            task_id: opts.task_id.to_string(),
            session_id: self.config.session_id.clone(),
            thread_id: opts
                .thread_id
                .map(str::to_string)
                .or_else(|| self.config.thread_id.clone()),
            plan_id: opts
                .plan_id
                .map(str::to_string)
                .or_else(|| self.config.plan_id.clone()),
            turn: opts.turn,
            payload: opts.payload.unwrap_or_default(),
        };
        self.send(&event).await
    }

    /// Send a fully-constructed event (advanced path).
    pub async fn send(&self, event: &AgentStatusEvent) -> Result<String> {
        if event.agent_id != self.agent_id {
            return Err(Error::InvalidArgument(format!(
                "event.agent_id={:?} does not match client agent_id={:?}; the platform will reject with 422",
                event.agent_id, self.agent_id
            )));
        }
        let body = serde_json::to_value(event)?;
        // The push URL is absolute (`config.url`), and CallbackTransport's
        // base_url was set to the same value; we POST to "" so the absolute
        // URL is respected by the path-or-url resolver.
        self.transport.push(&self.config.url, &body).await?;
        Ok(event.event_id.clone())
    }
}

fn default_transport_cfg() -> TransportConfig {
    TransportConfig {
        timeout: Duration::from_secs(DEFAULT_PUSH_TIMEOUT_SECS),
        max_retries: DEFAULT_PUSH_MAX_RETRIES,
        ..TransportConfig::default()
    }
}

/// Optional knobs for [`AgentStatusClient::report`].
///
/// Built as a struct (not many positional args) so the call site stays
/// readable when only a couple of fields are set.
#[derive(Debug, Default)]
pub struct ReportOptions<'a> {
    pub kind: AgentStatusKind,
    pub task_id: &'a str,
    pub payload: Option<Map<String, Value>>,
    pub turn: Option<i32>,
    pub event_id: Option<&'a str>,
    pub occurred_at: Option<DateTime<Utc>>,
    pub thread_id: Option<&'a str>,
    pub plan_id: Option<&'a str>,
}
