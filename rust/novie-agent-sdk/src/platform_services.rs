//! `PlatformServicesClient` — Rust port of `HttpCallbackPlatformServices`.
//!
//! Exposes typed proxies that share one [`CallbackTransport`]. Notebook-style
//! `memory.recall` / `memory.remember` RPCs were removed from the callback bus;
//! curated knowledge uses [`WikiProxy::search`].
//!
//! This file is intentionally one big module so that the wire surface stays
//! together — adding a new method should only need touching one place, just
//! like the Python `client.py`.

use std::sync::Arc;

use serde_json::{Map, Value, json};

use crate::error::{Error, Result};
use crate::headers::RequestHeaders;
use crate::memory::CheckpointSnapshot;
use crate::payload::AgentInvokePayload;
use crate::session::{Session, SessionEvent, SessionEventsPage};
use crate::transport::{CallbackTransport, TransportConfig};

/// Top-level client. Cheap to clone — internal state is one `Arc<CallbackTransport>`.
#[derive(Debug, Clone)]
pub struct PlatformServicesClient {
    transport: Arc<CallbackTransport>,
    pub wiki: WikiProxy,
    pub usage: UsageProxy,
    pub review: ReviewProxy,
    pub events: EventsProxy,
    pub audit: AuditProxy,
    pub policy: PolicyProxy,
    pub checkpoint: CheckpointProxy,
    pub time_travel: TimeTravelProxy,
    pub sessions: SessionsProxy,
}

impl PlatformServicesClient {
    pub fn new(base_url: impl Into<String>, headers: RequestHeaders) -> Result<Self> {
        Self::with_config(base_url, headers, TransportConfig::default())
    }

    pub fn with_config(
        base_url: impl Into<String>,
        headers: RequestHeaders,
        cfg: TransportConfig,
    ) -> Result<Self> {
        let transport = Arc::new(CallbackTransport::with_signed_headers(
            base_url, headers, cfg,
        )?);
        Ok(Self::from_transport(transport))
    }

    /// Build using a pre-existing transport (e.g. shared between
    /// `AgentStatusClient` and the RPC client in tests).
    pub fn from_transport(transport: Arc<CallbackTransport>) -> Self {
        Self {
            wiki: WikiProxy {
                t: transport.clone(),
            },
            usage: UsageProxy {
                t: transport.clone(),
            },
            review: ReviewProxy {
                t: transport.clone(),
            },
            events: EventsProxy {
                t: transport.clone(),
            },
            audit: AuditProxy {
                t: transport.clone(),
            },
            policy: PolicyProxy {
                t: transport.clone(),
            },
            checkpoint: CheckpointProxy {
                t: transport.clone(),
            },
            time_travel: TimeTravelProxy {
                t: transport.clone(),
            },
            sessions: SessionsProxy {
                t: transport.clone(),
            },
            transport,
        }
    }

    /// Pull the RPC `platform_callback` base URL out of an invoke payload.
    pub fn from_invoke_payload(
        payload: &AgentInvokePayload,
        headers: RequestHeaders,
    ) -> Result<Self> {
        let cfg = payload.platform_callback.as_ref().ok_or_else(|| {
            Error::InvalidArgument(
                "invoke payload missing 'platform_callback' field; \
                 did DispatchService include the callback base URL?"
                    .into(),
            )
        })?;
        Self::new(cfg.base_url.clone(), headers)
    }

    pub fn transport(&self) -> &Arc<CallbackTransport> {
        &self.transport
    }
}

// ---------------------------------------------------------------------------
// Proxy structs
//
// Each proxy is a thin Arc<Transport> + a couple of methods. We keep them
// public-fields rather than encapsulated because they have no invariants of
// their own — all logic lives in the transport.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WikiProxy {
    t: Arc<CallbackTransport>,
}

impl WikiProxy {
    pub async fn search(
        &self,
        query: &str,
        top_k: u32,
        project_id: Option<&str>,
    ) -> Result<Vec<Value>> {
        let v = self
            .t
            .call(
                "wiki",
                "search",
                json!({ "query": query, "top_k": top_k, "project_id": project_id }),
            )
            .await?;
        decode_value_list(v)
    }
}

#[derive(Debug, Clone)]
pub struct UsageProxy {
    t: Arc<CallbackTransport>,
}

impl UsageProxy {
    pub async fn record(&self, record: &Value) -> Result<()> {
        self.t
            .call("usage", "record", json!({ "record": record }))
            .await?;
        Ok(())
    }

    pub async fn get_summary(&self, opts: GetSummaryOptions<'_>) -> Result<Value> {
        self.t
            .call(
                "usage",
                "get_summary",
                json!({
                    "scope": opts.scope,
                    "scope_value": opts.scope_value,
                    "breakdown_by": opts.breakdown_by,
                }),
            )
            .await
    }

    pub async fn list_records(&self, opts: ListUsageRecordsOptions<'_>) -> Result<Vec<Value>> {
        let v = self
            .t
            .call(
                "usage",
                "list_records",
                json!({
                    "session_id": opts.session_id,
                    "thread_id": opts.thread_id,
                    "agent_id": opts.agent_id,
                    "limit": opts.limit,
                }),
            )
            .await?;
        decode_value_list(v)
    }
}

/// `usage.scope` is a `Literal` in Python — keep it open for forward-compat
/// (new dimensions can be added without recompiling agents).
#[derive(Debug, Default)]
pub struct GetSummaryOptions<'a> {
    pub scope: &'a str,
    pub scope_value: Option<&'a str>,
    pub breakdown_by: Option<&'a str>,
}

#[derive(Debug, Default)]
pub struct ListUsageRecordsOptions<'a> {
    pub session_id: Option<&'a str>,
    pub thread_id: Option<&'a str>,
    pub agent_id: Option<&'a str>,
    pub limit: u32,
}

#[derive(Debug, Clone)]
pub struct ReviewProxy {
    t: Arc<CallbackTransport>,
}

impl ReviewProxy {
    pub async fn open_gate(&self, gate_payload: &Value) -> Result<String> {
        let v = self
            .t
            .call(
                "review",
                "open_gate",
                json!({ "gate_payload": gate_payload }),
            )
            .await?;
        v.as_str().map(String::from).ok_or_else(|| Error::Protocol {
            message: format!("review.open_gate returned non-string: {v}"),
            code: None,
            http_status: None,
            callback_id: None,
        })
    }

    pub async fn wait_for_resolution(&self, gate_id: &str) -> Result<Map<String, Value>> {
        let v = self
            .t
            .call(
                "review",
                "wait_for_resolution",
                json!({ "gate_id": gate_id }),
            )
            .await?;
        Ok(v.as_object().cloned().unwrap_or_default())
    }
}

#[derive(Debug, Clone)]
pub struct EventsProxy {
    t: Arc<CallbackTransport>,
}

impl EventsProxy {
    pub async fn publish(&self, topic: &str, payload: &Value) -> Result<()> {
        self.t
            .call(
                "events",
                "publish",
                json!({ "topic": topic, "payload": payload }),
            )
            .await?;
        Ok(())
    }

    /// Stub mirroring the Python implementation: subscribe is intentionally
    /// not exposed over HTTP callback. Use SSE / Redis from the
    /// `novie-cortex` crate instead.
    pub fn subscribe<T>(&self, _topic: &str) -> Result<T> {
        Err(Error::InvalidArgument(
            "events.subscribe is not available over HTTP callback (SPEC v1)".into(),
        ))
    }
}

#[derive(Debug, Clone)]
pub struct AuditProxy {
    t: Arc<CallbackTransport>,
}

impl AuditProxy {
    pub async fn record(&self, event: &Value) -> Result<()> {
        self.t
            .call("audit", "record", json!({ "event": event }))
            .await?;
        Ok(())
    }

    pub async fn query(&self, opts: AuditQueryOptions<'_>) -> Result<Vec<Value>> {
        let v = self
            .t
            .call(
                "audit",
                "query",
                json!({
                    "kinds": opts.kinds,
                    "thread_id": opts.thread_id,
                    "limit": opts.limit,
                }),
            )
            .await?;
        decode_value_list(v)
    }
}

#[derive(Debug, Default)]
pub struct AuditQueryOptions<'a> {
    pub kinds: &'a [&'a str],
    pub thread_id: Option<&'a str>,
    pub limit: u32,
}

#[derive(Debug, Clone)]
pub struct PolicyProxy {
    t: Arc<CallbackTransport>,
}

impl PolicyProxy {
    pub async fn evaluate(&self, scenario: &str, payload: &Value) -> Result<Value> {
        self.t
            .call(
                "policy",
                "evaluate",
                json!({
                    "request": {
                        "scenario": scenario,
                        "payload": payload,
                    }
                }),
            )
            .await
    }
}

#[derive(Debug, Clone)]
pub struct CheckpointProxy {
    t: Arc<CallbackTransport>,
}

impl CheckpointProxy {
    pub async fn get(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
    ) -> Result<Option<CheckpointSnapshot>> {
        let v = self
            .t
            .call(
                "checkpoint",
                "get",
                json!({
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                }),
            )
            .await?;
        if v.is_null() {
            return Ok(None);
        }
        Ok(Some(serde_json::from_value(v)?))
    }

    pub async fn list_history(
        &self,
        thread_id: &str,
        limit: u32,
    ) -> Result<Vec<CheckpointSnapshot>> {
        let v = self
            .t
            .call(
                "checkpoint",
                "list_history",
                json!({ "thread_id": thread_id, "limit": limit }),
            )
            .await?;
        decode_list(v)
    }
}

#[derive(Debug, Clone)]
pub struct TimeTravelProxy {
    t: Arc<CallbackTransport>,
}

impl TimeTravelProxy {
    pub async fn list_history(
        &self,
        thread_id: &str,
        limit: u32,
    ) -> Result<Vec<CheckpointSnapshot>> {
        let v = self
            .t
            .call(
                "time_travel",
                "list_history",
                json!({ "thread_id": thread_id, "limit": limit }),
            )
            .await?;
        decode_list(v)
    }

    pub async fn fork_from(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
        reason: &str,
    ) -> Result<String> {
        let v = self
            .t
            .call(
                "time_travel",
                "fork_from",
                json!({
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                    "reason": reason,
                }),
            )
            .await?;
        v.as_str().map(String::from).ok_or_else(|| Error::Protocol {
            message: format!("time_travel.fork_from returned non-string: {v}"),
            code: None,
            http_status: None,
            callback_id: None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SessionsProxy {
    t: Arc<CallbackTransport>,
}

impl SessionsProxy {
    /// Append an event. Server-side overrides `session_id`/`tenant_id`/
    /// `workspace_id` from the token claims and forces `source = "callback"`,
    /// so cosmetic mismatches between caller and token won't fail the call.
    pub async fn record(&self, event: &SessionEvent) -> Result<SessionEvent> {
        let v = self
            .t
            .call("sessions", "record", json!({ "event": event }))
            .await?;
        Ok(serde_json::from_value(v)?)
    }

    pub async fn get_session(&self, session_id: &str) -> Result<Option<Session>> {
        let v = self
            .t
            .call(
                "sessions",
                "get_session",
                json!({ "session_id": session_id }),
            )
            .await?;
        if v.is_null() {
            return Ok(None);
        }
        Ok(Some(serde_json::from_value(v)?))
    }

    pub async fn list_sessions(&self, limit: u32) -> Result<Vec<Session>> {
        let v = self
            .t
            .call("sessions", "list_sessions", json!({ "limit": limit }))
            .await?;
        decode_list(v)
    }

    pub async fn list_events(
        &self,
        session_id: &str,
        since: i64,
        limit: u32,
    ) -> Result<SessionEventsPage> {
        let v = self
            .t
            .call(
                "sessions",
                "list_events",
                json!({
                    "session_id": session_id,
                    "since": since,
                    "limit": limit,
                }),
            )
            .await?;
        Ok(serde_json::from_value(v)?)
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn decode_list<T: for<'de> serde::Deserialize<'de>>(v: Value) -> Result<Vec<T>> {
    if v.is_null() {
        return Ok(Vec::new());
    }
    Ok(serde_json::from_value(v)?)
}

fn decode_value_list(v: Value) -> Result<Vec<Value>> {
    if v.is_null() {
        return Ok(Vec::new());
    }
    match v {
        Value::Array(items) => Ok(items),
        other => Err(Error::Protocol {
            message: format!("expected JSON array, got {other}"),
            code: None,
            http_status: None,
            callback_id: None,
        }),
    }
}
