//! Session timeline contracts mirrored from `novie_protocol.contracts.session`.
//!
//! Only the shapes the agent SDK needs to send / receive are encoded here —
//! the platform-side `SessionTimelineService` keeps richer types but they are
//! not exposed across the callback boundary.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionEventSource {
    Chat,
    Planning,
    Dispatch,
    Gate,
    Callback,
    System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    Active,
    Waiting,
    Completed,
    Failed,
    Cancelled,
}

/// Lifecycle phase distinct from `SessionStatus`. Mirrors
/// `novie_protocol.contracts.session.SessionLifecycleState` introduced
/// in protocol v0.1.2.
///
/// `#[serde(default)]` on `Session.lifecycle_state` keeps backward-compatible
/// deserialisation: pre-v0.1.2 payloads (no field) decode as `Active`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionLifecycleState {
    Active,
    Idle,
    Closed,
    Archived,
}

impl Default for SessionLifecycleState {
    fn default() -> Self {
        Self::Active
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub session_id: String,
    pub tenant_id: String,
    pub workspace_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default = "default_status")]
    pub status: SessionStatus,
    #[serde(default)]
    pub lifecycle_state: SessionLifecycleState,
    #[serde(default)]
    pub thread_id: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub last_event_seq: i64,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

fn default_status() -> SessionStatus {
    SessionStatus::Active
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEvent {
    pub seq: i64,
    pub event_id: String,
    pub occurred_at: DateTime<Utc>,
    pub session_id: String,
    pub source: SessionEventSource,
    pub kind: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub tenant_id: String,
    #[serde(default)]
    pub workspace_id: String,
    #[serde(default)]
    pub thread_id: Option<String>,
    #[serde(default)]
    pub payload: Map<String, Value>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

impl SessionEvent {
    /// Build an event with `seq=0` (server assigns final seq) and a fresh
    /// `event_id`. Mirrors `new_session_event(...)` in the protocol crate.
    pub fn new_for_record(
        session_id: impl Into<String>,
        source: SessionEventSource,
        kind: impl Into<String>,
    ) -> Self {
        Self {
            seq: 0,
            event_id: format!("sev-{}", &uuid::Uuid::new_v4().simple().to_string()[..16]),
            occurred_at: Utc::now(),
            session_id: session_id.into(),
            source,
            kind: kind.into(),
            summary: String::new(),
            tenant_id: String::new(),
            workspace_id: String::new(),
            thread_id: None,
            payload: Map::new(),
            metadata: Map::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEventsPage {
    pub session_id: String,
    #[serde(default)]
    pub events: Vec<SessionEvent>,
    #[serde(default)]
    pub next_since: i64,
    #[serde(default)]
    pub has_more: bool,
}
