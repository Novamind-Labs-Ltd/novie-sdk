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
    pub correlation: Option<RunCorrelation>,
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
            correlation: None,
            payload: Map::new(),
            metadata: Map::new(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunCorrelation {
    pub tenant_id: String,
    pub workspace_id: String,
    #[serde(default)]
    pub project_id: String,
    pub principal_id: String,
    pub session_id: String,
    pub turn_id: String,
    pub root_run_id: String,
    pub thread_id: String,
    pub request_id: String,
    #[serde(default)]
    pub workflow_id: String,
    #[serde(default)]
    pub workflow_run_id: String,
    #[serde(default)]
    pub attempt_id: String,
    #[serde(default)]
    pub entity_type: String,
    #[serde(default)]
    pub entity_id: String,
    #[serde(default)]
    pub causation_event_id: String,
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

#[cfg(test)]
mod tests {
    use super::{RunCorrelation, SessionEvent, SessionEventSource};

    #[test]
    fn session_event_roundtrips_optional_run_correlation() {
        let mut event = SessionEvent::new_for_record(
            "session-1",
            SessionEventSource::System,
            "run.started",
        );
        event.correlation = Some(RunCorrelation {
            tenant_id: "tenant-1".to_owned(),
            workspace_id: "workspace-1".to_owned(),
            project_id: "project-1".to_owned(),
            principal_id: "user-1".to_owned(),
            session_id: "session-1".to_owned(),
            turn_id: "turn-1".to_owned(),
            root_run_id: "root-1".to_owned(),
            thread_id: "thread-1".to_owned(),
            request_id: "request-1".to_owned(),
            ..RunCorrelation::default()
        });

        let encoded = serde_json::to_string(&event).expect("encode session event");
        let decoded: SessionEvent =
            serde_json::from_str(&encoded).expect("decode session event");

        assert_eq!(
            decoded.correlation.as_ref().map(|c| c.root_run_id.as_str()),
            Some("root-1")
        );
    }
}
