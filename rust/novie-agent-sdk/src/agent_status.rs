//! `AgentStatusEvent` contract — agent → platform observability event.
//!
//! Mirrors `novie_protocol.contracts.agent_status` so server-side validation
//! treats Python-emitted and Rust-emitted events identically.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Discriminator for the agent-reported lifecycle event.
///
/// Kept as a closed enum so the SDK refuses to send unknown kinds; new kinds
/// must land in both Python and Rust at the same time. Wire format is the
/// snake_case literal (matches `Literal[...]` in Python).
///
/// `Default` is `StatusUpdate` — the most semantically neutral kind, picked so
/// that `ReportOptions::default()` in `agent_status_client` is safe even
/// without an explicit kind.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatusKind {
    TurnStart,
    TurnEnd,
    ToolCall,
    ToolResult,
    ProgressNote,
    #[default]
    StatusUpdate,
    ArtifactCreated,
}

/// Event posted to `POST /internal/callbacks/agent-status`.
///
/// Field order intentionally tracks the dataclass in `agent_status.py`. The
/// server stores this verbatim into the session timeline so the round-trip
/// `mint` → `record` → SSE replay remains lossless.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatusEvent {
    pub event_id: String,
    pub occurred_at: DateTime<Utc>,
    pub kind: AgentStatusKind,
    pub agent_id: String,
    pub task_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn: Option<i32>,
    #[serde(default)]
    pub payload: Map<String, Value>,
}

impl AgentStatusEvent {
    /// Convenience constructor that fills `event_id` (uuid4 hex) and
    /// `occurred_at` (now UTC) — matches the Python `AgentStatusClient.report`
    /// defaults so opt-out is explicit.
    pub fn now(
        kind: AgentStatusKind,
        agent_id: impl Into<String>,
        task_id: impl Into<String>,
    ) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4().simple().to_string(),
            occurred_at: Utc::now(),
            kind,
            agent_id: agent_id.into(),
            task_id: task_id.into(),
            session_id: None,
            thread_id: None,
            plan_id: None,
            turn: None,
            payload: Map::new(),
        }
    }
}
