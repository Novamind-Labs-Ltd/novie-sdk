//! Checkpoint contracts mirrored from `novie_protocol.contracts.memory`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThreadKind {
    Dispatch,
    Agent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointSnapshot {
    pub checkpoint_id: String,
    pub thread_id: String,
    pub thread_kind: ThreadKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_checkpoint_id: Option<String>,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub state: Map<String, Value>,
    #[serde(default)]
    pub pending_writes: Vec<Map<String, Value>>,
}
