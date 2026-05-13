//! `AgentCallScope` — per-invocation workspace + credential hint contract.
//!
//! Mirrors `novie_protocol.contracts.call_scope`. See the Python module for
//! the full design rationale; in short, the platform does **not** enforce
//! these fields — they are advisory hints the agent uses to pick a workspace
//! layout, cache TTL, and cleanup hook.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceScope {
    PerTask,
    PerSession,
    #[default]
    Shared,
    None,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenScopeKind {
    PerTask,
    PerSession,
    #[default]
    None,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CleanupWhen {
    OnStepComplete,
    OnPlanComplete,
    AgentManaged,
    #[default]
    NoCleanup,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CredentialHint {
    #[serde(default)]
    pub kind: TokenScopeKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorized_by: Option<String>,
    #[serde(default)]
    pub allowed_resources: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentCallScope {
    #[serde(default)]
    pub workspace_scope: WorkspaceScope,
    #[serde(default)]
    pub cleanup_when: CleanupWhen,
    #[serde(default)]
    pub tenant_id: String,
    #[serde(default)]
    pub workspace_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    #[serde(default)]
    pub scope_key: String,
    #[serde(default)]
    pub credentials: CredentialHint,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cleanup_callback_url: Option<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

/// Pull `AgentCallScope` from either a top-level invoke payload or the
/// `inputs` dict.
///
/// Matches `extract_call_scope(source)` in Python — accepts either path
/// (HTTP invoke vs in-process) and silently returns `None` instead of
/// raising on shape errors so agents fall back to safe defaults.
pub fn extract_call_scope(source: &Value) -> Option<AgentCallScope> {
    let map = source.as_object()?;
    let raw = map
        .get("call_scope")
        .or_else(|| map.get("__call_scope__"))?;
    if !raw.is_object() {
        tracing::warn!(
            type_name = ?raw,
            "call_scope present but not a JSON object; ignoring",
        );
        return None;
    }
    match serde_json::from_value(raw.clone()) {
        Ok(scope) => Some(scope),
        Err(err) => {
            tracing::warn!(error = %err, "failed to parse call_scope; ignoring");
            None
        }
    }
}
