//! `AgentManifestV2` types — aligned with Python `novie_protocol.contracts.agent_sdk_v2`.
//!
//! The JSON representation is the canonical interchange format for agent
//! discovery and invocation.  Fields match the Python dataclass 1-to-1 so
//! that `.well-known/agent.json` files serialised by either SDK are consumed
//! transparently.

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

/// Wire format version tag written into every manifest.
pub const MANIFEST_SCHEMA: &str = "https://novie.dev/schemas/agent-manifest-v2.json";
const DESCRIPTION_MIN_LENGTH: usize = 30;

// ─────────────────────────────────────────────────────────────────────────────
// Protocol mode
// ─────────────────────────────────────────────────────────────────────────────

/// A2A protocol invocation mode.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolMode {
    /// Synchronous `POST /invoke` -> `200 { output }`.
    #[default]
    Simple,
    /// Streaming `POST /stream` -> NDJSON event sequence.
    Stream,
    /// Async task `POST /tasks` -> poll `GET /tasks/{id}` / events / result.
    Tasks,
}

/// Agent-side accepted-work durability claim.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DurabilityLevel {
    /// Stateless one-shot; no accepted work survives past the response.
    #[default]
    None,
    /// One-shot idempotency/result cache survives retries/restart.
    ResultCache,
    /// Async task records/events/results survive restart.
    TaskStore,
}

// ─────────────────────────────────────────────────────────────────────────────
// Execution hints
// ─────────────────────────────────────────────────────────────────────────────

/// Execution behaviour hints communicated through the manifest.
///
/// These let the Platform configure retry policy, timeout budget, and
/// scheduling without agent-specific logic in the workflow.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionHints {
    /// Expected wall-clock duration in seconds (Platform uses as soft timeout).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_duration_seconds: Option<u32>,

    /// Hard timeout after which the Platform cancels the task.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_duration_seconds: Option<u32>,

    /// Whether the same `Idempotency-Key` always produces the same result.
    #[serde(default)]
    pub idempotent: bool,

    /// Whether the agent supports `POST /tasks/{id}/cancel`.
    /// Only valid for `tasks` protocol mode.
    #[serde(default)]
    pub supports_cancel: bool,

    /// Whether the agent can resume a task after a worker restart.
    #[serde(default)]
    pub supports_resume: bool,

    /// Whether the agent emits A2A events (stream or task events endpoint).
    #[serde(default)]
    pub emits_events: bool,

    /// Agent-side durability guarantee for accepted work.
    #[serde(default)]
    pub durability: DurabilityLevel,
}

// ─────────────────────────────────────────────────────────────────────────────
// Agent kind / runtime
// ─────────────────────────────────────────────────────────────────────────────

/// Broad capability category.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentKind {
    ExpertBasic,
    ExpertComplex,
    Coordinator,
    Observer,
}

/// How the agent binary is packaged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentRuntime {
    #[serde(rename = "external_a2a")]
    ExternalA2A,
    #[serde(rename = "in_process")]
    InProcess,
}

// ─────────────────────────────────────────────────────────────────────────────
// Manifest
// ─────────────────────────────────────────────────────────────────────────────

/// Unified agent descriptor — aligns 1:1 with Python `AgentManifestV2`.
#[derive(Debug, Clone, Serialize)]
pub struct AgentManifestV2 {
    /// JSON Schema URI for tooling validation.
    #[serde(rename = "$schema", default = "default_schema")]
    pub schema: String,

    /// Stable, globally unique agent identifier (e.g. `"novie-cortex"`).
    pub agent_id: String,

    /// Human-readable display name.
    pub name: String,

    /// Semantic version string (e.g. `"0.2.0"`).
    pub version: String,

    /// Broad capability category.
    pub kind: AgentKind,

    /// How the agent is packaged.
    pub runtime: AgentRuntime,

    /// A2A protocol mode used to invoke this agent.
    #[serde(default)]
    pub protocol_mode: ProtocolMode,

    /// Base URL where this agent is reachable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,

    /// String capability tags (e.g. `["code_review", "github"]`).
    #[serde(default)]
    pub capabilities: Vec<String>,

    /// First-class capability contracts advertised by this agent.
    #[serde(default)]
    pub capability_manifest: Vec<Value>,

    /// HITL gate identifiers declared by the agent.
    #[serde(default)]
    pub declared_gates: Vec<String>,

    /// Execution behaviour hints.
    #[serde(default)]
    pub execution: ExecutionHints,

    /// Secrets the agent requires the Platform to inject.
    #[serde(default)]
    pub required_secrets: Vec<String>,

    /// Whether the agent has streaming behavior.
    #[serde(default)]
    pub supports_streaming: bool,

    /// Sandbox isolation hint.
    #[serde(default = "default_sandbox_isolation")]
    pub sandbox_isolation: String,

    /// Optional path where task bundles are mounted.
    #[serde(default)]
    pub task_bundles_path: String,

    /// Extension metadata that remains outside the stable top-level contract.
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

fn default_schema() -> String {
    MANIFEST_SCHEMA.to_owned()
}

fn default_sandbox_isolation() -> String {
    "shared".to_owned()
}

#[derive(Debug, Deserialize)]
struct RawAgentManifestV2 {
    #[serde(rename = "$schema", default = "default_schema")]
    schema: String,
    #[serde(default)]
    agent_id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    version: String,
    #[serde(default = "default_agent_kind")]
    kind: AgentKind,
    #[serde(default = "default_agent_runtime")]
    runtime: AgentRuntime,
    #[serde(default)]
    protocol_mode: Option<ProtocolMode>,
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default)]
    capabilities: Vec<String>,
    #[serde(default)]
    capability_manifest: Vec<Value>,
    #[serde(default)]
    declared_gates: Vec<String>,
    #[serde(default)]
    execution: Option<ExecutionHints>,
    #[serde(default)]
    required_secrets: Vec<String>,
    #[serde(default)]
    supports_streaming: bool,
    #[serde(default = "default_sandbox_isolation")]
    sandbox_isolation: String,
    #[serde(default)]
    task_bundles_path: String,
    #[serde(default)]
    metadata: Map<String, Value>,
    #[serde(default)]
    supports_cancel: Option<bool>,
    #[serde(default)]
    supports_resume: Option<bool>,
}

fn default_agent_kind() -> AgentKind {
    AgentKind::ExpertBasic
}

fn default_agent_runtime() -> AgentRuntime {
    AgentRuntime::ExternalA2A
}

fn metadata_protocol_mode(metadata: &Map<String, Value>) -> Option<ProtocolMode> {
    metadata
        .get("protocol_mode")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
}

fn metadata_bool(metadata: &Map<String, Value>, key: &str) -> Option<bool> {
    metadata.get(key).and_then(Value::as_bool)
}

fn metadata_durability(metadata: &Map<String, Value>) -> Option<DurabilityLevel> {
    metadata
        .get("durability")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
}

fn clean_metadata(metadata: Map<String, Value>) -> Map<String, Value> {
    metadata
        .into_iter()
        .filter(|(key, _)| {
            !matches!(
                key.as_str(),
                "protocol_mode"
                    | "supports_cancel"
                    | "supports_resume"
                    | "supports_streaming"
                    | "emits_events"
                    | "durability"
            )
        })
        .collect()
}

impl<'de> Deserialize<'de> for AgentManifestV2 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawAgentManifestV2::deserialize(deserializer)?;
        let protocol_mode = raw
            .protocol_mode
            .or_else(|| metadata_protocol_mode(&raw.metadata))
            .unwrap_or_default();
        let execution = raw.execution.unwrap_or_else(|| ExecutionHints {
            supports_cancel: raw
                .supports_cancel
                .or_else(|| metadata_bool(&raw.metadata, "supports_cancel"))
                .unwrap_or(false),
            supports_resume: raw
                .supports_resume
                .or_else(|| metadata_bool(&raw.metadata, "supports_resume"))
                .unwrap_or(false),
            emits_events: metadata_bool(&raw.metadata, "emits_events").unwrap_or(matches!(
                protocol_mode,
                ProtocolMode::Stream | ProtocolMode::Tasks
            )),
            durability: metadata_durability(&raw.metadata).unwrap_or_default(),
            ..Default::default()
        });

        Ok(Self {
            schema: raw.schema,
            agent_id: raw.agent_id,
            name: raw.name,
            version: raw.version,
            kind: raw.kind,
            runtime: raw.runtime,
            protocol_mode,
            endpoint: raw.endpoint,
            capabilities: raw.capabilities,
            capability_manifest: raw.capability_manifest,
            declared_gates: raw.declared_gates,
            execution,
            required_secrets: raw.required_secrets,
            supports_streaming: raw.supports_streaming,
            sandbox_isolation: raw.sandbox_isolation,
            task_bundles_path: raw.task_bundles_path,
            metadata: clean_metadata(raw.metadata),
        })
    }
}

impl AgentManifestV2 {
    /// Validate the manifest for internal consistency.
    ///
    /// Returns all validation errors in a single call so callers can surface
    /// the full set of problems rather than one at a time.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.agent_id.is_empty() {
            errors.push("`agent_id` must not be empty".to_owned());
        }
        if self.name.is_empty() {
            errors.push("`name` must not be empty".to_owned());
        }
        if self.version.is_empty() {
            errors.push("`version` must not be empty".to_owned());
        }
        if self.execution.supports_cancel && self.protocol_mode != ProtocolMode::Tasks {
            errors.push("`supports_cancel` requires `protocol_mode = tasks`".to_owned());
        }
        if self.execution.supports_resume && self.protocol_mode != ProtocolMode::Tasks {
            errors.push("`supports_resume` requires `protocol_mode = tasks`".to_owned());
        }
        if self.execution.emits_events && self.protocol_mode == ProtocolMode::Simple {
            errors.push("`emits_events` requires `protocol_mode = stream` or `tasks`".to_owned());
        }
        if self.execution.durability == DurabilityLevel::TaskStore
            && self.protocol_mode != ProtocolMode::Tasks
        {
            errors.push("`durability = task_store` requires `protocol_mode = tasks`".to_owned());
        }
        for capability in &self.capability_manifest {
            errors.extend(validate_capability_manifest_entry(capability));
        }
        errors
    }

    /// Parse a manifest from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialise to a pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

fn validate_capability_manifest_entry(entry: &Value) -> Vec<String> {
    let Some(obj) = entry.as_object() else {
        return vec!["capability_manifest entries must be JSON objects".to_owned()];
    };
    let capability_id = obj
        .get("capability_id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if capability_id.is_empty() {
        return vec!["capability_manifest entries must have non-empty capability_id".to_owned()];
    }

    let description = obj
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    let display_name = obj
        .get("display_name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();

    if description.len() < DESCRIPTION_MIN_LENGTH {
        return vec![format!(
            "capability {capability_id:?}: description too short ({} chars; minimum {DESCRIPTION_MIN_LENGTH}). Describe what the capability does, its inputs, and its side effects.",
            description.len()
        )];
    }
    if !display_name.is_empty() && description.eq_ignore_ascii_case(display_name) {
        return vec![format!(
            "capability {capability_id:?}: description duplicates display_name; write a real explanation of what the capability does."
        )];
    }
    Vec::new()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tasks_manifest() -> AgentManifestV2 {
        AgentManifestV2 {
            schema: MANIFEST_SCHEMA.to_owned(),
            agent_id: "test-agent".to_owned(),
            name: "Test Agent".to_owned(),
            version: "0.1.0".to_owned(),
            kind: AgentKind::ExpertComplex,
            runtime: AgentRuntime::ExternalA2A,
            protocol_mode: ProtocolMode::Tasks,
            endpoint: Some("http://localhost:8080".to_owned()),
            capabilities: vec!["code_review".to_owned()],
            capability_manifest: vec![],
            declared_gates: vec![],
            execution: ExecutionHints {
                supports_cancel: true,
                emits_events: true,
                durability: DurabilityLevel::TaskStore,
                expected_duration_seconds: Some(30),
                max_duration_seconds: Some(300),
                ..Default::default()
            },
            required_secrets: vec![],
            supports_streaming: false,
            sandbox_isolation: "shared".to_owned(),
            task_bundles_path: String::new(),
            metadata: Map::new(),
        }
    }

    #[test]
    fn valid_manifest_passes_validation() {
        let m = tasks_manifest();
        let errs = m.validate();
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");
    }

    #[test]
    fn supports_cancel_without_tasks_mode_fails() {
        let mut m = tasks_manifest();
        m.protocol_mode = ProtocolMode::Simple;
        let errs = m.validate();
        assert!(
            errs.iter().any(|e| e.contains("supports_cancel")),
            "expected supports_cancel error"
        );
    }

    #[test]
    fn empty_agent_id_fails() {
        let mut m = tasks_manifest();
        m.agent_id = String::new();
        let errs = m.validate();
        assert!(errs.iter().any(|e| e.contains("agent_id")));
    }

    #[test]
    fn roundtrip_json() {
        let m = tasks_manifest();
        let json = m.to_json_pretty().unwrap();
        let parsed: AgentManifestV2 = AgentManifestV2::from_json(&json).unwrap();
        assert_eq!(parsed.agent_id, m.agent_id);
        assert_eq!(parsed.protocol_mode, ProtocolMode::Tasks);
        assert!(parsed.execution.supports_cancel);
        assert_eq!(parsed.execution.durability, DurabilityLevel::TaskStore);
    }

    #[test]
    fn default_protocol_mode_is_simple() {
        let m: AgentManifestV2 = serde_json::from_str(
            r#"{
            "agent_id": "x",
            "name": "X",
            "version": "0.1.0",
            "kind": "expert_basic",
            "runtime": "external_a2a"
        }"#,
        )
        .unwrap();
        assert_eq!(m.protocol_mode, ProtocolMode::Simple);
    }

    #[test]
    fn parses_current_cortex_manifest_shape() {
        let m =
            AgentManifestV2::from_json(include_str!("../tests/fixtures/novie-cortex.agent.json"))
                .unwrap();

        assert_eq!(m.agent_id, "novie-cortex");
        assert_eq!(m.protocol_mode, ProtocolMode::Tasks);
        assert_eq!(m.execution.durability, DurabilityLevel::TaskStore);
        assert_eq!(
            m.required_secrets,
            vec!["GITHUB_TOKEN", "ANTHROPIC_API_KEY"]
        );
        assert_eq!(m.task_bundles_path, "/task-bundles");
        assert!(!m.capability_manifest.is_empty());
        assert!(m.metadata.contains_key("owner_team"));
        assert!(!m.metadata.contains_key("protocol_mode"));
    }

    #[test]
    fn validates_capability_manifest_description_quality() {
        let mut m = tasks_manifest();
        m.capability_manifest = vec![serde_json::json!({
            "capability_id": "agent.test.short_description",
            "version": "0.1.0",
            "display_name": "Short",
            "description": "too short",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "risk": "read",
            "side_effect": "none",
            "exec_kind": "async",
            "runtime_ref": "agent:test:short_description"
        })];

        let errors = m.validate();

        assert!(
            errors
                .iter()
                .any(|error| error.contains("description too short")),
            "expected description quality error, got {errors:?}"
        );
    }

    #[test]
    fn validates_capability_manifest_requires_capability_id() {
        let mut m = tasks_manifest();
        m.capability_manifest = vec![serde_json::json!({
            "capability_id": "",
            "version": "0.1.0",
            "display_name": "Missing Id",
            "description": "This description is long enough to pass the quality floor.",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "risk": "read",
            "side_effect": "none",
            "exec_kind": "async",
            "runtime_ref": "agent:test:missing_id"
        })];

        let errors = m.validate();

        assert!(
            errors
                .iter()
                .any(|error| error.contains("non-empty capability_id")),
            "expected capability_id error, got {errors:?}"
        );
    }

    #[test]
    fn reads_legacy_protocol_mode_from_metadata() {
        let m: AgentManifestV2 = serde_json::from_str(
            r#"{
            "agent_id": "legacy",
            "name": "Legacy",
            "version": "0.1.0",
            "kind": "expert_complex",
            "runtime": "external_a2a",
            "metadata": {
                "protocol_mode": "tasks",
                "supports_cancel": true,
                "emits_events": true,
                "durability": "task_store"
            }
        }"#,
        )
        .unwrap();

        assert_eq!(m.protocol_mode, ProtocolMode::Tasks);
        assert!(m.execution.supports_cancel);
        assert!(m.execution.emits_events);
        assert_eq!(m.execution.durability, DurabilityLevel::TaskStore);
        assert!(!m.metadata.contains_key("protocol_mode"));
    }
}
