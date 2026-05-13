//! Invoke / task-request payload parsers — `payload.py` Rust port.
//!
//! These types only describe the **callback configuration** fragments
//! (`platform_callback`, `agent_status_callback`) that the agent SDK needs to
//! bootstrap its HTTP clients. The full agent invoke payload is left as
//! `serde_json::Value` because every agent has its own `inputs` schema.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};

/// Signed RPC channel injected by `DispatchService`. Talks to the 9 platform
/// services (memory / wiki / usage / review / events / audit / policy /
/// checkpoint / time_travel / sessions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCallbackConfig {
    pub base_url: String,
    pub token: String,
    #[serde(default = "default_v1")]
    pub version: String,
}

/// Push channel for `POST /internal/callbacks/agent-status`. Token is **scoped
/// to the agent-status endpoint only** — distinct from `platform_callback.token`.
///
/// See `PLATFORM_CALLBACK_SPEC §7.1` and `§11.3`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatusCallbackConfig {
    pub url: String,
    pub token: String,
    #[serde(default = "default_v1")]
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_id: Option<String>,
}

fn default_v1() -> String {
    "v1".to_string()
}

/// Slim view of `platform → agent` invoke payload — only the bits the SDK
/// itself needs. The rest of `inputs` / `context` is kept opaque as `Value`.
#[derive(Debug, Clone)]
pub struct AgentInvokePayload {
    pub raw: Value,
    pub platform_callback: Option<PlatformCallbackConfig>,
    pub agent_status_callback: Option<AgentStatusCallbackConfig>,
}

impl AgentInvokePayload {
    /// Parse from a raw JSON object. Validates that `inputs` is an object,
    /// matches Python `AgentInvokePayload.from_dict` semantics.
    pub fn from_value(raw: Value) -> Result<Self> {
        let obj = raw
            .as_object()
            .ok_or_else(|| Error::InvalidArgument("invoke payload must be a JSON object".into()))?;

        if let Some(inputs) = obj.get("inputs")
            && !inputs.is_object()
            && !inputs.is_null()
        {
            return Err(Error::InvalidArgument(
                "invoke payload.inputs must be a JSON object".into(),
            ));
        }

        let platform_callback = match obj.get("platform_callback") {
            Some(v) if v.is_object() => Some(parse_platform_callback(v)?),
            _ => None,
        };
        let agent_status_callback = match obj.get("agent_status_callback") {
            Some(v) if v.is_object() => Some(parse_agent_status_callback(v)?),
            _ => None,
        };

        Ok(Self {
            raw,
            platform_callback,
            agent_status_callback,
        })
    }

    /// `inputs` block as a JSON value (object or `null`).
    pub fn inputs(&self) -> &Value {
        self.raw.get("inputs").unwrap_or(&Value::Null)
    }

    /// Best-effort agent-id lookup used by `AgentStatusClient::from_invoke_payload`.
    /// Looks at top-level `agent_id` first, then `inputs.agent_id`.
    pub fn agent_id(&self) -> Option<&str> {
        if let Some(s) = self.raw.get("agent_id").and_then(|v| v.as_str())
            && !s.is_empty()
        {
            return Some(s);
        }
        self.raw
            .get("inputs")
            .and_then(|v| v.get("agent_id"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
    }
}

fn parse_platform_callback(v: &Value) -> Result<PlatformCallbackConfig> {
    let cfg: PlatformCallbackConfig = serde_json::from_value(v.clone())?;
    if cfg.base_url.is_empty() {
        return Err(Error::InvalidArgument(
            "platform_callback.base_url must be a non-empty string".into(),
        ));
    }
    if cfg.token.is_empty() {
        return Err(Error::InvalidArgument(
            "platform_callback.token must be a non-empty string".into(),
        ));
    }
    if cfg.version != "v1" {
        return Err(Error::InvalidArgument(format!(
            "unsupported callback SPEC version: {:?}",
            cfg.version
        )));
    }
    Ok(cfg)
}

fn parse_agent_status_callback(v: &Value) -> Result<AgentStatusCallbackConfig> {
    let cfg: AgentStatusCallbackConfig = serde_json::from_value(v.clone())?;
    if cfg.url.is_empty() {
        return Err(Error::InvalidArgument(
            "agent_status_callback.url must be a non-empty string".into(),
        ));
    }
    if cfg.token.is_empty() {
        return Err(Error::InvalidArgument(
            "agent_status_callback.token must be a non-empty string".into(),
        ));
    }
    if cfg.version != "v1" {
        return Err(Error::InvalidArgument(format!(
            "unsupported agent-status SPEC version: {:?}",
            cfg.version
        )));
    }
    Ok(cfg)
}
