//! Long-task completion webhook helper — `notify_long_task_complete` Rust port.
//!
//! Posts to `POST {base_url}/long-task-complete` so the platform can resume
//! the corresponding dispatch step without waiting for the next poller tick.

use serde::Serialize;
use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::transport::CallbackTransport;

/// Final status of a long-running task. Mirrors the Python `Literal` so the
/// platform validator accepts both implementations.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LongTaskStatus {
    #[default]
    Succeeded,
    Failed,
}

/// Options for [`notify_long_task_complete`]. Only `agent_task_id` and
/// `status` are required; the rest mirror the Python defaults.
#[derive(Debug, Default)]
pub struct LongTaskCompletion<'a> {
    pub agent_task_id: &'a str,
    pub status: LongTaskStatus,
    pub tracker_id: Option<&'a str>,
    pub thread_id: Option<&'a str>,
    pub output: Option<Map<String, Value>>,
    pub error: Option<&'a str>,
}

/// Notify the platform that a long-running task finished. Uses the standard
/// callback bearer token (same as `PlatformServicesClient`).
pub async fn notify_long_task_complete(
    transport: &CallbackTransport,
    completion: LongTaskCompletion<'_>,
) -> Result<()> {
    if completion.agent_task_id.is_empty() {
        return Err(Error::InvalidArgument("agent_task_id is required".into()));
    }
    let mut body = serde_json::Map::new();
    body.insert(
        "agent_task_id".into(),
        Value::String(completion.agent_task_id.to_string()),
    );
    body.insert("status".into(), serde_json::to_value(completion.status)?);
    if let Some(t) = completion.tracker_id {
        body.insert("tracker_id".into(), Value::String(t.to_string()));
    }
    if let Some(t) = completion.thread_id {
        body.insert("thread_id".into(), Value::String(t.to_string()));
    }
    if let Some(o) = completion.output {
        body.insert("output".into(), Value::Object(o));
    }
    if let Some(e) = completion.error {
        body.insert("error".into(), Value::String(e.to_string()));
    }
    let value = Value::Object(body);
    transport.push("/long-task-complete", &value).await?;
    Ok(())
}
