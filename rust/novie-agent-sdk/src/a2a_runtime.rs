//! A2A Agent Runtime for Rust.
//!
//! Hosts the full A2A endpoint surface so agent authors only need to implement
//! the business handler — not the protocol machinery.
//!
//! # Supported endpoints
//!
//! | Endpoint                        | Mode          |
//! |--------------------------------|---------------|
//! | `GET  /healthz`                | all           |
//! | `GET  /.well-known/agent.json` | all           |
//! | `POST /invoke`                 | simple        |
//! | `POST /stream`                 | stream        |
//! | `POST /tasks`                  | tasks         |
//! | `GET  /tasks/{id}`             | tasks         |
//! | `GET  /tasks/{id}/events`      | tasks         |
//! | `GET  /tasks/{id}/result`      | tasks         |
//! | `POST /tasks/{id}/cancel`      | tasks         |
//!
//! # Quickstart
//!
//! ```ignore
//! use novie_agent_sdk::a2a_runtime::{Agent, TaskContext};
//! use novie_agent_sdk::manifest::AgentManifestV2;
//!
//! // Load manifest from .well-known/agent.json
//! let manifest: AgentManifestV2 = serde_json::from_str(r#"
//!     {"agent_id":"my-agent","name":"My Agent","version":"0.1.0",
//!      "endpoint":"http://localhost:8080","protocol_mode":"tasks"}
//! "#).unwrap();
//!
//! let agent = Agent::new(manifest)
//!     .task_handler(|ctx: TaskContext| async move {
//!         ctx.emit_message("Starting work").await;
//!         Ok(serde_json::json!({ "result": 42 }))
//!     });
//!
//! // agent.serve("0.0.0.0:8080".parse().unwrap()).await.unwrap();
//! ```

use std::{
    collections::HashMap,
    future::Future,
    path::{Path as FsPath, PathBuf},
    pin::Pin,
    sync::{Arc, Mutex as StdMutex},
    time::Duration,
};

use axum::{
    Json, Router,
    extract::{Path, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use chrono::Utc;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tracing::{error, info};
use uuid::Uuid;

use crate::{
    headers::{RequestHeaders, verify_agent_request_headers},
    manifest::{AgentManifestV2, DurabilityLevel},
};

// ─────────────────────────────────────────────────────────────────────────────
// Task types
// ─────────────────────────────────────────────────────────────────────────────

/// Possible lifecycle states of an A2A task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Queued,
    Running,
    WaitingForInput,
    WaitingForHuman,
    Completed,
    Failed,
    Cancelled,
}

/// Terminal states — transitions out of these are ignored.
fn is_terminal(s: &TaskStatus) -> bool {
    matches!(
        s,
        TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Cancelled
    )
}

/// A single task record stored in the in-process task store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRecord {
    pub task_id: String,
    pub status: TaskStatus,
    pub input: Value,
    pub error: Option<String>,
    #[serde(default)]
    pub events: Vec<TaskEvent>,
    pub result: Option<Value>,
    pub created_at: String,
    pub updated_at: String,
}

/// An event emitted by a task handler during execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEvent {
    pub event_id: String,
    pub task_id: String,
    pub kind: String,
    pub timestamp: String,
    pub data: Value,
}

#[derive(Debug)]
struct SqliteTaskRow {
    task_id: String,
    status: String,
    input_json: String,
    error: Option<String>,
    result_json: Option<String>,
    created_at: String,
    updated_at: String,
}

/// Structured HITL wait payload emitted by worker task contexts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanWaitRequest {
    pub gate_id: String,
    pub prompt: String,
    #[serde(default)]
    pub allowed_actions: Vec<String>,
    #[serde(default)]
    pub resume_reference: String,
    #[serde(default)]
    pub timeout_policy: Value,
    #[serde(default)]
    pub metadata: Value,
}

/// Author-returned task result envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    #[serde(default)]
    pub output: Value,
    #[serde(default)]
    pub artifacts: Vec<Value>,
    #[serde(default)]
    pub metadata: Value,
}

/// Author-returned one-shot artifact envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactResult {
    pub artifact_type: String,
    pub summary: String,
    #[serde(default)]
    pub content: Value,
    #[serde(default)]
    pub metadata: Value,
}

impl ArtifactResult {
    pub fn into_output(self) -> Value {
        json!({
            "kind": "artifact",
            "artifact_type": self.artifact_type,
            "summary": self.summary,
            "content": self.content,
            "metadata": self.metadata
        })
    }
}

/// One-shot response that asks the platform to confirm before continuing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeedsConfirmationResult {
    pub prompt: String,
    #[serde(default)]
    pub allowed_actions: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}

impl NeedsConfirmationResult {
    pub fn into_output(self) -> Value {
        json!({
            "kind": "needs_confirmation",
            "status": "needs_confirmation",
            "prompt": self.prompt,
            "allowed_actions": self.allowed_actions,
            "metadata": self.metadata
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Task store trait + in-memory implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Persistent store for task records.  Replace with a database-backed
/// implementation for production deployments that require restart-safe storage.
#[async_trait::async_trait]
pub trait TaskStore: Send + Sync + 'static {
    fn backend_name(&self) -> &'static str;
    fn durable(&self) -> bool;
    async fn create(
        &self,
        task_id: &str,
        input: Value,
        idempotency_key: Option<&str>,
    ) -> TaskRecord;
    async fn get(&self, task_id: &str) -> Option<TaskRecord>;
    async fn update_status(&self, task_id: &str, status: TaskStatus);
    async fn set_result(&self, task_id: &str, result: Value);
    async fn set_error(&self, task_id: &str, error: String);
    async fn append_event(&self, task_id: &str, event: TaskEvent);
    async fn get_events(&self, task_id: &str) -> Vec<TaskEvent>;
    /// Returns `true` if the task was successfully cancelled (was non-terminal).
    async fn cancel(&self, task_id: &str) -> bool;
    /// Returns `true` if the task has been requested to cancel.
    async fn is_cancelled(&self, task_id: &str) -> bool;
}

#[derive(Default)]
struct StoreInner {
    tasks: HashMap<String, TaskRecord>,
    idempotency: HashMap<String, String>,
    cancelled: HashMap<String, bool>,
}

/// Simple in-memory [`TaskStore`] — not restart-safe.
pub struct InMemoryTaskStore {
    inner: Mutex<StoreInner>,
}

impl Default for InMemoryTaskStore {
    fn default() -> Self {
        Self {
            inner: Mutex::new(StoreInner::default()),
        }
    }
}

impl std::fmt::Debug for InMemoryTaskStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryTaskStore").finish()
    }
}

#[async_trait::async_trait]
impl TaskStore for InMemoryTaskStore {
    fn backend_name(&self) -> &'static str {
        "memory"
    }

    fn durable(&self) -> bool {
        false
    }

    async fn create(
        &self,
        task_id: &str,
        input: Value,
        idempotency_key: Option<&str>,
    ) -> TaskRecord {
        let mut inner = self.inner.lock().await;
        if let Some(key) = idempotency_key {
            if let Some(existing_id) = inner.idempotency.get(key) {
                if let Some(existing) = inner.tasks.get(existing_id) {
                    return existing.clone();
                }
            }
        }
        let now = Utc::now().to_rfc3339();
        let record = TaskRecord {
            task_id: task_id.to_owned(),
            status: TaskStatus::Queued,
            input,
            error: None,
            events: vec![],
            result: None,
            created_at: now.clone(),
            updated_at: now,
        };
        inner.tasks.insert(task_id.to_owned(), record.clone());
        if let Some(key) = idempotency_key {
            inner.idempotency.insert(key.to_owned(), task_id.to_owned());
        }
        record
    }

    async fn get(&self, task_id: &str) -> Option<TaskRecord> {
        self.inner.lock().await.tasks.get(task_id).cloned()
    }

    async fn update_status(&self, task_id: &str, status: TaskStatus) {
        let mut inner = self.inner.lock().await;
        if let Some(rec) = inner.tasks.get_mut(task_id) {
            if is_terminal(&rec.status) {
                return;
            }
            rec.status = status;
            rec.updated_at = Utc::now().to_rfc3339();
        }
    }

    async fn set_result(&self, task_id: &str, result: Value) {
        let mut inner = self.inner.lock().await;
        if let Some(rec) = inner.tasks.get_mut(task_id) {
            rec.result = Some(result);
            rec.status = TaskStatus::Completed;
            rec.updated_at = Utc::now().to_rfc3339();
        }
    }

    async fn set_error(&self, task_id: &str, error: String) {
        let mut inner = self.inner.lock().await;
        if let Some(rec) = inner.tasks.get_mut(task_id) {
            if is_terminal(&rec.status) {
                return;
            }
            rec.error = Some(error);
            rec.status = TaskStatus::Failed;
            rec.updated_at = Utc::now().to_rfc3339();
        }
    }

    async fn append_event(&self, task_id: &str, event: TaskEvent) {
        let mut inner = self.inner.lock().await;
        if let Some(rec) = inner.tasks.get_mut(task_id) {
            rec.events.push(event);
        }
    }

    async fn get_events(&self, task_id: &str) -> Vec<TaskEvent> {
        self.inner
            .lock()
            .await
            .tasks
            .get(task_id)
            .map(|r| r.events.clone())
            .unwrap_or_default()
    }

    async fn cancel(&self, task_id: &str) -> bool {
        let mut inner = self.inner.lock().await;
        if let Some(rec) = inner.tasks.get_mut(task_id) {
            if is_terminal(&rec.status) {
                return false;
            }
            rec.status = TaskStatus::Cancelled;
            rec.updated_at = Utc::now().to_rfc3339();
            inner.cancelled.insert(task_id.to_owned(), true);
            return true;
        }
        false
    }

    async fn is_cancelled(&self, task_id: &str) -> bool {
        *self
            .inner
            .lock()
            .await
            .cancelled
            .get(task_id)
            .unwrap_or(&false)
    }
}

/// SQLite-backed [`TaskStore`] for restart-safe worker task state.
pub struct SqliteTaskStore {
    conn: StdMutex<Connection>,
}

impl std::fmt::Debug for SqliteTaskStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteTaskStore").finish()
    }
}

impl SqliteTaskStore {
    pub fn open(path: impl AsRef<FsPath>) -> Result<Self, String> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let conn = Connection::open(path).map_err(|e| e.to_string())?;
        let store = Self {
            conn: StdMutex::new(conn),
        };
        store.init()?;
        Ok(store)
    }

    fn init(&self) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                input_json TEXT NOT NULL,
                error TEXT,
                result_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                cancelled INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS task_events (
                task_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data_json TEXT NOT NULL,
                seq INTEGER PRIMARY KEY AUTOINCREMENT
            );
            CREATE TABLE IF NOT EXISTS task_idempotency (
                idempotency_key TEXT PRIMARY KEY,
                task_id TEXT NOT NULL
            );
            "#,
        )
        .map_err(|e| e.to_string())
    }

    fn record_from_row(row: SqliteTaskRow, events: Vec<TaskEvent>) -> TaskRecord {
        TaskRecord {
            task_id: row.task_id,
            status: serde_json::from_value(Value::String(row.status)).unwrap_or(TaskStatus::Failed),
            input: serde_json::from_str(&row.input_json).unwrap_or(Value::Null),
            error: row.error,
            events,
            result: row
                .result_json
                .and_then(|raw| serde_json::from_str(&raw).ok()),
            created_at: row.created_at,
            updated_at: row.updated_at,
        }
    }
}

#[async_trait::async_trait]
impl TaskStore for SqliteTaskStore {
    fn backend_name(&self) -> &'static str {
        "sqlite"
    }

    fn durable(&self) -> bool {
        true
    }

    async fn create(
        &self,
        task_id: &str,
        input: Value,
        idempotency_key: Option<&str>,
    ) -> TaskRecord {
        let conn = self.conn.lock().expect("sqlite task store lock poisoned");
        if let Some(key) = idempotency_key {
            if let Ok(existing_id) = conn.query_row(
                "SELECT task_id FROM task_idempotency WHERE idempotency_key = ?1",
                params![key],
                |row| row.get::<_, String>(0),
            ) {
                if let Some(record) = self.get_record_locked(&conn, &existing_id) {
                    return record;
                }
                let _ = conn.execute(
                    "DELETE FROM task_idempotency WHERE idempotency_key = ?1",
                    params![key],
                );
            }
        }

        let now = Utc::now().to_rfc3339();
        let input_json = serde_json::to_string(&input).unwrap_or_else(|_| "null".to_owned());
        conn.execute(
            "INSERT OR REPLACE INTO tasks (task_id, status, input_json, created_at, updated_at, cancelled)
             VALUES (?1, ?2, ?3, ?4, ?5, 0)",
            params![task_id, "queued", input_json, now, now],
        )
        .expect("sqlite task insert failed");
        if let Some(key) = idempotency_key {
            let _ = conn.execute(
                "INSERT OR REPLACE INTO task_idempotency (idempotency_key, task_id) VALUES (?1, ?2)",
                params![key, task_id],
            );
        }
        self.get_record_locked(&conn, task_id)
            .expect("created task must be readable")
    }

    async fn get(&self, task_id: &str) -> Option<TaskRecord> {
        let conn = self.conn.lock().ok()?;
        self.get_record_locked(&conn, task_id)
    }

    async fn update_status(&self, task_id: &str, status: TaskStatus) {
        if matches!(self.get(task_id).await.map(|r| r.status), Some(s) if is_terminal(&s)) {
            return;
        }
        let conn = self.conn.lock().expect("sqlite task store lock poisoned");
        let status_json =
            serde_json::to_value(status).unwrap_or(Value::String("failed".to_owned()));
        let status_str = status_json.as_str().unwrap_or("failed");
        let _ = conn.execute(
            "UPDATE tasks SET status = ?2, updated_at = ?3 WHERE task_id = ?1",
            params![task_id, status_str, Utc::now().to_rfc3339()],
        );
    }

    async fn set_result(&self, task_id: &str, result: Value) {
        let conn = self.conn.lock().expect("sqlite task store lock poisoned");
        let result_json = serde_json::to_string(&result).unwrap_or_else(|_| "null".to_owned());
        let _ = conn.execute(
            "UPDATE tasks SET status = 'completed', result_json = ?2, updated_at = ?3 WHERE task_id = ?1",
            params![task_id, result_json, Utc::now().to_rfc3339()],
        );
    }

    async fn set_error(&self, task_id: &str, error: String) {
        if matches!(self.get(task_id).await.map(|r| r.status), Some(s) if is_terminal(&s)) {
            return;
        }
        let conn = self.conn.lock().expect("sqlite task store lock poisoned");
        let _ = conn.execute(
            "UPDATE tasks SET status = 'failed', error = ?2, updated_at = ?3 WHERE task_id = ?1",
            params![task_id, error, Utc::now().to_rfc3339()],
        );
    }

    async fn append_event(&self, task_id: &str, event: TaskEvent) {
        let conn = self.conn.lock().expect("sqlite task store lock poisoned");
        let data_json = serde_json::to_string(&event.data).unwrap_or_else(|_| "null".to_owned());
        let _ = conn.execute(
            "INSERT INTO task_events (task_id, event_id, kind, timestamp, data_json)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                task_id,
                event.event_id,
                event.kind,
                event.timestamp,
                data_json
            ],
        );
    }

    async fn get_events(&self, task_id: &str) -> Vec<TaskEvent> {
        let Ok(conn) = self.conn.lock() else {
            return Vec::new();
        };
        self.get_events_locked(&conn, task_id)
    }

    async fn cancel(&self, task_id: &str) -> bool {
        if matches!(self.get(task_id).await.map(|r| r.status), Some(s) if is_terminal(&s)) {
            return false;
        }
        let conn = self.conn.lock().expect("sqlite task store lock poisoned");
        conn.execute(
            "UPDATE tasks SET status = 'cancelled', cancelled = 1, updated_at = ?2 WHERE task_id = ?1",
            params![task_id, Utc::now().to_rfc3339()],
        )
        .map(|rows| rows > 0)
        .unwrap_or(false)
    }

    async fn is_cancelled(&self, task_id: &str) -> bool {
        let Ok(conn) = self.conn.lock() else {
            return false;
        };
        conn.query_row(
            "SELECT cancelled FROM tasks WHERE task_id = ?1",
            params![task_id],
            |row| row.get::<_, i64>(0),
        )
        .map(|value| value != 0)
        .unwrap_or(false)
    }
}

impl SqliteTaskStore {
    fn get_events_locked(&self, conn: &Connection, task_id: &str) -> Vec<TaskEvent> {
        let Ok(mut stmt) = conn.prepare(
            "SELECT event_id, kind, timestamp, data_json FROM task_events
             WHERE task_id = ?1 ORDER BY seq ASC",
        ) else {
            return Vec::new();
        };
        let Ok(rows) = stmt.query_map(params![task_id], |row| {
            let data_json: String = row.get(3)?;
            Ok(TaskEvent {
                event_id: row.get(0)?,
                task_id: task_id.to_owned(),
                kind: row.get(1)?,
                timestamp: row.get(2)?,
                data: serde_json::from_str(&data_json).unwrap_or(Value::Null),
            })
        }) else {
            return Vec::new();
        };
        rows.filter_map(Result::ok).collect()
    }

    fn get_record_locked(&self, conn: &Connection, task_id: &str) -> Option<TaskRecord> {
        let events = self.get_events_locked(conn, task_id);
        conn.query_row(
            "SELECT task_id, status, input_json, error, result_json, created_at, updated_at
             FROM tasks WHERE task_id = ?1",
            params![task_id],
            |row| {
                Ok(Self::record_from_row(
                    SqliteTaskRow {
                        task_id: row.get(0)?,
                        status: row.get(1)?,
                        input_json: row.get(2)?,
                        error: row.get(3)?,
                        result_json: row.get(4)?,
                        created_at: row.get(5)?,
                        updated_at: row.get(6)?,
                    },
                    events.clone(),
                ))
            },
        )
        .ok()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Task context — passed to handlers
// ─────────────────────────────────────────────────────────────────────────────

/// Context provided to task handlers.
pub struct TaskContext {
    pub task_id: String,
    pub input: Value,
    pub headers: RequestHeaders,
    store: Arc<dyn TaskStore>,
}

/// Context provided to simple invoke handlers.
#[derive(Debug, Clone)]
pub struct InvokeContext {
    pub input: Value,
    pub headers: RequestHeaders,
    pub agent_manifest: AgentManifestV2,
}

/// Context provided to stream handlers.
#[derive(Debug, Clone)]
pub struct StreamContext {
    pub input: Value,
    pub headers: RequestHeaders,
    pub agent_manifest: AgentManifestV2,
}

impl InvokeContext {
    pub fn artifact(
        &self,
        artifact_type: impl Into<String>,
        summary: impl Into<String>,
        content: Value,
    ) -> Value {
        ArtifactResult {
            artifact_type: artifact_type.into(),
            summary: summary.into(),
            content,
            metadata: Value::Null,
        }
        .into_output()
    }

    pub fn needs_confirmation(
        &self,
        prompt: impl Into<String>,
        allowed_actions: Vec<String>,
        metadata: Value,
    ) -> Value {
        NeedsConfirmationResult {
            prompt: prompt.into(),
            allowed_actions,
            metadata,
        }
        .into_output()
    }
}

impl StreamContext {
    pub fn progress(&self, data: Value) -> StreamEvent {
        StreamEvent::progress(data)
    }

    pub fn done(&self, output: Value) -> StreamEvent {
        StreamEvent::done(output)
    }
}

/// NDJSON event returned by stream handlers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEvent {
    pub kind: String,
    #[serde(default)]
    pub data: Value,
}

impl StreamEvent {
    pub fn progress(data: Value) -> Self {
        Self {
            kind: "progress".to_owned(),
            data,
        }
    }

    pub fn done(output: Value) -> Self {
        Self {
            kind: "done".to_owned(),
            data: json!({ "output": output }),
        }
    }
}

impl std::fmt::Debug for TaskContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskContext")
            .field("task_id", &self.task_id)
            .finish()
    }
}

impl TaskContext {
    fn new(
        task_id: String,
        input: Value,
        headers: RequestHeaders,
        store: Arc<dyn TaskStore>,
    ) -> Self {
        Self {
            task_id,
            input,
            headers,
            store,
        }
    }

    /// Emit a structured event.
    pub async fn emit_event(&self, kind: impl Into<String>, data: Value) {
        let event = TaskEvent {
            event_id: Uuid::new_v4().to_string(),
            task_id: self.task_id.clone(),
            kind: kind.into(),
            timestamp: Utc::now().to_rfc3339(),
            data,
        };
        self.store.append_event(&self.task_id, event).await;
    }

    /// Emit a text message event.
    pub async fn emit_message(&self, text: impl Into<String>) {
        self.emit_event("message", json!({ "text": text.into() }))
            .await;
    }

    /// Emit a structured data event.
    pub async fn emit_data(&self, kind: impl Into<String>, data: Value) {
        self.emit_event(kind, data).await;
    }

    /// Emit an artifact event during task execution.
    pub async fn emit_artifact(&self, artifact: ArtifactResult) {
        self.emit_event("artifact", artifact.into_output()).await;
    }

    /// Report one LLM usage event in the platform A2A event shape.
    pub async fn report_llm_usage(
        &self,
        provider: impl Into<String>,
        model: impl Into<String>,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
        total_tokens: Option<u64>,
    ) {
        self.emit_event(
            "usage",
            json!({
                "provider": provider.into(),
                "model": model.into(),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }),
        )
        .await;
    }

    /// Set the task status explicitly.
    pub async fn set_status(&self, status: TaskStatus) {
        self.store.update_status(&self.task_id, status).await;
    }

    /// Emit a structured human wait event and move the task into waiting state.
    pub async fn wait_for_human(&self, request: HumanWaitRequest) {
        self.set_status(TaskStatus::WaitingForHuman).await;
        self.emit_event(
            "wait_prompt",
            serde_json::to_value(request).unwrap_or(Value::Null),
        )
        .await;
    }

    /// Build a standard task result envelope.
    pub fn result(&self, output: Value, artifacts: Vec<Value>, metadata: Value) -> Value {
        serde_json::to_value(TaskResult {
            output,
            artifacts,
            metadata,
        })
        .unwrap_or(Value::Null)
    }

    /// Check if cancellation has been requested.
    pub async fn is_cancelled(&self) -> bool {
        self.store.is_cancelled(&self.task_id).await
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Handler type alias
// ─────────────────────────────────────────────────────────────────────────────

pub type TaskHandlerFn = Arc<
    dyn Fn(TaskContext) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>>
        + Send
        + Sync,
>;

pub type InvokeHandlerFn = Arc<
    dyn Fn(InvokeContext) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>>
        + Send
        + Sync,
>;

pub type StreamHandlerFn = Arc<
    dyn Fn(StreamContext) -> Pin<Box<dyn Future<Output = Result<Vec<StreamEvent>, String>> + Send>>
        + Send
        + Sync,
>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvocationState {
    Started,
    InProgress,
    Completed(String),
}

#[async_trait::async_trait]
pub trait OneShotInvocationStore: Send + Sync + 'static {
    fn backend_name(&self) -> &'static str;
    fn durable(&self) -> bool;
    async fn start_or_get(&self, key: &str) -> InvocationState;
    async fn complete(&self, key: &str, response_body: String);
}

#[derive(Debug, Default)]
pub struct InMemoryOneShotInvocationStore {
    inner: Mutex<HashMap<String, Option<String>>>,
}

#[async_trait::async_trait]
impl OneShotInvocationStore for InMemoryOneShotInvocationStore {
    fn backend_name(&self) -> &'static str {
        "memory"
    }

    fn durable(&self) -> bool {
        false
    }

    async fn start_or_get(&self, key: &str) -> InvocationState {
        let mut inner = self.inner.lock().await;
        match inner.get(key) {
            Some(Some(body)) => InvocationState::Completed(body.clone()),
            Some(None) => InvocationState::InProgress,
            None => {
                inner.insert(key.to_owned(), None);
                InvocationState::Started
            }
        }
    }

    async fn complete(&self, key: &str, response_body: String) {
        self.inner
            .lock()
            .await
            .insert(key.to_owned(), Some(response_body));
    }
}

pub struct SqliteOneShotInvocationStore {
    conn: StdMutex<Connection>,
}

impl std::fmt::Debug for SqliteOneShotInvocationStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteOneShotInvocationStore").finish()
    }
}

impl SqliteOneShotInvocationStore {
    pub fn open(path: impl AsRef<FsPath>) -> Result<Self, String> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let conn = Connection::open(path).map_err(|e| e.to_string())?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS one_shot_invocations (
                key TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                response_body TEXT,
                updated_at TEXT NOT NULL
            );
            "#,
        )
        .map_err(|e| e.to_string())?;
        Ok(Self {
            conn: StdMutex::new(conn),
        })
    }
}

#[async_trait::async_trait]
impl OneShotInvocationStore for SqliteOneShotInvocationStore {
    fn backend_name(&self) -> &'static str {
        "sqlite"
    }

    fn durable(&self) -> bool {
        true
    }

    async fn start_or_get(&self, key: &str) -> InvocationState {
        let conn = self
            .conn
            .lock()
            .expect("sqlite invocation store lock poisoned");
        let existing = conn.query_row(
            "SELECT status, response_body FROM one_shot_invocations WHERE key = ?1",
            params![key],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
        );
        match existing {
            Ok((status, Some(body))) if status == "completed" => InvocationState::Completed(body),
            Ok(_) => InvocationState::InProgress,
            Err(_) => {
                let _ = conn.execute(
                    "INSERT INTO one_shot_invocations (key, status, updated_at) VALUES (?1, 'in_progress', ?2)",
                    params![key, Utc::now().to_rfc3339()],
                );
                InvocationState::Started
            }
        }
    }

    async fn complete(&self, key: &str, response_body: String) {
        let conn = self
            .conn
            .lock()
            .expect("sqlite invocation store lock poisoned");
        let _ = conn.execute(
            "UPDATE one_shot_invocations SET status = 'completed', response_body = ?2, updated_at = ?3 WHERE key = ?1",
            params![key, response_body, Utc::now().to_rfc3339()],
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Registration client
// ─────────────────────────────────────────────────────────────────────────────

/// Client for registering this agent with the Platform's Manifest Registry.
#[derive(Clone, Debug)]
pub struct RegistrationClient {
    registry_url: String,
    agent_id: String,
    registration_token: Option<String>,
    http: reqwest::Client,
}

impl RegistrationClient {
    pub fn new(registry_url: impl Into<String>, agent_id: impl Into<String>) -> Self {
        Self {
            registry_url: registry_url.into(),
            agent_id: agent_id.into(),
            registration_token: std::env::var("NOVIE_AGENT_REGISTRATION_TOKEN")
                .ok()
                .or_else(|| std::env::var("NOVIE_AGENT_SECRET").ok())
                .filter(|v| !v.trim().is_empty()),
            http: reqwest::Client::new(),
        }
    }

    fn request_with_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.registration_token {
            Some(token) => req.header("Agent-Secret", token).bearer_auth(token),
            None => req,
        }
    }

    pub async fn register(&self, manifest: &AgentManifestV2) -> Result<(), String> {
        let url = format!("{}/agents/register", self.registry_url);
        let body = serde_json::to_value(manifest).map_err(|e| e.to_string())?;
        self.request_with_auth(self.http.post(&url).json(&body))
            .send()
            .await
            .map_err(|e| e.to_string())?
            .error_for_status()
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub async fn heartbeat(&self) -> Result<(), String> {
        let url = format!("{}/agents/{}/heartbeat", self.registry_url, self.agent_id);
        self.request_with_auth(self.http.post(&url))
            .send()
            .await
            .map_err(|e| e.to_string())?
            .error_for_status()
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub async fn deregister(&self) -> Result<(), String> {
        let url = format!("{}/agents/{}", self.registry_url, self.agent_id);
        self.request_with_auth(self.http.delete(&url))
            .send()
            .await
            .map_err(|e| e.to_string())?
            .error_for_status()
            .map_err(|e| e.to_string())?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared application state
// ─────────────────────────────────────────────────────────────────────────────

struct AppState {
    manifest: AgentManifestV2,
    store: Arc<dyn TaskStore>,
    invocation_store: Arc<dyn OneShotInvocationStore>,
    invoke_handler: Option<InvokeHandlerFn>,
    stream_handler: Option<StreamHandlerFn>,
    task_handler: Option<TaskHandlerFn>,
}

type SharedState = Arc<AppState>;

// ─────────────────────────────────────────────────────────────────────────────
// Request / response types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct InvokeRequest {
    input: Value,
    #[serde(default, rename = "context")]
    _context: Value,
}

#[derive(Debug, Serialize)]
struct InvokeResponse {
    status: &'static str,
    output: Value,
}

#[derive(Debug, Deserialize)]
struct CreateTaskRequest {
    input: Value,
    #[serde(default, rename = "context")]
    _context: Value,
}

#[derive(Debug, Serialize)]
struct CreateTaskResponse {
    task_id: String,
    status: TaskStatus,
}

#[derive(Debug, Serialize)]
struct TaskStatusResponse {
    task_id: String,
    status: TaskStatus,
    error: Option<String>,
    created_at: String,
    updated_at: String,
}

#[derive(Debug, Serialize)]
struct EventsResponse {
    task_id: String,
    events: Vec<TaskEvent>,
}

#[derive(Debug, Serialize)]
struct ResultResponse {
    task_id: String,
    status: TaskStatus,
    output: Value,
}

#[derive(Debug, Serialize)]
struct CancelResponse {
    task_id: String,
    status: TaskStatus,
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_id: Option<String>,
}

fn error_response(status: StatusCode, msg: impl Into<String>, task_id: Option<String>) -> Response {
    (
        status,
        Json(ErrorBody {
            error: msg.into(),
            task_id,
        }),
    )
        .into_response()
}

fn json_body_response(status: StatusCode, body: String) -> Response {
    (status, [(header::CONTENT_TYPE, "application/json")], body).into_response()
}

fn ndjson_body_response(status: StatusCode, body: String) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "application/x-ndjson")],
        body,
    )
        .into_response()
}

fn retry_in_progress_response() -> Response {
    (
        StatusCode::CONFLICT,
        Json(json!({
            "status": "retry_in_progress",
            "error": "retry_in_progress"
        })),
    )
        .into_response()
}

fn invocation_key(prefix: &str, headers: &RequestHeaders) -> Option<String> {
    (!headers.idempotency_key.is_empty()).then(|| format!("{prefix}:{}", headers.idempotency_key))
}

fn stream_events_to_ndjson(mut events: Vec<StreamEvent>) -> String {
    if !events
        .iter()
        .any(|event| event.kind == "done" || event.kind == "final")
    {
        events.push(StreamEvent::done(Value::Null));
    }
    events
        .into_iter()
        .map(|event| serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_owned()))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

fn state_dir() -> PathBuf {
    std::env::var("NOVIE_AGENT_STATE_DIR")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".novie-agent-state"))
}

fn sanitized_agent_id(agent_id: &str) -> String {
    agent_id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn default_task_store(manifest: &AgentManifestV2) -> Arc<dyn TaskStore> {
    let production = env_is_production();
    if manifest.execution.durability == DurabilityLevel::TaskStore || production {
        let path = std::env::var("NOVIE_AGENT_TASK_STORE_PATH")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                state_dir().join(format!(
                    "{}-tasks.sqlite3",
                    sanitized_agent_id(&manifest.agent_id)
                ))
            });
        return Arc::new(
            SqliteTaskStore::open(path).expect("failed to open default SQLite task store"),
        );
    }
    Arc::new(InMemoryTaskStore::default())
}

fn default_invocation_store(manifest: &AgentManifestV2) -> Arc<dyn OneShotInvocationStore> {
    if manifest.execution.durability == DurabilityLevel::ResultCache {
        let path = std::env::var("NOVIE_AGENT_INVOCATION_STORE_PATH")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                state_dir().join(format!(
                    "{}-invocations.sqlite3",
                    sanitized_agent_id(&manifest.agent_id)
                ))
            });
        return Arc::new(
            SqliteOneShotInvocationStore::open(path)
                .expect("failed to open default SQLite invocation store"),
        );
    }
    Arc::new(InMemoryOneShotInvocationStore::default())
}

fn env_is_production() -> bool {
    std::env::var("NOVIE_RUNTIME_MODE")
        .unwrap_or_default()
        .eq_ignore_ascii_case("production")
        || std::env::var("NOVIE_ENV")
            .unwrap_or_default()
            .eq_ignore_ascii_case("production")
}

fn verified_headers(headers: &HeaderMap) -> Result<RequestHeaders, Box<Response>> {
    let request_headers = RequestHeaders::from_header_map(headers);
    verify_agent_request_headers(&request_headers)
        .map_err(|err| Box::new(error_response(StatusCode::UNAUTHORIZED, err.code(), None)))?;
    Ok(request_headers)
}

// ─────────────────────────────────────────────────────────────────────────────
// Route handlers
// ─────────────────────────────────────────────────────────────────────────────

async fn healthz(State(state): State<SharedState>) -> impl IntoResponse {
    let production = env_is_production();
    let task_store_ready =
        state.manifest.execution.durability != DurabilityLevel::TaskStore || state.store.durable();
    let status = if production && !task_store_ready {
        "not_ready"
    } else {
        "ok"
    };
    Json(serde_json::json!({
        "status": status,
        "agent_id": state.manifest.agent_id,
        "version": state.manifest.version,
        "protocol_mode": state.manifest.protocol_mode,
        "durability": state.manifest.execution.durability,
        "task_store_backend": state.store.backend_name(),
        "task_store_durable": state.store.durable(),
        "invocation_store_backend": state.invocation_store.backend_name(),
        "invocation_store_durable": state.invocation_store.durable(),
        "signed_headers_required": crate::headers::requires_signed_agent_headers(),
        "ready": task_store_ready,
    }))
}

async fn well_known_manifest(State(state): State<SharedState>) -> impl IntoResponse {
    Json(serde_json::to_value(&state.manifest).unwrap_or_default())
}

async fn post_invoke(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(req): Json<InvokeRequest>,
) -> Response {
    let hdrs = match verified_headers(&headers) {
        Ok(headers) => headers,
        Err(response) => return *response,
    };
    let Some(handler) = &state.invoke_handler else {
        return error_response(
            StatusCode::NOT_IMPLEMENTED,
            "no invoke handler registered",
            None,
        );
    };

    let idem_key = invocation_key("invoke", &hdrs);
    if let Some(key) = &idem_key {
        match state.invocation_store.start_or_get(key).await {
            InvocationState::Completed(body) => return json_body_response(StatusCode::OK, body),
            InvocationState::InProgress => return retry_in_progress_response(),
            InvocationState::Started => {}
        }
    }

    let ctx = InvokeContext {
        input: req.input,
        headers: hdrs,
        agent_manifest: state.manifest.clone(),
    };
    match handler(ctx).await {
        Ok(output) => {
            let body = serde_json::to_string(&InvokeResponse {
                status: "completed",
                output,
            })
            .unwrap_or_else(|_| r#"{"status":"failed","output":null}"#.to_owned());
            if let Some(key) = &idem_key {
                state.invocation_store.complete(key, body.clone()).await;
            }
            json_body_response(StatusCode::OK, body)
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e, None),
    }
}

async fn post_stream(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(req): Json<InvokeRequest>,
) -> Response {
    let hdrs = match verified_headers(&headers) {
        Ok(headers) => headers,
        Err(response) => return *response,
    };
    let Some(handler) = &state.stream_handler else {
        return error_response(
            StatusCode::NOT_IMPLEMENTED,
            "no stream handler registered",
            None,
        );
    };

    let idem_key = invocation_key("stream", &hdrs);
    if let Some(key) = &idem_key {
        match state.invocation_store.start_or_get(key).await {
            InvocationState::Completed(body) => return ndjson_body_response(StatusCode::OK, body),
            InvocationState::InProgress => return retry_in_progress_response(),
            InvocationState::Started => {}
        }
    }

    let ctx = StreamContext {
        input: req.input,
        headers: hdrs,
        agent_manifest: state.manifest.clone(),
    };
    match handler(ctx).await {
        Ok(events) => {
            let body = stream_events_to_ndjson(events);
            if let Some(key) = &idem_key {
                state.invocation_store.complete(key, body.clone()).await;
            }
            ndjson_body_response(StatusCode::OK, body)
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e, None),
    }
}

async fn post_tasks(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(req): Json<CreateTaskRequest>,
) -> Response {
    let hdrs = match verified_headers(&headers) {
        Ok(headers) => headers,
        Err(response) => return *response,
    };
    let handler = match &state.task_handler {
        Some(h) => Arc::clone(h),
        None => {
            return error_response(
                StatusCode::NOT_IMPLEMENTED,
                "no task handler registered",
                None,
            );
        }
    };

    let idempotency_key =
        (!hdrs.idempotency_key.is_empty()).then_some(hdrs.idempotency_key.clone());

    let task_id = Uuid::new_v4().to_string();
    let record = state
        .store
        .create(&task_id, req.input.clone(), idempotency_key.as_deref())
        .await;

    // If idempotent hit, return existing
    if record.task_id != task_id {
        return (
            StatusCode::ACCEPTED,
            Json(CreateTaskResponse {
                task_id: record.task_id,
                status: record.status,
            }),
        )
            .into_response();
    }

    let store = Arc::clone(&state.store);
    let ctx = TaskContext::new(task_id.clone(), req.input, hdrs, Arc::clone(&store));

    tokio::spawn(async move {
        store.update_status(&task_id, TaskStatus::Running).await;
        match handler(ctx).await {
            Ok(output) => {
                store.set_result(&task_id, output).await;
            }
            Err(e) => {
                error!("task {} failed: {}", task_id, e);
                store.set_error(&task_id, e).await;
            }
        }
    });

    (
        StatusCode::ACCEPTED,
        Json(CreateTaskResponse {
            task_id: record.task_id,
            status: TaskStatus::Queued,
        }),
    )
        .into_response()
}

async fn get_task(State(state): State<SharedState>, Path(task_id): Path<String>) -> Response {
    match state.store.get(&task_id).await {
        None => error_response(StatusCode::NOT_FOUND, "task not found", Some(task_id)),
        Some(r) => Json(TaskStatusResponse {
            task_id: r.task_id,
            status: r.status,
            error: r.error,
            created_at: r.created_at,
            updated_at: r.updated_at,
        })
        .into_response(),
    }
}

async fn get_task_events(
    State(state): State<SharedState>,
    Path(task_id): Path<String>,
) -> Response {
    if state.store.get(&task_id).await.is_none() {
        return error_response(StatusCode::NOT_FOUND, "task not found", Some(task_id));
    }
    let events = state.store.get_events(&task_id).await;
    Json(EventsResponse { task_id, events }).into_response()
}

async fn get_task_result(
    State(state): State<SharedState>,
    Path(task_id): Path<String>,
) -> Response {
    match state.store.get(&task_id).await {
        None => error_response(
            StatusCode::NOT_FOUND,
            "task not found",
            Some(task_id.clone()),
        ),
        Some(r) => {
            if r.status != TaskStatus::Completed {
                return error_response(
                    StatusCode::CONFLICT,
                    format!("task is not completed (status: {:?})", r.status),
                    Some(task_id),
                );
            }
            Json(ResultResponse {
                task_id: r.task_id,
                status: r.status,
                output: r.result.unwrap_or(Value::Null),
            })
            .into_response()
        }
    }
}

async fn post_cancel(State(state): State<SharedState>, Path(task_id): Path<String>) -> Response {
    if state.store.get(&task_id).await.is_none() {
        return error_response(StatusCode::NOT_FOUND, "task not found", Some(task_id));
    }
    let cancelled = state.store.cancel(&task_id).await;
    if cancelled {
        (
            StatusCode::ACCEPTED,
            Json(CancelResponse {
                task_id,
                status: TaskStatus::Cancelled,
            }),
        )
            .into_response()
    } else {
        error_response(
            StatusCode::CONFLICT,
            "task already in terminal state",
            Some(task_id),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Agent builder
// ─────────────────────────────────────────────────────────────────────────────

/// The main entry point for building an A2A agent in Rust.
///
/// ```no_run
/// use novie_agent_sdk::a2a_runtime::Agent;
/// # use novie_agent_sdk::manifest::AgentManifestV2;
///
/// # fn build_manifest() -> AgentManifestV2 { todo!() }
/// let agent = Agent::new(build_manifest())
///     .task_handler(|ctx| async move {
///         ctx.emit_message("Done").await;
///         Ok(serde_json::json!({"result": "ok"}))
///     });
/// ```
pub struct Agent {
    manifest: AgentManifestV2,
    store: Arc<dyn TaskStore>,
    invocation_store: Arc<dyn OneShotInvocationStore>,
    invoke_handler: Option<InvokeHandlerFn>,
    stream_handler: Option<StreamHandlerFn>,
    task_handler: Option<TaskHandlerFn>,
    registry_client: Option<RegistrationClient>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("agent_id", &self.manifest.agent_id)
            .field("version", &self.manifest.version)
            .finish()
    }
}

impl Agent {
    /// Create a new agent from a manifest.
    pub fn new(manifest: AgentManifestV2) -> Self {
        let errors = manifest.validate();
        if !errors.is_empty() {
            panic!("Invalid agent manifest: {errors:?}");
        }
        Self {
            store: default_task_store(&manifest),
            invocation_store: default_invocation_store(&manifest),
            manifest,
            invoke_handler: None,
            stream_handler: None,
            task_handler: None,
            registry_client: None,
        }
    }

    /// Override the default in-memory task store.
    pub fn with_store(mut self, store: impl TaskStore) -> Self {
        self.store = Arc::new(store);
        self
    }

    /// Override the default one-shot idempotency/result store.
    pub fn with_invocation_store(mut self, store: impl OneShotInvocationStore) -> Self {
        self.invocation_store = Arc::new(store);
        self
    }

    /// Register a handler for `simple` protocol mode.
    pub fn invoke_handler<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(InvokeContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, String>> + Send + 'static,
    {
        self.invoke_handler = Some(Arc::new(move |ctx| Box::pin(f(ctx))));
        self
    }

    /// Register a handler for `stream` protocol mode.
    pub fn stream_handler<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(StreamContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Vec<StreamEvent>, String>> + Send + 'static,
    {
        self.stream_handler = Some(Arc::new(move |ctx| Box::pin(f(ctx))));
        self
    }

    /// Register a handler for `tasks` protocol mode.
    pub fn task_handler<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(TaskContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, String>> + Send + 'static,
    {
        self.task_handler = Some(Arc::new(move |ctx| Box::pin(f(ctx))));
        self
    }

    /// Configure a registration client so the agent auto-registers with the
    /// Platform Manifest Registry on startup and deregisters on shutdown.
    pub fn with_registry(mut self, client: RegistrationClient) -> Self {
        self.registry_client = Some(client);
        self
    }

    /// Build the Axum router without starting an HTTP server.
    /// Useful for testing with `axum_test::TestServer`.
    pub fn build_router(self) -> Router {
        let state = Arc::new(AppState {
            manifest: self.manifest,
            store: self.store,
            invocation_store: self.invocation_store,
            invoke_handler: self.invoke_handler,
            stream_handler: self.stream_handler,
            task_handler: self.task_handler,
        });

        Router::new()
            .route("/healthz", get(healthz))
            .route("/.well-known/agent.json", get(well_known_manifest))
            .route("/invoke", post(post_invoke))
            .route("/stream", post(post_stream))
            .route("/tasks", post(post_tasks))
            .route("/tasks/{task_id}", get(get_task))
            .route("/tasks/{task_id}/events", get(get_task_events))
            .route("/tasks/{task_id}/result", get(get_task_result))
            .route("/tasks/{task_id}/cancel", post(post_cancel))
            .with_state(state)
    }

    /// Start serving on the given address.  Blocks until a shutdown signal is
    /// received (`CTRL-C` / `SIGTERM`).
    pub async fn serve(
        self,
        addr: impl Into<std::net::SocketAddr>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let manifest = self.manifest.clone();
        let registry_client = self.registry_client.clone();

        // Register with Platform registry if configured
        if let Some(ref rc) = registry_client {
            rc.register(&manifest).await.map_err(|e| {
                format!(
                    "failed to register agent {} with Platform registry: {}",
                    manifest.agent_id, e
                )
            })?;
            info!(
                "registered agent {} with Platform registry",
                manifest.agent_id
            );

            // Background heartbeat task
            let hb_client = rc.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(30));
                loop {
                    interval.tick().await;
                    if let Err(e) = hb_client.heartbeat().await {
                        error!("heartbeat failed: {e}");
                    }
                }
            });
        }

        let router = self.build_router();
        let listener = tokio::net::TcpListener::bind(addr.into()).await?;
        info!("A2A agent listening on {}", listener.local_addr()?);

        axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_signal())
            .await?;

        // Deregister on clean shutdown
        if let Some(rc) = registry_client {
            if let Err(e) = rc.deregister().await {
                error!("deregister failed: {e}");
            }
        }

        Ok(())
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install CTRL+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {}
        () = terminate => {}
    }
    info!("shutdown signal received");
}
