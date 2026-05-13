//! A2A runtime conformance tests.
//!
//! Verifies that the Axum router built by `Agent::build_router()` correctly
//! implements the full A2A HTTP endpoint surface.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use novie_agent_sdk::{
    a2a_runtime::{
        Agent, ArtifactResult, HumanWaitRequest, InvokeContext, SqliteTaskStore, StreamContext,
        StreamEvent, TaskContext, TaskStatus,
    },
    manifest::{
        AgentKind, AgentManifestV2, AgentRuntime, DurabilityLevel, ExecutionHints, ProtocolMode,
    },
};
use serde_json::{Value, json};
use tower::util::ServiceExt;

// ─────────────────────────────────────────────────────────────────────────────
// Test fixtures
// ─────────────────────────────────────────────────────────────────────────────

fn tasks_manifest() -> AgentManifestV2 {
    AgentManifestV2 {
        schema: "https://novie.dev/schemas/agent-manifest-v2.json".to_owned(),
        agent_id: "test-conformance".to_owned(),
        name: "Conformance Test Agent".to_owned(),
        version: "0.1.0".to_owned(),
        kind: AgentKind::ExpertComplex,
        runtime: AgentRuntime::ExternalA2A,
        protocol_mode: ProtocolMode::Tasks,
        endpoint: Some("http://localhost:8080".to_owned()),
        capabilities: vec![],
        capability_manifest: vec![],
        declared_gates: vec![],
        execution: ExecutionHints {
            supports_cancel: true,
            emits_events: true,
            durability: DurabilityLevel::None,
            expected_duration_seconds: Some(10),
            ..Default::default()
        },
        required_secrets: vec![],
        supports_streaming: false,
        sandbox_isolation: "shared".to_owned(),
        task_bundles_path: String::new(),
        metadata: serde_json::Map::new(),
    }
}

fn simple_manifest() -> AgentManifestV2 {
    AgentManifestV2 {
        schema: "https://novie.dev/schemas/agent-manifest-v2.json".to_owned(),
        agent_id: "test-simple".to_owned(),
        name: "Simple Test Agent".to_owned(),
        version: "0.1.0".to_owned(),
        kind: AgentKind::ExpertBasic,
        runtime: AgentRuntime::ExternalA2A,
        protocol_mode: ProtocolMode::Simple,
        endpoint: None,
        capabilities: vec![],
        capability_manifest: vec![],
        declared_gates: vec![],
        execution: ExecutionHints::default(),
        required_secrets: vec![],
        supports_streaming: false,
        sandbox_isolation: "shared".to_owned(),
        task_bundles_path: String::new(),
        metadata: serde_json::Map::new(),
    }
}

fn stream_manifest() -> AgentManifestV2 {
    AgentManifestV2 {
        protocol_mode: ProtocolMode::Stream,
        supports_streaming: true,
        execution: ExecutionHints {
            emits_events: true,
            ..Default::default()
        },
        ..simple_manifest()
    }
}

fn json_request(method: &str, uri: &str, body: Option<Value>) -> Request<Body> {
    let body_bytes = body
        .map(|v| serde_json::to_vec(&v).unwrap())
        .unwrap_or_default();
    Request::builder()
        .method(method)
        .uri(uri)
        .header("Content-Type", "application/json")
        .body(Body::from(body_bytes))
        .unwrap()
}

async fn body_json(resp: axum::response::Response) -> Value {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap_or(Value::Null)
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Health and manifest
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_healthz() {
    let router = Agent::new(tasks_manifest())
        .task_handler(|_ctx: TaskContext| async { Ok(json!({})) })
        .build_router();

    let resp = router
        .oneshot(json_request("GET", "/healthz", None))
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp).await;
    assert_eq!(body["status"], "ok");
    assert_eq!(body["agent_id"], "test-conformance");
    assert_eq!(body["task_store_backend"], "memory");
}

#[tokio::test]
async fn test_well_known_manifest() {
    let router = Agent::new(tasks_manifest())
        .task_handler(|_ctx: TaskContext| async { Ok(json!({})) })
        .build_router();

    let resp = router
        .oneshot(json_request("GET", "/.well-known/agent.json", None))
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp).await;
    assert_eq!(body["agent_id"], "test-conformance");
    assert_eq!(body["protocol_mode"], "tasks");
    assert!(
        body["execution"]["supports_cancel"]
            .as_bool()
            .unwrap_or(false)
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Simple mode
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_simple_invoke_ok() {
    let router = Agent::new(simple_manifest())
        .invoke_handler(|ctx: InvokeContext| async move {
            let name = ctx.input["name"].as_str().unwrap_or("world");
            Ok(json!({ "greeting": format!("Hello {name}") }))
        })
        .build_router();

    let resp = router
        .oneshot(json_request(
            "POST",
            "/invoke",
            Some(json!({"input": {"name": "Alice"}})),
        ))
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp).await;
    assert_eq!(body["status"], "completed");
    assert_eq!(body["output"]["greeting"], "Hello Alice");
}

#[tokio::test]
async fn test_simple_invoke_handler_error_returns_500() {
    let router = Agent::new(simple_manifest())
        .invoke_handler(|_ctx: InvokeContext| async { Err("boom".to_owned()) })
        .build_router();

    let resp = router
        .oneshot(json_request("POST", "/invoke", Some(json!({"input": {}}))))
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[tokio::test]
async fn test_simple_invoke_idempotency_replays_completed_response() {
    let router = Agent::new(simple_manifest())
        .invoke_handler(|ctx: InvokeContext| async move {
            Ok(json!({ "name": ctx.input["name"], "nonce": "stable" }))
        })
        .build_router();

    let request = || {
        Request::builder()
            .method("POST")
            .uri("/invoke")
            .header("Content-Type", "application/json")
            .header("Idempotency-Key", "invoke-key-1")
            .body(Body::from(
                serde_json::to_vec(&json!({"input": {"name": "Alice"}})).unwrap(),
            ))
            .unwrap()
    };

    let first = body_json(router.clone().oneshot(request()).await.unwrap()).await;
    let second = body_json(router.oneshot(request()).await.unwrap()).await;

    assert_eq!(first, second);
    assert_eq!(second["output"]["nonce"], "stable");
}

#[tokio::test]
async fn test_stream_endpoint_returns_ndjson_done_event() {
    let router = Agent::new(stream_manifest())
        .stream_handler(|ctx: StreamContext| async move {
            Ok(vec![
                StreamEvent::progress(json!({"received": ctx.input["topic"]})),
                StreamEvent::done(json!({"ok": true})),
            ])
        })
        .build_router();

    let resp = router
        .oneshot(json_request(
            "POST",
            "/stream",
            Some(json!({"input": {"topic": "rust"}})),
        ))
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let content_type = resp.headers()["content-type"].to_str().unwrap().to_owned();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();

    assert!(content_type.contains("application/x-ndjson"));
    assert!(body.contains(r#""kind":"progress""#));
    assert!(body.contains(r#""kind":"done""#));
}

#[tokio::test]
async fn test_stream_idempotency_replays_completed_sequence() {
    let router = Agent::new(stream_manifest())
        .stream_handler(|_ctx: StreamContext| async move {
            Ok(vec![StreamEvent::done(json!({"answer": 42}))])
        })
        .build_router();

    let request = || {
        Request::builder()
            .method("POST")
            .uri("/stream")
            .header("Content-Type", "application/json")
            .header("Idempotency-Key", "stream-key-1")
            .body(Body::from(
                serde_json::to_vec(&json!({"input": {}})).unwrap(),
            ))
            .unwrap()
    };

    let first = axum::body::to_bytes(
        router.clone().oneshot(request()).await.unwrap().into_body(),
        usize::MAX,
    )
    .await
    .unwrap();
    let second = axum::body::to_bytes(
        router.oneshot(request()).await.unwrap().into_body(),
        usize::MAX,
    )
    .await
    .unwrap();

    assert_eq!(first, second);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Tasks mode — create and status
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_create_task_returns_202() {
    let router = Agent::new(tasks_manifest())
        .task_handler(|_ctx: TaskContext| async { Ok(json!({"done": true})) })
        .build_router();

    let resp = router
        .oneshot(json_request(
            "POST",
            "/tasks",
            Some(json!({"input": {"q": "test"}})),
        ))
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let body = body_json(resp).await;
    assert!(body["task_id"].is_string());
    let status = body["status"].as_str().unwrap();
    assert!(["queued", "running", "completed"].contains(&status));
}

#[tokio::test]
async fn test_get_task_not_found() {
    let router = Agent::new(tasks_manifest())
        .task_handler(|_ctx: TaskContext| async { Ok(json!({})) })
        .build_router();

    let resp = router
        .oneshot(json_request("GET", "/tasks/does-not-exist", None))
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_task_status_transitions_to_completed() {
    use tokio::time::{Duration, sleep};

    let router = Agent::new(tasks_manifest())
        .task_handler(|ctx: TaskContext| async move {
            ctx.emit_message("working").await;
            Ok(json!({"answer": 42}))
        })
        .build_router();

    // Create task
    let create_resp = router
        .clone()
        .oneshot(json_request("POST", "/tasks", Some(json!({"input": {}}))))
        .await
        .unwrap();
    let create_body = body_json(create_resp).await;
    let task_id = create_body["task_id"].as_str().unwrap().to_owned();

    sleep(Duration::from_millis(200)).await;

    // Check status
    let status_resp = router
        .clone()
        .oneshot(json_request("GET", &format!("/tasks/{task_id}"), None))
        .await
        .unwrap();
    assert_eq!(status_resp.status(), StatusCode::OK);
    let status_body = body_json(status_resp).await;
    assert_eq!(status_body["status"], "completed");

    // Check events
    let events_resp = router
        .clone()
        .oneshot(json_request(
            "GET",
            &format!("/tasks/{task_id}/events"),
            None,
        ))
        .await
        .unwrap();
    let events_body = body_json(events_resp).await;
    let events = events_body["events"].as_array().unwrap();
    assert!(!events.is_empty());
    assert_eq!(events[0]["kind"], "message");

    // Check result
    let result_resp = router
        .oneshot(json_request(
            "GET",
            &format!("/tasks/{task_id}/result"),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(result_resp.status(), StatusCode::OK);
    let result_body = body_json(result_resp).await;
    assert_eq!(result_body["output"]["answer"], 42);
}

#[tokio::test]
async fn test_worker_helpers_emit_artifact_usage_and_human_wait() {
    use tokio::time::{Duration, sleep};

    let router = Agent::new(tasks_manifest())
        .task_handler(|ctx: TaskContext| async move {
            ctx.emit_artifact(ArtifactResult {
                artifact_type: "diff".to_owned(),
                summary: "Diff ready".to_owned(),
                content: json!({"files": []}),
                metadata: Value::Null,
            })
            .await;
            ctx.report_llm_usage("example", "model", Some(1), Some(2), Some(3))
                .await;
            ctx.wait_for_human(HumanWaitRequest {
                gate_id: "approve".to_owned(),
                prompt: "Approve?".to_owned(),
                allowed_actions: vec!["approve".to_owned(), "reject".to_owned()],
                resume_reference: "resume-1".to_owned(),
                timeout_policy: json!({"on_timeout": "cancel"}),
                metadata: Value::Null,
            })
            .await;
            Ok(ctx.result(json!({"approved": true}), vec![], Value::Null))
        })
        .build_router();

    let create_resp = router
        .clone()
        .oneshot(json_request("POST", "/tasks", Some(json!({"input": {}}))))
        .await
        .unwrap();
    let task_id = body_json(create_resp).await["task_id"]
        .as_str()
        .unwrap()
        .to_owned();

    sleep(Duration::from_millis(200)).await;

    let events_resp = router
        .clone()
        .oneshot(json_request(
            "GET",
            &format!("/tasks/{task_id}/events"),
            None,
        ))
        .await
        .unwrap();
    let events_body = body_json(events_resp).await;
    let kinds: Vec<&str> = events_body["events"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|event| event["kind"].as_str())
        .collect();

    assert!(kinds.contains(&"artifact"));
    assert!(kinds.contains(&"usage"));
    assert!(kinds.contains(&"wait_prompt"));
}

#[tokio::test]
async fn test_sqlite_task_store_persists_terminal_result() {
    use novie_agent_sdk::a2a_runtime::TaskStore;

    let db_path = std::env::temp_dir().join(format!(
        "novie-rust-sdk-test-{}.sqlite3",
        uuid::Uuid::new_v4()
    ));
    let store = SqliteTaskStore::open(&db_path).unwrap();
    let record = store
        .create("task-1", json!({"q": "persist"}), Some("idem"))
        .await;
    store
        .set_result(&record.task_id, json!({"answer": 42}))
        .await;
    drop(store);

    let reopened = SqliteTaskStore::open(&db_path).unwrap();
    let loaded = reopened.get("task-1").await.unwrap();

    assert_eq!(loaded.status, TaskStatus::Completed);
    assert_eq!(loaded.result.unwrap()["answer"], 42);

    let _ = std::fs::remove_file(db_path);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Cancel
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_cancel_running_task() {
    use tokio::time::{Duration, sleep};

    let router = Agent::new(tasks_manifest())
        .task_handler(|ctx: TaskContext| async move {
            for _ in 0..1000 {
                if ctx.is_cancelled().await {
                    return Err("cancelled".to_owned());
                }
                sleep(Duration::from_millis(10)).await;
            }
            Ok(json!({"done": true}))
        })
        .build_router();

    let create_resp = router
        .clone()
        .oneshot(json_request("POST", "/tasks", Some(json!({"input": {}}))))
        .await
        .unwrap();
    let task_id = body_json(create_resp).await["task_id"]
        .as_str()
        .unwrap()
        .to_owned();

    sleep(Duration::from_millis(50)).await;

    let cancel_resp = router
        .clone()
        .oneshot(json_request(
            "POST",
            &format!("/tasks/{task_id}/cancel"),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(cancel_resp.status(), StatusCode::ACCEPTED);

    sleep(Duration::from_millis(100)).await;

    let status_resp = router
        .oneshot(json_request("GET", &format!("/tasks/{task_id}"), None))
        .await
        .unwrap();
    let s = body_json(status_resp).await;
    assert_eq!(s["status"], "cancelled");
}

#[tokio::test]
async fn test_cancel_completed_task_returns_409() {
    use tokio::time::{Duration, sleep};

    let router = Agent::new(tasks_manifest())
        .task_handler(|_ctx: TaskContext| async { Ok(json!({"done": true})) })
        .build_router();

    let create_resp = router
        .clone()
        .oneshot(json_request("POST", "/tasks", Some(json!({"input": {}}))))
        .await
        .unwrap();
    let task_id = body_json(create_resp).await["task_id"]
        .as_str()
        .unwrap()
        .to_owned();

    sleep(Duration::from_millis(200)).await;

    let cancel_resp = router
        .oneshot(json_request(
            "POST",
            &format!("/tasks/{task_id}/cancel"),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(cancel_resp.status(), StatusCode::CONFLICT);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Idempotency
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_idempotency_key_returns_same_task() {
    let router = Agent::new(tasks_manifest())
        .task_handler(|_ctx: TaskContext| async { Ok(json!({"done": true})) })
        .build_router();

    let req1 = Request::builder()
        .method("POST")
        .uri("/tasks")
        .header("Content-Type", "application/json")
        .header("Idempotency-Key", "idem-key-xyz")
        .body(Body::from(
            serde_json::to_vec(&json!({"input": {}})).unwrap(),
        ))
        .unwrap();

    let resp1 = router.clone().oneshot(req1).await.unwrap();
    let id1 = body_json(resp1).await["task_id"]
        .as_str()
        .unwrap()
        .to_owned();

    let req2 = Request::builder()
        .method("POST")
        .uri("/tasks")
        .header("Content-Type", "application/json")
        .header("Idempotency-Key", "idem-key-xyz")
        .body(Body::from(
            serde_json::to_vec(&json!({"input": {}})).unwrap(),
        ))
        .unwrap();

    let resp2 = router.oneshot(req2).await.unwrap();
    let id2 = body_json(resp2).await["task_id"]
        .as_str()
        .unwrap()
        .to_owned();

    assert_eq!(id1, id2, "idempotency key must return the same task id");
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Result 409 when task not completed
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_result_409_when_task_running() {
    use tokio::time::{Duration, sleep};

    let router = Agent::new(tasks_manifest())
        .task_handler(|_ctx: TaskContext| async {
            sleep(Duration::from_secs(10)).await;
            Ok(json!({}))
        })
        .build_router();

    let create_resp = router
        .clone()
        .oneshot(json_request("POST", "/tasks", Some(json!({"input": {}}))))
        .await
        .unwrap();
    let task_id = body_json(create_resp).await["task_id"]
        .as_str()
        .unwrap()
        .to_owned();

    let result_resp = router
        .oneshot(json_request(
            "GET",
            &format!("/tasks/{task_id}/result"),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(result_resp.status(), StatusCode::CONFLICT);
}
