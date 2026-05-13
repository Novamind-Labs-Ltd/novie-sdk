//! End-to-end tests for `AgentStatusClient` against a wiremock server.

use std::time::Duration;

use novie_agent_sdk::agent_status::AgentStatusKind;
use novie_agent_sdk::transport::TransportConfig;
use novie_agent_sdk::{AgentStatusCallbackConfig, AgentStatusClient, Error, ReportOptions};
use serde_json::json;
use wiremock::matchers::{bearer_token, body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn quick_cfg() -> TransportConfig {
    TransportConfig {
        timeout: Duration::from_secs(2),
        max_retries: 2,
        ..TransportConfig::default()
    }
}

fn make_config(server: &MockServer) -> AgentStatusCallbackConfig {
    AgentStatusCallbackConfig {
        url: format!("{}/internal/callbacks/agent-status", server.uri()),
        token: "scoped-token".into(),
        version: "v1".into(),
        session_id: Some("sess-1".into()),
        thread_id: Some("thr-1".into()),
        plan_id: None,
    }
}

#[tokio::test]
async fn report_posts_bare_event_and_returns_event_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/internal/callbacks/agent-status"))
        .and(bearer_token("scoped-token"))
        .and(header("content-type", "application/json"))
        // event must be sent as-is, NOT wrapped in {"kwargs": ...}
        .and(body_partial_json(json!({
            "kind": "tool_call",
            "agent_id": "agent-1",
            "task_id": "task-1",
            "session_id": "sess-1",
            "thread_id": "thr-1",
            "payload": {"tool": "search"},
        })))
        .respond_with(ResponseTemplate::new(202))
        .mount(&server)
        .await;

    let client =
        AgentStatusClient::with_transport_config(make_config(&server), "agent-1", quick_cfg())
            .unwrap();
    let event_id = client
        .report(ReportOptions {
            kind: AgentStatusKind::ToolCall,
            task_id: "task-1",
            payload: Some(json!({"tool": "search"}).as_object().cloned().unwrap()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(!event_id.is_empty());
}

#[tokio::test]
async fn retries_on_503_then_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/internal/callbacks/agent-status"))
        .respond_with(ResponseTemplate::new(503).set_body_json(json!({
            "error": {"code": "PLATFORM_UNAVAILABLE", "message": "spinning up", "retry_after_ms": 10},
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/internal/callbacks/agent-status"))
        .respond_with(ResponseTemplate::new(202))
        .mount(&server)
        .await;

    let client =
        AgentStatusClient::with_transport_config(make_config(&server), "agent-1", quick_cfg())
            .unwrap();
    let event_id = client
        .report(ReportOptions {
            kind: AgentStatusKind::TurnEnd,
            task_id: "task-2",
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(!event_id.is_empty());
}

#[tokio::test]
async fn does_not_retry_on_422() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/internal/callbacks/agent-status"))
        .respond_with(ResponseTemplate::new(422).set_body_json(json!({
            "error": {"code": "PROTOCOL_INVALID_BODY", "message": "bad event"},
        })))
        .expect(1) // must NOT be called twice
        .mount(&server)
        .await;

    let client =
        AgentStatusClient::with_transport_config(make_config(&server), "agent-1", quick_cfg())
            .unwrap();
    let err = client
        .report(ReportOptions {
            kind: AgentStatusKind::ToolCall,
            task_id: "task-3",
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Protocol { .. }), "got {err:?}");
}

#[tokio::test]
async fn auth_errors_are_distinct() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/internal/callbacks/agent-status"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "error": {"code": "AUTH_INVALID_TOKEN", "message": "bad token"},
        })))
        .mount(&server)
        .await;

    let client =
        AgentStatusClient::with_transport_config(make_config(&server), "agent-1", quick_cfg())
            .unwrap();
    let err = client
        .report(ReportOptions {
            kind: AgentStatusKind::ToolCall,
            task_id: "task-4",
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Auth { .. }), "got {err:?}");
}

#[tokio::test]
async fn rejects_event_with_mismatched_agent_id() {
    let server = MockServer::start().await;
    let client =
        AgentStatusClient::with_transport_config(make_config(&server), "agent-1", quick_cfg())
            .unwrap();
    let mut event = novie_agent_sdk::agent_status::AgentStatusEvent::now(
        AgentStatusKind::TurnStart,
        "OTHER-AGENT",
        "task-x",
    );
    event.agent_id = "OTHER-AGENT".into();
    let err = client.send(&event).await.unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "got {err:?}");
}
