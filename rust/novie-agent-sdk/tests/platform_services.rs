//! End-to-end tests for `PlatformServicesClient` against a wiremock server.

use std::time::Duration;

use novie_agent_sdk::headers::RequestHeaders;
use novie_agent_sdk::session::{SessionEvent, SessionEventSource};
use novie_agent_sdk::transport::TransportConfig;
use novie_agent_sdk::{Error, PlatformServicesClient};
use serde_json::json;
use wiremock::matchers::{body_json, header, header_exists, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn quick_cfg() -> TransportConfig {
    TransportConfig {
        timeout: Duration::from_secs(2),
        max_retries: 2,
        ..TransportConfig::default()
    }
}

fn make_client(server: &MockServer) -> PlatformServicesClient {
    PlatformServicesClient::with_config(
        format!("{}/internal/callbacks", server.uri()),
        RequestHeaders {
            tenant_id: "tenant-1".into(),
            workspace_id: "workspace-1".into(),
            project_id: "project-1".into(),
            user_id: "user-1".into(),
            session_id: "session-1".into(),
            request_id: "request-1".into(),
            ..Default::default()
        },
        quick_cfg(),
    )
    .unwrap()
}

#[tokio::test]
async fn events_publish_wraps_payload_in_kwargs() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/internal/callbacks/events/publish"))
        .and(header("x-novie-org-id", "tenant-1"))
        .and(header("x-novie-workspace-id", "workspace-1"))
        .and(header("x-novie-project-id", "project-1"))
        .and(header("x-novie-user-id", "user-1"))
        .and(header_exists("x-novie-sig"))
        .and(body_json(json!({
            "kwargs": {"topic": "agent.startup", "payload": {"agent": "x"}},
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true, "result": null})))
        .mount(&server)
        .await;

    let client = make_client(&server);
    client
        .events
        .publish("agent.startup", &json!({"agent": "x"}))
        .await
        .unwrap();
}

#[tokio::test]
async fn sessions_record_round_trip() {
    let server = MockServer::start().await;
    let response_event = json!({
        "seq": 7,
        "event_id": "sev-server-assigned",
        "occurred_at": "2026-04-24T10:00:00Z",
        "session_id": "sess-1",
        "source": "callback",
        "kind": "agent_status",
        "summary": "",
        "tenant_id": "tnt",
        "workspace_id": "ws",
        "thread_id": null,
        "payload": {},
        "metadata": {},
    });
    Mock::given(method("POST"))
        .and(path("/internal/callbacks/sessions/record"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ok": true,
            "result": response_event,
        })))
        .mount(&server)
        .await;

    let client = make_client(&server);
    let evt = SessionEvent::new_for_record("sess-local", SessionEventSource::Callback, "ping");
    let returned = client.sessions.record(&evt).await.unwrap();
    assert_eq!(returned.seq, 7);
    assert_eq!(returned.event_id, "sev-server-assigned");
    assert_eq!(returned.session_id, "sess-1");
}

#[tokio::test]
async fn rpc_retries_on_503() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/internal/callbacks/events/publish"))
        .respond_with(ResponseTemplate::new(503).set_body_json(json!({
            "error": {"code": "PLATFORM_UNAVAILABLE", "message": "warming up", "retry_after_ms": 5},
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/internal/callbacks/events/publish"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true, "result": null})))
        .mount(&server)
        .await;

    let client = make_client(&server);
    client.events.publish("startup", &json!({})).await.unwrap();
}

#[tokio::test]
async fn rpc_does_not_retry_on_422() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/internal/callbacks/events/publish"))
        .respond_with(ResponseTemplate::new(422).set_body_json(json!({
            "error": {"code": "PROTOCOL_INVALID_BODY", "message": "bad topic"},
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client = make_client(&server);
    let err = client
        .events
        .publish("startup", &json!({}))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Protocol { .. }), "got {err:?}");
}

#[tokio::test]
async fn list_sessions_decodes_array() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/internal/callbacks/sessions/list_sessions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ok": true,
            "result": [
                {
                    "session_id": "s1",
                    "tenant_id": "t",
                    "workspace_id": "w",
                    "created_at": "2026-04-24T10:00:00Z",
                    "updated_at": "2026-04-24T10:00:00Z",
                },
            ],
        })))
        .mount(&server)
        .await;

    let client = make_client(&server);
    let sessions = client.sessions.list_sessions(10).await.unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].session_id, "s1");
}
