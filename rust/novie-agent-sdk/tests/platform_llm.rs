//! Tests for Rust PlatformLlmClient.

use novie_agent_sdk::{ChatMessage, ChatOptions, Error, PlatformLlmClient, PlatformLlmIdentity};
use serde_json::json;
use wiremock::matchers::{bearer_token, body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn make_client(server: &MockServer) -> PlatformLlmClient {
    PlatformLlmClient::with_identity(
        server.uri(),
        "platform-token",
        "rust-agent",
        PlatformLlmIdentity {
            org_id: "org-1".to_owned(),
            project_id: "project-1".to_owned(),
            workspace_id: "workspace-1".to_owned(),
            user_id: "user-1".to_owned(),
            session_id: "session-1".to_owned(),
            request_id: "request-1".to_owned(),
            auth_source: "agent_callback".to_owned(),
            ..Default::default()
        },
    )
    .unwrap()
}

#[tokio::test]
async fn chat_invokes_platform_llm_capability() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/capabilities/platform.llm.chat/invoke"))
        .and(bearer_token("platform-token"))
        .and(header("x-novie-org-id", "org-1"))
        .and(header("x-novie-project-id", "project-1"))
        .and(header("x-novie-workspace-id", "workspace-1"))
        .and(header("x-novie-user-id", "user-1"))
        .and(body_json(json!({
            "arguments": {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "test-model",
                "temperature": 0.2
            },
            "caller_type": "agent",
            "caller_id": "agent:rust-agent",
            "caller_mode": "execute",
            "mode": "execute"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "ok",
            "result": {
                "content": "hi",
                "usage_metadata": {"total_tokens": 3}
            }
        })))
        .mount(&server)
        .await;

    let client = make_client(&server);
    let result = client
        .chat(
            vec![ChatMessage::user("hello")],
            ChatOptions {
                model: Some("test-model"),
                temperature: Some(0.2),
            },
        )
        .await
        .unwrap();
    assert_eq!(result["content"], "hi");
}

#[tokio::test]
async fn quota_exceeded_maps_to_typed_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/capabilities/platform.llm.chat/invoke"))
        .respond_with(ResponseTemplate::new(403).set_body_json(json!({
            "detail": {
                "status": "denied",
                "error_code": "quota_exceeded",
                "explanation": "org token pool exhausted",
                "metadata": {
                    "quota": {
                        "org_id": "org-1",
                        "remaining_tokens": 0
                    }
                }
            }
        })))
        .mount(&server)
        .await;

    let client = make_client(&server);
    let err = client
        .chat(vec![ChatMessage::user("hello")], ChatOptions::default())
        .await
        .unwrap_err();
    match err {
        Error::QuotaExceeded {
            org_id,
            remaining_tokens,
            ..
        } => {
            assert_eq!(org_id.as_deref(), Some("org-1"));
            assert_eq!(remaining_tokens, Some(0));
        }
        other => panic!("expected quota error, got {other:?}"),
    }
}

#[tokio::test]
async fn budget_check_returns_result_map() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/capabilities/platform.llm.budget_check/invoke"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "ok",
            "result": {
                "allow": true,
                "org_pool": {"remaining_tokens": 500}
            }
        })))
        .mount(&server)
        .await;

    let client = make_client(&server);
    let result = client.budget_check().await.unwrap();
    assert_eq!(result["allow"], true);
    assert_eq!(result["org_pool"]["remaining_tokens"], 500);
}
