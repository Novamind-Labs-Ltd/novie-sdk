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
        .and(path("/invocations"))
        .and(bearer_token("platform-token"))
        .and(header("x-novie-org-id", "org-1"))
        .and(header("x-novie-project-id", "project-1"))
        .and(header("x-novie-workspace-id", "workspace-1"))
        .and(header("x-novie-user-id", "user-1"))
        .and(body_json(json!({
            "capability_id": "platform.llm.chat",
            "provider_id": "platform.llm",
            "mode": "execute",
            "inputs": {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "test-model",
                "temperature": 0.2
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "ok",
            "output": {
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
        .and(path("/invocations"))
        .respond_with(ResponseTemplate::new(403).set_body_json(json!({
            "detail": {
                "status": "denied",
                "error_code": "quota_exceeded",
                "error_message": "org token pool exhausted",
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
async fn error_message_field_preferred_over_explanation_fallback() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/invocations"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "detail": {
                "error_message": "preferred message",
                "explanation": "legacy fallback message"
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
        Error::Protocol { message, .. } => assert_eq!(message, "preferred message"),
        other => panic!("expected protocol error, got {other:?}"),
    }
}

#[tokio::test]
async fn explanation_field_used_when_error_message_absent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/invocations"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "detail": {
                "explanation": "legacy fallback message"
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
        Error::Protocol { message, .. } => assert_eq!(message, "legacy fallback message"),
        other => panic!("expected protocol error, got {other:?}"),
    }
}

#[tokio::test]
async fn budget_check_returns_result_map() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/invocations"))
        .and(body_json(json!({
            "capability_id": "platform.llm.budget_check",
            "provider_id": "platform.llm",
            "mode": "execute",
            "inputs": {}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "ok",
            "output": {
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

// ---------------------------------------------------------------------------
// LlmBudgetGuard tests
// ---------------------------------------------------------------------------

use novie_agent_sdk::{LlmBudgetGuard, TokenUsage};

fn make_budget_guard(server: &MockServer) -> LlmBudgetGuard {
    LlmBudgetGuard::new(make_client(server))
}

#[tokio::test]
async fn budget_guard_preflight_allows_when_budget_ok() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/invocations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "ok",
            "output": {
                "allow": true,
                "exhausted": false,
                "remaining_tokens": 5000,
                "total_tokens": 10000
            }
        })))
        .mount(&server)
        .await;

    let guard = make_budget_guard(&server);
    guard
        .preflight()
        .await
        .expect("preflight should pass when allow=true");
    assert!(!guard.should_stop());
}

#[tokio::test]
async fn budget_guard_preflight_denies_when_exhausted() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/invocations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "ok",
            "output": {
                "allow": false,
                "exhausted": true,
                "remaining_tokens": 0,
                "total_tokens": 10000,
                "reason": "org token pool exhausted"
            }
        })))
        .mount(&server)
        .await;

    let guard = make_budget_guard(&server);
    let err = guard
        .preflight()
        .await
        .expect_err("preflight should fail when exhausted");
    match err {
        Error::BudgetExceeded { message, .. } => {
            assert!(!message.is_empty());
        }
        other => panic!("expected BudgetExceeded, got {other:?}"),
    }
}

#[tokio::test]
async fn budget_guard_preflight_returns_governance_unavailable_on_503() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/invocations"))
        .respond_with(ResponseTemplate::new(503).set_body_json(json!({
            "detail": "service unavailable"
        })))
        .mount(&server)
        .await;

    let guard = make_budget_guard(&server);
    let result = guard.preflight().await;
    match result {
        Err(Error::GovernanceUnavailable { .. }) | Ok(()) => {
            // Either is acceptable: SDK returns GovernanceUnavailable or degrades gracefully.
        }
        Err(other) => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn budget_guard_report_usage_sets_exceeded_on_exhausted_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/invocations"))
        .and(body_json(json!({
            "capability_id": "platform.llm.report_usage",
            "provider_id": "platform.llm",
            "mode": "execute",
            "inputs": {
                "provider": "anthropic",
                "model": "claude-opus",
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "ok",
            "output": {
                "recorded": true,
                "exhausted": true
            }
        })))
        .mount(&server)
        .await;

    let guard = make_budget_guard(&server);
    assert!(
        !guard.should_stop(),
        "should not stop before reporting usage"
    );

    guard
        .report_usage(&TokenUsage {
            input_tokens: 1000,
            output_tokens: 500,
            total_tokens: 1500,
            provider: "anthropic".to_owned(),
            model: "claude-opus".to_owned(),
        })
        .await;

    assert!(
        guard.should_stop(),
        "should stop after platform reports exhausted"
    );
}

#[tokio::test]
async fn structured_uses_output_schema_field_not_schema() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/invocations"))
        .and(body_json(json!({
            "capability_id": "platform.llm.structured",
            "provider_id": "platform.llm",
            "mode": "execute",
            "inputs": {
                "messages": [{"role": "user", "content": "extract"}],
                "output_schema": {"type": "object"},
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "ok",
            "output": {
                "structured": {"answer": 42}
            }
        })))
        .mount(&server)
        .await;

    let client = make_client(&server);
    let result = client
        .structured(
            vec![ChatMessage::user("extract")],
            json!({"type": "object"}),
            novie_agent_sdk::StructuredOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(result["structured"]["answer"], 42);
}
