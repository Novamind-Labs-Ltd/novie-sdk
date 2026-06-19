use novie_agent_sdk::{Error, PmsIssueClient, PmsIssueIdentity};
use serde_json::json;
use wiremock::matchers::{bearer_token, body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn pms_client(base_url: String) -> PmsIssueClient {
    PmsIssueClient::with_identity(
        base_url,
        "runtime-token",
        PmsIssueIdentity::service("tenant-1", "project-1", "agent:pms-test"),
    )
    .unwrap()
}

#[tokio::test]
async fn list_candidate_issues_calls_pms_api_boundary() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/pms/issues/candidates"))
        .and(bearer_token("runtime-token"))
        .and(header("x-novie-org-id", "tenant-1"))
        .and(header("x-novie-project-id", "project-1"))
        .and(header("x-novie-workspace-id", "tenant-1"))
        .and(header("x-novie-service-principal", "agent:pms-test"))
        .and(body_json(json!({
            "projectIds": ["project-1"],
            "automationActions": ["Review", "Rework"],
            "includeHumanReview": true
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {
                "issues": [{
                    "id": "issue-1",
                    "identifier": "NOV-1",
                    "title": "Implement feature",
                    "status": {
                        "id": "status-review",
                        "title": "QA Gate",
                        "stage": "InProgress",
                        "automationAction": "Review"
                    },
                    "organizationId": "tenant-1",
                    "workspaceId": "workspace-1",
                    "projectId": "project-1",
                    "linkedPullRequestUrls": ["https://github.com/org/repo/pull/12"],
                    "agenticOrchestrationValues": {
                        "rework": { "instructionSource": "comment" }
                    }
                }]
            }
        })))
        .mount(&server)
        .await;

    let client = pms_client(server.uri());
    let issues = client
        .list_candidate_issues(
            vec!["project-1".to_string()],
            vec!["Review".to_string(), "Rework".to_string()],
            true,
            Some("tenant-1"),
            Some("workspace-1"),
        )
        .await
        .unwrap();

    assert_eq!(issues[0].id, "issue-1");
    assert_eq!(issues[0].automation_action, "Review");
    assert_eq!(issues[0].status_title, "QA Gate");
    assert_eq!(
        issues[0].linked_pr_urls,
        vec!["https://github.com/org/repo/pull/12".to_string()]
    );
    assert_eq!(
        issues[0].agentic_orchestration_values["rework"]["instructionSource"],
        "comment"
    );
}

#[tokio::test]
async fn list_issues_by_states_calls_pms_api_boundary() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/pms/issues/by-states"))
        .and(bearer_token("runtime-token"))
        .and(header("x-novie-org-id", "tenant-1"))
        .and(header("x-novie-project-id", "project-1"))
        .and(header("x-novie-workspace-id", "tenant-1"))
        .and(header("x-novie-service-principal", "agent:pms-test"))
        .and(body_json(json!({
            "states": ["Done"],
            "projectIds": ["project-1"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": { "issues": [{ "id": "issue-1", "state": "Done" }] }
        })))
        .mount(&server)
        .await;

    let client = pms_client(server.uri());
    let issues = client
        .list_issues_by_states(
            vec!["Done".to_string()],
            vec!["project-1".to_string()],
            Some("tenant-1"),
            Some("workspace-1"),
        )
        .await
        .unwrap();

    assert_eq!(issues[0].id, "issue-1");
}

#[tokio::test]
async fn fetch_active_cycle_id_calls_pms_api_boundary() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/pms/issues/active-cycle"))
        .and(bearer_token("runtime-token"))
        .and(header("x-novie-org-id", "tenant-1"))
        .and(header("x-novie-project-id", "project-1"))
        .and(header("x-novie-workspace-id", "tenant-1"))
        .and(header("x-novie-service-principal", "agent:pms-test"))
        .and(body_json(json!({})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": { "activeCycleId": "cycle-1" }
        })))
        .mount(&server)
        .await;

    let client = pms_client(server.uri());
    let cycle_id = client
        .fetch_active_cycle_id(Some("tenant-1"), Some("workspace-1"))
        .await
        .unwrap();

    assert_eq!(cycle_id.as_deref(), Some("cycle-1"));
}

#[tokio::test]
async fn update_agentic_orchestration_values_uses_durable_contract() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/pms/issues/update-agentic-orchestration-values"))
        .and(bearer_token("runtime-token"))
        .and(header("x-novie-org-id", "tenant-1"))
        .and(header("x-novie-project-id", "project-1"))
        .and(header("x-novie-workspace-id", "tenant-1"))
        .and(header("x-novie-service-principal", "agent:pms-test"))
        .and(body_json(json!({
            "issueId": "issue-1",
            "patch": {
                "rework": { "lastDeniedReason": "rework_required_missing_human_input" }
            },
            "actorUserId": "agent-user"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {
                "agenticOrchestrationValues": {
                    "rework": { "lastDeniedReason": "rework_required_missing_human_input" }
                }
            }
        })))
        .mount(&server)
        .await;

    let client = pms_client(server.uri());
    let values = client
        .update_agentic_orchestration_values(
            "issue-1",
            json!({"rework": {"lastDeniedReason": "rework_required_missing_human_input"}}),
            Some("agent-user"),
            Some("tenant-1"),
            Some("workspace-1"),
        )
        .await
        .unwrap();

    assert_eq!(
        values["rework"]["lastDeniedReason"],
        "rework_required_missing_human_input"
    );
}

#[tokio::test]
async fn list_comments_maps_created_at_and_author() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/pms/issues/comments"))
        .and(bearer_token("runtime-token"))
        .and(header("x-novie-org-id", "tenant-1"))
        .and(header("x-novie-project-id", "project-1"))
        .and(header("x-novie-workspace-id", "tenant-1"))
        .and(header("x-novie-service-principal", "agent:pms-test"))
        .and(body_json(json!({
            "issueId": "issue-1",
            "first": 20
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {
                "comments": [{
                    "id": "comment-1",
                    "content": "Please rework",
                    "createdAt": "2026-06-19T01:02:03Z",
                    "author": { "id": "human-1", "name": "Human" }
                }]
            }
        })))
        .mount(&server)
        .await;

    let client = pms_client(server.uri());
    let comments = client
        .list_comments("issue-1", Some(20), None, None)
        .await
        .unwrap();

    assert_eq!(comments[0].id, "comment-1");
    assert_eq!(comments[0].author_id, "human-1");
    assert_eq!(comments[0].created_at, "2026-06-19T01:02:03Z");
}

#[tokio::test]
async fn constructor_rejects_missing_runtime_token() {
    let err = PmsIssueClient::new("http://platform.test", "")
        .err()
        .unwrap();
    assert!(matches!(err, Error::InvalidArgument(_)));
}
