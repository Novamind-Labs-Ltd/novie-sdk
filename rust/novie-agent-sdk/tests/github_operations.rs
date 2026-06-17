use novie_agent_sdk::{Error, GitHubOperationsClient};
use serde_json::json;
use wiremock::matchers::{bearer_token, body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn merge_pull_request_calls_run_scoped_gateway() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(
            "/agent-runs/run-1/github/operations/merge-pull-request",
        ))
        .and(bearer_token("runtime-token"))
        .and(body_json(json!({
            "repositoryFullName": "org/repo",
            "number": 42,
            "mergeMethod": "squash"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {
                "merged": true,
                "repositoryFullName": "org/repo",
                "pullRequestNumber": 42
            }
        })))
        .mount(&server)
        .await;

    let client = GitHubOperationsClient::new(server.uri(), "runtime-token", "run-1").unwrap();
    let result = client
        .merge_pull_request("org/repo", 42, Some("squash"))
        .await
        .unwrap();

    assert_eq!(result["merged"], true);
    assert_eq!(result["repositoryFullName"], "org/repo");
}

#[tokio::test]
async fn constructor_rejects_missing_runtime_token_or_run_id() {
    let missing_token = GitHubOperationsClient::new("http://platform.test", "", "run-1")
        .err()
        .unwrap();
    assert!(matches!(missing_token, Error::InvalidArgument(_)));

    let missing_run = GitHubOperationsClient::new("http://platform.test", "runtime-token", "")
        .err()
        .unwrap();
    assert!(matches!(missing_run, Error::InvalidArgument(_)));
}

#[tokio::test]
async fn pull_request_calls_run_scoped_gateway() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/agent-runs/run-1/github/operations/pull-request"))
        .and(bearer_token("runtime-token"))
        .and(body_json(json!({
            "repositoryFullName": "org/repo",
            "number": 42
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {
                "number": 42,
                "repositoryFullName": "org/repo",
                "state": "open",
                "baseRefName": "main"
            }
        })))
        .mount(&server)
        .await;

    let client = GitHubOperationsClient::new(server.uri(), "runtime-token", "run-1").unwrap();
    let result = client.pull_request("org/repo", 42).await.unwrap();

    assert_eq!(result["number"], 42);
    assert_eq!(result["baseRefName"], "main");
}

#[tokio::test]
async fn ref_calls_run_scoped_gateway() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/agent-runs/run-1/github/operations/ref"))
        .and(bearer_token("runtime-token"))
        .and(body_json(json!({
            "repositoryFullName": "org/repo",
            "ref": "heads/main"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {
                "ref": "refs/heads/main",
                "treeSha": "tree-sha"
            }
        })))
        .mount(&server)
        .await;

    let client = GitHubOperationsClient::new(server.uri(), "runtime-token", "run-1").unwrap();
    let result = client.ref_("org/repo", "heads/main").await.unwrap();

    assert_eq!(result["treeSha"], "tree-sha");
}

#[tokio::test]
async fn update_pull_request_branch_calls_run_scoped_gateway() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(
            "/agent-runs/run-1/github/operations/update-pull-request-branch",
        ))
        .and(bearer_token("runtime-token"))
        .and(body_json(json!({
            "repositoryFullName": "org/repo",
            "number": 42,
            "expectedHeadSha": "old-head"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {
                "number": 42,
                "repositoryFullName": "org/repo",
                "headSha": "new-head"
            }
        })))
        .mount(&server)
        .await;

    let client = GitHubOperationsClient::new(server.uri(), "runtime-token", "run-1").unwrap();
    let result = client
        .update_pull_request_branch("org/repo", 42, Some("old-head"))
        .await
        .unwrap();

    assert_eq!(result["headSha"], "new-head");
}

#[tokio::test]
async fn gateway_error_maps_to_auth_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/agent-runs/run-1/github/operations/read-file"))
        .respond_with(ResponseTemplate::new(403).set_body_json(json!({
            "detail": {
                "error": "github_repository_not_granted",
                "message": "denied"
            }
        })))
        .mount(&server)
        .await;

    let client = GitHubOperationsClient::new(server.uri(), "runtime-token", "run-1").unwrap();
    let err = client
        .read_file("org/other", "README.md", None, None)
        .await
        .unwrap_err();

    assert_eq!(err.http_status(), Some(403));
    assert_eq!(err.code(), Some("github_repository_not_granted"));
}
