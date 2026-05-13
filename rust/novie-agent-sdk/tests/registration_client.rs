use novie_agent_sdk::{
    a2a_runtime::RegistrationClient,
    manifest::{AgentKind, AgentManifestV2, AgentRuntime, ExecutionHints, ProtocolMode},
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, method, path},
};

fn manifest() -> AgentManifestV2 {
    AgentManifestV2 {
        schema: "https://novie.dev/schemas/agent-manifest-v2.json".to_owned(),
        agent_id: "registration-test".to_owned(),
        name: "Registration Test".to_owned(),
        version: "0.1.0".to_owned(),
        kind: AgentKind::ExpertBasic,
        runtime: AgentRuntime::ExternalA2A,
        protocol_mode: ProtocolMode::Simple,
        endpoint: Some("http://localhost:8080".to_owned()),
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

#[tokio::test]
async fn registration_lifecycle_calls_expected_routes() {
    let server = MockServer::start().await;
    let manifest = manifest();

    Mock::given(method("POST"))
        .and(path("/agents/register"))
        .and(body_json(serde_json::to_value(&manifest).unwrap()))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/agents/registration-test/heartbeat"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/agents/registration-test"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    let client = RegistrationClient::new(server.uri(), "registration-test");

    client.register(&manifest).await.unwrap();
    client.heartbeat().await.unwrap();
    client.deregister().await.unwrap();
}
