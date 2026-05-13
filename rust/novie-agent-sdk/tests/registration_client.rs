use novie_agent_sdk::{
    a2a_runtime::{RegistrationClient, RegistrationLifecycle, RegistrationOptions},
    manifest::{AgentKind, AgentManifestV2, AgentRuntime, ExecutionHints, ProtocolMode},
};
use std::{
    sync::{Mutex, MutexGuard},
    time::Duration,
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, method, path},
};

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn clear_registration_env() {
    unsafe {
        std::env::remove_var("NOVIE_PLATFORM_BASE_URL");
        std::env::remove_var("NOVIE_AGENT_PUBLIC_ENDPOINT");
        std::env::remove_var("NOVIE_AGENT_REGISTRATION_TOKEN");
        std::env::remove_var("NOVIE_AGENT_SECRET");
        std::env::remove_var("NOVIE_AGENT_REGISTRATION_REQUIRED");
        std::env::remove_var("NOVIE_RUNTIME_MODE");
        std::env::remove_var("NOVIE_ENV");
        std::env::remove_var("NOVIE_AGENT_HEARTBEAT_INTERVAL_SECONDS");
        std::env::remove_var("NOVIE_AGENT_REGISTER_MAX_ATTEMPTS");
        std::env::remove_var("NOVIE_AGENT_REGISTER_BACKOFF_SECONDS");
    }
}

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

#[test]
fn registration_lifecycle_from_env_applies_python_style_settings() {
    let _guard = env_lock();
    clear_registration_env();
    unsafe {
        std::env::set_var("NOVIE_PLATFORM_BASE_URL", " http://platform:8000/ ");
        std::env::set_var("NOVIE_AGENT_PUBLIC_ENDPOINT", " http://agent:8080/ ");
        std::env::set_var("NOVIE_AGENT_REGISTRATION_TOKEN", " token ");
        std::env::set_var("NOVIE_AGENT_REGISTRATION_REQUIRED", "true");
        std::env::set_var("NOVIE_AGENT_HEARTBEAT_INTERVAL_SECONDS", "12.5");
        std::env::set_var("NOVIE_AGENT_REGISTER_MAX_ATTEMPTS", "7");
        std::env::set_var("NOVIE_AGENT_REGISTER_BACKOFF_SECONDS", "3.5");
    }

    let lifecycle = RegistrationLifecycle::from_env(manifest()).expect("lifecycle");

    assert_eq!(
        lifecycle.manifest().endpoint.as_deref(),
        Some("http://agent:8080")
    );

    clear_registration_env();
}

#[tokio::test]
async fn registration_lifecycle_registers_heartbeats_and_deregisters() {
    {
        let _guard = env_lock();
        clear_registration_env();
    }

    let server = MockServer::start().await;
    let manifest = manifest();

    Mock::given(method("POST"))
        .and(path("/agents/register"))
        .and(body_json(serde_json::to_value(&manifest).unwrap()))
        .respond_with(ResponseTemplate::new(201))
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

    let lifecycle = RegistrationLifecycle::new(
        manifest,
        RegistrationOptions {
            platform_url: server.uri(),
            registration_token: None,
            heartbeat_interval: Duration::from_millis(5),
            required: true,
            register_max_attempts: 1,
            register_backoff: Duration::from_millis(1),
        },
    )
    .unwrap();

    lifecycle.register_on_startup().await.unwrap();
    lifecycle.client().heartbeat().await.unwrap();
    lifecycle.client().deregister().await.unwrap();
}
