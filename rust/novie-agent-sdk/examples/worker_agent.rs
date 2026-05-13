use novie_agent_sdk::{
    a2a_runtime::{Agent, TaskContext},
    manifest::{
        AgentKind, AgentManifestV2, AgentRuntime, DurabilityLevel, ExecutionHints, ProtocolMode,
    },
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = AgentManifestV2 {
        schema: "https://novie.dev/schemas/agent-manifest-v2.json".to_owned(),
        agent_id: "rust-worker-example".to_owned(),
        name: "Rust Worker Example".to_owned(),
        version: "0.1.0".to_owned(),
        kind: AgentKind::ExpertComplex,
        runtime: AgentRuntime::ExternalA2A,
        protocol_mode: ProtocolMode::Tasks,
        endpoint: Some("http://localhost:8080".to_owned()),
        capabilities: vec!["example.worker".to_owned()],
        capability_manifest: vec![],
        declared_gates: vec![],
        execution: ExecutionHints {
            supports_cancel: true,
            emits_events: true,
            durability: DurabilityLevel::TaskStore,
            expected_duration_seconds: Some(60),
            ..Default::default()
        },
        required_secrets: vec![],
        supports_streaming: false,
        sandbox_isolation: "shared".to_owned(),
        task_bundles_path: String::new(),
        metadata: serde_json::Map::new(),
    };

    Agent::new(manifest)
        .task_handler(|ctx: TaskContext| async move {
            ctx.emit_message("worker started").await;
            ctx.report_llm_usage("example", "none", Some(1), Some(1), Some(2))
                .await;
            Ok(ctx.result(
                serde_json::json!({ "ok": true }),
                vec![],
                serde_json::json!({ "example": true }),
            ))
        })
        .serve("0.0.0.0:8080".parse::<std::net::SocketAddr>()?)
        .await
}
