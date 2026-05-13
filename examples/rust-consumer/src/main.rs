//! Minimal A2A agent demonstrating consumer-side usage of `novie-agent-sdk` (Rust).
//!
//! Run: `cargo run`

use novie_agent_sdk::{
    a2a_runtime::{Agent, TaskContext},
    manifest::{
        AgentKind, AgentManifestV2, AgentRuntime, DurabilityLevel, ExecutionHints, ProtocolMode,
    },
};

fn build_manifest() -> AgentManifestV2 {
    AgentManifestV2 {
        schema: "https://novie.dev/schemas/agent-manifest-v2.json".to_owned(),
        agent_id: "novie-rust-consumer-demo".to_owned(),
        name: "Novie Rust Consumer Demo".to_owned(),
        version: "0.0.1".to_owned(),
        kind: AgentKind::ExpertBasic,
        runtime: AgentRuntime::ExternalA2A,
        protocol_mode: ProtocolMode::Tasks,
        endpoint: Some("http://localhost:8080".to_owned()),
        capabilities: vec!["demo.echo".to_owned()],
        capability_manifest: vec![],
        declared_gates: vec![],
        execution: ExecutionHints {
            supports_cancel: true,
            emits_events: true,
            durability: DurabilityLevel::TaskStore,
            expected_duration_seconds: Some(30),
            ..Default::default()
        },
        required_secrets: vec![],
        supports_streaming: false,
        sandbox_isolation: "shared".to_owned(),
        task_bundles_path: String::new(),
        metadata: serde_json::Map::new(),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    Agent::new(build_manifest())
        .task_handler(|ctx: TaskContext| async move {
            ctx.emit_message("demo: received task").await;

            let result = ctx.result(
                serde_json::json!({ "status": "ok" }),
                vec![],
                serde_json::json!({ "demo": true }),
            );
            Ok(result)
        })
        .serve("0.0.0.0:8080".parse::<std::net::SocketAddr>()?)
        .await
}
