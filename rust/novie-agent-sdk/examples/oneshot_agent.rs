use novie_agent_sdk::{
    a2a_runtime::{Agent, InvokeContext},
    manifest::{AgentKind, AgentManifestV2, AgentRuntime, ExecutionHints, ProtocolMode},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = AgentManifestV2 {
        schema: "https://novie.dev/schemas/agent-manifest-v2.json".to_owned(),
        agent_id: "rust-oneshot-example".to_owned(),
        name: "Rust One-Shot Example".to_owned(),
        version: "0.1.0".to_owned(),
        kind: AgentKind::ExpertBasic,
        runtime: AgentRuntime::ExternalA2A,
        protocol_mode: ProtocolMode::Simple,
        endpoint: Some("http://localhost:8080".to_owned()),
        capabilities: vec!["example.oneshot".to_owned()],
        capability_manifest: vec![],
        declared_gates: vec![],
        execution: ExecutionHints::default(),
        required_secrets: vec![],
        supports_streaming: false,
        sandbox_isolation: "shared".to_owned(),
        task_bundles_path: String::new(),
        metadata: serde_json::Map::new(),
    };

    Agent::new(manifest)
        .invoke_handler(|ctx: InvokeContext| async move {
            Ok(ctx.artifact(
                "example_summary",
                "Rust one-shot example complete",
                serde_json::json!({ "input": ctx.input }),
            ))
        })
        .serve("0.0.0.0:8080".parse::<std::net::SocketAddr>()?)
        .await
}
