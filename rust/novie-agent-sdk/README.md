# novie-agent-sdk (Rust)

Rust A2A Agent Runtime SDK for the Novie Platform.

This crate is the Rust counterpart of the Python `novie_agent_sdk` package.
The primary path is the hosted A2A runtime: an agent loads an
`AgentManifestV2`, registers business handlers, and lets the SDK own the HTTP
protocol machinery.

## What's inside

| Capability | Type |
| --- | --- |
| A2A HTTP runtime (`/healthz`, manifest, `/invoke`, `/tasks/*`) | [`Agent`] |
| Manifest v2 loading and validation | [`AgentManifestV2`] |
| Typed signed A2A request headers | [`RequestHeaders`] |
| In-memory and SQLite task lifecycle/event stores | [`TaskStore`] |
| Legacy callback payload parsing | [`AgentInvokePayload`] |
| Legacy platform callback services | [`PlatformServicesClient`] |

## Python Parity Matrix

| Python SDK capability | Rust status | Notes |
| --- | --- | --- |
| Manifest V2 wire shape | must / done | Rust parses current Cortex manifests and legacy metadata-promoted fields. |
| Signed A2A headers | must / done | Canonical HMAC-SHA256 input matches Python. |
| `Agent.invoke` | must / done | Rust exposes `invoke_handler`. |
| `Agent.stream` | must / done | Rust exposes `stream_handler` and NDJSON replay. |
| `Agent.task` | must / done | Rust exposes `task_handler` and `/tasks/*`. |
| One-shot idempotency cache | must / done | In-memory and SQLite-backed stores. |
| Durable task store | must / done | SQLite-backed `SqliteTaskStore`. |
| Worker helpers | must / done | Message, artifact, usage, human-wait, status, and result helpers. |
| Artifact helpers | must / done | One-shot artifact and needs-confirmation envelopes. |
| Registration lifecycle | must / partial | Register, heartbeat, and deregister exist; richer operator telemetry can keep evolving. |
| External conformance CLI | must / partial | Examples compile and cover oneshot, stream, worker shapes; CLI remains Python-owned. |
| Legacy callback client | defer | Kept exported for Cortex/tests until migration removes callback dependency. |

## Quickstart

```rust
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
        agent_id: "my-rust-agent".to_owned(),
        name: "My Rust Agent".to_owned(),
        version: "0.1.0".to_owned(),
        kind: AgentKind::ExpertComplex,
        runtime: AgentRuntime::ExternalA2A,
        protocol_mode: ProtocolMode::Tasks,
        endpoint: Some("http://my-rust-agent:8080".to_owned()),
        capabilities: vec!["code_review".to_owned()],
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
            ctx.emit_message("Starting").await;
            Ok(serde_json::json!({ "result": "done" }))
        })
        .serve("0.0.0.0:8080".parse::<std::net::SocketAddr>()?)
        .await
}
```

## Compatibility notes

- **Manifest wire format** follows Python `AgentManifestV2`, including
  `execution.durability`, `capability_manifest`, `metadata`, and compatibility
  parsing for legacy metadata-promoted fields.
- **Signed A2A headers** use the same canonical HMAC-SHA256 input as Python:
  tenant, workspace, project, user, service principal, session, step,
  idempotency key, and timestamp.
- **Legacy callback modules** remain exported for existing tests and Cortex
  compatibility while the SDK migrates fully to hosted A2A runtime.
- **Token wire format** is locked to byte-equality with Python's
  `tokens.py::mint_callback_token`: HS256 JWT, fixed canonical header,
  `iss=novie-platform`, `aud=novie-agent-callback`, base64url without
  padding. See [`token`] module docs.
- **Retry policy** matches Python's `HttpCallbackPlatformServices`: only
  `503` and transport errors are retried; honours `retry_after_ms` from the
  server, otherwise exponential `200ms * 2^n`; max 2 retries by default
  (1 retry for the agent-status push channel since it's already idempotent).
- **`subscribe`** semantics are intentionally **not** exposed -
  `events.subscribe` and `sessions.subscribe` would require SSE/Redis,
  which lives in the platform-side `novie-cortex` crate. Use the SSE
  endpoint (`GET /sessions/{id}/stream`) or poll `list_events(since=...)`.

## Testing

```bash
cd sdk/rust/novie-agent-sdk
cargo test
```

Integration tests use [`wiremock`](https://docs.rs/wiremock) to stand up a
fake platform server, so they exercise the full HTTP path including retry
behaviour without needing a real platform deployment.

[`Agent`]: src/a2a_runtime.rs
[`AgentManifestV2`]: src/manifest.rs
[`RequestHeaders`]: src/headers.rs
[`TaskStore`]: src/a2a_runtime.rs
[`AgentInvokePayload`]: src/payload.rs
[`PlatformServicesClient`]: src/platform_services.rs
[`token`]: src/token.rs
