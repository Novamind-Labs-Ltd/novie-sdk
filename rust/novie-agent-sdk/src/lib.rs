//! # novie-agent-sdk
//!
//! Rust A2A Agent Runtime SDK for the Novie Platform.
//!
//! ## A2A Runtime (v2 — primary path)
//!
//! Agent authors implement a single async handler and the SDK hosts the full
//! A2A HTTP endpoint surface via Axum.
//!
//! ```no_run
//! use novie_agent_sdk::{
//!     a2a_runtime::{Agent, TaskContext},
//!     manifest::{AgentManifestV2, AgentKind, AgentRuntime, ProtocolMode, ExecutionHints, DurabilityLevel},
//! };
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let manifest = AgentManifestV2 {
//!     schema: "https://novie.dev/schemas/agent-manifest-v2.json".to_owned(),
//!     agent_id: "my-rust-agent".to_owned(),
//!     name: "My Rust Agent".to_owned(),
//!     version: "0.1.0".to_owned(),
//!     kind: AgentKind::ExpertComplex,
//!     runtime: AgentRuntime::ExternalA2A,
//!     protocol_mode: ProtocolMode::Tasks,
//!     endpoint: Some("http://my-rust-agent:8080".to_owned()),
//!     capabilities: vec!["code_review".to_owned()],
//!     execution: ExecutionHints {
//!         supports_cancel: true,
//!         emits_events: true,
//!         durability: DurabilityLevel::TaskStore,
//!         expected_duration_seconds: Some(60),
//!         ..Default::default()
//!     },
//!     required_secrets: vec![],
//!     declared_gates: vec![],
//!     capability_manifest: vec![],
//!     supports_streaming: false,
//!     sandbox_isolation: "shared".to_owned(),
//!     task_bundles_path: String::new(),
//!     metadata: serde_json::Map::new(),
//! };
//!
//! Agent::new(manifest)
//!     .task_handler(|ctx: TaskContext| async move {
//!         ctx.emit_message("Starting").await;
//!         Ok(serde_json::json!({ "result": "done" }))
//!     })
//!     .serve("0.0.0.0:8080".parse::<std::net::SocketAddr>()?)
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Protocol mirror
//!
//! The Rust SDK does **not** depend on the `novie-protocol` Python package via
//! Cargo. Instead, the contract types in this crate (manifest, call_scope,
//! session, memory, agent_status, …) are hand-mirrored from the canonical
//! Python definitions at `novie-protocol` tag [`MIRRORED_PROTOCOL_VERSION`].
//!
//! When bumping `novie-protocol`, also bump [`MIRRORED_PROTOCOL_VERSION`] and
//! re-sync any changed contracts in the same PR. CI greps for this constant.
#![warn(missing_debug_implementations)]

/// The `novie-protocol` tag whose Python contracts this crate hand-mirrors.
///
/// See module-level docs for the mirror policy. Bump this in lockstep with
/// `novie-protocol` releases.
pub const MIRRORED_PROTOCOL_VERSION: &str = "0.1.0";

// ── A2A Runtime (v2 primary path) ────────────────────────────────────────────
#[cfg(feature = "http")]
pub mod a2a_runtime;
pub mod headers;
pub mod manifest;

// ── Legacy callback compatibility modules ────────────────────────────────────
//
// These remain exported for existing Cortex/tests while the SDK migrates fully
// to the hosted A2A runtime path.
pub mod agent_status;
pub mod agent_status_client;
pub mod call_scope;
pub mod error;
pub mod long_task;
pub mod memory;
pub mod payload;
pub mod platform_llm;
pub mod platform_services;
pub mod session;
pub mod token;
pub mod transport;

pub use agent_status_client::{AgentStatusClient, ReportOptions};
pub use call_scope::{AgentCallScope, extract_call_scope};
pub use error::{Error, Result};
pub use long_task::{LongTaskCompletion, LongTaskStatus, notify_long_task_complete};
pub use payload::{AgentInvokePayload, AgentStatusCallbackConfig, PlatformCallbackConfig};
pub use platform_llm::{
    ChatMessage, ChatOptions, EmbedOptions, PlatformLlmClient, PlatformLlmIdentity,
    StructuredOptions,
};
pub use platform_services::PlatformServicesClient;
pub use token::{
    CallbackTokenClaims, MintContext, mint_callback_token, mint_with_clock, verify_callback_token,
    verify_with_clock,
};
