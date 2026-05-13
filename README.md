# novie-sdk

Official SDKs for building **A2A (Agent-to-Agent) agents** on the Novie Platform.

This repository hosts language-specific implementations of the Novie Agent SDK. Each SDK ships the same A2A runtime contract — manifest loading, signed request headers, task lifecycle, and HTTP surface — so agents written in any supported language can register with the Novie Platform and exchange tasks, events, and artifacts on equal footing.

## Repository layout

| Path | Language | Package | Status |
| --- | --- | --- | --- |
| [`python/novie-agent-sdk`](./python/novie-agent-sdk) | Python ≥ 3.14 | `novie-agent-sdk` (v0.3.0) | Reference implementation |
| [`rust/novie-agent-sdk`](./rust/novie-agent-sdk) | Rust (edition 2024, MSRV 1.85) | `novie-agent-sdk` (v0.2.0) | A2A runtime parity |
| [`examples/`](./examples) | — | — | Reference **consumer** projects for both SDKs (private Git-dependency setup) |

Both SDKs target the same wire protocol described in `docs/openapi/platform_callback.v1.yaml` (hosted alongside the platform). Pick the language that fits your agent's runtime and follow the SDK-local README for installation and usage.

## What an SDK gives you

- **A2A HTTP runtime** — health, manifest, `/invoke`, `/invoke/stream`, and `/tasks/*` endpoints with idempotency and event replay built in.
- **Manifest v2 loading** — parse and validate `AgentManifestV2` (capabilities, execution hints, durability, declared gates).
- **Signed request headers** — canonical HMAC-SHA256 input shared across languages (tenant, workspace, project, user, service principal, session, step, idempotency key, timestamp).
- **Task lifecycle and event stores** — in-memory and SQLite-backed stores for durable runs and resumable streams.
- **Worker helpers** — emit messages, artifacts, usage events, status updates, and human-wait envelopes from inside your task handler.
- **Project brief / context helpers** — pull the platform-injected project brief out of `TaskContext` and render it into your system prompt.

## Quickstart — Python

```python
import asyncio
from novie_agent_sdk import Agent, TaskContext

agent = Agent.from_manifest(".well-known/agent.json")

@agent.task
async def handle(ctx: TaskContext) -> dict:
    await ctx.emit_message("Processing…")
    return {"status": "ok"}

if __name__ == "__main__":
    asyncio.run(agent.serve())
```

See [`python/novie-agent-sdk/README.md`](./python/novie-agent-sdk/README.md) for the full guide, including observability hooks, contract test helpers, and project brief injection.

## Quickstart — Rust

```rust
use novie_agent_sdk::{
    a2a_runtime::{Agent, TaskContext},
    manifest::AgentManifestV2,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = AgentManifestV2 { /* …fields… */ ..Default::default() };

    Agent::new(manifest)
        .task_handler(|ctx: TaskContext| async move {
            ctx.emit_message("Starting").await;
            Ok(serde_json::json!({ "result": "done" }))
        })
        .serve("0.0.0.0:8080".parse()?)
        .await
}
```

See [`rust/novie-agent-sdk/README.md`](./rust/novie-agent-sdk/README.md) for the full quickstart, the Python parity matrix, and compatibility notes on token wire format, retry policy, and unsupported `subscribe` semantics.

## Cross-language parity

The two SDKs are intentionally kept on the same wire contract so a task started against a Python agent and a Rust agent looks identical to the platform:

- **Manifest wire format** — Rust parses the same `AgentManifestV2` JSON Python produces, including legacy metadata-promoted fields.
- **Signed A2A headers** — identical canonical HMAC-SHA256 input across both SDKs.
- **Token format** — locked to byte-equality with Python's `tokens.py::mint_callback_token` (HS256 JWT, fixed canonical header, `iss=novie-platform`, `aud=novie-agent-callback`, base64url unpadded).
- **Retry policy** — both retry only `503`s and transport errors, honour `retry_after_ms`, and default to two retries (one for the idempotent agent-status push channel).
- **Removed `/memory/recall` and `/memory/remember`** — curated knowledge goes through `WikiService.search` / `/wiki/search` in both SDKs.

See the **Python Parity Matrix** in the Rust README for a capability-by-capability status.

## Consuming the SDK (private Git dependency)

The SDKs are **not published** to PyPI or crates.io. Consumers pull them as Git dependencies pinned to language-prefixed tags (`python-v<x>`, `rust-v<x>`). Full setup, auth, and CI snippets live in [`examples/`](./examples) — start there.

## Cutting a release

Release tags are produced by [`.github/workflows/release.yml`](./.github/workflows/release.yml). Trigger it from the **Actions** tab → "Release SDK" → "Run workflow", choose `python` or `rust`, and enter the new semver. The workflow:

1. Validates the version string is semver and isn't already tagged.
2. Confirms the version matches `pyproject.toml` / `Cargo.toml`.
3. Runs the SDK's tests.
4. Builds wheel/sdist (Python) or runs `cargo package --no-verify` (Rust) as a sanity check.
5. Creates and pushes the annotated tag (`python-v<x>` / `rust-v<x>`).
6. Drafts a GitHub Release with the consumer dependency snippet pre-filled.

Bumping a version is therefore a two-step PR: edit the version in `pyproject.toml` / `Cargo.toml`, merge, then run the workflow.

## Implementing the SDK in another language

If you need to ship an agent in a language that isn't yet covered:

1. Implement the A2A HTTP surface (`/healthz`, manifest GET, `/invoke`, `/invoke/stream`, `/tasks/{id}`, `/tasks/{id}/events`).
2. Mirror the callback envelope and signed-header canonicalization from `docs/openapi/platform_callback.v1.yaml`.
3. Match the token wire format exactly — the platform validates bytes, not semantics.
4. Use the Python SDK as the reference implementation; the Rust SDK is a good second reference for non-GC languages.

## License

- **Python SDK**: Proprietary (Novie).
- **Rust SDK**: Apache-2.0.

Refer to each SDK's package metadata (`pyproject.toml` / `Cargo.toml`) for the authoritative license.

## Links

- Remote: <https://github.com/Novamind-Labs-Ltd/novie-sdk>
- Python SDK: [`python/novie-agent-sdk`](./python/novie-agent-sdk)
- Rust SDK: [`rust/novie-agent-sdk`](./rust/novie-agent-sdk)
