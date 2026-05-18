# novie-rust-consumer-demo

Reference consumer of the **Rust** Novie Agent SDK, pulled in as a **private Git dependency**.

The interesting part is the dependency declaration in [`Cargo.toml`](./Cargo.toml):

```toml
[dependencies]
novie-agent-sdk = { git = "ssh://git@github.com/Novamind-Labs-Ltd/novie-sdk.git", tag = "rust-v0.3.2" }
```

Three things worth noting:

| Piece | Why |
| --- | --- |
| `git = "ssh://git@github.com/…"` | Uses your local SSH key. For HTTPS + token, see [`../README.md`](../README.md). |
| `tag = "rust-v0.3.2"` | Pins the **language-prefixed** tag. Use `rev = "<sha>"` to lock to a commit. |
| _No subdirectory field_ | cargo scans the whole git repo and matches by package name, so `rust/novie-agent-sdk/Cargo.toml` is found automatically. This differs from pip. |

`Cargo.lock` records the resolved commit SHA, so builds remain reproducible even if the tag is later moved (don't do that).

## Run the demo locally

```bash
cd examples/rust-consumer
cargo run
```

Smoke check:

```bash
curl http://localhost:8080/healthz
```

## Local-development override (don't commit)

When you're iterating on the SDK and demo at the same time, swap the dependency to a path:

```toml
# Cargo.toml — DEV ONLY, do not commit
novie-agent-sdk = { path = "../../rust/novie-agent-sdk" }
```

Or, in a workspace, pin globally without editing the demo:

```toml
# Top-level Cargo.toml
[patch."ssh://git@github.com/Novamind-Labs-Ltd/novie-sdk.git"]
novie-agent-sdk = { path = "../../rust/novie-agent-sdk" }
```

## Cargo cache gotcha

`cargo` caches git sources under `~/.cargo/git/db/`. If you bump the tag but `cargo update -p novie-agent-sdk` returns stale data, force a refresh:

```bash
cargo update -p novie-agent-sdk --precise <new-commit-sha>
# nuclear option:
rm -rf ~/.cargo/git/db/novie-sdk-*
```

## CI snippet (GitHub Actions)

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: webfactory/ssh-agent@v0.9.0
        with:
          ssh-private-key: ${{ secrets.NOVIE_SDK_DEPLOY_KEY }}
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo build --release
```

## What the demo actually does

[`src/main.rs`](./src/main.rs) builds an `AgentManifestV2` in code, registers one `task_handler`, and serves the A2A HTTP surface (`/healthz`, manifest, `/invoke`, `/tasks/*`) on `0.0.0.0:8080`.

If you'd rather load the manifest from disk, mirror the Python demo: parse JSON into `AgentManifestV2` via `serde_json::from_str` and call `Agent::new(...)`.
