# Consumer Demos

End-to-end examples of how a downstream project consumes the Novie Agent SDK as a **private Git dependency**.

| Demo | Language | SDK |
| --- | --- | --- |
| [`python-consumer`](./python-consumer) | Python ≥ 3.14 | `novie-agent-sdk` (Python) |
| [`rust-consumer`](./rust-consumer) | Rust (edition 2024) | `novie-agent-sdk` (Rust) |

Both demos do the same thing: register a minimal A2A agent with the Novie runtime and respond to one task. The interesting part is the **dependency declaration** in `pyproject.toml` / `Cargo.toml` and the **auth setup** for pulling from a private GitHub repo.

> ⚠️ **Python consumers need access to two private repos**, not one.
> The Python SDK transitively depends on [`Novamind-Labs-Ltd/novie-protocol`](https://github.com/Novamind-Labs-Ltd/novie-protocol). When `pip install` clones `novie-sdk`, it then clones `novie-protocol` to resolve the dependency. Whatever auth method you pick (SSH key / deploy key / PAT) must grant read access to **both** repositories. The Rust SDK has its protocol types in-tree and is not affected.

## Tag convention used in this repo

This is a monorepo with two SDKs, so releases are tagged with a language prefix:

| SDK | Tag pattern | Example |
| --- | --- | --- |
| Python | `python-v<semver>` | `python-v0.3.0` |
| Rust | `rust-v<semver>` | `rust-v0.2.0` |

Consumers pin to a specific tag. The release workflow at [`.github/workflows/release.yml`](../.github/workflows/release.yml) creates these tags automatically.

## Auth setup (shared between demos)

Both `pip` and `cargo` will follow whatever auth GitHub already trusts on the machine. Pick one of the two patterns and use it consistently across all consumers.

### Local dev — SSH (recommended)

1. Make sure GitHub trusts your SSH key:
   ```bash
   ssh -T git@github.com
   ```
2. Use `ssh://git@github.com/...` URLs in `pyproject.toml` / `Cargo.toml`.

### CI — GitHub Actions

**Option A — Deploy key (preferred)**

1. Generate a key pair locally.
2. On `Novamind-Labs-Ltd/novie-sdk` → Settings → Deploy keys → add the **public** key, read-only.
3. **Repeat steps 1–2 for `Novamind-Labs-Ltd/novie-protocol`** — same key pair can be reused; same key needs to be added to **both** repos' deploy-key lists.
4. On the consumer repo → Settings → Secrets → add the **private** key as `NOVIE_SDK_DEPLOY_KEY`.
5. In the consumer's workflow:
   ```yaml
   - uses: webfactory/ssh-agent@v0.9.0
     with:
       ssh-private-key: ${{ secrets.NOVIE_SDK_DEPLOY_KEY }}
   ```

**Option B — Personal Access Token (HTTPS)**

```yaml
- run: |
    git config --global url."https://x-access-token:${{ secrets.NOVIE_SDK_TOKEN }}@github.com/".insteadOf "https://github.com/"
```
Then declare the dependency with `https://github.com/...` instead of `ssh://git@github.com/...`. The PAT must have `repo` scope on **both** `novie-sdk` and `novie-protocol`.

## Local-development override

If you're hacking on both the SDK and a consumer at the same time, the Git URL is slow to iterate on. Each demo's README documents how to swap to a **path dependency** temporarily — that change should never be committed.
