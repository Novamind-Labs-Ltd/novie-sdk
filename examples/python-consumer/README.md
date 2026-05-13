# novie-python-consumer-demo

Reference consumer of the **Python** Novie Agent SDK, pulled in as a **private Git dependency**.

The interesting part is the dependency declaration in [`pyproject.toml`](./pyproject.toml):

```toml
dependencies = [
    "novie-agent-sdk[server] @ git+ssh://git@github.com/Novamind-Labs-Ltd/novie-sdk.git@python-v0.3.0#subdirectory=python/novie-agent-sdk",
]
```

Three things make this work:

| Piece | Why |
| --- | --- |
| `git+ssh://git@github.com/…` | Uses your local SSH key. For HTTPS + token, swap the URL — see [`../README.md`](../README.md). |
| `@python-v0.3.0` | Pins the **language-prefixed** tag. Bumping = change this string. |
| `#subdirectory=python/novie-agent-sdk` | The SDK lives in a subdirectory; without this, pip looks in the repo root and fails. |
| `[server]` | Pulls in FastAPI (needed for `agent.serve()`). Drop if you only need the contract types. |

> ⚠️ The SDK transitively depends on the private [`novie-protocol`](https://github.com/Novamind-Labs-Ltd/novie-protocol) repo. Your auth (SSH key / deploy key / PAT) must grant read access to both `novie-sdk` and `novie-protocol`.

## Run the demo locally

```bash
cd examples/python-consumer
python -m venv .venv && source .venv/bin/activate
pip install -e .
novie-demo                       # uvicorn binds to 0.0.0.0:8000
```

Smoke check:

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/.well-known/agent.json
```

## Local-development override (don't commit)

When you're iterating on the SDK and the demo at the same time, the Git URL is too slow. Install the SDK editable from the local checkout instead:

```bash
pip install -e ../../python/novie-agent-sdk
pip install -e . --no-deps    # keep the demo install, skip re-resolving the SDK
```

Or temporarily edit `pyproject.toml`:

```toml
# DEV ONLY — do not commit
dependencies = [
    "novie-agent-sdk[server] @ file://${PROJECT_ROOT}/../../python/novie-agent-sdk",
]
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
          # Same private key must be registered as a deploy key on
          # BOTH novie-sdk and novie-protocol.
          ssh-private-key: ${{ secrets.NOVIE_SDK_DEPLOY_KEY }}
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
      - run: pip install -e .
      - run: python -c "from novie_python_consumer_demo import build_agent; build_agent()"
```

## What the demo actually does

[`src/novie_python_consumer_demo/main.py`](./src/novie_python_consumer_demo/main.py) registers one `@agent.task` handler that:

1. Pulls the platform-injected project brief out of `ctx.input` via `extract_project_brief` / `render_brief_for_prompt`.
2. Emits an event with `ctx.emit_message`.
3. Returns a JSON result.

That's the entire surface area you need to learn to ship a real agent.
