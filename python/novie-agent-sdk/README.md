# novie-agent-sdk

Python SDK for Novie **A2A agents**: task lifecycle, HTTP surface (FastAPI),
optional observability hooks, and prompt helpers that mirror `novie-protocol`.

Wire-level callback RPC paths live in `docs/openapi/platform_callback.v1.yaml`.
**Notebook-style `/memory/recall` and `/memory/remember` callbacks were removed**;
curated knowledge goes through **`WikiService.search`** / `/wiki/search`.

> **Renamed from `novie-agent-runtime` in v0.2.0.** The legacy `novie_agent_runtime`
> import shim was removed in v0.3.0 — use `novie_agent_sdk` directly.

## Install

```toml
# agent's pyproject.toml
dependencies = [
    "novie-protocol",
    "novie-agent-sdk",
    # "novie-agent-sdk[observability]",  # LangChain usage reporting helpers
]
```

## A2A runtime (primary path)

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

`Agent.serve()` hosts health, manifest, invoke/stream/tasks endpoints when FastAPI
is installed.

## Project brief injection

```python
from novie_agent_sdk import TaskContext, extract_project_brief, render_brief_for_prompt


@agent.task
async def handle(ctx: TaskContext) -> dict:
    brief = extract_project_brief(ctx.input)
    extra = render_brief_for_prompt(brief) if brief is not None else ""
    # Append ``extra`` to your system prompt when non-empty.
    return {"ok": True}
```

When `minimal=True`, `render_brief_for_prompt` instructs the model to fall back to
`services.wiki.search` and platform-injected context instead of a removed memory
recall path.

## Observability and usage

For LangChain/LangGraph agents, attach observability through `AgentObservability`
(see `novie_agent_sdk.observability`). Novie platform usage events remain the
authoritative ledger; Langfuse is optional behind env flags.

Non-LangChain agents can call `TaskContext.report_llm_usage(...)` to emit
usage-shaped task events.

## Contract tests

`novie_agent_sdk.testing` exposes small helpers (`assert_http_json_*`) for smoke
testing an agent HTTP surface in CI.

## Non-Python agents

Implement the same callback envelope described in
`docs/openapi/platform_callback.v1.yaml`. A Rust mirror lives under
`sdk/rust/novie-agent-sdk` (currently focuses on the A2A runtime slice exported
from its `lib.rs`).
