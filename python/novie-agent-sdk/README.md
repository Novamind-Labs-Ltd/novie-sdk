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

## Skill runtime contracts

Skill instructions live in `SKILL.md`. If a skill also needs to drive runtime
behaviour, put structured policy in `contract.yaml` next to the skill:

```yaml
version: 1
name: report-synthesis
runtime:
  strategy: sectioned_document
  context_policy: artifact_ref_context_pack
  finalization: bounded_polish
task_profile:
  selected_by: llm_structured
  schema:
    length_profile:
      enum: [short, medium, long]
document:
  outline:
    min_sections: 2
    max_sections: 9
  section:
    min_units: 90
    default_units: 180
    max_units: 280
    max_revision_rounds: 1
  final:
    min_retention_ratio: 0.8
artifacts:
  outline_type: management_report.outline
  section_type: management_report.section
  final_type: management_report
workpad:
  record_outline_ref: true
  record_section_refs: true
  record_final_deliverable_ref: true
```

Resolve one or more skill contracts from agent code:

```python
from novie_agent_sdk import SkillContractResolver

resolver = SkillContractResolver(root_dir="/app/my-agent")
contract = resolver.resolve(
    ["/skills/shared/", "/skills/report_synthesis/"],
    required=True,
)

if contract.strategy == "sectioned_document":
    artifact_type = contract.artifacts.final_type
```

`contract.yaml` is preferred because it keeps runtime policy separate from
LLM-facing prose. As a fallback, `SkillContractResolver` also reads
`runtime_contract` or `contract` from `SKILL.md` frontmatter.

## Legacy agent bridge

New agents should use `ctx.platform` and `ctx.llm` directly. Existing external
agents that still have a pre-SDK internal runtime can use the bridge helpers:

```python
from novie_agent_sdk import (
    build_gateway_client,
    build_http_platform_services,
    env_float,
    format_stream_event,
    legacy_request_from_sdk_context,
)


@agent.invoke
async def handle_invoke(ctx):
    request = legacy_request_from_sdk_context(
        ctx,
        agent_id="pm",
        capability_timeout_seconds=env_float(
            "NOVIE_PM_PLATFORM_CAPABILITY_TIMEOUT_S",
            default=30.0,
            minimum=20.0,
        ),
    )
    services = build_http_platform_services(ctx.headers.raw, agent_id="pm")
    gateway = build_gateway_client(ctx.headers.raw, agent_id="pm")
    return await runtime.ainvoke(
        request.execution_context,
        request.inputs,
        services=services,
        gateway=gateway,
        llm_facade=request.llm_context,
    )
```

The bridge only projects protocol data: headers, context, capability id,
stream-event formatting, and platform callback services. It does not classify
business intent or infer artifact/task semantics.

For final outputs, use manifest projection when an agent needs to surface
explicitly declared artifact refs without inlining every artifact:

```python
from novie_agent_sdk import provided_artifacts_for_capability

output["provides_artifacts"] = provided_artifacts_for_capability(
    agent_manifest,
    capability_id=capability_id,
    artifact_type=artifact_type,
    structured_output=structured_output,
)
```

When multiple skills are resolved, later sources override earlier sources. This
lets a shared skill provide defaults while a business skill overrides artifact
types, task profiles, or quality bounds.

## Artifact ledger and context packs

Long-running agents should keep workpads compact: store artifact refs and short
previews, then rebuild bounded prompt context by resolving refs through platform
artifact APIs.

```python
from novie_agent_sdk import ArtifactLedger, ContextPackBuilder, ContextPackBudget

ledger = ArtifactLedger(ctx.llm.platform_ns)
result = await ledger.create_and_record(
    artifact_type="design_doc.section",
    content=section_markdown,
    kind="section_draft",
    title="Architecture",
    workflow_id=ctx.workflow_id,
    step_id="s2",
)

context_pack = await ContextPackBuilder(
    ctx.llm.platform_ns,
    budget=ContextPackBudget(max_total_chars=24000),
).build(
    workflow_id=ctx.workflow_id,
    upstream=ctx.input.get("upstream", {}),
    query="architecture tradeoffs",
    purpose="draft section",
)
```

`EvidencePackBuilder` remains available as a backwards-compatible alias for
research/analyst agents. New generic agents should prefer `ContextPackBuilder`.

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
