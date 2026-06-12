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

## SDK responsibility

The SDK is the platform access layer for external agents. It helps any agent type
expose an A2A-compatible runtime, report lifecycle/progress/workpad/artifact
state, and call platform-owned capabilities such as LLM, search, artifacts,
checkpoints, knowledge, and future tool namespaces.

Agent packages own their domain workflow, prompts, skills, artifact taxonomy, and
business-specific fallback policy. SDK modules should stay generic unless the
specific behavior is supplied by an agent or skill runtime contract.

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
name: document-authoring
runtime:
  strategy: sectioned_longform
  context_policy: evidence_pack_v1
  finalization: section_ledger_polish
task_profile:
  selected_by: llm
  defaults:
    length_profile: adaptive
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
  outline_type: document.outline
  section_type: document.section
  final_type: document
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
    ["/skills/shared/", "/skills/document_authoring/"],
    required=True,
)

if contract.strategy == "sectioned_longform":
    artifact_type = contract.artifacts.final_type
```

`contract.yaml` is preferred because it keeps runtime policy separate from
LLM-facing prose. As a fallback, `SkillContractResolver` also reads
`runtime_contract` or `contract` from `SKILL.md` frontmatter.

## Sectioned longform authoring

Document-style agents can use `SectionedLongformAuthor` to write long outputs
through the same durable flow:

1. Ask the LLM for an outline from the original task and upstream/workpad refs.
2. Rebuild a bounded evidence pack for each section.
3. Draft, quality-check, and record each section as an artifact/workpad ref.
4. Polish the joined sections and record the final deliverable ref.

```python
from novie_agent_sdk import (
    SectionedLongformAuthor,
    sectioned_authoring_contract_from_skill,
)

author = SectionedLongformAuthor(
    llm_facade=llm_facade,
    platform=platform_ns,
    artifact_type="document",
    step_id=step_id,
    capability_id=capability_id,
    context_budget=context_budget,
    authoring_contract=sectioned_authoring_contract_from_skill(
        contract,
        artifact_type="document",
    ),
)
result = await author.author(
    brief=brief,
    upstream=upstream,
    workflow_id=ctx.workflow_id,
    thread_id=ctx.thread_id,
    agent_id="writer",
)
```

The feature is on by default. Set `NOVIE_SECTIONED_AUTHORING_DISABLED=1`, or an
agent-specific disabled variable passed to `sectioned_authoring_enabled()`, to
fail closed before authoring starts.

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

Use `execution_context_from_sdk_request()` for SDK request objects. If a
legacy runtime already has a raw context block, use
`execution_context_from_runtime_block(ctx_data, agent_id="...")`. The older
top-level `execution_context_from_block(ctx_data)` name remains the
document-agent helper and does not require `agent_id`.

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
    strict=True,
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

Use `strict=True` when the artifact and workpad entry are part of the same
durable authoring ledger. In strict mode, `create_and_record` raises if artifact
creation or workpad ref recording fails, instead of returning a partial result.

`EvidencePackBuilder` remains available as a backwards-compatible alias for older
agent packages. New agents should prefer `ContextPackBuilder`.

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
