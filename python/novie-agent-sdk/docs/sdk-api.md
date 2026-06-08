# Novie Agent SDK API

This document lists the current author-facing SDK APIs used by external agents.
The goal is to let new agents depend on platform-owned runtime capabilities
instead of re-implementing transport, artifact, workpad, LLM, and output
contracts.

## API Categories For Agent Authors

Use the SDK in these layers. New agents should start from the first two
categories and only drop lower when they need tighter control.

| Category | Primary APIs | Use When |
|---|---|---|
| Agent authoring facades | `Agent`, `artifact_agent`, `worker_agent` | You need to expose an A2A-compatible external agent. |
| Platform context | `extract_call_scope`, `extract_project_brief`, `execution_workpad_context`, `upstream_context` | You need tenant/session/project context or platform-provided work context. |
| Platform capabilities | `build_platform_namespace`, `PlatformNamespace`, `ArtifactsNamespace`, `KnowledgeNamespace`, `WebNamespace`, `LlmNamespace`, `CheckpointsNamespace` | You need to call platform-owned tools instead of direct vendor/service APIs. |
| Document A2A adapters | `load_runtime_manifest`, `context_block_from_request`, `execution_context_from_request`, `stream_event_to_wire` | You are building a document-style agent with custom invoke/stream runtime glue. |
| Document agent templates | `DocumentAgentTemplate`, `DocumentCapabilitySpec`, `resolve_document_agent_input`, `document_final_output`, `get_matching_document_checkpoint`, `skipped_quality_result` | You are building a custom document-agent runtime and want shared budget, stream, progress, final output, resume, quality, and capability semantics. |
| Skill-driven document agents | `compile_skill_scope`, `SkillScope`, `SkillMetadata`, `build_deep_agent_executor` | You need bounded `SKILL.md` navigation and DeepAgents assembly for a capability-selected skill set. |
| Platform services bridge | `build_http_platform_services`, `build_gateway_client`, `CapabilityClient`, `HttpWikiService`, `HttpExternalAgentCheckpointService` | You need `novie_protocol.services.PlatformServices` backed by platform capability callbacks. |
| LLM integration | `PlatformChatModel`, `build_llm_facade`, `langchain_runnable_config`, LLM normalization helpers | You need LangChain or direct platform LLM calls with usage/callback propagation. |
| Context budget | `default_context_budget`, `context_budget_from_inputs`, `budget_status`, `adaptive_phase_timeout_seconds`, `budget_summary` | You need shared token/wall-clock budget behavior for long document runs. |
| Artifacts and deliverables | `ArtifactReader`, `format_artifact_read_result`, `markdown_deliverable_output`, `bounded_handoff_output` | You need agent-friendly artifact reads or standardized final/internal outputs. |
| Step role and output contracts | `step_run_policy`, `resolve_step_role`, `handoff_max_bytes` | You need to branch behavior for upstream handoffs vs final deliverables. |
| Streaming and workpad events | `progress_event`, `content_delta_event`, `tool_call_event`, `tool_result_event`, `workpad_entry_event`, `execution_workpad_entries`, `with_keepalive`, `SubtaskEventMapper` | You need platform-readable progress, tool, content, workpad traces, keepalives, subtask summaries, or workpad recovery reads. |
| Checkpoints and degradation | `checkpoint_step_id`, `checkpoint_input_digest`, `checkpoint_matches_invocation`, `external_agent_checkpoint_service`, `put_external_agent_checkpoint`, `get_matching_document_checkpoint`, `skipped_phase_events`, `DegradationTracker` | You need resumable external-agent checkpoints, cross-version checkpoint writes/reads, skipped-phase traces, or symbolic degradation flags. |
| Provider tooling | provider authoring/conformance/OpenAPI helpers | You are adding platform capability providers, not building a normal agent. |
| Testing/conformance | HTTP assertion helpers, `run_conformance`, provider conformance probes | You need to validate an agent/provider implementation. |

### Duplication Review

The SDK intentionally has two levels for several concepts:

- Facades (`artifact_agent`, `worker_agent`) are the high-level authoring path.
- Primitive helpers (`progress_event`, `workpad_entry_event`, `bounded_handoff_output`) are for custom agents that implement their own runtime loop.

Current API overlap to watch:

- `markdown_deliverable_output` and `bounded_handoff_output` are related but not duplicates: one is user-visible final output, the other is internal DAG handoff output.
- `PlatformChatModel`, `build_llm_facade`, and `LlmNamespace` are three access levels over platform LLM. Prefer `PlatformChatModel` for LangChain, `build_llm_facade` for agent runtime abstraction, and `LlmNamespace` for direct platform calls.
- `ArtifactReader` and `ArtifactsNamespace` are not duplicates: `ArtifactReader` is the agent-friendly text/cache layer over lower-level artifact capability calls.
- `workpad_entry_event` and `workpad_checkpoint_event` are aliases by design; `workpad_checkpoint_event` exists for long-running agent readability.

## Runtime

### `Agent`

Import:

```python
from novie_agent_sdk import Agent
```

Use:

```python
agent = Agent.from_manifest(".well-known/agent.json")

@agent.task
async def handle(ctx):
    return {"status": "ok"}

agent.serve()
```

Purpose:

- Loads the agent manifest.
- Provides A2A invoke, stream, and task runtime wiring.
- Owns HTTP route shape, idempotency, and protocol-compatible responses.

## Artifact Agent Facade

### `artifact_agent`

Import:

```python
from novie_agent_sdk import artifact_agent
```

Use:

```python
app = artifact_agent(manifest=".well-known/agent.json")

@app.handle
async def handle(ctx):
    await ctx.progress("Reading context")
    return ctx.artifact(
        artifact_type="market_report",
        summary="Market report complete",
        content="# Report",
        metadata={"confidence": "medium"},
    )
```

Purpose:

- Gives document/artifact agents a smaller authoring surface.
- Projects handler returns into platform-compatible invoke and stream outputs.
- Provides `ctx.progress(...)`, `ctx.artifact(...)`, and `ctx.needs_confirmation(...)`.

## Worker Agent Facade

### `worker_agent`

Import:

```python
from novie_agent_sdk import worker_agent
```

Purpose:

- Authoring facade for durable worker-style agents.
- Supports task execution, failure envelopes, and human wait requests.
- Intended for agents like task splitters that produce PMS work rather than a final document.

## Document Agent Template

### `DocumentCapabilitySpec`

Import:

```python
from novie_agent_sdk import DocumentCapabilitySpec
```

Purpose:

- Shared runtime-facing capability contract for document agents.
- Carries `capability_id`, `skill_sources`, `mode`, `phase`, `artifact_type`,
  `artifact_family`, `consumes`, `consumes_strict`, `optional_consumes`,
  `provides`, `artifact_access`, `synthesis_path`, and side-effect metadata.
- Agent packages still own their concrete registry values.

### `resolve_document_agent_input`

Import:

```python
from novie_agent_sdk import resolve_document_agent_input
```

Purpose:

- Converts a capability's `artifact_access` into prompt-facing input.
- Hides upstream summaries when `artifact_access == "none"`.
- Leaves dependency enforcement to the platform.

### `DocumentAgentTemplate`

Import:

```python
from novie_agent_sdk import DocumentAgentTemplate
```

Purpose:

- Base helper for custom document-agent runtimes.
- Provides shared context-budget parsing, budget estimate/degraded events,
  keepalive-wrapped graph streaming, graph stream normalization, and
  `final_deliverable_progress` events.
- Keeps domain workflow, artifact schemas, checkpoint payloads, and governance
  in the concrete agent.

Common methods:

- `context_budget(inputs)`
- `stream_graph_run(...)`
- `budget_estimate_event(...)`
- `budget_degraded_event(...)`
- `final_deliverable_progress_event(...)`

### Final document output

Imports:

```python
from novie_agent_sdk import document_final_output, document_final_event
```

Purpose:

- Builds the standard final output shape for document-style agents:
  `analysis`, `content`, `narrative`, `structured_output`,
  `final_payload`, `provides_artifacts`, optional budget, quality, and
  checkpoint metadata.
- Keeps agent-specific mode/phase keys configurable, e.g. `analysis_mode`,
  `pm_mode`, or `architect_mode`.

### Document quality outcome

Imports:

```python
from novie_agent_sdk import DocumentQualityOutcome, skipped_quality_result
```

Purpose:

- Gives agents a shared quality-loop result shape.
- Lets lightweight agents mark quality as intentionally skipped without
  inventing local metadata fields.
- Analyst can still run its LLM review/revise loop and map the outcome into
  the same metadata contract.

## External-Agent Checkpoints

### `checkpoint_step_id`

Import:

```python
from novie_agent_sdk import checkpoint_step_id
```

Purpose:

- Resolves the platform step id from an `ExecutionContext`.
- Uses `ctx.parent_step_id` first, then `ctx.metadata["step_id"]`.

### `checkpoint_input_digest`

Import:

```python
from novie_agent_sdk import checkpoint_input_digest
```

Purpose:

- Builds a deterministic digest for the invocation inputs that should identify
  a resumable checkpoint.
- Keeps digest construction consistent across Analyst, PM, Architect, and
  future document agents.

### `external_agent_checkpoint_service`

Import:

```python
from novie_agent_sdk import external_agent_checkpoint_service
```

Purpose:

- Selects the platform checkpoint adapter from `PlatformServices`.
- Supports both `services.external_agent_checkpoints` and the older
  `services.checkpoint` name.

### `put_external_agent_checkpoint`

Import:

```python
from novie_agent_sdk import put_external_agent_checkpoint
```

Use:

```python
service = external_agent_checkpoint_service(services)
step_id = checkpoint_step_id(ctx)
input_digest = checkpoint_input_digest(
    brief=brief,
    upstream=upstream,
    capability_id=spec.capability_id,
)

record = await put_external_agent_checkpoint(
    service,
    ctx,
    owner_agent_id="architect",
    payload=checkpoint_payload,
    workflow_id=ctx.workflow_id or None,
    step_id=step_id or None,
    summary="draft complete",
    metadata={
        "capability_id": spec.capability_id,
        "step_id": step_id,
        "input_digest": input_digest,
    },
)
```

Purpose:

- Writes an external-agent checkpoint across old and new platform service
  shapes.

### `get_matching_document_checkpoint`

Import:

```python
from novie_agent_sdk import get_matching_document_checkpoint
```

Purpose:

- Reads the latest checkpoint for the current thread/step.
- Validates the payload model.
- Verifies phase, narrative presence, capability id, workflow id, step id, and
  input digest before allowing an agent to resume.

### `skipped_phase_events`

Import:

```python
from novie_agent_sdk import skipped_phase_events
```

Purpose:

- Emits standard trace events for phases skipped because a checkpoint was
  reused.
- Keeps timeline behavior understandable after retry/resume instead of
  showing missing phases.

## Execution Workpad Reads

Imports:

```python
from novie_agent_sdk import (
    execution_workpad_context,
    execution_workpad_entries,
    latest_workpad_entry,
    workpad_entries_by_kind,
)
```

Purpose:

- Reads the compact Execution Workpad injected by the platform.
- Lets agents recover prior run goals, drafts, evidence notes, or quality
  checkpoints without parsing platform-specific envelopes.
- Lets document agents stop carrying private checkpoint write compatibility
  code in their runtimes.

## Skill-Driven Document Agents

### `compile_skill_scope`

Import:

```python
from novie_agent_sdk import compile_skill_scope
```

Purpose:

- Reads selected `SKILL.md` files from a capability spec.
- Parses frontmatter into `SkillMetadata`.
- Builds a bounded `SkillScope` with `skill_sources`, `allowed_tools`, and a
  prompt hint containing the relevant skill instructions.
- Supports synthesis adapters through `source_resolver`.

### `build_deep_agent_executor`

Import:

```python
from novie_agent_sdk import build_deep_agent_executor
```

Purpose:

- Builds a DeepAgents executor for a bounded document capability.
- Provides a virtual filesystem backend.
- Can materialize selected skill directories when native skill loading is
  explicitly enabled.
- Defaults to prompt-loaded skill instructions so agents do not spend runtime
  context reading `/skills`.

## Platform Namespace

### `build_platform_namespace`

Import:

```python
from novie_agent_sdk import build_platform_namespace
```

Purpose:

- Builds the platform client namespace from incoming request headers.
- Exposes platform capabilities to agents through typed namespaces.

Common namespaces:

| Namespace | Main calls | Purpose |
| --- | --- | --- |
| `platform.llm` | `chat`, `structured`, `embed` | Use platform-governed LLM endpoints. |
| `platform.web` | `search` | Public web search through platform policy. |
| `platform.knowledge` | `search` | Project/wiki/curated knowledge search. |
| `platform.artifacts` | `read`, `read_text` | Budgeted artifact retrieval. |
| `platform.checkpoints` | `get`, `list_history` | Runtime checkpoint access. |

## Artifact Reading

### `ArtifactReader`

Import:

```python
from novie_agent_sdk import ArtifactReader
```

Use:

```python
reader = ArtifactReader(
    platform,
    max_uncached_reads=8,
    purpose="analyst evidence retrieval",
)

text = await reader.read_text(
    "artifact://artifact-1",
    mode="search",
    query="pricing evidence",
    max_bytes=4096,
)
```

Purpose:

- Reads platform artifacts as prompt-safe text.
- Normalizes `artifact://` refs.
- Prefers `platform.artifacts.read_text` when available.
- Falls back to formatting `platform.artifacts.read` envelopes.
- Supports exact-read LRU cache.
- Enforces a per-step uncached read budget.
- Handles `content.data`, `excerpts`, base64 JSON/text, nested artifact envelopes,
  and `Next offset`.

### `format_artifact_read_result`

Import:

```python
from novie_agent_sdk import format_artifact_read_result
```

Purpose:

- Converts a platform artifact read envelope into a stable string.
- Useful for tests or custom readers.

### `ArtifactReadCache`

Import:

```python
from novie_agent_sdk import ArtifactReadCache
```

Purpose:

- Small in-process LRU cache for repeated bounded artifact reads.

## Execution Workpad

### `workpad_checkpoint_event`

Import:

```python
from novie_agent_sdk import workpad_checkpoint_event
```

Use:

```python
yield workpad_checkpoint_event(
    kind="section_draft",
    title="Market sizing",
    content="## Market sizing\n\n...",
    base_metadata={"runtime_phase": "draft_sections"},
    metadata={"section_index": 2},
)
```

Purpose:

- Emits a standard `execution_workpad_entry` trace event.
- Platform persists the entry into Execution Workpad.
- Lets agents publish resumable intermediate state without knowing platform storage.

### `execution_workpad_context`

Import:

```python
from novie_agent_sdk import execution_workpad_context
```

Use:

```python
workpad = execution_workpad_context(inputs)
```

Purpose:

- Reads the compact Execution Workpad injected by platform.
- Supports direct `inputs["execution_workpad"]` and nested
  `inputs["platform_context"]["execution_workpad"]`.

### `upstream_context`

Import:

```python
from novie_agent_sdk import upstream_context
```

Use:

```python
upstream = upstream_context(inputs)
direct = upstream.get("direct_handoffs", {})
```

Purpose:

- Reads platform `upstream_context.v1`.
- Gives agents direct handoffs, dependency metadata, transitive refs, and read policy.

## Document A2A Runtime Adapters

These helpers are for document-style agents that keep a custom runtime loop but
should not re-implement A2A context, manifest, or stream wire conversion.

### `load_runtime_manifest`

Import:

```python
from novie_agent_sdk import load_runtime_manifest
```

Use:

```python
manifest = load_runtime_manifest(".well-known/agent.json")
```

Purpose:

- Loads an agent card JSON file.
- Replaces `endpoint` from `NOVIE_AGENT_PUBLIC_ENDPOINT` when set.
- Lets deployed agents publish a runtime endpoint without mutating the checked-in manifest.

### `context_block_from_request`

Import:

```python
from novie_agent_sdk import context_block_from_request
```

Use:

```python
ctx_block = context_block_from_request(
    ctx,
    default_request_id="req-analyst-local",
    default_session_id="sess-analyst-local",
    default_thread_id="thread-analyst-local",
)
```

Purpose:

- Merges inline `input["context"]` with A2A request headers.
- Fills request, session, thread, tenant, workspace, project, identity, workflow, and parent-step fields.
- Preserves `on_behalf_of_user_id` from context, headers, or input.

### `execution_context_from_block` / `execution_context_from_request`

Import:

```python
from novie_agent_sdk import execution_context_from_block, execution_context_from_request
```

Purpose:

- Converts a context block into `novie_protocol.contracts.ExecutionContext`.
- `execution_context_from_request(...)` combines `context_block_from_request(...)` and conversion in one call.
- Use this before constructing protocol services, checkpoints, or platform-scoped operations.

### `resolve_capability_id`

Import:

```python
from novie_agent_sdk import resolve_capability_id
```

Purpose:

- Reads `inputs["capability_id"]`.
- Falls back to the first `capability_grants[*].capability_id`.
- Returns an empty string when no capability id is present.

### `is_internal_stream_visibility`

Import:

```python
from novie_agent_sdk import is_internal_stream_visibility
```

Purpose:

- Checks stream metadata fields such as `visibility`, `content_visibility`,
  `tool_result_visibility`, `output_visibility`, and `internal`.
- Lets agents suppress internal tool/subtask output before writing stream events to clients.

### `stream_event_to_wire`

Import:

```python
from novie_agent_sdk import stream_event_to_wire
```

Use:

```python
body = stream_event_to_wire(event)
```

Purpose:

- Converts `AgentStreamEvent` into the JSON wire body expected by A2A stream endpoints.
- Preserves `metadata["_wire_event"]` when a lower layer already built the wire event.
- Suppresses internal `content` and `tool_result` by default while recording suppression metadata.

## Context Budget

### `GENERIC_DEFAULT_CONTEXT_BUDGET`

Purpose:

- SDK default budget for long-running document agents.
- Includes input/output/total token limits, wall-clock seconds, phase timeouts,
  revision rounds, inline artifact bytes, and a `source` label.

### `default_context_budget`

Import:

```python
from novie_agent_sdk import default_context_budget
```

Use:

```python
defaults = default_context_budget(
    env_prefix="NOVIE_ANALYST",
    source="analyst_default",
    max_revision_rounds=2,
    overrides={"phase_timeouts": {"research": 300}},
)
```

Purpose:

- Builds a default budget with optional agent-specific env prefix.
- Reads `{PREFIX}_MAX_INPUT_TOKENS`, `{PREFIX}_MAX_OUTPUT_TOKENS`, and `{PREFIX}_MAX_TOTAL_TOKENS`.
- Merges override phase timeouts without dropping SDK defaults.

### `context_budget_from_inputs`

Import:

```python
from novie_agent_sdk import context_budget_from_inputs
```

Purpose:

- Reads `inputs["context_budget"]` when present.
- Merges request-supplied values over defaults.
- Merges nested `phase_timeouts` instead of replacing the whole map.

### Budget Measurement Helpers

Imports:

```python
from novie_agent_sdk import (
    adaptive_phase_timeout_seconds,
    budget_limit,
    budget_status,
    budget_summary,
    effective_phase_timeout_seconds,
    estimated_tokens,
    is_over_budget_status,
    max_revision_rounds_from_budget,
    phase_timeout_seconds,
    wall_clock_deadline,
)
```

Purpose:

- `estimated_tokens(value)` gives a conservative text/JSON token estimate.
- `budget_status(...)` returns `within_budget`, `input_over_budget`,
  `output_over_budget`, or `total_over_budget`.
- `is_over_budget_status(status)` converts that status into a boolean gate.
- `budget_limit(...)` and `max_revision_rounds_from_budget(...)` read bounded numeric settings.
- `wall_clock_deadline(...)`, `phase_timeout_seconds(...)`,
  `effective_phase_timeout_seconds(...)`, and `adaptive_phase_timeout_seconds(...)`
  coordinate per-phase timeouts with total run time.
- `budget_summary(...)` produces metadata for final outputs and traces.

## Platform Services Bridge

### `build_http_platform_services`

Import:

```python
from novie_agent_sdk import build_http_platform_services
```

Use:

```python
services = build_http_platform_services(
    incoming_headers,
    agent_id="analyst",
    tracker=degradation_tracker,
)
```

Purpose:

- Builds `novie_protocol.services.PlatformServices` from incoming A2A headers.
- Backs wiki search and external-agent checkpoints with signed platform capability callbacks.
- Returns `None` when `NOVIE_PLATFORM_BASE_URL` or required tenant/project headers are missing.
- Provides no-op protocol services for capabilities that are not wired through callback transport.

### `build_gateway_client`

Import:

```python
from novie_agent_sdk import build_gateway_client
```

Purpose:

- Builds the lower-level signed `CapabilityClient`.
- Useful when an agent needs a direct capability invoke path instead of full `PlatformServices`.

### `build_forward_headers`

Purpose:

- Produces signed-platform-forwardable headers from incoming request headers and `agent_id`.
- Used by `build_gateway_client(...)` and `build_http_platform_services(...)`.

### `CapabilityClient`

Purpose:

- Calls `/capabilities/{capability_id}/invoke`.
- Provides `invoke(...)` for result-only calls and `invoke_with_diagnostics(...)` for degradation-aware calls.
- Also provides `get_json(...)` for signed gateway GET requests.

### `HttpWikiService`

Purpose:

- Implements platform wiki search through `platform.knowledge.search`.
- Marks degradation flags through `DegradationTracker` for binding, transport,
  platform, or no-result states.

### `HttpExternalAgentCheckpointService`

Purpose:

- Implements external-agent checkpoint `put`, `get`, and `list_history`.
- Uses `platform.external_agent_checkpoint.put/get/list` capabilities.
- Returns protocol `ExternalAgentCheckpointRecord` values.

### Platform Service Constants

Imports:

```python
from novie_agent_sdk import (
    CHECKPOINT_GET_CAP,
    CHECKPOINT_LIST_CAP,
    CHECKPOINT_PUT_CAP,
    DEFAULT_TIMEOUT_SECONDS,
    KNOWLEDGE_SEARCH_CAP,
    record_from_dict,
)
```

Purpose:

- Capability id constants keep tests and custom adapters aligned with SDK callback routing.
- `record_from_dict(...)` converts platform checkpoint envelopes into protocol records.

## External Agent Checkpoint Identity

### `checkpoint_step_id`

Import:

```python
from novie_agent_sdk import checkpoint_step_id
```

Purpose:

- Resolves the checkpoint step id from `ctx.parent_step_id` or `ctx.metadata["step_id"]`.

### `checkpoint_input_digest`

Import:

```python
from novie_agent_sdk import checkpoint_input_digest
```

Use:

```python
digest = checkpoint_input_digest(
    {
        "brief": brief,
        "capability_id": capability_id,
        "mode": mode,
        "phase": phase,
    }
)
```

Purpose:

- Produces a stable SHA-256 digest from sorted JSON input parts.
- Lets agents decide whether a checkpoint matches the current invocation.

### `checkpoint_matches_invocation`

Import:

```python
from novie_agent_sdk import checkpoint_matches_invocation
```

Purpose:

- Compares checkpoint record step id, workflow id, capability id, and input digest against the current invocation.
- Accepts optional payload metadata for agents that store digest fields inside the checkpoint payload.

## Document Streaming

### `with_keepalive`

Import:

```python
from novie_agent_sdk import with_keepalive
```

Purpose:

- Wraps an async event iterator.
- Emits `agent_keepalive` trace events after idle periods.
- Reads `NOVIE_AGENT_KEEPALIVE_INTERVAL_S` by default; pass `env_var` for an agent-specific override.

### `content_stream_closed_event` / `keepalive_event`

Purpose:

- Build standard trace events for content-stream closure and idle keepalives.
- Accept an agent-provided `phase_metadata(...)` callback so runtime, mode, phase,
  and capability metadata stay consistent.

### LangGraph Stream Helpers

Imports:

```python
from novie_agent_sdk import (
    extract_chunk_text,
    is_assistant_content_chunk,
    normalize_langgraph_stream_item,
    text_blob,
)
```

Purpose:

- Normalize LangGraph stream tuples into `(namespace, stream_mode, payload)`.
- Extract text from LangChain message chunks and nested content blocks.
- Filter tool chunks so only assistant content is streamed as document text.

### `SubtaskEventMapper`

Import:

```python
from novie_agent_sdk import SubtaskEventMapper
```

Purpose:

- Maps DeepAgents `task` tool calls/results into platform-visible subtask lifecycle traces.
- Emits summary-visible `subtask.started`, `subtask.completed`, `subtask.incomplete`,
  `subtask.tool_call`, `subtask.tool_result`, and `subtask.stream_content` events.
- Emits internal workpad evidence cards for subtask results.

### `with_subtask_keepalive`

Purpose:

- Works like `with_keepalive(...)`.
- Emits active-subtask keepalive traces when a subtask is running.

### `assess_subtask_result`

Import:

```python
from novie_agent_sdk import assess_subtask_result, SubtaskResultAssessment
```

Purpose:

- Classifies subtask results as `complete` or `incomplete`.
- Checks empty/short output, budget or incomplete markers, and evidence signals.
- Uses `DEFAULT_MIN_SUBTASK_RESULT_CHARS` unless overridden.

### Streaming Constants

Purpose:

- `DEFAULT_KEEPALIVE_INTERVAL_SECONDS` is the SDK default idle interval.
- `DEFAULT_MIN_SUBTASK_RESULT_CHARS` is the default evidence/result size threshold.
- `KEEPALIVE_DONE` is an advanced sentinel used internally by keepalive wrappers.

## Degradation Tracking

### `DegradationTracker`

Import:

```python
from novie_agent_sdk import DegradationTracker
```

Use:

```python
tracker = DegradationTracker()
tracker.mark("platform_knowledge_search", "no_results")
flags = tracker.flags()
```

Purpose:

- Accumulates per-run symbolic degradation flags without duplicates.
- `mark(prefix, kind)` records `{prefix}.{kind}`.
- `mark_diagnostics(prefix, diagnostics)` records non-OK capability diagnostics
  and successful `no_results` diagnostics.
- `has(flag_or_prefix)` checks an exact flag or any flag with that prefix.

## Output Contract

### `resolve_step_role`

Import:

```python
from novie_agent_sdk import resolve_step_role
```

Use:

```python
role = resolve_step_role(inputs)
```

Purpose:

- Reads the platform-provided step role from `output_contract`.
- Lets agents distinguish `upstream_handoff` from `terminal_deliverable`.
- Keeps DAG-derived role interpretation in one SDK API instead of per-agent code.

### `is_upstream_handoff` / `is_terminal_deliverable`

Import:

```python
from novie_agent_sdk import is_upstream_handoff, is_terminal_deliverable
```

Purpose:

- Branches output behavior based on the platform contract.
- Intermediate steps should produce bounded handoffs.
- Sink steps should produce final user-visible artifacts.

### `step_run_policy`

Import:

```python
from novie_agent_sdk import step_run_policy
```

Use:

```python
policy = step_run_policy(inputs)

if policy.is_upstream_handoff:
    # Produce a bounded internal handoff, then complete the step.
    # Do not stream long user-visible content or run final-report quality loops.
    ...
```

Purpose:

- Converts the platform `output_contract` into concrete agent runtime behavior.
- For `upstream_handoff`, disables user-visible content streaming, quality loops, and final deliverable generation.
- For terminal steps, keeps user-visible streaming, quality loops, and final deliverable generation enabled.
- Gives document agents, task splitter agents, and future agents one shared boundary between platform DAG semantics and agent-specific business logic.

### `handoff_max_bytes`

Import:

```python
from novie_agent_sdk import handoff_max_bytes
```

Purpose:

- Reads the byte budget for bounded handoffs.
- Applies SDK-level safety bounds so agents do not create oversized internal artifacts.

### `fit_text_to_utf8_bytes`

Import:

```python
from novie_agent_sdk import fit_text_to_utf8_bytes
```

Purpose:

- Trims markdown/text to a UTF-8 byte budget.
- Avoids splitting multi-byte characters.

## LangChain Runtime Helpers

### `langchain_runnable_config`

Import:

```python
from novie_agent_sdk import langchain_runnable_config
```

Use:

```python
config = langchain_runnable_config(
    agent_id="analyst",
    ctx=ctx,
    callbacks=callbacks,
    runtime_phase="draft",
    capability_id="agent.analyst.report_synthesis",
    stage="section_1",
)
```

Purpose:

- Builds LangChain `RunnableConfig`.
- Creates a stable run id for the agent segment.
- Adds request/session/workflow/capability metadata.
- Keeps usage callbacks tied to stable run identity.

### `agent_run_id`

Purpose:

- Returns a deterministic UUID for one agent runtime segment.

### `runnable_run_id`

Purpose:

- Reads a run id from LangChain config-like values.
- Generates a fallback UUID when no run id is present.

### `notify_usage_callbacks`

Use:

```python
await notify_usage_callbacks(config, {"input_tokens": 100, "output_tokens": 20})
```

Purpose:

- Sends platform LLM usage metadata to LangChain callbacks via `on_llm_end`.
- Used by platform LLM adapters that receive usage outside normal LangChain
  provider callbacks.

## Platform LLM

### `PlatformChatModel`

Import:

```python
from novie_agent_sdk import PlatformChatModel
```

Use:

```python
model = PlatformChatModel(platform, model="anthropic/claude-sonnet-4.6")
message = await model.ainvoke("Summarize this")
```

Purpose:

- LangChain-compatible chat model backed by `platform.llm`.
- Supports `ainvoke`, `astream`, `bind_tools`, and `with_structured_output`.
- Uses `platform.llm.chat` directly for normal chat calls.
- Uses capability diagnostics when tool binding requires `platform.llm.chat`
  through the platform capability gateway.
- Calls `notify_usage_callbacks(...)` after platform LLM responses that include usage metadata.
- Lets agents use platform-governed model routing without provider keys.

### `PlatformStructuredChatModel`

Import:

```python
from novie_agent_sdk import PlatformStructuredChatModel
```

Purpose:

- Public structured-output model returned by `PlatformChatModel.with_structured_output(...)`.
- Delegates to `platform.llm.structured`.
- Accepts JSON Schema dicts and Pydantic model classes through `with_structured_output(...)`.
- Propagates usage metadata through `notify_usage_callbacks(...)`.

## LLM Facade

### `build_llm_facade`

Import:

```python
from novie_agent_sdk import build_llm_facade
```

Purpose:

- Builds an LLM facade that can use platform LLMs when available and BYOK/direct
  configuration when appropriate.
- Provides `chat`, `structured`, `embed`, `budget_check`, and `usage_summary`.

## Deliverables

### `markdown_deliverable_output`

Import:

```python
from novie_agent_sdk import markdown_deliverable_output
```

Use:

```python
output = markdown_deliverable_output(
    title="Final Report",
    markdown="# Final Report",
    artifact_type="management_report",
    artifact_family="document",
    metadata={"phase": "final"},
)
```

Purpose:

- Produces the standard final document deliverable shape.
- Platform can project this into chat cards, downloads, and downstream handoffs.

### `bounded_handoff_output`

Import:

```python
from novie_agent_sdk import bounded_handoff_output
```

Use:

```python
output = bounded_handoff_output(
    handoff_markdown="# Evidence Handoff\n\nKey facts...",
    artifact_type="research_dossier",
    artifact_family="document",
    capability_id="agent.example.research",
    output_contract={"kind": "bounded_handoff", "max_bytes": 12000},
    metadata={"analysis_phase": "market_map"},
)
```

Purpose:

- Produces the standard internal output shape for intermediate DAG steps.
- Marks the output as `output_visibility="internal"`.
- Keeps upstream handoff envelope construction out of individual agents.
- Lets terminal steps consume a consistent `final_payload` / `structured_output` shape.

## Stream Event Helpers

Imports:

```python
from novie_agent_sdk import (
    content_delta_event,
    progress_event,
    tool_call_event,
    tool_result_event,
)
```

Purpose:

- Build platform-compatible stream events.
- Keep tool/progress/content event shapes consistent across agents.

## Manifest and Conformance

Common imports:

```python
from novie_agent_sdk import (
    generate_agent_manifest,
    validate_agent_yaml,
    run_conformance,
)
```

Purpose:

- Generate `.well-known/agent.json` from `agent.yaml`.
- Validate agent manifests.
- Run SDK/platform protocol conformance checks.
