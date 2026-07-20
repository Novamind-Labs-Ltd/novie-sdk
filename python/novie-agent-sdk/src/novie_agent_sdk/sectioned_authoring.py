"""Section-by-section longform authoring for document deliverables."""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import re
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any, NamedTuple

from novie_protocol.agents import AgentStreamEvent

from .artifact_ledger import ArtifactLedger, EvidencePackBuilder
from .context_budget import wall_clock_deadline as context_wall_clock_deadline
from .document_authoring_budget import (
    DocumentAuthoringDeadlineExceeded,
    DocumentOutputBudget,
)
from .document_quality import DocumentQualityLoopResult, skipped_quality_result
from .skill_contracts import SkillRuntimeContract

_SECTIONED_AUTHORING_ENV = "NOVIE_SECTIONED_AUTHORING_V2"
_SECTIONED_AUTHORING_DISABLED_ENV = "NOVIE_SECTIONED_AUTHORING_DISABLED"
_LLM_STREAM_MAX_ATTEMPTS_ENV = "NOVIE_LLM_STREAM_MAX_ATTEMPTS"
_LLM_STREAM_RETRY_BACKOFF_ENV = "NOVIE_LLM_STREAM_RETRY_BACKOFF_S"
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSY_ENV_VALUES = {"0", "false", "no", "off", "disabled"}
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_INTERNAL_PROCESS_RE = re.compile(
    r"(compact upstream|compact handoff|evidence pack|tool status|"
    r"now writing|now draft|fetch_artifact|raw json|"
    r"上游证据包|紧凑交接|现在撰写)",
    re.IGNORECASE,
)
_PLACEHOLDER_SECTION_RE = re.compile(
    r"\bno\s+section\s+draft\s+was\s+returned\b",
    re.IGNORECASE,
)

# Quality-gate failures that indicate the section is structurally unusable.
# These always block (even under ``degrade`` enforcement) because a downstream
# merge/polish step cannot recover an empty or un-headed section.
_STRUCTURAL_GATE_FAILURES = frozenset({
    "empty_section",
    "missing_section_heading",
    "placeholder_section",
})

# Gate enforcement modes. ``strict`` keeps the legacy behaviour of hard-failing
# the step on any unmet gate; ``degrade`` records a best-effort section with an
# explicit gap marker for soft, evidence-bound failures so the plan completes
# instead of dead-ending on an unsatisfiable quality bar.
_GATE_ENFORCEMENT_STRICT = "strict"
_GATE_ENFORCEMENT_DEGRADE = "degrade"
_GATE_ENFORCEMENT_MODES = frozenset({_GATE_ENFORCEMENT_STRICT, _GATE_ENFORCEMENT_DEGRADE})

# Finalization modes implemented by ``SectionedLongformAuthor._polish_final``.
# Contract construction rejects anything else: an unrecognised mode used to
# fall through to ``single_polish`` silently, which shipped truncated
# full-document rewrites while the skill author believed a different
# finalize path was active.
KNOWN_FINALIZATION_MODES = frozenset({
    "single_polish",
    "boundary_stitch",
    "progressive_section_merge",
})


# Provider stop/finish values that mean "output hit max_output_tokens".
# OpenAI-compatible providers report finish_reason="length"; Anthropic-style
# metadata reports stop_reason="max_tokens" (OpenRouter forwards either,
# depending on the routed model).
_TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens", "max_output_tokens"})


class _StreamedLlmText(NamedTuple):
    """Text plus completion metadata from one ``_stream_llm_text`` call.

    ``truncated`` means the provider stopped at the output-token limit, so
    ``text`` ends mid-thought. Callers must decide per call site whether a
    cut-off result is acceptable (running summary), droppable (seam bridge),
    replaceable (final polish falls back to the combined sections), or a
    quality failure (section drafts feed the revision loop).
    """

    text: str
    finish_reason: str = ""
    truncated: bool = False


def _finish_reason_of(completed_result: Mapping[str, Any] | None) -> str:
    if not isinstance(completed_result, Mapping):
        return ""
    metadata = completed_result.get("response_metadata")
    if not isinstance(metadata, Mapping):
        return ""
    reason = metadata.get("finish_reason") or metadata.get("stop_reason")
    return str(reason or "").strip().lower()


def _remaining_deadline_seconds(deadline: float) -> float:
    return max(0.0, deadline - asyncio.get_running_loop().time())


def _validated_finalization(value: Any) -> str:
    mode = str(value or "single_polish").strip()
    if mode not in KNOWN_FINALIZATION_MODES:
        raise ValueError(
            "sectioned_authoring contract: unknown finalization mode "
            f"{mode!r}. Valid modes: {sorted(KNOWN_FINALIZATION_MODES)}. "
            "Check the skill runtime contract "
            "(SKILL.md metadata.novie.runtime_contract runtime/length_profiles)."
        )
    return mode
_TRANSIENT_LLM_ERROR_CODES = frozenset({
    "internal_error",
    "platform_unavailable",
    "stream_heartbeat_timeout",
    "transport_error",
})


@dataclass(frozen=True, slots=True)
class SectionPlan:
    section_id: str
    title: str
    objective: str = ""
    evidence_query: str = ""
    min_words: int = 180


@dataclass(frozen=True, slots=True)
class SectionedAuthoringContract:
    """Shape and quality contract for sectioned longform authoring."""

    coverage_model: str = "document"
    length_profile: str = "adaptive"
    profile_source: str = ""
    profile_confidence: str = ""
    context_policy: str = "evidence_pack_v1"
    quality_contract_ref: str = "document.generic_quality"
    finalization: str = "single_polish"
    evidence_depth: str = "standard"
    min_outline_sections: int = 2
    max_outline_sections: int = 9
    min_section_words: int = 90
    default_section_words: int = 180
    max_section_words: int = 280
    max_section_revision_rounds: int = 1
    max_document_output_tokens: int = 0
    final_retention_ratio: float = 0.8
    seam_context_chars: int = 1500
    finalize_model: str = ""
    running_context: bool = True
    running_context_window_k: int = 2
    running_summary_max_tokens: int = 400
    running_summary_model: str = ""
    require_evidence_refs: bool = True
    require_confidence_layer: bool = False
    forbid_step_artifact_only_citations: bool = False
    min_unique_sources_per_core_section: int = 0
    gate_enforcement: str = _GATE_ENFORCEMENT_DEGRADE
    outline_artifact_type: str = ""
    section_artifact_type: str = ""
    final_artifact_type: str = ""
    record_outline_ref: bool = True
    record_section_refs: bool = True
    record_final_deliverable_ref: bool = True

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "SectionedAuthoringContract":
        raw = dict(value or {})
        return cls(
            coverage_model=str(raw.get("coverage_model") or "document"),
            length_profile=str(raw.get("length_profile") or "adaptive"),
            profile_source=str(raw.get("profile_source") or ""),
            profile_confidence=str(raw.get("profile_confidence") or ""),
            context_policy=str(raw.get("context_policy") or "evidence_pack_v1"),
            quality_contract_ref=str(
                raw.get("quality_contract_ref") or "document.generic_quality"
            ),
            finalization=_validated_finalization(raw.get("finalization")),
            evidence_depth=str(raw.get("evidence_depth") or "standard"),
            min_outline_sections=_positive_int(raw.get("min_outline_sections"), 2),
            max_outline_sections=_positive_int(raw.get("max_outline_sections"), 9),
            min_section_words=_positive_int(raw.get("min_section_words"), 90),
            default_section_words=_positive_int(raw.get("default_section_words"), 180),
            max_section_words=_positive_int(raw.get("max_section_words"), 280),
            max_section_revision_rounds=_positive_int(
                raw.get("max_section_revision_rounds"),
                1,
            ),
            max_document_output_tokens=_positive_int(
                raw.get("max_document_output_tokens"),
                0,
            ),
            final_retention_ratio=_ratio(raw.get("final_retention_ratio"), 0.8),
            seam_context_chars=_positive_int(raw.get("seam_context_chars"), 1500),
            finalize_model=str(raw.get("finalize_model") or ""),
            running_context=_bool(raw.get("running_context"), True),
            running_context_window_k=_positive_int(
                raw.get("running_context_window_k"), 2
            ),
            running_summary_max_tokens=_positive_int(
                raw.get("running_summary_max_tokens"), 400
            ),
            running_summary_model=str(raw.get("running_summary_model") or ""),
            require_evidence_refs=_bool(raw.get("require_evidence_refs"), True),
            require_confidence_layer=_bool(
                raw.get("require_confidence_layer"),
                False,
            ),
            forbid_step_artifact_only_citations=_bool(
                raw.get("forbid_step_artifact_only_citations"),
                False,
            ),
            min_unique_sources_per_core_section=_positive_int(
                raw.get("min_unique_sources_per_core_section"),
                0,
            ),
            gate_enforcement=_gate_enforcement(raw.get("gate_enforcement")),
            outline_artifact_type=str(raw.get("outline_artifact_type") or ""),
            section_artifact_type=str(raw.get("section_artifact_type") or ""),
            final_artifact_type=str(raw.get("final_artifact_type") or ""),
            record_outline_ref=_bool(raw.get("record_outline_ref"), True),
            record_section_refs=_bool(raw.get("record_section_refs"), True),
            record_final_deliverable_ref=_bool(
                raw.get("record_final_deliverable_ref"),
                True,
            ),
        )


@dataclass(frozen=True, slots=True)
class SectionDraft:
    plan: SectionPlan
    markdown: str
    artifact_ref: Mapping[str, Any] = field(default_factory=dict)
    quality: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SectionQualityGateResult:
    failures: tuple[str, ...]
    information_units: int
    citation_count: int
    evidence_item_count: int
    revision_rounds: int = 0
    unique_sources_available: int = 0
    unique_sources_cited: int = 0
    degraded: bool = False

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def hard_failures(self) -> tuple[str, ...]:
        """Structural failures that always block the deliverable."""
        return tuple(f for f in self.failures if f in _STRUCTURAL_GATE_FAILURES)

    @property
    def soft_failures(self) -> tuple[str, ...]:
        """Evidence/quality-bound failures eligible for graceful degradation."""
        return tuple(f for f in self.failures if f not in _STRUCTURAL_GATE_FAILURES)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "hard_failures": list(self.hard_failures),
            "soft_failures": list(self.soft_failures),
            "information_units": self.information_units,
            "citation_count": self.citation_count,
            "evidence_item_count": self.evidence_item_count,
            "revision_rounds": self.revision_rounds,
            "unique_sources_available": self.unique_sources_available,
            "unique_sources_cited": self.unique_sources_cited,
            "degraded": self.degraded,
        }


@dataclass(frozen=True, slots=True)
class SectionedAuthoringResult:
    markdown: str
    length_profile: str
    outline: tuple[SectionPlan, ...]
    drafts: tuple[SectionDraft, ...]
    ledger: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SectionedDocumentFinalizationResult:
    """Output of the high-level sectioned finalization runner."""

    authoring_result: SectionedAuthoringResult
    quality_result: DocumentQualityLoopResult
    started_event: AgentStreamEvent
    completed_event: AgentStreamEvent
    finalize_strategy: str = "sectioned_longform"
    finalize_attempts: int = 1

    @property
    def authoring_ledger(self) -> dict[str, Any]:
        return dict(self.authoring_result.ledger)


def sectioned_authoring_enabled(
    env: Mapping[str, str] | None = None,
    *,
    enabled_env_var: str = _SECTIONED_AUTHORING_ENV,
    disabled_env_var: str = _SECTIONED_AUTHORING_DISABLED_ENV,
    agent_enabled_env_var: str | None = None,
    agent_disabled_env_var: str | None = None,
) -> bool:
    values = os.environ if env is None else env
    disabled_keys = [disabled_env_var, agent_disabled_env_var]
    for key in disabled_keys:
        if not key:
            continue
        disabled = str(values.get(key, "")).strip().lower()
        if disabled in _TRUTHY_ENV_VALUES:
            return False

    enabled_keys = [agent_enabled_env_var, enabled_env_var]
    for key in enabled_keys:
        if not key:
            continue
        raw = str(values.get(key, "")).strip().lower()
        if raw in _FALSY_ENV_VALUES:
            return False
    return True


def platform_namespace_from_llm_facade(llm_facade: Any | None) -> Any | None:
    """Return the platform namespace exposed by an SDK LLM facade."""
    if llm_facade is None:
        return None
    platform_ns = getattr(llm_facade, "platform_ns", None)
    if platform_ns is None:
        platform_ns = getattr(llm_facade, "_platform_ns", None)
    if platform_ns is not None:
        return platform_ns
    if getattr(llm_facade, "artifacts", None) is not None:
        return llm_facade
    return None


async def run_sectioned_document_finalization(
    *,
    llm_facade: Any | None,
    skill_contract: SkillRuntimeContract | None,
    artifact_type: str,
    step_id: str,
    capability_id: str,
    context_budget: Mapping[str, Any],
    brief: Mapping[str, Any],
    upstream: Mapping[str, Any],
    workflow_id: str | None = None,
    thread_id: str | None = None,
    agent_id: str | None = None,
    mode_metadata: Mapping[str, Any] | None = None,
    draft_narrative: str = "",
    draft_narrative_key: str = "",
    draft_narrative_artifact_type: str = "",
    draft_narrative_summary: str = "",
    document_input: Mapping[str, Any] | None = None,
    authoring_instructions: str = "",
    agent_disabled_env_var: str | None = None,
    agent_enabled_env_var: str | None = None,
    required_strategy: str = "sectioned_longform",
    quality_reason: str = "sectioned_authoring_quality_gates",
    quality_metadata: Mapping[str, Any] | None = None,
    defer_intermediate_artifacts: bool = False,
    defer_final_artifact: bool = False,
    wall_clock_deadline: float | None = None,
) -> SectionedDocumentFinalizationResult:
    """Run sectioned longform finalization for document agents.

    This is intentionally a coarse runner: agents still own the graph, prompt,
    structured artifact construction, and final event envelope. The SDK owns
    the repeated sectioned-authoring checks, author construction, trace
    metadata, and skipped-quality result wiring.
    """
    if skill_contract is None or skill_contract.is_empty:
        raise RuntimeError(
            "sectioned_authoring_required: document finalization requires "
            "a skill runtime contract"
        )
    if skill_contract.strategy != required_strategy:
        raise RuntimeError(
            "sectioned_authoring_required: document finalization requires "
            f"runtime.strategy={required_strategy} skill contracts"
        )
    if not sectioned_authoring_enabled(
        agent_disabled_env_var=agent_disabled_env_var,
        agent_enabled_env_var=agent_enabled_env_var,
    ):
        raise RuntimeError("sectioned_authoring_disabled")
    if llm_facade is None:
        raise RuntimeError("sectioned_authoring_llm_unavailable")
    platform_ns = platform_namespace_from_llm_facade(llm_facade)
    if platform_ns is None:
        raise RuntimeError("sectioned_authoring_platform_namespace_unavailable")

    mode_meta = dict(mode_metadata or {})
    started_event = AgentStreamEvent(
        kind="trace",
        metadata={
            "event": "sectioned_authoring_started",
            "runtime_phase": "sectioned_authoring",
            "semantic_phase": "finalizing_output",
            **mode_meta,
            "capability_id": capability_id,
            "authoring_strategy": required_strategy,
            "skill_contract": skill_contract.to_metadata(),
        },
    )
    author = SectionedLongformAuthor(
        llm_facade=llm_facade,
        platform=platform_ns,
        artifact_type=artifact_type,
        step_id=step_id or "",
        capability_id=capability_id,
        context_budget=dict(context_budget),
        authoring_contract=sectioned_authoring_contract_from_skill(
            skill_contract,
            artifact_type=artifact_type,
        ),
        authoring_instructions=authoring_instructions,
        defer_intermediate_artifacts=defer_intermediate_artifacts,
        defer_final_artifact=defer_final_artifact,
    )
    authoring_upstream: dict[str, Any] = dict(upstream)
    if draft_narrative_key:
        authoring_upstream[draft_narrative_key] = {
            "artifact_type": draft_narrative_artifact_type or draft_narrative_key,
            "summary": draft_narrative_summary,
            "content": draft_narrative,
        }
    if document_input:
        authoring_upstream["_document_input"] = dict(document_input)

    deadline = wall_clock_deadline or context_wall_clock_deadline(dict(context_budget))
    try:
        if deadline is None:
            authoring_result = await author.author(
                brief=brief,
                upstream=authoring_upstream,
                workflow_id=workflow_id,
                thread_id=thread_id,
                agent_id=agent_id,
            )
        else:
            authoring_result = await asyncio.wait_for(
                author.author(
                    brief=brief,
                    upstream=authoring_upstream,
                    workflow_id=workflow_id,
                    thread_id=thread_id,
                    agent_id=agent_id,
                ),
                timeout=_remaining_deadline_seconds(deadline),
            )
    except TimeoutError as exc:
        raise DocumentAuthoringDeadlineExceeded(
            "document_authoring_deadline_exceeded: sectioned authoring exceeded "
            "its absolute wall-clock deadline"
        ) from exc
    completed_event = AgentStreamEvent(
        kind="trace",
        metadata={
            "event": "sectioned_authoring_completed",
            "runtime_phase": "sectioned_authoring",
            "semantic_phase": "finalizing_output",
            **mode_meta,
            "capability_id": capability_id,
            "finalize_strategy": required_strategy,
            "section_count": len(authoring_result.drafts),
            "authoring_ledger": dict(authoring_result.ledger),
        },
    )
    quality_result = skipped_quality_result(
        authoring_result.markdown,
        reason=quality_reason,
        metadata={
            **dict(quality_metadata or {}),
            "section_count": len(authoring_result.drafts),
            "authoring_ledger": dict(authoring_result.ledger),
        },
    )
    return SectionedDocumentFinalizationResult(
        authoring_result=authoring_result,
        quality_result=quality_result,
        started_event=started_event,
        completed_event=completed_event,
        finalize_strategy=required_strategy,
        finalize_attempts=1,
    )


async def astream_sectioned_document_finalization(
    *,
    llm_facade: Any | None,
    skill_contract: SkillRuntimeContract | None,
    artifact_type: str,
    step_id: str,
    capability_id: str,
    context_budget: Mapping[str, Any],
    brief: Mapping[str, Any],
    upstream: Mapping[str, Any],
    workflow_id: str | None = None,
    thread_id: str | None = None,
    agent_id: str | None = None,
    mode_metadata: Mapping[str, Any] | None = None,
    draft_narrative: str = "",
    draft_narrative_key: str = "",
    draft_narrative_artifact_type: str = "",
    draft_narrative_summary: str = "",
    document_input: Mapping[str, Any] | None = None,
    authoring_instructions: str = "",
    agent_disabled_env_var: str | None = None,
    agent_enabled_env_var: str | None = None,
    required_strategy: str = "sectioned_longform",
    quality_reason: str = "sectioned_authoring_quality_gates",
    quality_metadata: Mapping[str, Any] | None = None,
    defer_intermediate_artifacts: bool = False,
    defer_final_artifact: bool = False,
    length_profile: str | None = None,
    profile_source: str = "skill_default",
    profile_confidence: str = "confirmed",
    resume_state: Mapping[str, Any] | None = None,
    phase_checkpoint_sink: Callable[[Mapping[str, Any]], Any] | None = None,
    rebase_artifact_types_to_runtime: bool = False,
    wall_clock_deadline: float | None = None,
) -> AsyncIterator[AgentStreamEvent | SectionedDocumentFinalizationResult]:
    """Gold-path sectioned finalize used by report_synthesis-style document agents.

    Streams phase/content events while ``SectionedLongformAuthor`` runs, then
    yields a terminal ``SectionedDocumentFinalizationResult``. Callers own the
    typed artifact envelope and final event.
    """
    if skill_contract is None or skill_contract.is_empty:
        raise RuntimeError(
            "sectioned_authoring_required: document finalization requires "
            "a skill runtime contract"
        )
    if skill_contract.strategy != required_strategy:
        raise RuntimeError(
            "sectioned_authoring_required: document finalization requires "
            f"runtime.strategy={required_strategy} skill contracts"
        )
    if not sectioned_authoring_enabled(
        agent_disabled_env_var=agent_disabled_env_var,
        agent_enabled_env_var=agent_enabled_env_var,
    ):
        raise RuntimeError("sectioned_authoring_disabled")
    if llm_facade is None:
        raise RuntimeError("sectioned_authoring_llm_unavailable")
    platform_ns = platform_namespace_from_llm_facade(llm_facade)
    if platform_ns is None:
        raise RuntimeError("sectioned_authoring_platform_namespace_unavailable")

    resolved_profile = _resolve_length_profile_for_finalize(
        skill_contract,
        length_profile=length_profile,
        profile_source=profile_source,
        profile_confidence=profile_confidence,
    )
    authoring_contract = sectioned_authoring_contract_from_skill(
        skill_contract,
        artifact_type=artifact_type,
        length_profile=resolved_profile["profile"],
        profile_source=resolved_profile["source"],
        profile_confidence=resolved_profile["confidence"],
    )
    if rebase_artifact_types_to_runtime and isinstance(authoring_contract, dict):
        declared_final = authoring_contract.get("final_artifact_type")
        if artifact_type and declared_final != artifact_type:
            authoring_contract = {
                **authoring_contract,
                "outline_artifact_type": f"{artifact_type}.outline",
                "section_artifact_type": f"{artifact_type}.section",
                "final_artifact_type": artifact_type,
            }

    mode_meta = dict(mode_metadata or {})
    yield AgentStreamEvent(
        kind="trace",
        metadata={
            "event": "sectioned_authoring_started",
            "runtime_phase": "sectioned_authoring",
            "semantic_phase": "finalizing_output",
            **mode_meta,
            "capability_id": capability_id,
            "authoring_strategy": required_strategy,
            "length_profile": resolved_profile["profile"],
            "profile_source": resolved_profile["source"],
            "profile_confidence": resolved_profile["confidence"],
            "skill_contract": skill_contract.to_metadata(),
        },
    )

    sectioned_done = object()
    sectioned_events: asyncio.Queue[dict[str, Any] | object] = asyncio.Queue()

    def _collect_sectioned_event(event: Mapping[str, Any]) -> None:
        sectioned_events.put_nowait(dict(event))

    author = SectionedLongformAuthor(
        llm_facade=llm_facade,
        platform=platform_ns,
        artifact_type=artifact_type,
        step_id=step_id or "",
        capability_id=capability_id,
        context_budget=dict(context_budget),
        authoring_contract=authoring_contract,
        authoring_instructions=authoring_instructions,
        defer_intermediate_artifacts=defer_intermediate_artifacts,
        defer_final_artifact=defer_final_artifact,
        phase_event_sink=_collect_sectioned_event,
        phase_checkpoint_sink=phase_checkpoint_sink,
    )
    authoring_upstream: dict[str, Any] = dict(upstream)
    if draft_narrative_key:
        authoring_upstream[draft_narrative_key] = {
            "artifact_type": draft_narrative_artifact_type or draft_narrative_key,
            "summary": draft_narrative_summary,
            "content": draft_narrative,
        }
    if document_input:
        authoring_upstream["_document_input"] = dict(document_input)

    async def _author_document() -> SectionedAuthoringResult:
        try:
            return await author.author(
                brief=brief,
                upstream=authoring_upstream,
                workflow_id=workflow_id,
                thread_id=thread_id,
                agent_id=agent_id,
                resume_state=dict(resume_state) if resume_state is not None else None,
            )
        finally:
            sectioned_events.put_nowait(sectioned_done)

    deadline = wall_clock_deadline or context_wall_clock_deadline(dict(context_budget))
    author_task = asyncio.create_task(_author_document())
    try:
        while True:
            try:
                if deadline is None:
                    sectioned_event = await sectioned_events.get()
                else:
                    sectioned_event = await asyncio.wait_for(
                        sectioned_events.get(),
                        timeout=_remaining_deadline_seconds(deadline),
                    )
            except TimeoutError as exc:
                yield AgentStreamEvent(
                    kind="trace",
                    metadata={
                        "event": "sectioned_authoring_deadline_exceeded",
                        "runtime_phase": "sectioned_authoring",
                        "semantic_phase": "finalizing_output",
                        "error_code": DocumentAuthoringDeadlineExceeded.code,
                        **mode_meta,
                        "capability_id": capability_id,
                    },
                )
                raise DocumentAuthoringDeadlineExceeded(
                    "document_authoring_deadline_exceeded: sectioned authoring "
                    "exceeded its absolute wall-clock deadline"
                ) from exc
            if sectioned_event is sectioned_done:
                break
            event_metadata = {
                **mode_meta,
                "capability_id": capability_id,
                **dict(sectioned_event),
            }
            for stream_event in project_sectioned_phase_event(event_metadata):
                yield stream_event
        authoring_result = await author_task
    finally:
        if not author_task.done():
            author_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await author_task

    completed_event = AgentStreamEvent(
        kind="trace",
        metadata={
            "event": "sectioned_authoring_completed",
            "runtime_phase": "sectioned_authoring",
            "semantic_phase": "finalizing_output",
            **mode_meta,
            "capability_id": capability_id,
            "finalize_strategy": required_strategy,
            "authoring_strategy": required_strategy,
            "length_profile": resolved_profile["profile"],
            "profile_source": resolved_profile["source"],
            "section_count": len(authoring_result.drafts),
            "authoring_ledger": dict(authoring_result.ledger),
        },
    )
    yield completed_event
    quality_result = skipped_quality_result(
        authoring_result.markdown,
        reason=quality_reason,
        metadata={
            **dict(quality_metadata or {}),
            "section_count": len(authoring_result.drafts),
            "authoring_ledger": dict(authoring_result.ledger),
            "length_profile": resolved_profile["profile"],
        },
    )
    yield SectionedDocumentFinalizationResult(
        authoring_result=authoring_result,
        quality_result=quality_result,
        started_event=AgentStreamEvent(
            kind="trace",
            metadata={
                "event": "sectioned_authoring_started",
                "runtime_phase": "sectioned_authoring",
                "semantic_phase": "finalizing_output",
                **mode_meta,
                "capability_id": capability_id,
                "authoring_strategy": required_strategy,
            },
        ),
        completed_event=completed_event,
        finalize_strategy=required_strategy,
        finalize_attempts=1,
    )


def project_sectioned_phase_event(
    metadata: Mapping[str, Any],
) -> list[AgentStreamEvent]:
    """Project SDK sectioned-authoring phase events onto A2A stream primitives."""
    event_name = str(metadata.get("event") or "").strip()
    meta = dict(metadata)
    if event_name == "agent.llm_call.delta":
        delta = str(meta.get("text_delta") or "")
        if not delta:
            return []
        content_metadata = dict(meta)
        content_metadata["event"] = "agent.stream_content"
        content_metadata["source_event"] = event_name
        return [
            AgentStreamEvent(
                kind="content",
                content=delta,
                metadata=content_metadata,
            )
        ]
    if event_name == "agent.tool_call":
        return [
            AgentStreamEvent(
                kind="tool_call",
                tool_name=str(meta.get("tool_name") or ""),
                tool_call_id=str(meta.get("tool_call_id") or ""),
                tool_args={
                    key: value
                    for key, value in meta.items()
                    if key
                    not in {
                        "event",
                        "tool_name",
                        "tool_call_id",
                        "tool_result",
                        "result_preview",
                    }
                },
                metadata=meta,
            )
        ]
    if event_name == "agent.tool_result":
        result_preview = str(
            meta.get("tool_result")
            or meta.get("result_preview")
            or meta.get("summary")
            or ""
        )
        return [
            AgentStreamEvent(
                kind="tool_result",
                tool_name=str(meta.get("tool_name") or ""),
                tool_call_id=str(meta.get("tool_call_id") or ""),
                tool_result=result_preview,
                metadata=meta,
            )
        ]
    if event_name == "agent.tool_error":
        result_preview = str(
            meta.get("message")
            or meta.get("error")
            or meta.get("result_preview")
            or "tool failed"
        )
        return [
            AgentStreamEvent(
                kind="tool_result",
                tool_name=str(meta.get("tool_name") or ""),
                tool_call_id=str(meta.get("tool_call_id") or ""),
                tool_result=result_preview,
                metadata={**meta, "ok": False},
            )
        ]
    return [AgentStreamEvent(kind="trace", metadata=meta)]


def _resolve_length_profile_for_finalize(
    skill_contract: SkillRuntimeContract,
    *,
    length_profile: str | None,
    profile_source: str,
    profile_confidence: str,
) -> dict[str, str]:
    supported = set(skill_contract.document.length_profiles) or {
        "short",
        "medium",
        "long",
    }
    defaults = dict(skill_contract.task_profile.defaults or {})
    selected = str(length_profile or "").strip().lower()
    source = profile_source
    if not selected:
        selected = str(defaults.get("length_profile") or "adaptive").strip().lower()
        source = "skill_default"
    if selected == "adaptive":
        selected = "medium"
    if selected not in supported and "medium" in supported:
        selected = "medium"
    if selected not in supported:
        selected = next(iter(sorted(supported)))
    return {
        "profile": selected,
        "source": source,
        "confidence": profile_confidence or "confirmed",
    }


def sectioned_authoring_contract_from_skill(
    contract: SkillRuntimeContract,
    *,
    artifact_type: str,
    length_profile: str | None = None,
    profile_source: str = "",
    profile_confidence: str = "",
) -> dict[str, Any]:
    """Map a generic skill runtime contract to sectioned authoring settings."""
    document = contract.document
    raw_document = dict(document.raw or {})
    defaults = dict(contract.task_profile.defaults or {})
    base_length_profile = (
        str(length_profile or "").strip().lower()
        or str(defaults.get("length_profile") or "").strip().lower()
        or str(raw_document.get("length_profile") or "").strip().lower()
        or "adaptive"
    )
    profile = document.length_profiles.get(base_length_profile)
    if profile is not None:
        finalization = profile.finalization or contract.runtime.finalization or "single_polish"
        evidence_depth = profile.evidence_depth or "standard"
        min_outline_sections = profile.min_sections or document.outline.min_sections or 2
        max_outline_sections = profile.max_sections or document.outline.max_sections or 9
        min_section_words = profile.min_units or document.section.min_units or 90
        default_section_words = (
            profile.default_units or document.section.default_units or 180
        )
        max_section_words = profile.max_units or document.section.max_units or 280
        max_section_revision_rounds = (
            profile.max_revision_rounds
            or document.section.max_revision_rounds
            or 1
        )
        final_retention_ratio = (
            profile.final_retention_ratio
            or document.final.min_retention_ratio
            or 0.8
        )
    else:
        finalization = contract.runtime.finalization or "single_polish"
        evidence_depth = str(raw_document.get("evidence_depth") or "standard")
        min_outline_sections = document.outline.min_sections or 2
        max_outline_sections = document.outline.max_sections or 9
        min_section_words = document.section.min_units or 90
        default_section_words = document.section.default_units or 180
        max_section_words = document.section.max_units or 280
        max_section_revision_rounds = document.section.max_revision_rounds or 1
        final_retention_ratio = document.final.min_retention_ratio or 0.8

    document_output_limit = _positive_int(raw_document.get("max_document_output_tokens"), 0)
    if profile is not None:
        document_output_limit = profile.max_document_output_tokens or document_output_limit

    settings: dict[str, Any] = {
        "coverage_model": raw_document.get("coverage_model") or artifact_type,
        "length_profile": base_length_profile,
        "profile_source": profile_source,
        "profile_confidence": profile_confidence,
        "context_policy": contract.runtime.context_policy or "evidence_pack_v1",
        "quality_contract_ref": raw_document.get("quality_contract_ref")
        or contract.name
        or artifact_type,
        "finalization": finalization,
        "evidence_depth": evidence_depth,
        "min_outline_sections": min_outline_sections,
        "max_outline_sections": max_outline_sections,
        "min_section_words": min_section_words,
        "default_section_words": default_section_words,
        "max_section_words": max_section_words,
        "max_section_revision_rounds": max_section_revision_rounds,
        "max_document_output_tokens": document_output_limit,
        "final_retention_ratio": final_retention_ratio,
        "require_evidence_refs": (
            contract.quality_gates.require_evidence_refs
            if contract.quality_gates.raw
            else True
        ),
        "require_confidence_layer": contract.quality_gates.require_confidence_layer,
        "forbid_step_artifact_only_citations": (
            contract.quality_gates.forbid_step_artifact_only_citations
        ),
        "min_unique_sources_per_core_section": (
            contract.quality_gates.min_unique_sources_per_core_section
        ),
        "gate_enforcement": _gate_enforcement(raw_document.get("gate_enforcement")),
        "outline_artifact_type": contract.artifacts.outline_type
        or f"{artifact_type}.outline",
        "section_artifact_type": contract.artifacts.section_type
        or f"{artifact_type}.section",
        "final_artifact_type": contract.artifacts.final_type or artifact_type,
        "record_outline_ref": contract.workpad.record_outline_ref,
        "record_section_refs": contract.workpad.record_section_refs,
        "record_final_deliverable_ref": contract.workpad.record_final_deliverable_ref,
    }
    # PR1 boundary-stitch / PR2 running-context tuning knobs. Forwarded only when
    # the skill contract sets them — per length profile first, then the runtime
    # block — so unspecified knobs fall through to the SectionedAuthoringContract
    # defaults (running_context stays on, finalization unchanged, etc.).
    runtime_raw = dict(contract.runtime.raw or {})
    profile_raw = dict(profile.raw) if profile is not None else {}
    for knob in (
        "seam_context_chars",
        "finalize_model",
        "running_context",
        "running_context_window_k",
        "running_summary_max_tokens",
        "running_summary_model",
    ):
        if profile_raw.get(knob) is not None:
            settings[knob] = profile_raw[knob]
        elif runtime_raw.get(knob) is not None:
            settings[knob] = runtime_raw[knob]
    return settings


class SectionedLongformAuthor:
    """Outline, draft, record, and polish a longform document section by section."""

    def __init__(
        self,
        *,
        llm_facade: Any,
        platform: Any,
        artifact_type: str,
        step_id: str,
        capability_id: str,
        context_budget: Mapping[str, Any] | None = None,
        authoring_contract: Mapping[str, Any] | SectionedAuthoringContract | None = None,
        authoring_instructions: str = "",
        phase_event_sink: Callable[[Mapping[str, Any]], Any] | None = None,
        phase_checkpoint_sink: Callable[[Mapping[str, Any]], Any] | None = None,
        defer_intermediate_artifacts: bool = False,
        defer_final_artifact: bool = False,
    ) -> None:
        self._llm = llm_facade
        self._platform = platform
        self._artifact_type = artifact_type
        self._step_id = step_id
        self._capability_id = capability_id
        self._defer_intermediate_artifacts = defer_intermediate_artifacts
        self._defer_final_artifact = defer_final_artifact
        self._context_budget = dict(context_budget or {})
        self._contract = (
            authoring_contract
            if isinstance(authoring_contract, SectionedAuthoringContract)
            else SectionedAuthoringContract.from_mapping(authoring_contract)
        )
        self._authoring_instructions = str(authoring_instructions or "").strip()[:12000]
        if defer_intermediate_artifacts:
            self._contract = replace(
                self._contract,
                record_outline_ref=False,
                record_section_refs=False,
            )
        self._ledger = ArtifactLedger(platform)
        self._evidence = EvidencePackBuilder(platform, budget=context_budget)
        self._max_section_revision_rounds = _positive_int(
            self._context_budget.get("max_section_revision_rounds"),
            self._contract.max_section_revision_rounds,
        )
        # Output ceiling for content-bearing LLM calls (section drafts and
        # finalize rewrites). This is the run's budget-contract limit — the
        # platform sends it per run, sized to the tenant's model — NOT a
        # per-section length control: content length is governed by prompt
        # targets and the quality gate, and hitting this ceiling is a loud
        # truncation event, never a silent cut. None lets the platform apply
        # its own default.
        self._output_token_ceiling = (
            _positive_int(self._context_budget.get("max_output_tokens"), 0) or None
        )
        self._document_output_budget = DocumentOutputBudget.from_limits(
            self._context_budget,
            contract_limit=self._contract.max_document_output_tokens,
        )
        self._max_llm_stream_attempts = _positive_int(
            self._context_budget.get("llm_stream_max_attempts")
            or os.getenv(_LLM_STREAM_MAX_ATTEMPTS_ENV),
            2,
        )
        self._llm_stream_retry_backoff_seconds = _non_negative_float(
            self._context_budget.get("llm_stream_retry_backoff_seconds")
            or os.getenv(_LLM_STREAM_RETRY_BACKOFF_ENV),
            1.0,
        )
        self._phase_event_sink = phase_event_sink
        self._phase_checkpoint_sink = phase_checkpoint_sink
        self._llm_call_seq = 0
        self._tool_call_seq = 0

    def _skill_instruction_block(self) -> str:
        if not self._authoring_instructions:
            return ""
        return (
            "Skill authoring instructions (follow these when planning and writing):\n"
            f"{self._authoring_instructions}\n\n"
        )

    async def author(
        self,
        *,
        brief: Mapping[str, Any],
        upstream: Mapping[str, Any],
        workflow_id: str | None = None,
        thread_id: str | None = None,
        agent_id: str | None = None,
        resume_state: Mapping[str, Any] | None = None,
    ) -> SectionedAuthoringResult:
        state = dict(resume_state or {})
        await self._emit(
            "document.profile.selected",
            status="complete",
            length_profile=self._contract.length_profile,
            profile_source=self._contract.profile_source,
            profile_confidence=self._contract.profile_confidence,
            finalization=self._contract.finalization,
            evidence_depth=self._contract.evidence_depth,
            min_outline_sections=self._contract.min_outline_sections,
            max_outline_sections=self._contract.max_outline_sections,
            min_section_words=self._contract.min_section_words,
            default_section_words=self._contract.default_section_words,
            max_section_words=self._contract.max_section_words,
        )
        resume_outline = _section_plans_from_resume_state(state)
        drafts: list[SectionDraft] = []
        if resume_outline:
            outline = resume_outline
            length_profile = str(state.get("length_profile") or self._contract.length_profile)
            outline_ref = _mapping(state.get("outline_ref"))
            drafts = await self._resume_drafts_from_state(state, outline=outline)
            await self._emit(
                "document.sectioned_authoring.resumed",
                status="running",
                artifact_type=self._artifact_type,
                length_profile=length_profile,
                profile_source=self._contract.profile_source,
                section_count=len(outline),
                resumed_section_count=len(drafts),
            )
        else:
            await self._emit(
                "document.outline.started",
                status="running",
                artifact_type=self._artifact_type,
                length_profile=self._contract.length_profile,
                profile_source=self._contract.profile_source,
            )
            length_profile, outline = await self._build_outline(brief=brief, upstream=upstream)
            outline_ref = await self._record_outline(
                outline,
                length_profile=length_profile,
                workflow_id=workflow_id,
                thread_id=thread_id,
                agent_id=agent_id,
            )
            await self._emit(
                "document.outline.completed",
                status="complete",
                artifact_type=self._artifact_type,
                length_profile=length_profile,
                profile_source=self._contract.profile_source,
                section_count=len(outline),
                artifact_ref=outline_ref,
            )
            await self._checkpoint(
                current_phase="draft_sections",
                length_profile=length_profile,
                outline=[asdict(plan) for plan in outline],
                outline_ref=outline_ref,
                drafts=[],
            )
        degraded_sections: list[dict[str, Any]] = list(
            state.get("degraded_sections") if isinstance(state.get("degraded_sections"), list) else []
        )
        window_k = max(0, self._contract.running_context_window_k)
        running_summary = str(state.get("running_summary") or "")
        if self._contract.running_context and not running_summary and len(drafts) > window_k:
            # Resuming a checkpoint that predates running-summary state: rebuild it
            # by folding the sections already outside the recent-body window.
            for dropped in drafts[: len(drafts) - window_k]:
                running_summary = await self._fold_running_summary(running_summary, dropped)
        for index, plan in enumerate(outline[len(drafts) :], start=len(drafts) + 1):
            await self._emit(
                "document.section.started",
                status="running",
                section_id=plan.section_id,
                section_title=plan.title,
                section_index=index,
                length_profile=self._contract.length_profile,
            )
            evidence_call_id = self._next_tool_call_id("evidence-build")
            await self._emit(
                "agent.tool_call",
                tool_name="evidence.build",
                tool_call_id=evidence_call_id,
                status="running",
                section_id=plan.section_id,
                section_title=plan.title,
                section_index=index,
                query=plan.evidence_query or plan.title,
            )
            evidence_pack = await self._evidence.build(
                workflow_id=workflow_id,
                upstream=upstream,
                query=plan.evidence_query or plan.title,
                purpose=f"draft section {index}: {plan.title}",
                exclude_workpad_step_ids={self._step_id} if self._step_id else set(),
            )
            evidence_pack_input = evidence_pack.to_prompt_input()
            await self._emit(
                "agent.tool_result",
                tool_name="evidence.build",
                tool_call_id=evidence_call_id,
                status="complete",
                section_id=plan.section_id,
                section_title=plan.title,
                section_index=index,
                result_preview=(
                    f"{len(evidence_pack.items)} evidence items, "
                    f"{evidence_pack.total_chars} chars"
                ),
                evidence_item_count=len(evidence_pack.items),
                evidence_total_chars=evidence_pack.total_chars,
                warnings=list(evidence_pack.warnings),
            )
            await self._emit(
                "document.section.evidence_pack_built",
                status="complete",
                section_id=plan.section_id,
                section_title=plan.title,
                section_index=index,
                evidence_item_count=len(evidence_pack.items),
                evidence_total_chars=evidence_pack.total_chars,
                warnings=list(evidence_pack.warnings),
            )
            if evidence_pack.warnings or not evidence_pack.items:
                await self._emit(
                    "document.section.gap_detected",
                    status="incomplete",
                    section_id=plan.section_id,
                    section_title=plan.title,
                    section_index=index,
                    reasons=list(evidence_pack.warnings)
                    or ["empty_evidence_pack"],
                )
            running_context = (
                self._compose_running_context(drafts, running_summary)
                if self._contract.running_context
                else ""
            )
            markdown, output_truncated = await self._draft_section(
                brief=brief,
                plan=plan,
                section_index=index,
                previous=drafts,
                evidence_pack=evidence_pack_input,
                running_context=running_context,
                output_slots_remaining=len(outline) - index + 2,
            )
            if output_truncated:
                await self._emit(
                    "document.section.truncation_detected",
                    status="degraded",
                    section_id=plan.section_id,
                    section_title=plan.title,
                    section_index=index,
                    revision_round=0,
                )
            quality = _evaluate_section_quality(
                plan=plan,
                markdown=markdown,
                evidence_pack=evidence_pack_input,
                contract=self._contract,
                revision_rounds=0,
                output_truncated=output_truncated,
            )
            await self._emit(
                "document.section.quality_checked",
                # ponytail: use gate_failed (not failed) so A2A stream guards
                # that scan metadata.status do not treat a soft quality miss as
                # a terminal agent failure; degrade/revise still runs below.
                status="passed" if quality.passed else "gate_failed",
                section_id=plan.section_id,
                section_title=plan.title,
                section_index=index,
                quality=quality.to_metadata(),
            )
            revision_rounds = 0
            while (
                not quality.passed
                and revision_rounds < self._max_section_revision_rounds
            ):
                revision_rounds += 1
                markdown, output_truncated = await self._draft_section(
                    brief=brief,
                    plan=plan,
                    section_index=index,
                    previous=drafts,
                    evidence_pack=evidence_pack_input,
                    running_context=running_context,
                    revision_feedback=quality,
                    output_slots_remaining=len(outline) - index + 2,
                )
                if output_truncated:
                    await self._emit(
                        "document.section.truncation_detected",
                        status="degraded",
                        section_id=plan.section_id,
                        section_title=plan.title,
                        section_index=index,
                        revision_round=revision_rounds,
                    )
                quality = _evaluate_section_quality(
                    plan=plan,
                    markdown=markdown,
                    evidence_pack=evidence_pack_input,
                    contract=self._contract,
                    revision_rounds=revision_rounds,
                    output_truncated=output_truncated,
                )
                await self._emit(
                    "document.section.quality_checked",
                    status="passed" if quality.passed else "gate_failed",
                    section_id=plan.section_id,
                    section_title=plan.title,
                    section_index=index,
                    quality=quality.to_metadata(),
                )
            if not quality.passed:
                hard_failures = quality.hard_failures
                if (
                    hard_failures
                    or self._contract.gate_enforcement == _GATE_ENFORCEMENT_STRICT
                ):
                    raise RuntimeError(
                        "section_quality_gate_failed:"
                        f"{plan.section_id}:"
                        + ",".join(quality.failures)
                    )
                # Graceful degradation: soft, evidence-bound gate failures must
                # not dead-end the plan. Record the best-effort section with an
                # explicit gap marker and continue so the deliverable completes.
                markdown = _append_quality_gap_note(markdown, quality.soft_failures)
                quality = SectionQualityGateResult(
                    failures=quality.failures,
                    information_units=quality.information_units,
                    citation_count=quality.citation_count,
                    evidence_item_count=quality.evidence_item_count,
                    revision_rounds=quality.revision_rounds,
                    unique_sources_available=quality.unique_sources_available,
                    unique_sources_cited=quality.unique_sources_cited,
                    degraded=True,
                )
                degraded_sections.append(
                    {
                        "section_id": plan.section_id,
                        "section_title": plan.title,
                        "failures": list(quality.soft_failures),
                    }
                )
                await self._emit(
                    "document.section.quality_degraded",
                    status="degraded",
                    section_id=plan.section_id,
                    section_title=plan.title,
                    section_index=index,
                    quality=quality.to_metadata(),
                )
            artifact_ref = await self._record_section(
                plan,
                markdown,
                index=index,
                workflow_id=workflow_id,
                thread_id=thread_id,
                agent_id=agent_id,
                quality=quality,
            )
            await self._emit(
                "document.section.completed",
                status="complete",
                section_id=plan.section_id,
                section_title=plan.title,
                section_index=index,
                artifact_ref=artifact_ref,
                quality=quality.to_metadata(),
            )
            drafts.append(
                SectionDraft(
                    plan=plan,
                    markdown=markdown,
                    artifact_ref=artifact_ref,
                    quality=quality.to_metadata(),
                )
            )
            if self._contract.running_context and len(drafts) > window_k:
                # The section that just left the recent-body window is folded once
                # into the running summary (incremental fold).
                dropped = drafts[len(drafts) - window_k - 1]
                running_summary = await self._fold_running_summary(
                    running_summary,
                    dropped,
                    output_slots_remaining=len(outline) - index + 2,
                )
            await self._checkpoint(
                current_phase="draft_sections",
                length_profile=length_profile,
                outline=[asdict(item) for item in outline],
                outline_ref=outline_ref,
                drafts=[_draft_resume_record(draft) for draft in drafts],
                degraded_sections=degraded_sections,
                running_summary=running_summary,
            )

        await self._emit(
            "document.final.polish_started",
            status="running",
            section_count=len(drafts),
            length_profile=self._contract.length_profile,
            finalization=self._contract.finalization,
        )
        final_markdown = await self._polish_final(brief=brief, drafts=drafts)
        # Fail-closed document agents publish the final deliverable through
        # the platform's completed-output materializer. Keeping the SDK-side
        # artifact in memory until that boundary avoids a durable final
        # artifact surviving cancellation between authoring and transport
        # success. Legacy callers retain the original ledger behavior.
        final_ref: dict[str, Any] = {}
        if not (
            self._defer_intermediate_artifacts or self._defer_final_artifact
        ):
            final_ref = await self._record_final(
                final_markdown,
                workflow_id=workflow_id,
                thread_id=thread_id,
                agent_id=agent_id,
            )
        await self._checkpoint(
            current_phase="finalize",
            length_profile=length_profile,
            outline=[asdict(item) for item in outline],
            outline_ref=outline_ref,
            drafts=[_draft_resume_record(draft) for draft in drafts],
            final_ref=final_ref,
            narrative=final_markdown,
            degraded_sections=degraded_sections,
        )
        await self._emit(
            "document.final.created",
            status="complete",
            artifact_ref=final_ref,
            artifact_committed=bool(final_ref),
            final_commit_deferred=bool(
                self._defer_intermediate_artifacts or self._defer_final_artifact
            ),
            section_count=len(drafts),
            length_profile=self._contract.length_profile,
            finalization=self._contract.finalization,
        )
        artifact_refs = [{"role": "outline", **dict(outline_ref)}] if outline_ref else []
        artifact_refs.extend(
            {
                "role": "section_draft",
                "section_id": draft.plan.section_id,
                **dict(draft.artifact_ref),
            }
            for draft in drafts
            if draft.artifact_ref
        )
        if final_ref:
            artifact_refs.append({"role": "final_deliverable", **dict(final_ref)})
        return SectionedAuthoringResult(
            markdown=final_markdown,
            length_profile=length_profile,
            outline=outline,
            drafts=tuple(drafts),
            ledger={
                "enabled": True,
                "status": "recorded",
                "outline_ref": outline_ref,
                "final_ref": final_ref,
                "length_profile": length_profile,
                "profile_source": self._contract.profile_source,
                "profile_confidence": self._contract.profile_confidence,
                "finalization": self._contract.finalization,
                "section_count": len(outline),
                "created_count": len(artifact_refs),
                "artifact_refs": artifact_refs,
                "degraded": bool(degraded_sections),
                "degraded_sections": degraded_sections,
            },
        )

    async def _emit(self, event: str, **metadata: Any) -> None:
        sink = self._phase_event_sink
        if sink is None:
            return
        payload = {
            "event": event,
            "runtime_phase": "sectioned_authoring",
            "authoring_strategy": "sectioned_longform",
            "artifact_type": self._artifact_type,
            "capability_id": self._capability_id,
            **metadata,
        }
        result = sink(payload)
        if inspect.isawaitable(result):
            await result

    async def _checkpoint(self, **payload: Any) -> None:
        sink = self._phase_checkpoint_sink
        if sink is None:
            return
        result = sink(
            {
                "authoring_strategy": "sectioned_longform",
                "artifact_type": self._artifact_type,
                "capability_id": self._capability_id,
                **payload,
            }
        )
        if inspect.isawaitable(result):
            await result

    async def _resume_drafts_from_state(
        self,
        state: Mapping[str, Any],
        *,
        outline: tuple[SectionPlan, ...],
    ) -> list[SectionDraft]:
        raw_drafts = state.get("drafts")
        if not isinstance(raw_drafts, list):
            return []
        drafts: list[SectionDraft] = []
        for index, raw in enumerate(raw_drafts, start=1):
            if not isinstance(raw, Mapping):
                continue
            plan = (
                _section_plan_from_mapping(raw.get("plan"))
                or (outline[index - 1] if index - 1 < len(outline) else None)
            )
            if plan is None:
                continue
            markdown = str(raw.get("markdown") or "")
            artifact_ref = _mapping(raw.get("artifact_ref"))
            if not markdown:
                markdown = await self._read_resume_artifact_text(artifact_ref)
            if not markdown.strip():
                raise RuntimeError(
                    "sectioned_resume_artifact_unavailable:"
                    f"{plan.section_id or index}"
                )
            drafts.append(
                SectionDraft(
                    plan=plan,
                    markdown=markdown,
                    artifact_ref=artifact_ref,
                    quality=_mapping(raw.get("quality")),
                )
            )
        return drafts

    async def _read_resume_artifact_text(self, artifact_ref: Mapping[str, Any]) -> str:
        artifacts = getattr(self._platform, "artifacts", None)
        read_text = getattr(artifacts, "read_text", None)
        if not callable(read_text):
            return ""
        artifact_id = _artifact_id_from_ref(artifact_ref)
        if not artifact_id:
            return ""
        return str(
            await read_text(
                artifact_id,
                mode="chunks",
                purpose="resume sectioned authoring draft",
                max_bytes=64000,
            )
            or ""
        )

    def _next_llm_call_id(self, purpose: str) -> str:
        self._llm_call_seq += 1
        return f"llm-{_slug(purpose, fallback='call')}-{self._llm_call_seq:04d}"

    def _next_tool_call_id(self, tool_name: str) -> str:
        self._tool_call_seq += 1
        return f"tool-{_slug(tool_name, fallback='call')}-{self._tool_call_seq:04d}"

    async def _stream_llm_text(
        self,
        *,
        purpose: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_output_tokens: int | None,
        output_slots_remaining: int = 1,
        model: str | None = None,
        section: SectionPlan | None = None,
        section_index: int | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> _StreamedLlmText:
        allocated_output_tokens = self._document_output_budget.reserve(
            max_output_tokens,
            slots_remaining=output_slots_remaining,
        )
        call_id = self._next_llm_call_id(purpose)
        base_metadata = {
            "call_id": call_id,
            "llm_purpose": purpose,
            "max_output_tokens": allocated_output_tokens,
            **self._document_output_budget.metadata(),
            **(dict(extra_metadata or {})),
        }
        if section is not None:
            base_metadata.update(
                {
                    "section_id": section.section_id,
                    "section_title": section.title,
                    "section_index": section_index,
                }
            )
        await self._emit("agent.llm_call.started", **base_metadata, status="running")
        max_attempts = max(1, self._max_llm_stream_attempts)
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            chunks: list[str] = []
            completed_result: Mapping[str, Any] | None = None
            saw_delta = False
            attempt_metadata = {
                **base_metadata,
                "attempt": attempt,
                "max_attempts": max_attempts,
            }
            try:
                stream_text = getattr(self._llm, "stream_text", None)
                if not callable(stream_text):
                    result = await self._llm.chat(
                        messages=messages,
                        temperature=temperature,
                        max_output_tokens=allocated_output_tokens,
                        model=model, **{"reasoning_mode": "disabled"},
                    )
                    content = str(result.get("content") or "")
                    if content:
                        saw_delta = True
                        chunks.append(content)
                        await self._emit(
                            "agent.llm_call.delta",
                            **attempt_metadata,
                            text_delta=content,
                            preview=_preview(content, limit=240),
                            chars_in_chunk=len(content),
                            chars_total=len(content),
                        )
                    completed_result = result if isinstance(result, Mapping) else {}
                else:
                    async for event in stream_text(
                        messages,
                        temperature=temperature,
                        max_output_tokens=allocated_output_tokens,
                        model=model, **{"reasoning_mode": "disabled"},
                    ):
                        delta = _llm_stream_event_delta(event)
                        if delta:
                            saw_delta = True
                            chunks.append(delta)
                            await self._emit(
                                "agent.llm_call.delta",
                                **attempt_metadata,
                                text_delta=delta,
                                preview=_preview(delta, limit=240),
                                chars_in_chunk=len(delta),
                                chars_total=sum(len(chunk) for chunk in chunks),
                            )
                        result = _llm_stream_event_result(event)
                        if result is not None:
                            completed_result = result
            except Exception as exc:
                last_exc = exc
                chars_total = sum(len(chunk) for chunk in chunks)
                if attempt < max_attempts and _is_transient_llm_stream_error(exc):
                    await self._emit(
                        "agent.llm_call.retrying",
                        **attempt_metadata,
                        status="retrying",
                        error=type(exc).__name__,
                        message=str(exc),
                        chars_total=chars_total,
                        next_attempt=attempt + 1,
                    )
                    delay = self._llm_stream_retry_backoff_seconds * (2 ** (attempt - 1))
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue
                await self._emit(
                    "agent.llm_call.failed",
                    **attempt_metadata,
                    status="failed",
                    error=type(exc).__name__,
                    message=str(exc),
                    chars_total=chars_total,
                )
                raise

            content = "".join(chunks)
            if not content and completed_result is not None:
                final_content = str(completed_result.get("content") or "")
                if final_content:
                    content = final_content
                    if not saw_delta:
                        await self._emit(
                            "agent.llm_call.delta",
                            **attempt_metadata,
                            text_delta=final_content,
                            preview=_preview(final_content, limit=240),
                            chars_in_chunk=len(final_content),
                            chars_total=len(final_content),
                        )
            finish_reason = _finish_reason_of(completed_result)
            truncated = finish_reason in _TRUNCATION_FINISH_REASONS
            await self._emit(
                "agent.llm_call.completed",
                **attempt_metadata,
                status="complete",
                chars_total=len(content),
                finish_reason=finish_reason,
                truncated=truncated,
                usage_metadata=(
                    dict(completed_result.get("usage_metadata") or {})
                    if isinstance(completed_result, Mapping)
                    else {}
                ),
            )
            return _StreamedLlmText(content, finish_reason, truncated)
        if last_exc is not None:
            raise last_exc
        return _StreamedLlmText("")

    async def _build_outline(
        self,
        *,
        brief: Mapping[str, Any],
        upstream: Mapping[str, Any],
    ) -> tuple[str, tuple[SectionPlan, ...]]:
        prompt = (
            "Design a longform document outline for the selected length profile.\n"
            f"Coverage model: {self._contract.coverage_model}.\n"
            f"Selected length profile: {self._contract.length_profile}.\n"
            f"Evidence depth: {self._contract.evidence_depth}.\n"
            f"{self._skill_instruction_block()}"
            "The length profile is already selected by the runtime. Do not "
            "change it. Plan within the declared section and unit bounds. "
            f"Return {self._contract.min_outline_sections}-"
            f"{self._contract.max_outline_sections} sections. "
            "Each section must have a focused evidence query and an appropriate "
            "minimum information-unit target.\n\n"
            f"Original task:\n{_json_block(brief, limit=8000)}\n\n"
            f"Available upstream/workpad refs:\n{_json_block(upstream, limit=12000)}"
        )
        call_id = self._next_llm_call_id("build_outline")
        await self._emit(
            "agent.llm_call.started",
            call_id=call_id,
            llm_purpose="build_outline",
            status="running",
            max_outline_sections=self._contract.max_outline_sections,
            length_profile=self._contract.length_profile,
        )
        try:
            result = await self._llm.structured(
                messages=[{"role": "user", "content": prompt}],
                output_schema=_outline_schema(self._contract),
                temperature=0.2,
            )
        except Exception as exc:
            await self._emit(
                "agent.llm_call.failed",
                call_id=call_id,
                llm_purpose="build_outline",
                status="failed",
                error=type(exc).__name__,
                message=str(exc),
            )
            raise
        structured = result.get("structured") if isinstance(result, Mapping) else None
        raw_sections = structured.get("sections") if isinstance(structured, Mapping) else None
        length_profile = self._contract.length_profile
        plans: list[SectionPlan] = []
        if isinstance(raw_sections, list):
            for index, raw in enumerate(raw_sections, start=1):
                if not isinstance(raw, Mapping):
                    continue
                title = str(raw.get("title") or "").strip()
                if not title:
                    continue
                plans.append(
                    SectionPlan(
                        section_id=_slug(raw.get("section_id") or title, fallback=f"section-{index}"),
                        title=title,
                        objective=str(raw.get("objective") or "").strip(),
                        evidence_query=str(raw.get("evidence_query") or title).strip(),
                        min_words=_clamp_int(
                            _int(
                                raw.get("min_words"),
                                self._contract.default_section_words,
                            ),
                            minimum=self._contract.min_section_words,
                            maximum=self._contract.max_section_words,
                        ),
                    )
                )
        await self._emit(
            "agent.llm_call.completed",
            call_id=call_id,
            llm_purpose="build_outline",
            status="complete",
            section_count=len(plans),
        )
        if len(plans) >= self._contract.min_outline_sections:
            return (
                length_profile or "medium",
                tuple(plans[: self._contract.max_outline_sections]),
            )
        fallback = [
            SectionPlan(
                section_id="overview",
                title="Overview",
                objective="Summarize the requested document scope and key points.",
                evidence_query=str(brief.get("title") or brief.get("goal") or "overview"),
                min_words=self._contract.default_section_words,
            ),
            SectionPlan(
                section_id="details",
                title="Details",
                objective="Develop the main body from the available context and evidence.",
                evidence_query="document details",
                min_words=self._contract.default_section_words,
            ),
            SectionPlan(
                section_id="next-steps",
                title="Next Steps",
                objective="Capture follow-up actions, decisions, or open questions.",
                evidence_query="next steps open questions",
                min_words=self._contract.default_section_words,
            ),
        ]
        while len(fallback) < self._contract.min_outline_sections:
            index = len(fallback) + 1
            fallback.append(
                SectionPlan(
                    section_id=f"section-{index}",
                    title=f"Section {index}",
                    objective="Develop an additional required section from the available context.",
                    evidence_query=f"section {index} supporting evidence",
                    min_words=self._contract.default_section_words,
                )
            )
        return (
            length_profile or "medium",
            tuple(fallback[: max(self._contract.min_outline_sections, 1)]),
        )

    def _compose_running_context(
        self,
        drafts: list[SectionDraft],
        running_summary: str,
    ) -> str:
        """Build the bounded "document so far" context for the next section.

        Recent sections (last ``running_context_window_k``) are included verbatim
        so the most relevant continuity cues are exact; earlier sections are
        represented by the rolling summary. The footprint is bounded by the window
        plus the summary cap, independent of total document length.
        """
        window_k = max(0, self._contract.running_context_window_k)
        window = drafts[-window_k:] if window_k else []
        parts: list[str] = []
        summary = running_summary.strip()
        if summary:
            parts.append(
                "Earlier sections (running summary — covered points, key claims, "
                f"terminology):\n{summary}"
            )
        for draft in window:
            body = draft.markdown.strip()
            if body:
                parts.append(f"Recent section — {draft.plan.title}:\n{body}")
        return "\n\n".join(parts)

    async def _fold_running_summary(
        self,
        prior_summary: str,
        dropped: SectionDraft,
        output_slots_remaining: int = 1,
    ) -> str:
        """Incrementally fold one section into the rolling running summary."""
        prompt = (
            "Maintain a running summary of a document being written section by "
            "section. Fold the new section into the prior running summary. Keep it "
            "factual and structural — covered points, key claims and numbers, "
            "defined terms, and still-open threads. Do not add information that is "
            "not in the sections and do not editorialize. Be concise.\n\n"
            f"Prior running summary:\n{prior_summary.strip() or '(none yet)'}\n\n"
            f"New section to fold in — {dropped.plan.title}:\n"
            f"{dropped.markdown.strip()[:16000]}"
        )
        # A summary cut at its token budget is still a usable lossy summary,
        # so truncation is accepted here.
        streamed = await self._stream_llm_text(
            purpose="running_summary",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_output_tokens=max(1, self._contract.running_summary_max_tokens),
            output_slots_remaining=output_slots_remaining,
            model=self._contract.running_summary_model or None,
            extra_metadata={"section_id": dropped.plan.section_id},
        )
        summary = streamed.text.strip()
        if not summary:
            return prior_summary
        await self._emit(
            "document.running_summary.updated",
            status="complete",
            section_id=dropped.plan.section_id,
            summary_chars=len(summary),
        )
        return summary

    async def _draft_section(
        self,
        *,
        brief: Mapping[str, Any],
        plan: SectionPlan,
        section_index: int | None,
        previous: list[SectionDraft],
        evidence_pack: Mapping[str, Any],
        running_context: str = "",
        revision_feedback: SectionQualityGateResult | None = None,
        output_slots_remaining: int = 1,
    ) -> tuple[str, bool]:
        """Draft one section; returns ``(markdown, output_truncated)``."""
        previous_index = [
            {
                "section_id": draft.plan.section_id,
                "title": draft.plan.title,
                "artifact_ref": draft.artifact_ref.get("artifact_ref"),
            }
            for draft in previous
        ]
        story_block = ""
        if running_context.strip():
            story_block = (
                "Document so far (maintain continuity; do not repeat what is already "
                f"covered):\n{running_context.strip()}\n\n"
            )
        prompt = (
            "Write exactly this document section in Markdown.\n"
            "Use the original task, the document-so-far context, the prior section "
            "ledger, and the bounded evidence pack. Continue naturally from what "
            "earlier sections established and avoid repeating their content. "
            "Do not include process notes. Cite artifact refs or source refs when evidence is used. "
            f"Write at least {plan.min_words} substantive information units.\n\n"
            f"{self._skill_instruction_block()}"
            f"Original task:\n{_json_block(brief, limit=8000)}\n\n"
            f"Section plan:\n{_json_block(asdict(plan), limit=4000)}\n\n"
            f"{story_block}"
            f"Prior section refs:\n{_json_block(previous_index, limit=4000)}\n\n"
            f"Evidence pack:\n{_json_block(evidence_pack, limit=24000)}"
        )
        if revision_feedback is not None:
            prompt += (
                "\n\nSection quality gate failed. Rewrite the same section only, "
                "fixing these deterministic failures without dropping evidence:\n"
                f"{_json_block(revision_feedback.to_metadata(), limit=4000)}"
            )
            if "output_truncated" in revision_feedback.failures:
                prompt += (
                    "\n\nThe previous draft was cut off at the output length limit "
                    "before it could finish. Write a tighter version of this "
                    "section that reaches a proper conclusion: prioritise the most "
                    "important content, compress or drop secondary detail, and "
                    "always end with a complete sentence."
                )
        streamed = await self._stream_llm_text(
            purpose="revise_section" if revision_feedback is not None else "draft_section",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_output_tokens=self._output_token_ceiling,
            output_slots_remaining=output_slots_remaining,
            section=plan,
            section_index=section_index,
            extra_metadata={
                "revision_round": revision_feedback.revision_rounds + 1
                if revision_feedback is not None
                else 0,
            },
        )
        content = streamed.text.strip()
        if not content:
            return "", streamed.truncated
        return _ensure_section_heading(content, plan.title), streamed.truncated

    async def _bounded_for_final_prompt(
        self,
        text: str,
        *,
        limit: int,
        phase: str,
        seam_index: int | None = None,
    ) -> str:
        """Bound prompt input, emitting a warning event when truncation occurs.

        Finalize prompts still cap their input, but truncation is never silent:
        a ``document.final.truncation_warning`` event records the dropped span so
        a degraded long-document finalize is observable rather than hidden.
        """
        if len(text) <= limit:
            return text
        metadata: dict[str, Any] = {
            "status": "degraded",
            "phase": phase,
            "input_chars": len(text),
            "limit": limit,
            "dropped_chars": len(text) - limit,
        }
        if seam_index is not None:
            metadata["cluster_index"] = seam_index
        await self._emit("document.final.truncation_warning", **metadata)
        return text[:limit]

    async def _polish_final(
        self,
        *,
        brief: Mapping[str, Any],
        drafts: list[SectionDraft],
    ) -> str:
        combined = _join_markdown(draft.markdown for draft in drafts)
        if self._contract.finalization == "boundary_stitch":
            return await self._boundary_stitch_final(
                brief=brief,
                drafts=drafts,
                combined=combined,
            )
        if self._contract.finalization == "progressive_section_merge":
            return await self._progressive_merge_final(
                brief=brief,
                drafts=drafts,
                combined=combined,
            )
        if self._contract.finalization != "single_polish":
            # Contracts built via from_mapping cannot reach here; this guards
            # direct dataclass construction so an unknown mode is at least
            # loud before the single_polish fallback runs.
            await self._emit(
                "document.finalize.mode_fallback",
                status="degraded",
                requested_mode=self._contract.finalization,
                effective_mode="single_polish",
            )
        return await self._single_polish_final(
            brief=brief,
            drafts=drafts,
            combined=combined,
        )

    async def _boundary_stitch_final(
        self,
        *,
        brief: Mapping[str, Any],
        drafts: list[SectionDraft],
        combined: str,
    ) -> str:
        """Smooth seams between sections without rewriting their bodies.

        Section bodies are preserved verbatim; the LLM only produces a short
        transition between adjacent sections. The prompt footprint is O(number
        of seams) and independent of total document length, so this path never
        truncates and cannot drop section content.
        """
        bodies = [draft.markdown.strip() for draft in drafts if draft.markdown.strip()]
        if len(bodies) <= 1:
            return combined
        if self._document_output_budget.remaining_tokens == 0:
            await self._emit(
                "document.final.output_budget_exhausted",
                status="degraded",
                **self._document_output_budget.metadata(),
            )
            return combined
        finalize_model = self._contract.finalize_model or None
        limit = max(200, self._contract.seam_context_chars)

        async def _bridge(seam_index: int) -> str:
            before = bodies[seam_index]
            after = bodies[seam_index + 1]
            await self._emit(
                "document.final.seam_stitch_started",
                status="running",
                seam_index=seam_index + 1,
                seam_count=len(bodies) - 1,
                length_profile=self._contract.length_profile,
            )
            prompt = (
                "You are smoothing the seam between two adjacent sections of one "
                "Markdown document. Write ONLY a short transition of at most two "
                "sentences (no heading, no list) that leads from the first section "
                "into the second. Do not repeat or summarize their content and do "
                "not introduce new facts or citations. If the flow already reads "
                "smoothly, return an empty string.\n\n"
                f"Original task:\n{_json_block(brief, limit=4000)}\n\n"
                f"End of preceding section:\n{before[-limit:]}\n\n"
                f"Start of following section:\n{after[:limit]}"
            )
            streamed = await self._stream_llm_text(
                purpose="seam_stitch",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_output_tokens=200,
                output_slots_remaining=len(bodies) - seam_index - 1,
                model=finalize_model,
                extra_metadata={"seam_index": seam_index + 1},
            )
            bridge = streamed.text.strip()
            if streamed.truncated:
                # A cut-off transition reads worse than a plain join; sections
                # stand on their own, so drop the bridge entirely.
                bridge = ""
            await self._emit(
                "document.final.seam_stitch_completed",
                status="degraded" if streamed.truncated else "complete",
                seam_index=seam_index + 1,
                bridged=bool(bridge),
                truncated=streamed.truncated,
                length_profile=self._contract.length_profile,
            )
            return bridge

        bridges = await asyncio.gather(
            *(_bridge(seam_index) for seam_index in range(len(bodies) - 1))
        )
        parts: list[str] = [bodies[0]]
        for index in range(1, len(bodies)):
            bridge = bridges[index - 1]
            if bridge:
                parts.append(bridge)
            parts.append(bodies[index])
        return _join_markdown(parts)

    async def _single_polish_final(
        self,
        *,
        brief: Mapping[str, Any],
        drafts: list[SectionDraft],
        combined: str,
    ) -> str:
        refs = [
            {
                "section_id": draft.plan.section_id,
                "title": draft.plan.title,
                "artifact_ref": draft.artifact_ref.get("artifact_ref"),
            }
            for draft in drafts
        ]
        draft_sections = await self._bounded_for_final_prompt(
            combined,
            limit=48000,
            phase="single_polish",
        )
        prompt = (
            "Polish the concatenated sections into one coherent final Markdown deliverable. "
            "Preserve factual claims, source refs, and section substance. "
            "Improve transitions and remove repetition without shortening materially.\n\n"
            f"{self._skill_instruction_block()}"
            f"Original task:\n{_json_block(brief, limit=8000)}\n\n"
            f"Section artifact refs:\n{_json_block(refs, limit=6000)}\n\n"
            f"Draft sections:\n{draft_sections}"
        )
        if self._document_output_budget.remaining_tokens == 0:
            await self._emit(
                "document.final.output_budget_exhausted",
                status="degraded",
                **self._document_output_budget.metadata(),
            )
            return combined
        streamed = await self._stream_llm_text(
            purpose="final_polish",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_output_tokens=self._output_token_ceiling,
            model=self._contract.finalize_model or None,
            extra_metadata={"section_count": len(drafts)},
        )
        if streamed.truncated:
            # A rewrite cut at the output limit ends mid-document. The combined
            # sections are complete, so they always beat a truncated polish.
            # (The retention guard below cannot catch this case: a rewrite that
            # EXPANDED the drafts before being cut still passes the shrinkage
            # check — that is exactly how a half-written deliverable shipped as
            # a success in prod.)
            await self._emit(
                "document.final.truncation_warning",
                status="degraded",
                phase="single_polish_output",
                finish_reason=streamed.finish_reason,
                polished_chars=len(streamed.text),
            )
            return combined
        polished = streamed.text.strip()
        if not polished:
            return combined
        combined_units = _information_units(combined)
        polished_units = _information_units(polished)
        if (
            combined_units > 0
            and polished_units < int(combined_units * self._contract.final_retention_ratio)
        ):
            return combined
        return polished

    async def _progressive_merge_final(
        self,
        *,
        brief: Mapping[str, Any],
        drafts: list[SectionDraft],
        combined: str,
    ) -> str:
        if len(drafts) <= 3:
            return await self._single_polish_final(
                brief=brief,
                drafts=drafts,
                combined=combined,
            )

        if self._document_output_budget.remaining_tokens == 0:
            await self._emit(
                "document.final.output_budget_exhausted",
                status="degraded",
                **self._document_output_budget.metadata(),
            )
            return combined
        merged_clusters: list[str] = []
        cluster_size = 4
        for cluster_index, start in enumerate(range(0, len(drafts), cluster_size), start=1):
            cluster = drafts[start : start + cluster_size]
            cluster_markdown = _join_markdown(draft.markdown for draft in cluster)
            cluster_refs = [
                {
                    "section_id": draft.plan.section_id,
                    "title": draft.plan.title,
                    "artifact_ref": draft.artifact_ref.get("artifact_ref"),
                }
                for draft in cluster
            ]
            await self._emit(
                "document.final.merge_cluster_started",
                status="running",
                cluster_index=cluster_index,
                section_count=len(cluster),
                length_profile=self._contract.length_profile,
            )
            cluster_sections = await self._bounded_for_final_prompt(
                cluster_markdown,
                limit=36000,
                phase="merge_cluster",
                seam_index=cluster_index,
            )
            prompt = (
                "Merge this cluster of adjacent report sections into one "
                "coherent Markdown chapter block. Preserve all factual claims, "
                "source refs, headings, and confidence markers. Reduce only "
                "clear repetition.\n\n"
                f"{self._skill_instruction_block()}"
                f"Original task:\n{_json_block(brief, limit=6000)}\n\n"
                f"Cluster section refs:\n{_json_block(cluster_refs, limit=5000)}\n\n"
                f"Cluster draft sections:\n{cluster_sections}"
            )
            streamed = await self._stream_llm_text(
                purpose="merge_cluster",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_output_tokens=self._output_token_ceiling,
                model=self._contract.finalize_model or None,
                extra_metadata={
                    "cluster_index": cluster_index,
                    "section_count": len(cluster),
                },
            )
            if streamed.truncated:
                # Same rule as single_polish: complete originals beat a
                # merge that was cut mid-cluster.
                await self._emit(
                    "document.final.truncation_warning",
                    status="degraded",
                    phase="merge_cluster_output",
                    cluster_index=cluster_index,
                    finish_reason=streamed.finish_reason,
                )
                merged = cluster_markdown
            else:
                merged = streamed.text.strip() or cluster_markdown
            if _information_units(merged) < int(
                _information_units(cluster_markdown)
                * self._contract.final_retention_ratio
            ):
                merged = cluster_markdown
            merged_clusters.append(merged)
            await self._emit(
                "document.final.merge_cluster_completed",
                status="complete",
                cluster_index=cluster_index,
                section_count=len(cluster),
                length_profile=self._contract.length_profile,
            )

        cluster_drafts = [
            SectionDraft(
                plan=SectionPlan(
                    section_id=f"merged-cluster-{index}",
                    title=f"Merged Cluster {index}",
                    min_words=1,
                ),
                markdown=markdown,
            )
            for index, markdown in enumerate(merged_clusters, start=1)
        ]
        return await self._single_polish_final(
            brief=brief,
            drafts=cluster_drafts,
            combined=_join_markdown(merged_clusters),
        )

    async def _record_outline(
        self,
        outline: tuple[SectionPlan, ...],
        *,
        length_profile: str,
        workflow_id: str | None,
        thread_id: str | None,
        agent_id: str | None,
    ) -> dict[str, Any]:
        if not self._contract.record_outline_ref:
            return {}
        if not self._ledger.is_available:
            return {"available": False, "error": "artifact_ledger_unavailable"}
        content = {
            "artifact_type": self._artifact_type,
            "authoring_strategy": "sectioned_longform",
            "authoring_contract": asdict(self._contract),
            "length_profile": length_profile,
            "sections": [asdict(plan) for plan in outline],
        }
        result = await self._record_artifact_with_events(
            artifact_type=self._contract.outline_artifact_type
            or f"{self._artifact_type}.outline",
            role="outline",
            content=content,
            kind="outline",
            title=f"{self._artifact_type} outline",
            content_type="application/json",
            summary=f"{self._artifact_type} outline with {len(outline)} sections",
            workflow_id=workflow_id,
            thread_id=thread_id,
            agent_id=agent_id,
            metadata={
                "role": "outline",
                "authoring_strategy": "sectioned_longform",
                "capability_id": self._capability_id,
                "authoring_contract": asdict(self._contract),
                "length_profile": length_profile,
            },
            workpad_metadata={
                "section_count": len(outline),
                "length_profile": length_profile,
                "authoring_strategy": "sectioned_longform",
            },
        )
        return dict(result.get("artifact") or {})

    async def _record_section(
        self,
        plan: SectionPlan,
        markdown: str,
        *,
        index: int,
        workflow_id: str | None,
        thread_id: str | None,
        agent_id: str | None,
        quality: SectionQualityGateResult,
    ) -> dict[str, Any]:
        if not self._contract.record_section_refs:
            return {}
        if not self._ledger.is_available:
            return {"available": False, "error": "artifact_ledger_unavailable"}
        result = await self._record_artifact_with_events(
            artifact_type=self._contract.section_artifact_type
            or f"{self._artifact_type}.section",
            role="section_draft",
            content=markdown,
            kind="section_draft",
            title=plan.title,
            content_type="text/markdown",
            summary=_preview(markdown),
            workflow_id=workflow_id,
            thread_id=thread_id,
            agent_id=agent_id,
            metadata={
                "role": "section_draft",
                "section_id": plan.section_id,
                "section_title": plan.title,
                "section_index": index,
                "authoring_strategy": "sectioned_longform",
                "capability_id": self._capability_id,
                "quality": quality.to_metadata(),
            },
            workpad_metadata={
                "section_id": plan.section_id,
                "section_index": index,
                "authoring_strategy": "sectioned_longform",
                "quality": quality.to_metadata(),
            },
        )
        return dict(result.get("artifact") or {})

    async def _record_final(
        self,
        markdown: str,
        *,
        workflow_id: str | None,
        thread_id: str | None,
        agent_id: str | None,
    ) -> dict[str, Any]:
        if not self._contract.record_final_deliverable_ref:
            return {}
        if not self._ledger.is_available:
            return {"available": False, "error": "artifact_ledger_unavailable"}
        result = await self._record_artifact_with_events(
            artifact_type=self._contract.final_artifact_type or self._artifact_type,
            role="final_deliverable",
            content=markdown,
            kind="final_deliverable",
            title=f"{self._artifact_type} final deliverable",
            content_type="text/markdown",
            summary=_preview(markdown),
            workflow_id=workflow_id,
            thread_id=thread_id,
            agent_id=agent_id,
            metadata={
                "role": "final_deliverable",
                "authoring_strategy": "sectioned_longform",
                "capability_id": self._capability_id,
                "authoring_contract": asdict(self._contract),
            },
            workpad_metadata={
                "role": "final_deliverable",
                "authoring_strategy": "sectioned_longform",
            },
        )
        artifact = dict(result.get("artifact") or {})
        artifact_ref = str(artifact.get("artifact_ref") or "")
        if artifact_ref:
            final_result = await self._ledger.set_final_deliverable(
                artifact_ref,
                workflow_id=workflow_id,
                step_id=self._step_id,
                metadata={
                    "authoring_strategy": "sectioned_longform",
                    "capability_id": self._capability_id,
                },
            )
            if isinstance(final_result, Mapping) and final_result.get("available", True) is False:
                raise RuntimeError(
                    "workpad_final_deliverable_failed:"
                    f"{final_result.get('error') or 'workpad_final_unavailable'}"
                )
        return artifact

    async def _record_artifact_with_events(
        self,
        *,
        artifact_type: str,
        role: str,
        content: Any,
        kind: str,
        title: str,
        content_type: str,
        summary: str,
        workflow_id: str | None,
        thread_id: str | None,
        agent_id: str | None,
        metadata: Mapping[str, Any],
        workpad_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        tool_call_id = self._next_tool_call_id("artifact-write")
        await self._emit(
            "agent.tool_call",
            tool_name="artifact.write",
            tool_call_id=tool_call_id,
            status="running",
            artifact_type=artifact_type,
            role=role,
            title=title,
        )
        try:
            result = await _create_and_record_strict(
                self._ledger,
                artifact_type=artifact_type,
                content=content,
                kind=kind,
                title=title,
                content_type=content_type,
                summary=summary,
                workflow_id=workflow_id,
                thread_id=thread_id,
                step_id=self._step_id,
                agent_id=agent_id,
                capability_id=self._capability_id,
                metadata=dict(metadata),
                workpad_metadata=dict(workpad_metadata),
            )
        except Exception as exc:
            await self._emit(
                "agent.tool_error",
                tool_name="artifact.write",
                tool_call_id=tool_call_id,
                status="failed",
                artifact_type=artifact_type,
                role=role,
                error=type(exc).__name__,
                message=str(exc),
            )
            raise
        artifact = result.get("artifact") if isinstance(result, Mapping) else None
        artifact_ref = (
            str(artifact.get("artifact_ref") or artifact.get("artifact_id") or "")
            if isinstance(artifact, Mapping)
            else ""
        )
        workpad = result.get("workpad") if isinstance(result, Mapping) else None
        if isinstance(workpad, Mapping) and workpad.get("available", True) is False:
            await self._emit(
                "artifact.write.workpad_degraded",
                tool_name="artifact.write",
                tool_call_id=tool_call_id,
                status="degraded",
                artifact_type=artifact_type,
                role=role,
                artifact_ref=artifact_ref,
                error=workpad.get("error") or "workpad_record_unavailable",
                message=workpad.get("message") or "",
            )
        await self._emit(
            "agent.tool_result",
            tool_name="artifact.write",
            tool_call_id=tool_call_id,
            status="complete",
            artifact_type=artifact_type,
            role=role,
            artifact_ref=artifact_ref,
            result_preview=summary,
        )
        return result


async def _create_and_record_strict(ledger: ArtifactLedger, **kwargs: Any) -> dict[str, Any]:
    try:
        result = await ledger.create_and_record(**kwargs, strict=True)
    except TypeError as exc:
        if "strict" not in str(exc):
            raise
        result = await ledger.create_and_record(**kwargs)
    _assert_ledger_recorded(result)
    return result


def _assert_ledger_recorded(result: Mapping[str, Any]) -> None:
    artifact = result.get("artifact")
    if isinstance(artifact, Mapping) and artifact.get("available", True) is False:
        raise RuntimeError(
            "artifact_create_failed:"
            f"{artifact.get('error') or 'artifact_create_unavailable'}"
        )
    if not isinstance(artifact, Mapping) or not artifact.get("artifact_ref"):
        raise RuntimeError("artifact_create_failed:artifact_ref_missing")


def _evaluate_section_quality(
    *,
    plan: SectionPlan,
    markdown: str,
    evidence_pack: Mapping[str, Any],
    contract: SectionedAuthoringContract,
    revision_rounds: int,
    output_truncated: bool = False,
) -> SectionQualityGateResult:
    text = str(markdown or "").strip()
    failures: list[str] = []
    information_units = _information_units(text)
    evidence_items = _evidence_items(evidence_pack)
    citation_count = _citation_count(text, evidence_items)
    url_citation_count = len(_URL_RE.findall(text))
    unique_sources_available = _unique_source_count(evidence_items)
    unique_sources_cited = _unique_sources_cited(text, evidence_items)
    # The gate can only ask for as many unique sources as the evidence pack
    # actually contains. Counting cited (not merely available) sources measures
    # the section, and capping by availability keeps the bar satisfiable when
    # upstream evidence is thin — otherwise the step dead-ends forever.
    required_unique_sources = min(
        contract.min_unique_sources_per_core_section,
        unique_sources_available,
    )

    if not text:
        failures.append("empty_section")
    if output_truncated:
        # The provider stopped at max_output_tokens, so the section ends
        # mid-thought. Soft failure: the revision loop asks for a rewrite
        # that fits the budget; if that also truncates, degrade enforcement
        # records the cut in the gap note instead of shipping it silently.
        failures.append("output_truncated")
    if _PLACEHOLDER_SECTION_RE.search(text):
        failures.append("placeholder_section")
    if not _section_has_heading(text, plan.title):
        failures.append("missing_section_heading")
    if information_units < plan.min_words:
        failures.append("insufficient_section_depth")
    if _INTERNAL_PROCESS_RE.search(text):
        failures.append("internal_process_language")
    if contract.require_evidence_refs and evidence_items and citation_count == 0:
        failures.append("missing_evidence_reference")
    if (
        contract.forbid_step_artifact_only_citations
        and evidence_items
        and citation_count > 0
        and url_citation_count == 0
    ):
        failures.append("artifact_only_citations")
    if (
        required_unique_sources
        and evidence_items
        and unique_sources_cited < required_unique_sources
    ):
        failures.append("insufficient_unique_sources")
    if contract.require_confidence_layer and not _has_confidence_layer(text):
        failures.append("missing_confidence_layer")

    return SectionQualityGateResult(
        failures=tuple(dict.fromkeys(failures)),
        information_units=information_units,
        citation_count=citation_count,
        evidence_item_count=len(evidence_items),
        revision_rounds=revision_rounds,
        unique_sources_available=unique_sources_available,
        unique_sources_cited=unique_sources_cited,
    )


def _section_has_heading(markdown: str, title: str) -> bool:
    wanted = _normalise_heading(title)
    for match in _HEADING_RE.finditer(str(markdown or "")):
        if _normalise_heading(match.group(1)) == wanted:
            return True
    return False


def _ensure_section_heading(markdown: str, title: str) -> str:
    text = str(markdown or "").strip()
    if _section_has_heading(text, title):
        return text
    return f"## {title}\n\n{text}" if text else f"## {title}"


def _normalise_heading(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())


def _information_units(markdown: str) -> int:
    text = re.sub(r"`[^`]*`", " ", str(markdown or ""))
    text = re.sub(r"https?://\S+", " ", text)
    return len(_WORD_RE.findall(text)) + len(_CJK_RE.findall(text))


def _evidence_items(evidence_pack: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_items = evidence_pack.get("items")
    if not isinstance(raw_items, list):
        return []
    return [dict(item) for item in raw_items if isinstance(item, Mapping)]


def _citation_count(markdown: str, evidence_items: list[dict[str, Any]]) -> int:
    text = str(markdown or "")
    count = len(_URL_RE.findall(text))
    if "artifact://" in text:
        count += text.count("artifact://")
    for item in evidence_items:
        artifact_id = str(item.get("artifact_id") or "").strip()
        ref = str(item.get("ref") or "").strip()
        if artifact_id and artifact_id in text:
            count += 1
        elif ref and ref in text:
            count += 1
    return count


def _unique_source_count(evidence_items: list[dict[str, Any]]) -> int:
    sources: set[str] = set()
    for item in evidence_items:
        for key in ("url", "source_url", "ref", "artifact_id", "title", "source"):
            value = str(item.get(key) or "").strip().lower()
            if value:
                sources.add(value)
                break
    return len(sources)


def _unique_sources_cited(markdown: str, evidence_items: list[dict[str, Any]]) -> int:
    """Count distinct evidence sources actually referenced in the section text.

    Unlike :func:`_unique_source_count` (which measures what the evidence pack
    *offers*), this measures what the drafted section *uses* — inline URLs plus
    any evidence identifier (url/ref/artifact_id/title) that appears verbatim in
    the markdown. This is what the quality gate should grade.
    """
    text = str(markdown or "")
    if not text:
        return 0
    cited: set[str] = set()
    for url in _URL_RE.findall(text):
        normalised = url.strip().lower()
        if normalised:
            cited.add(normalised)
    for item in evidence_items:
        for key in ("url", "source_url", "ref", "artifact_id", "title", "source"):
            value = str(item.get(key) or "").strip()
            if value and value in text:
                cited.add(value.lower())
                break
    return len(cited)


def _has_confidence_layer(markdown: str) -> bool:
    lowered = str(markdown or "").lower()
    return any(
        marker in lowered
        for marker in (
            "confirmed",
            "inferred",
            "open gap",
            "open_gap",
            "confidence:",
            "evidence strength",
        )
    )


def _join_markdown(items: Any) -> str:
    return "\n\n".join(str(item or "").strip() for item in items if str(item or "").strip())


def _llm_stream_event_delta(event: Any) -> str:
    if not isinstance(event, Mapping):
        return ""
    delta = event.get("delta")
    if isinstance(delta, Mapping):
        content = delta.get("content") or delta.get("text_delta") or delta.get("text")
        text = _llm_content_to_text(content)
        if text:
            return text
    content = event.get("content") or event.get("text_delta") or event.get("text")
    return _llm_content_to_text(content)


def _llm_stream_event_result(event: Any) -> Mapping[str, Any] | None:
    if not isinstance(event, Mapping):
        return None
    event_type = str(event.get("type") or event.get("event") or "").strip().lower()
    result = event.get("result")
    if isinstance(result, Mapping):
        normalised = dict(result)
        content = _llm_content_to_text(
            normalised.get("content")
            or normalised.get("text")
            or normalised.get("output_text")
            or normalised.get("message")
        )
        if content:
            normalised["content"] = content
        return normalised
    if event_type in {"completed", "complete", "done"}:
        content = _llm_content_to_text(event.get("content") or event.get("message"))
        if content:
            return {"content": content}
        return {}
    return None


def _llm_content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "content", "output_text"):
            text = _llm_content_to_text(value.get(key))
            if text:
                return text
        return ""
    if isinstance(value, (list, tuple)):
        return "".join(_llm_content_to_text(item) for item in value)
    return str(value)


def _is_transient_llm_stream_error(exc: Exception) -> bool:
    if bool(getattr(exc, "is_transient", False)):
        return True
    error_code = str(getattr(exc, "error_code", "") or "").strip()
    if error_code in _TRANSIENT_LLM_ERROR_CODES:
        return True
    detail = str(getattr(exc, "detail", "") or exc).lower()
    return any(
        marker in detail
        for marker in (
            "incomplete chunked read",
            "peer closed connection",
            "stream_heartbeat_timeout",
            "readtimeout",
            "connection reset",
            "transport error",
        )
    )


def _outline_schema(contract: SectionedAuthoringContract) -> dict[str, Any]:
    profile = str(contract.length_profile or "").strip().lower()
    profile_enum = [profile] if profile in {"short", "medium", "long"} else ["short", "medium", "long"]
    return {
        # LangChain uses the title as the structured-output function name;
        # a title-less dict schema is rejected by with_structured_output.
        "title": "document_outline_plan",
        "type": "object",
        "additionalProperties": False,
        "required": ["length_profile", "sections"],
        "properties": {
            "length_profile": {
                "type": "string",
                "enum": profile_enum,
            },
            "sections": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "section_id",
                        "title",
                        "objective",
                        "evidence_query",
                        "min_words",
                    ],
                    "properties": {
                        "section_id": {"type": "string"},
                        "title": {"type": "string"},
                        "objective": {"type": "string"},
                        "evidence_query": {"type": "string"},
                        "min_words": {"type": "integer"},
                    },
                },
            }
        },
    }


def _json_block(value: Any, *, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _section_plan_from_mapping(value: Any) -> SectionPlan | None:
    raw = _mapping(value)
    section_id = str(raw.get("section_id") or "").strip()
    title = str(raw.get("title") or "").strip()
    if not section_id or not title:
        return None
    return SectionPlan(
        section_id=section_id,
        title=title,
        objective=str(raw.get("objective") or ""),
        evidence_query=str(raw.get("evidence_query") or ""),
        min_words=_positive_int(raw.get("min_words"), 1),
    )


def _section_plans_from_resume_state(
    state: Mapping[str, Any],
) -> tuple[SectionPlan, ...]:
    raw_outline = state.get("outline")
    if not isinstance(raw_outline, list):
        return ()
    plans = [
        plan
        for item in raw_outline
        if (plan := _section_plan_from_mapping(item)) is not None
    ]
    return tuple(plans)


def _draft_resume_record(draft: SectionDraft) -> dict[str, Any]:
    return {
        "plan": asdict(draft.plan),
        "artifact_ref": dict(draft.artifact_ref),
        "quality": dict(draft.quality),
    }


def _artifact_id_from_ref(ref: Mapping[str, Any]) -> str:
    for key in ("artifact_id", "id"):
        value = str(ref.get(key) or "").strip()
        if value:
            return value
    artifact_ref = str(ref.get("artifact_ref") or ref.get("ref") or "").strip()
    if artifact_ref.startswith("artifact://"):
        return artifact_ref.removeprefix("artifact://").strip()
    return artifact_ref


def _slug(value: Any, *, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return slug[:80] or fallback


def _preview(value: Any, *, limit: int = 900) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _positive_int(value: Any, default: int) -> int:
    parsed = _int(value, default)
    return parsed if parsed >= 0 else default


def _non_negative_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _clamp_int(value: Any, *, minimum: int, maximum: int) -> int:
    parsed = _int(value, minimum)
    return max(minimum, min(parsed, maximum))


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    raw = str(value).strip().lower()
    if raw in _TRUTHY_ENV_VALUES:
        return True
    if raw in _FALSY_ENV_VALUES:
        return False
    return default


def _ratio(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < 0:
        return default
    return min(parsed, 1.0)


def _gate_enforcement(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return raw if raw in _GATE_ENFORCEMENT_MODES else _GATE_ENFORCEMENT_DEGRADE


def _append_quality_gap_note(markdown: str, soft_failures: tuple[str, ...]) -> str:
    """Append an explicit, reader-visible gap marker to a degraded section.

    Keeps the deliverable honest: the section is recorded best-effort, but the
    unmet quality dimensions are surfaced so downstream readers (and the merge
    step) treat the affected claims as provisional rather than confirmed.
    """
    text = str(markdown or "").rstrip()
    if not soft_failures:
        return text
    reasons = ", ".join(soft_failures)
    note = (
        "\n\n> **Evidence gap (auto-flagged):** this section did not meet the "
        f"full quality bar ({reasons}). Treat the affected claims as "
        "provisional pending stronger evidence."
    )
    return f"{text}{note}"
