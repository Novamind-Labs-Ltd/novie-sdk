"""Section-by-section longform authoring for document deliverables."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import os
import re
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType
from typing import Any, NamedTuple

from novie_protocol.agents import AgentStreamEvent

from .artifact_ledger import ArtifactLedger, EvidencePackBuilder
from .artifact_text import scrub_artifact_scaffolding
from .context_budget import wall_clock_deadline as context_wall_clock_deadline
from .document_authoring_budget import (
    AuthoringCallBudget,
    DocumentAuthoringDeadlineExceeded,
    DocumentOutputBudget,
)
from .document_authoring_compaction import compact_authoring_context
from .document_authoring_context import (
    build_authoring_context_envelope,
    deterministic_authoring_summary,
)
from .document_authoring_plan import (
    PlannedPart,
    build_authoring_execution_plan,
)
from .document_authoring_plan_codec import authoring_execution_plan_from_mapping
from .document_authoring_plan_recovery import (
    append_recovery_part,
    merge_executed_section_parts,
)
from .document_authoring_recovery import SectionCoverageCursor
from .document_completeness import (
    markdown_structure_violation,
    review_section_completeness,
)
from .document_quality import (
    DocumentQualityLoopResult,
    completed_document_quality_result,
)
from .document_part_assembly import AcceptedPart
from .document_section_parts import (
    author_planned_section,
    resumed_parts_from_checkpoint,
)
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
# An ATX heading welded directly onto the end of a paragraph line
# ("...viability.## Findings"), which markdown renders as body text because the
# hashes are not at a line start.
#
# Deliberately narrow. The leading class excludes `#` so a legitimate
# line-start "### Title" cannot match itself, and excludes whitespace so prose
# that merely mentions "the ## marker", a table cell "| ## |", or "C# and F#"
# are all left alone. Requiring the hashes to touch the preceding character
# matches the failure actually observed from a final polish and keeps the
# pattern free of false positives on ordinary writing.
_GLUED_HEADING_RE = re.compile(r"[^#\s]#{2,6}[ \t]+\S")
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
    "glued_heading",
    "markdown_structure_unclosed",
    "missing_section_heading",
    "output_truncated",
    "placeholder_section",
    "semantic_incomplete",
})

# Gate enforcement modes. ``strict`` keeps the legacy behaviour of hard-failing
# the step on any unmet gate; ``degrade`` records a best-effort section with an
# explicit gap marker for soft, evidence-bound failures so the plan completes
# instead of dead-ending on an unsatisfiable quality bar.
_GATE_ENFORCEMENT_STRICT = "strict"
_GATE_ENFORCEMENT_DEGRADE = "degrade"
_GATE_ENFORCEMENT_MODES = frozenset({_GATE_ENFORCEMENT_STRICT, _GATE_ENFORCEMENT_DEGRADE})

# Accepted parts and sections are immutable at finalization time. Cohesion is
# created while drafting bounded parts; the final document is a deterministic
# assembly and never another whole-document generation.
KNOWN_FINALIZATION_MODES = frozenset({"deterministic_assembly"})


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
    nested = metadata if isinstance(metadata, Mapping) else {}
    reasons = {
        str(source.get(key) or "").strip().lower()
        for source in (completed_result, nested)
        for key in ("finish_reason", "stop_reason")
        if str(source.get(key) or "").strip()
    }
    if reasons & _TRUNCATION_FINISH_REASONS:
        return "length"
    if "tool_calls" in reasons or "tool_use" in reasons:
        return "tool_calls"
    return "stop" if reasons else ""


def _remaining_deadline_seconds(deadline: float) -> float:
    return max(0.0, deadline - asyncio.get_running_loop().time())


def _validated_finalization(value: Any) -> str:
    mode = str(value or "deterministic_assembly").strip()
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
    finalization: str = "deterministic_assembly"
    evidence_depth: str = "standard"
    min_outline_sections: int = 2
    max_outline_sections: int = 9
    canonical_section_titles: tuple[str, ...] = ()
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
            canonical_section_titles=tuple(
                str(title).strip()
                for title in (raw.get("canonical_section_titles") or [])
                if str(title).strip()
            ),
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
    completeness_review: Mapping[str, Any] = field(default_factory=dict)

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
            "completeness_review": dict(self.completeness_review),
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


@dataclass(frozen=True, slots=True)
class DocumentAuthoringRequest:
    """One skill-resolved request for a terminal document-authoring pass.

    Agents retain ownership of capability selection, graph work, and final
    artifact rendering.  They pass the already-resolved skill contract and its
    bounded authoring instructions here, so the shared SDK author never has to
    infer document intent from an artifact type or a generic template.
    """

    llm_facade: Any | None
    skill_contract: SkillRuntimeContract | None
    artifact_type: str
    step_id: str
    capability_id: str
    context_budget: Mapping[str, Any]
    brief: Mapping[str, Any]
    upstream: Mapping[str, Any]
    authoring_instructions: str
    workflow_id: str | None = None
    thread_id: str | None = None
    agent_id: str | None = None
    mode_metadata: Mapping[str, Any] | None = None
    draft_narrative: str = ""
    draft_narrative_key: str = ""
    draft_narrative_artifact_type: str = ""
    draft_narrative_summary: str = ""
    document_input: Mapping[str, Any] | None = None
    agent_disabled_env_var: str | None = None
    agent_enabled_env_var: str | None = None
    required_strategy: str = "sectioned_longform"
    quality_reason: str = "sectioned_authoring_quality_gates"
    quality_metadata: Mapping[str, Any] | None = None
    defer_intermediate_artifacts: bool = False
    defer_final_artifact: bool = True
    # Capability runtimes may require a structural assembly strategy even when
    # their shared skill contract uses a different default.
    finalization_override: str | None = None
    canonical_section_titles: tuple[str, ...] = ()
    length_profile: str | None = None
    profile_source: str = "skill_default"
    profile_confidence: str = "confirmed"
    resume_state: Mapping[str, Any] | None = None
    phase_checkpoint_sink: Callable[[Mapping[str, Any]], Any] | None = None
    rebase_artifact_types_to_runtime: bool = False
    wall_clock_deadline: float | None = None

    def __post_init__(self) -> None:
        if not self.artifact_type.strip():
            raise ValueError("document_authoring_request artifact_type is required")
        if not self.capability_id.strip():
            raise ValueError("document_authoring_request capability_id is required")
        if not self.authoring_instructions.strip():
            raise ValueError(
                "document_authoring_request skill authoring instructions are required"
            )
        if self.defer_intermediate_artifacts and self.defer_final_artifact:
            raise ValueError(
                "document_authoring_request cannot defer both intermediate and final artifacts"
            )
        for field_name in ("context_budget", "brief", "upstream"):
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(getattr(self, field_name))),
            )
        for field_name in (
            "mode_metadata",
            "document_input",
            "quality_metadata",
            "resume_state",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    MappingProxyType(dict(value)),
                )

    def to_metadata(self) -> dict[str, Any]:
        contract = self.skill_contract
        return {
            "document_authoring_request": {
                "artifact_type": self.artifact_type,
                "capability_id": self.capability_id,
                "skill_sources": list(contract.sources) if contract is not None else [],
                "skill_instructions_loaded": True,
                "resuming_sectioned_state": self.resume_state is not None,
                "defer_final_artifact": self.defer_final_artifact,
                "finalization_override": self.finalization_override,
                "canonical_section_titles": list(self.canonical_section_titles),
            }
        }


async def run_sectioned_document_finalization(
    *,
    request: DocumentAuthoringRequest,
) -> SectionedDocumentFinalizationResult:
    """Run sectioned longform finalization for document agents.

    This is intentionally a coarse runner: agents still own the graph, prompt,
    structured artifact construction, and final event envelope. The SDK owns
    the repeated sectioned-authoring checks, author construction, trace
    metadata, and skipped-quality result wiring.
    """
    llm_facade = request.llm_facade
    skill_contract = request.skill_contract
    artifact_type = request.artifact_type
    step_id = request.step_id
    capability_id = request.capability_id
    context_budget = request.context_budget
    brief = request.brief
    upstream = request.upstream
    workflow_id = request.workflow_id
    thread_id = request.thread_id
    agent_id = request.agent_id
    mode_metadata = request.mode_metadata
    draft_narrative = request.draft_narrative
    draft_narrative_key = request.draft_narrative_key
    draft_narrative_artifact_type = request.draft_narrative_artifact_type
    draft_narrative_summary = request.draft_narrative_summary
    document_input = request.document_input
    authoring_instructions = request.authoring_instructions
    agent_disabled_env_var = request.agent_disabled_env_var
    agent_enabled_env_var = request.agent_enabled_env_var
    required_strategy = request.required_strategy
    quality_reason = request.quality_reason
    quality_metadata = request.quality_metadata
    defer_intermediate_artifacts = request.defer_intermediate_artifacts
    defer_final_artifact = request.defer_final_artifact
    wall_clock_deadline = request.wall_clock_deadline
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
            **request.to_metadata(),
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
            finalization_override=request.finalization_override,
            canonical_section_titles=request.canonical_section_titles,
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
    quality_result = completed_document_quality_result(
        authoring_result.markdown,
        degraded=bool(authoring_result.ledger.get("degraded")),
        metadata={
            **dict(quality_metadata or {}),
            "quality_reason": quality_reason,
            "section_count": len(authoring_result.drafts),
            "reviewed_section_count": len(authoring_result.drafts),
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
    request: DocumentAuthoringRequest,
) -> AsyncIterator[AgentStreamEvent | SectionedDocumentFinalizationResult]:
    """Gold-path sectioned finalize used by report_synthesis-style document agents.

    Streams phase/content events while ``SectionedLongformAuthor`` runs, then
    yields a terminal ``SectionedDocumentFinalizationResult``. Callers own the
    typed artifact envelope and final event.
    """
    llm_facade = request.llm_facade
    skill_contract = request.skill_contract
    artifact_type = request.artifact_type
    step_id = request.step_id
    capability_id = request.capability_id
    context_budget = request.context_budget
    brief = request.brief
    upstream = request.upstream
    workflow_id = request.workflow_id
    thread_id = request.thread_id
    agent_id = request.agent_id
    mode_metadata = request.mode_metadata
    draft_narrative = request.draft_narrative
    draft_narrative_key = request.draft_narrative_key
    draft_narrative_artifact_type = request.draft_narrative_artifact_type
    draft_narrative_summary = request.draft_narrative_summary
    document_input = request.document_input
    authoring_instructions = request.authoring_instructions
    agent_disabled_env_var = request.agent_disabled_env_var
    agent_enabled_env_var = request.agent_enabled_env_var
    required_strategy = request.required_strategy
    quality_reason = request.quality_reason
    quality_metadata = request.quality_metadata
    defer_intermediate_artifacts = request.defer_intermediate_artifacts
    defer_final_artifact = request.defer_final_artifact
    length_profile = request.length_profile
    profile_source = request.profile_source
    profile_confidence = request.profile_confidence
    resume_state = request.resume_state
    phase_checkpoint_sink = request.phase_checkpoint_sink
    rebase_artifact_types_to_runtime = request.rebase_artifact_types_to_runtime
    wall_clock_deadline = request.wall_clock_deadline
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
        finalization_override=request.finalization_override,
        canonical_section_titles=request.canonical_section_titles,
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
            **request.to_metadata(),
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
    quality_result = completed_document_quality_result(
        authoring_result.markdown,
        degraded=bool(authoring_result.ledger.get("degraded")),
        metadata={
            **dict(quality_metadata or {}),
            "quality_reason": quality_reason,
            "section_count": len(authoring_result.drafts),
            "reviewed_section_count": len(authoring_result.drafts),
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
        # Accepted section bodies are durable artifacts. Re-emitting their
        # token deltas would reconstruct the full document in A2A/Temporal
        # transport and defeat the reference-only finalization contract.
        return []
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
    finalization_override: str | None = None,
    canonical_section_titles: tuple[str, ...] = (),
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
        finalization = (
            finalization_override
            or profile.finalization
            or contract.runtime.finalization
            or "deterministic_assembly"
        )
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
        finalization = (
            finalization_override
            or contract.runtime.finalization
            or "deterministic_assembly"
        )
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
        "canonical_section_titles": tuple(
            title.strip() for title in canonical_section_titles if title.strip()
        ),
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
        self._document_output_byte_limit = (
            _positive_int(self._context_budget.get("max_document_output_bytes"), 0)
            or None
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
        self._authoring_call_budget: AuthoringCallBudget | None = None
        self._last_authoring_context_pressure = "normal"
        self._persisted_part_index_loaded = False
        self._persisted_part_refs: dict[tuple[str, str], dict[str, Any]] = {}

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
            drafts = await self._revalidate_resumed_drafts(drafts)
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
        degraded_sections = list(
            state.get("degraded_sections")
            if isinstance(state.get("degraded_sections"), list)
            else []
        )
        scope = {
            "tenant_id": str(self._context_budget.get("tenant_id") or ""),
            "workspace_id": str(self._context_budget.get("workspace_id") or ""),
            "workflow_id": str(workflow_id or ""),
            "step_id": self._step_id,
            "capability_id": self._capability_id,
        }
        execution_plan = build_authoring_execution_plan(
            outline,
            scope=scope,
            context_budget=self._context_budget,
            revision=_positive_int(state.get("authoring_plan_revision"), 1),
        )
        stored_plan = state.get("authoring_execution_plan")
        if isinstance(stored_plan, Mapping):
            try:
                resumed_plan = authoring_execution_plan_from_mapping(stored_plan)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "document_authoring_checkpoint_plan_invalid"
                ) from exc
            if resumed_plan.outline_digest != execution_plan.outline_digest:
                raise RuntimeError(
                    "document_authoring_checkpoint_outline_mismatch"
                )
            execution_plan = resumed_plan
        self._authoring_call_budget = AuthoringCallBudget(
            total_limit=execution_plan.max_authoring_llm_calls,
            compaction_limit=execution_plan.max_authoring_compaction_calls,
            review_limit=execution_plan.max_section_review_calls,
            used=_positive_int(state.get("authoring_llm_calls_used"), 0),
            compactions_used=_positive_int(
                state.get("authoring_compaction_calls_used"),
                0,
            ),
            reviews_used=_positive_int(state.get("authoring_review_calls_used"), 0),
        )
        authoring_summary = _mapping(state.get("authoring_context_summary"))
        await self._emit(
            "document.authoring_plan.accepted",
            status="complete",
            plan_revision=execution_plan.revision,
            planned_part_count=execution_plan.part_count,
            mandatory_llm_call_count=execution_plan.mandatory_call_count,
            max_authoring_llm_calls=execution_plan.max_authoring_llm_calls,
            estimated_authoring_seconds=execution_plan.estimated_authoring_seconds,
            deadline_feasible=execution_plan.deadline_feasible,
        )
        for index, plan in enumerate(outline[len(drafts) :], start=len(drafts) + 1):
            execution_section = execution_plan.sections[index - 1]
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
            checkpoint_base = {
                "current_phase": "draft_sections",
                "length_profile": length_profile,
                "outline": [asdict(item) for item in outline],
                "outline_ref": outline_ref,
                "drafts": [_draft_resume_record(draft) for draft in drafts],
                "degraded_sections": degraded_sections,
                "authoring_context_summary": authoring_summary,
                **self._authoring_call_budget.metadata(),
            }
            resumed_parts = await resumed_parts_from_checkpoint(
                self,
                state,
                section_id=plan.section_id,
            )
            part_result = await author_planned_section(
                self,
                brief=brief,
                section_plan=plan,
                execution_plan=execution_plan,
                execution_section=execution_section,
                section_index=index,
                previous_drafts=drafts,
                evidence_pack=evidence_pack_input,
                authoring_summary=authoring_summary,
                workflow_id=workflow_id,
                thread_id=thread_id,
                agent_id=agent_id,
                checkpoint_base=checkpoint_base,
                resumed_parts=resumed_parts,
            )
            markdown = part_result.markdown
            quality = _evaluate_section_quality(
                plan=plan,
                markdown=markdown,
                evidence_pack=evidence_pack_input,
                contract=self._contract,
                revision_rounds=0,
                output_truncated=False,
            )
            quality = await self._apply_completeness_review(
                plan=plan,
                markdown=markdown,
                quality=quality,
            )
            if "semantic_incomplete" in quality.failures:
                review = quality.completeness_review
                execution_plan = merge_executed_section_parts(
                    execution_plan,
                    section_index=index - 1,
                    parts=[
                        item.accepted.plan
                        for item in part_result.execution.parts
                    ],
                    revision=part_result.execution.plan_revision,
                )
                execution_plan = append_recovery_part(
                    execution_plan,
                    section_index=index - 1,
                    issue_code=str(review.get("issue_code") or "missing_planned_content"),
                    reason=str(review.get("reason") or plan.objective),
                    scope=scope,
                )
                execution_section = execution_plan.sections[index - 1]
                await self._emit(
                    "document.section.recovery_part_planned",
                    status="running",
                    section_id=plan.section_id,
                    section_index=index,
                    plan_revision=execution_plan.revision,
                    issue_code=review.get("issue_code"),
                )
                part_result = await author_planned_section(
                    self,
                    brief=brief,
                    section_plan=plan,
                    execution_plan=execution_plan,
                    execution_section=execution_section,
                    section_index=index,
                    previous_drafts=drafts,
                    evidence_pack=evidence_pack_input,
                    authoring_summary=authoring_summary,
                    workflow_id=workflow_id,
                    thread_id=thread_id,
                    agent_id=agent_id,
                    checkpoint_base=checkpoint_base,
                    resumed_parts=part_result.execution.parts,
                )
                markdown = part_result.markdown
                quality = _evaluate_section_quality(
                    plan=plan,
                    markdown=markdown,
                    evidence_pack=evidence_pack_input,
                    contract=self._contract,
                    revision_rounds=1,
                    output_truncated=False,
                )
                quality = await self._apply_completeness_review(
                    plan=plan,
                    markdown=markdown,
                    quality=quality,
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
                # not dead-end the plan. Record the best-effort section and
                # continue so the deliverable completes.
                #
                # The degradation is reported structurally only — via the
                # `document.section.quality_degraded` event, the `degraded` /
                # `degraded_sections` result metadata below, and
                # `quality.to_metadata()`. It is deliberately NOT written into
                # the section markdown: internal gate identifiers
                # ("artifact_only_citations", "missing_confidence_layer") are
                # reviewer vocabulary, and the reader of the deliverable is not
                # the reviewer. Surfacing it in the UI is the consumer's call.
                quality = SectionQualityGateResult(
                    failures=quality.failures,
                    information_units=quality.information_units,
                    citation_count=quality.citation_count,
                    evidence_item_count=quality.evidence_item_count,
                    revision_rounds=quality.revision_rounds,
                    unique_sources_available=quality.unique_sources_available,
                    unique_sources_cited=quality.unique_sources_cited,
                    degraded=True,
                    completeness_review=quality.completeness_review,
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
            authoring_summary = await self._refresh_authoring_summary(
                prior_summary=authoring_summary,
                execution_plan=execution_plan,
                drafts=drafts,
                accepted_parts=part_result.execution.parts,
                evidence_pack=evidence_pack_input,
            )
            await self._checkpoint(
                checkpoint_schema="document-authoring-checkpoint.v2",
                current_phase="draft_sections",
                length_profile=length_profile,
                outline=[asdict(item) for item in outline],
                outline_ref=outline_ref,
                drafts=[_draft_resume_record(draft) for draft in drafts],
                degraded_sections=degraded_sections,
                authoring_execution_plan=execution_plan.to_mapping(),
                authoring_plan_revision=part_result.execution.plan_revision,
                authoring_context_summary=authoring_summary,
                current_section_id="",
                accepted_parts=[],
                **self._authoring_call_budget.metadata(),
            )

        await self._emit(
            "document.final.deterministic_assembly_started",
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

    async def _apply_completeness_review(
        self,
        *,
        plan: SectionPlan,
        markdown: str,
        quality: SectionQualityGateResult,
    ) -> SectionQualityGateResult:
        if quality.hard_failures:
            return quality
        self._reserve_authoring_call("review")
        review = await review_section_completeness(
            self._llm,
            section_title=plan.title,
            section_objective=plan.objective,
            markdown=markdown,
        )
        failures = quality.failures
        if not review.complete:
            failures = tuple(dict.fromkeys((*failures, "semantic_incomplete")))
        await self._emit(
            "document.section.completeness_reviewed",
            status="passed" if review.complete else "gate_failed",
            section_id=plan.section_id,
            section_title=plan.title,
            issue_code=review.issue_code,
            complete=review.complete,
        )
        return replace(
            quality,
            failures=failures,
            completeness_review=review.to_metadata(),
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

    async def _revalidate_resumed_drafts(
        self,
        drafts: list[SectionDraft],
    ) -> list[SectionDraft]:
        """Admit legacy checkpoint drafts only after ADR-114 completeness checks."""
        admitted: list[SectionDraft] = []
        for draft in drafts:
            quality = dict(draft.quality)
            hard_failures = quality.get("hard_failures")
            mechanical = (
                markdown_structure_violation(draft.markdown)
                or (
                    "glued_heading"
                    if _GLUED_HEADING_RE.search(draft.markdown)
                    else None
                )
                or (
                    None
                    if _section_has_heading(draft.markdown, draft.plan.title)
                    else "missing_section_heading"
                )
            )
            if (
                mechanical is not None
                or isinstance(hard_failures, (list, tuple))
                and bool(hard_failures)
            ):
                await self._emit(
                    "document.section.resume_invalidated",
                    status="gate_failed",
                    section_id=draft.plan.section_id,
                    section_title=draft.plan.title,
                    reason=mechanical or "blocking_quality_failure",
                )
                break
            review = quality.get("completeness_review")
            if not isinstance(review, Mapping) or review.get("complete") is not True:
                reviewed = await review_section_completeness(
                    self._llm,
                    section_title=draft.plan.title,
                    section_objective=draft.plan.objective,
                    markdown=draft.markdown,
                )
                if not reviewed.complete:
                    await self._emit(
                        "document.section.resume_invalidated",
                        status="gate_failed",
                        section_id=draft.plan.section_id,
                        section_title=draft.plan.title,
                        reason=reviewed.issue_code,
                    )
                    break
                quality["completeness_review"] = reviewed.to_metadata()
            admitted.append(replace(draft, quality=quality))
        return admitted

    async def _read_resume_artifact_text(self, artifact_ref: Mapping[str, Any]) -> str:
        artifacts = getattr(self._platform, "artifacts", None)
        read_raw_text = getattr(artifacts, "read_raw_text", None)
        if not callable(read_raw_text):
            return ""
        artifact_id = _artifact_id_from_ref(artifact_ref)
        if not artifact_id:
            return ""
        return str(
            await read_raw_text(
                artifact_id,
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
        if self._contract.canonical_section_titles:
            # Fixed-shape artifacts own their reader-facing section order. The
            # model still writes each body, but it must not invent headings or
            # change the number of sections.
            plans = tuple(
                SectionPlan(
                    section_id=_slug(title, fallback=f"section-{index}"),
                    title=title,
                    objective=f"Cover the {title.lower()} required by the artifact contract.",
                    evidence_query=title,
                    min_words=self._contract.default_section_words,
                )
                for index, title in enumerate(
                    self._contract.canonical_section_titles,
                    start=1,
                )
            )
            if plans:
                await self._emit(
                    "document.outline.fixed_shape",
                    status="complete",
                    section_count=len(plans),
                    section_titles=[plan.title for plan in plans],
                )
                return self._contract.length_profile or "medium", plans
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

    async def _refresh_authoring_summary(
        self,
        *,
        prior_summary: Mapping[str, Any],
        execution_plan: Any,
        drafts: list[SectionDraft],
        accepted_parts: Any,
        evidence_pack: Mapping[str, Any],
    ) -> dict[str, Any]:
        covered = list(prior_summary.get("covered_objectives") or [])
        covered.extend(
            part.accepted.plan.objective_digest for part in accepted_parts
        )
        covered = list(dict.fromkeys(str(item) for item in covered if str(item)))
        all_objectives = [
            part.objective_digest
            for section in execution_plan.sections
            for part in section.parts
        ]
        unresolved = [item for item in all_objectives if item not in covered]
        evidence_refs = [
            {
                "artifact_id": str(item.get("artifact_id") or ""),
                "artifact_ref": str(item.get("ref") or ""),
            }
            for item in evidence_pack.get("items", [])
            if isinstance(item, Mapping)
        ]
        if self._last_authoring_context_pressure == "normal":
            summary = deterministic_authoring_summary(
                covered_objectives=covered,
                unresolved_objectives=unresolved,
                evidence_refs=evidence_refs,
                continuity_notes=(
                    str(prior_summary.get("previous_part_handoff") or ""),
                ),
            )
            await self._emit(
                "document.authoring_context.compacted",
                status="complete",
                compaction_mode="deterministic",
                pressure="normal",
            )
            return summary
        self._reserve_authoring_call("compaction")
        if accepted_parts:
            latest = accepted_parts[-1].accepted.markdown
        elif drafts:
            latest = drafts[-1].markdown
        else:
            latest = ""
        compaction_input_tokens = _positive_int(
            self._context_budget.get("max_authoring_compaction_input_tokens"),
            12_000,
        )
        latest = latest[: compaction_input_tokens * 4]
        compacted = await compact_authoring_context(
            self._llm,
            prior_summary=prior_summary,
            accepted_part_markdown=latest,
            covered_objectives=covered,
            unresolved_objectives=unresolved,
            evidence_refs=evidence_refs,
            model=self._contract.running_summary_model or None,
            max_output_tokens=self._contract.running_summary_max_tokens,
        )
        await self._emit(
            "document.authoring_context.compacted",
            status="complete",
            compaction_mode=compacted.mode,
            source_digest=compacted.source_digest,
            fallback_reason=compacted.error,
            pressure=self._last_authoring_context_pressure,
        )
        return dict(compacted.summary)

    async def _draft_section(
        self,
        *,
        brief: Mapping[str, Any],
        plan: SectionPlan,
        section_index: int | None,
        previous: list[SectionDraft],
        evidence_pack: Mapping[str, Any],
        planned_part: PlannedPart | None = None,
        coverage_cursor: SectionCoverageCursor | None = None,
        previous_part_handoff: str = "",
        authoring_summary: Mapping[str, Any] | None = None,
        output_slots_remaining: int = 1,
    ) -> tuple[str, bool]:
        """Draft one bounded Planned Part; returns text and truncation state."""
        previous_index = [
            {
                "section_id": draft.plan.section_id,
                "title": draft.plan.title,
                "artifact_ref": draft.artifact_ref.get("artifact_ref"),
            }
            for draft in previous
        ]
        objective = planned_part.objective if planned_part else plan.objective
        target = (
            planned_part.target_information_units
            if planned_part
            else plan.min_words
        )
        maximum = (
            planned_part.max_information_units
            if planned_part
            else max(plan.min_words + 1, int(plan.min_words * 1.25))
        )
        cursor = coverage_cursor or SectionCoverageCursor(section_id=plan.section_id)
        envelope = build_authoring_context_envelope(
            context_budget=self._context_budget,
            authoring_contract={
                "artifact_type": self._artifact_type,
                "skill_instructions": self._authoring_instructions,
            },
            objective={
                "section": asdict(plan),
                "planned_part": asdict(planned_part) if planned_part else {},
                "objective": objective,
            },
            required_evidence=evidence_pack,
            coverage_cursor=asdict(cursor),
            previous_part_handoff=previous_part_handoff,
            authoring_summary=authoring_summary or {},
            optional_evidence={"prior_section_refs": previous_index},
        )
        self._last_authoring_context_pressure = envelope.pressure
        await self._emit(
            "document.authoring_context.envelope_built",
            status="complete",
            section_id=plan.section_id,
            part_identity=planned_part.part_identity if planned_part else "",
            envelope=envelope.to_metadata(),
        )
        prompt = (
            "Write one complete bounded part of a document section in Markdown.\n"
            f"The parent section is `## {plan.title}`. Do not emit a level-one or "
            "level-two heading; use level-three-or-deeper local headings only. "
            "Finish every sentence, list, table, and code block started in this "
            "part. Cover only the current objective and do not anticipate later "
            "parts. Use the previous-part handoff to open naturally without "
            "repeating accepted content. Cite supplied artifact/source refs when "
            "evidence is used. Return prose only, without process notes.\n"
            f"Target {target} substantive information units and do not exceed "
            f"{maximum} information units.\n\n"
            f"Original task:\n{_json_block(brief, limit=8000)}\n\n"
            f"{envelope.render()}"
        )
        # Token counts and word-like information units are not interchangeable.
        # Keep enough provider headroom to reach a semantic boundary; calibrating
        # this near the information-unit target caused real providers to stop
        # mid-word.  Acceptance has an independent runaway ceiling, and the
        # document-level budget remains the aggregate protection against
        # context/output explosion.
        part_output_tokens = max(256, maximum * 8)
        if self._output_token_ceiling is not None:
            part_output_tokens = min(part_output_tokens, self._output_token_ceiling)
        self._reserve_authoring_call("draft")
        streamed = await self._stream_llm_text(
            purpose="draft_planned_part",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_output_tokens=part_output_tokens,
            output_slots_remaining=output_slots_remaining,
            section=plan,
            section_index=section_index,
            extra_metadata={
                "part_identity": planned_part.part_identity if planned_part else "",
                "part_ordinal": planned_part.ordinal if planned_part else 1,
                "part_total": planned_part.total if planned_part else 1,
                "target_information_units": target,
                "max_information_units": maximum,
            },
        )
        continuation_attempt = 0
        max_continuations = _positive_int(
            self._context_budget.get("max_part_continuations"),
            2,
        )
        while streamed.truncated and continuation_attempt < max_continuations:
            continuation_attempt += 1
            await self._emit(
                "document.part.continuation_requested",
                status="running",
                section_id=plan.section_id,
                part_identity=planned_part.part_identity if planned_part else "",
                continuation_attempt=continuation_attempt,
            )
            continuation_prompt = (
                "The preceding planned-part output stopped at the provider token "
                "limit. Continue exactly where it ended. Do not restart, repeat, "
                "summarize, add a section heading, or expand the scope. Complete "
                "the current sentence and any open list, table, or code block, "
                "then finish only this planned part in the fewest words needed. "
                "Return continuation text only.\n\n"
                f"Current objective:\n{objective}\n\n"
                "Tail of interrupted output:\n"
                f"{streamed.text[-6000:]}"
            )
            self._reserve_authoring_call("draft")
            continuation = await self._stream_llm_text(
                purpose="continue_truncated_planned_part",
                messages=[{"role": "user", "content": continuation_prompt}],
                temperature=0.1,
                max_output_tokens=part_output_tokens,
                output_slots_remaining=output_slots_remaining,
                section=plan,
                section_index=section_index,
                extra_metadata={
                    "part_identity": planned_part.part_identity if planned_part else "",
                    "continuation_attempt": continuation_attempt,
                },
            )
            if not continuation.text:
                break
            streamed = _StreamedLlmText(
                text=streamed.text + continuation.text,
                finish_reason=continuation.finish_reason,
                truncated=continuation.truncated,
            )
            await self._emit(
                "document.part.continuation_completed",
                status="complete" if not streamed.truncated else "incomplete",
                section_id=plan.section_id,
                part_identity=planned_part.part_identity if planned_part else "",
                continuation_attempt=continuation_attempt,
                truncated=streamed.truncated,
            )
        content = streamed.text.strip()
        if not content:
            return "", streamed.truncated
        # Scrub at the point the draft is accepted, not only on the way to
        # storage: the quality gate, the section preview/summary and the final
        # merge all derive from this text. `_preview` in particular flattens
        # newlines, which would turn a copied tool-observation header into an
        # inline substring that the line-anchored scrub can no longer see.
        # `ArtifactLedger` still scrubs on write as the universal backstop for
        # authoring paths that do not come through here.
        section = scrub_artifact_scaffolding(
            _isolate_requested_section(content, plan.title)
        ).text
        if _PLACEHOLDER_SECTION_RE.search(section):
            raise RuntimeError(
                "document_planned_part_rejected:placeholder_section"
            )
        return section, streamed.truncated

    def _reserve_authoring_call(self, kind: str) -> None:
        budget = self._authoring_call_budget
        if budget is not None:
            budget.reserve(kind)

    async def _polish_final(
        self,
        *,
        brief: Mapping[str, Any],
        drafts: list[SectionDraft],
    ) -> str:
        del brief
        combined = _join_markdown(draft.markdown for draft in drafts)
        if self._contract.finalization not in KNOWN_FINALIZATION_MODES:
            raise RuntimeError(
                "document_finalization_mode_invalid:"
                f"{self._contract.finalization}"
            )
        violation = _document_integrity_violation(combined, drafts)
        if violation is not None:
            raise RuntimeError(
                f"document_final_integrity_invalid:{violation}"
            )
        mechanical_violation = markdown_structure_violation(combined)
        if mechanical_violation is not None:
            raise RuntimeError(
                f"document_final_integrity_invalid:{mechanical_violation}"
            )
        await self._emit(
            "document.final.deterministic_assembly_completed",
            status="complete",
            section_count=len(drafts),
        )
        return combined

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

    async def _record_part(
        self,
        section: SectionPlan,
        accepted: AcceptedPart,
        *,
        section_index: int,
        workflow_id: str | None,
        thread_id: str | None,
        agent_id: str | None,
    ) -> dict[str, Any]:
        if not self._ledger.is_available:
            raise RuntimeError("artifact_create_failed:artifact_ledger_unavailable")
        existing = await self._find_persisted_part(
            accepted,
            workflow_id=workflow_id,
            thread_id=thread_id,
        )
        if existing:
            return existing
        result = await self._ledger.create_artifact(
            artifact_type=f"{self._artifact_type}.section_part",
            content=accepted.markdown,
            content_type="text/markdown",
            summary=_preview(accepted.markdown),
            workflow_id=workflow_id,
            thread_id=thread_id,
            step_id=self._step_id,
            agent_id=agent_id,
            metadata={
                "role": "section_part",
                "internal_authoring_artifact": True,
                "section_id": section.section_id,
                "section_index": section_index,
                "part_identity": accepted.plan.part_identity,
                "part_ordinal": accepted.plan.ordinal,
                "part_total": accepted.plan.total,
                "objective_digest": accepted.plan.objective_digest,
                "evidence_digest": accepted.plan.evidence_digest,
                "content_digest": accepted.content_digest,
                "capability_id": self._capability_id,
            },
        )
        if result.get("available", True) is False or not result.get("artifact_ref"):
            raise RuntimeError(
                "artifact_create_failed:"
                f"{result.get('error') or 'artifact_ref_missing'}"
            )
        stored = await self._read_resume_artifact_text(result)
        stored_digest = hashlib.sha256(stored.encode()).hexdigest()
        if stored_digest != accepted.content_digest:
            raise RuntimeError(
                "document_part_artifact_digest_mismatch:"
                f"{accepted.plan.part_identity}"
            )
        self._persisted_part_refs[
            (accepted.plan.part_identity, accepted.content_digest)
        ] = dict(result)
        return result

    async def _find_persisted_part(
        self,
        accepted: AcceptedPart,
        *,
        workflow_id: str | None,
        thread_id: str | None,
    ) -> dict[str, Any]:
        if not self._persisted_part_index_loaded:
            self._persisted_part_index_loaded = True
            artifacts = getattr(self._platform, "artifacts", None)
            search_index = getattr(artifacts, "search_index", None)
            if callable(search_index) and (workflow_id or thread_id):
                items = await search_index(
                    workflow_id=workflow_id,
                    thread_id=thread_id,
                    artifact_type_prefix=f"{self._artifact_type}.section_part",
                    limit=200,
                )
                for item in items:
                    if not isinstance(item, Mapping):
                        continue
                    metadata = item.get("metadata")
                    if not isinstance(metadata, Mapping):
                        continue
                    part_identity = str(metadata.get("part_identity") or "").strip()
                    content_digest = str(metadata.get("content_digest") or "").strip()
                    if not part_identity or not content_digest:
                        continue
                    ref = dict(item)
                    artifact_id = _artifact_id_from_ref(ref)
                    if artifact_id and not ref.get("artifact_ref"):
                        ref["artifact_ref"] = f"artifact://{artifact_id}"
                    self._persisted_part_refs[
                        (part_identity, content_digest)
                    ] = ref

        key = (accepted.plan.part_identity, accepted.content_digest)
        candidate = self._persisted_part_refs.get(key)
        if not candidate:
            return {}
        stored = await self._read_resume_artifact_text(candidate)
        if hashlib.sha256(stored.encode()).hexdigest() != accepted.content_digest:
            self._persisted_part_refs.pop(key, None)
            return {}
        return dict(candidate)

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
        # A token-limit stop is blocking. The bounded revision loop may rewrite
        # it, but degrade enforcement must never publish it.
        failures.append("output_truncated")
    if _PLACEHOLDER_SECTION_RE.search(text):
        failures.append("placeholder_section")
    if _GLUED_HEADING_RE.search(text):
        failures.append("glued_heading")
    structure_failure = markdown_structure_violation(text)
    if structure_failure is not None:
        failures.append(structure_failure)
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


def _isolate_requested_section(markdown: str, title: str) -> str:
    """Keep only the planned level-two section from an over-broad LLM draft.

    The final document contract is intentionally available to every section
    writer. Models occasionally follow it too literally and emit the entire
    document for a single section. Preserving that response would duplicate
    headings when the drafts are joined, so retain the requested level-two
    section and drop sibling document sections deterministically.
    """
    text = str(markdown or "").strip()
    wanted = _normalise_heading(title)
    headings = list(_HEADING_RE.finditer(text))
    for index, heading in enumerate(headings):
        raw_heading = heading.group(0)
        level = len(raw_heading) - len(raw_heading.lstrip("#"))
        if level != 2 or _normalise_heading(heading.group(1)) != wanted:
            continue
        end = len(text)
        for sibling in headings[index + 1 :]:
            sibling_raw = sibling.group(0)
            sibling_level = len(sibling_raw) - len(sibling_raw.lstrip("#"))
            if sibling_level <= 2:
                end = sibling.start()
                break
        return text[heading.start() : end].strip()
    # The model may return one otherwise valid section under the wrong H2.
    # Wrapping that response with the requested heading would preserve the
    # foreign H2 and produce a structurally invalid final outline. Retitle the
    # first returned H2 section and discard sibling document sections instead.
    for index, heading in enumerate(headings):
        raw_heading = heading.group(0)
        level = len(raw_heading) - len(raw_heading.lstrip("#"))
        if level != 2:
            continue
        end = len(text)
        for sibling in headings[index + 1 :]:
            sibling_raw = sibling.group(0)
            sibling_level = len(sibling_raw) - len(sibling_raw.lstrip("#"))
            if sibling_level <= 2:
                end = sibling.start()
                break
        body = text[heading.end() : end].strip()
        return f"## {title}\n\n{body}".strip()
    return _ensure_section_heading(text, title)


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


def _section_structure_violation(
    markdown: str, drafts: list[SectionDraft]
) -> str | None:
    """Name the structural defect in a final rewrite, or ``None`` if it is sound.

    A final LLM polish can retain all the words while serializing the document
    with a heading glued to the preceding paragraph. Downstream fixed-shape
    artifact validation then (correctly) rejects that text. The original
    section drafts are already validated and joined with explicit separators,
    so use them whenever the rewrite no longer preserves their exact outline.

    Two independent checks, because the outline comparison alone has a blind
    spot: it only inspects level-two headings, so a glued ``### Subheading``
    leaves the level-two list identical and slips through. A reader then sees
    the subheading rendered as body text — the heading is not at a line start,
    so it is not a heading at all.

    Returns the specific violation rather than a bool so the caller can report
    *which* defect fired. That distinction is the only way to tell from
    production whether glued headings actually occur, or whether the reordering
    check is doing all the work — a bool would collapse both into one
    indistinguishable fallback event.

    A false positive is cheap and safe: the caller falls back to the joined
    section drafts, which are already validated. It costs the polish, never
    correctness.
    """
    text = str(markdown or "")
    if _GLUED_HEADING_RE.search(text):
        return "glued_heading"
    expected = [_normalise_heading(draft.plan.title) for draft in drafts]
    if not expected:
        return None if text.strip() else "empty_markdown"
    actual = [
        _normalise_heading(match.group(1))
        for match in _HEADING_RE.finditer(text)
        if len(match.group(0)) - len(match.group(0).lstrip("#")) == 2
    ]
    if actual != expected:
        return "missing_or_reordered_section_headings"
    return None


def _document_integrity_violation(
    markdown: str,
    drafts: list[SectionDraft],
) -> str | None:
    structure = _section_structure_violation(markdown, drafts)
    if structure is not None:
        return structure
    mechanical = markdown_structure_violation(markdown)
    if mechanical is not None:
        return mechanical
    for draft in drafts:
        quality = draft.quality if isinstance(draft.quality, Mapping) else {}
        review = quality.get("completeness_review")
        if not isinstance(review, Mapping) or review.get("complete") is not True:
            return f"section_completeness_missing:{draft.plan.section_id}"
        hard_failures = quality.get("hard_failures")
        if isinstance(hard_failures, (list, tuple)) and hard_failures:
            return f"section_blocking_failure:{draft.plan.section_id}"
    return None


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
