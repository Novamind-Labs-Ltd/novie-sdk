"""Section-by-section longform authoring for document deliverables."""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .artifact_ledger import ArtifactLedger, EvidencePackBuilder
from .skill_contracts import SkillRuntimeContract

_SECTIONED_AUTHORING_ENV = "NOVIE_SECTIONED_AUTHORING_V2"
_SECTIONED_AUTHORING_DISABLED_ENV = "NOVIE_SECTIONED_AUTHORING_DISABLED"
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

    coverage_model: str = "management_report"
    length_profile: str = "adaptive"
    context_policy: str = "evidence_pack_v1"
    quality_contract_ref: str = "research_report.deep_synthesis"
    min_outline_sections: int = 2
    max_outline_sections: int = 9
    min_section_words: int = 90
    default_section_words: int = 180
    max_section_words: int = 280
    max_section_revision_rounds: int = 1
    final_retention_ratio: float = 0.8
    require_evidence_refs: bool = True
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
            coverage_model=str(raw.get("coverage_model") or "management_report"),
            length_profile=str(raw.get("length_profile") or "adaptive"),
            context_policy=str(raw.get("context_policy") or "evidence_pack_v1"),
            quality_contract_ref=str(
                raw.get("quality_contract_ref") or "research_report.deep_synthesis"
            ),
            min_outline_sections=_positive_int(raw.get("min_outline_sections"), 2),
            max_outline_sections=_positive_int(raw.get("max_outline_sections"), 9),
            min_section_words=_positive_int(raw.get("min_section_words"), 90),
            default_section_words=_positive_int(raw.get("default_section_words"), 180),
            max_section_words=_positive_int(raw.get("max_section_words"), 280),
            max_section_revision_rounds=_positive_int(
                raw.get("max_section_revision_rounds"),
                1,
            ),
            final_retention_ratio=_ratio(raw.get("final_retention_ratio"), 0.8),
            require_evidence_refs=_bool(raw.get("require_evidence_refs"), True),
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

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_metadata(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "information_units": self.information_units,
            "citation_count": self.citation_count,
            "evidence_item_count": self.evidence_item_count,
            "revision_rounds": self.revision_rounds,
        }


@dataclass(frozen=True, slots=True)
class SectionedAuthoringResult:
    markdown: str
    length_profile: str
    outline: tuple[SectionPlan, ...]
    drafts: tuple[SectionDraft, ...]
    ledger: Mapping[str, Any]


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


def sectioned_authoring_contract_from_skill(
    contract: SkillRuntimeContract,
    *,
    artifact_type: str,
) -> dict[str, Any]:
    """Map a generic skill runtime contract to sectioned authoring settings."""
    document = contract.document
    raw_document = dict(document.raw or {})
    defaults = dict(contract.task_profile.defaults or {})
    return {
        "coverage_model": raw_document.get("coverage_model") or artifact_type,
        "length_profile": defaults.get("length_profile")
        or raw_document.get("length_profile")
        or "adaptive",
        "context_policy": contract.runtime.context_policy or "evidence_pack_v1",
        "quality_contract_ref": raw_document.get("quality_contract_ref")
        or contract.name
        or artifact_type,
        "min_outline_sections": document.outline.min_sections or 2,
        "max_outline_sections": document.outline.max_sections or 9,
        "min_section_words": document.section.min_units or 90,
        "default_section_words": document.section.default_units or 180,
        "max_section_words": document.section.max_units or 280,
        "max_section_revision_rounds": document.section.max_revision_rounds or 1,
        "final_retention_ratio": document.final.min_retention_ratio or 0.8,
        "require_evidence_refs": True,
        "outline_artifact_type": contract.artifacts.outline_type
        or f"{artifact_type}.outline",
        "section_artifact_type": contract.artifacts.section_type
        or f"{artifact_type}.section",
        "final_artifact_type": contract.artifacts.final_type or artifact_type,
        "record_outline_ref": contract.workpad.record_outline_ref,
        "record_section_refs": contract.workpad.record_section_refs,
        "record_final_deliverable_ref": contract.workpad.record_final_deliverable_ref,
    }


class SectionedLongformAuthor:
    """Outline, draft, record, and polish a longform report section by section."""

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
    ) -> None:
        self._llm = llm_facade
        self._platform = platform
        self._artifact_type = artifact_type
        self._step_id = step_id
        self._capability_id = capability_id
        self._context_budget = dict(context_budget or {})
        self._contract = (
            authoring_contract
            if isinstance(authoring_contract, SectionedAuthoringContract)
            else SectionedAuthoringContract.from_mapping(authoring_contract)
        )
        self._ledger = ArtifactLedger(platform)
        self._evidence = EvidencePackBuilder(platform, budget=context_budget)
        self._max_section_revision_rounds = _positive_int(
            self._context_budget.get("max_section_revision_rounds"),
            self._contract.max_section_revision_rounds,
        )

    async def author(
        self,
        *,
        brief: Mapping[str, Any],
        upstream: Mapping[str, Any],
        workflow_id: str | None = None,
        thread_id: str | None = None,
        agent_id: str | None = None,
    ) -> SectionedAuthoringResult:
        length_profile, outline = await self._build_outline(brief=brief, upstream=upstream)
        outline_ref = await self._record_outline(
            outline,
            length_profile=length_profile,
            workflow_id=workflow_id,
            thread_id=thread_id,
            agent_id=agent_id,
        )
        drafts: list[SectionDraft] = []
        for index, plan in enumerate(outline, start=1):
            evidence_pack = await self._evidence.build(
                workflow_id=workflow_id,
                upstream=upstream,
                query=plan.evidence_query or plan.title,
                purpose=f"draft section {index}: {plan.title}",
            )
            evidence_pack_input = evidence_pack.to_prompt_input()
            markdown = await self._draft_section(
                brief=brief,
                plan=plan,
                previous=drafts,
                evidence_pack=evidence_pack_input,
            )
            quality = _evaluate_section_quality(
                plan=plan,
                markdown=markdown,
                evidence_pack=evidence_pack_input,
                contract=self._contract,
                revision_rounds=0,
            )
            revision_rounds = 0
            while (
                not quality.passed
                and revision_rounds < self._max_section_revision_rounds
            ):
                revision_rounds += 1
                markdown = await self._draft_section(
                    brief=brief,
                    plan=plan,
                    previous=drafts,
                    evidence_pack=evidence_pack_input,
                    revision_feedback=quality,
                )
                quality = _evaluate_section_quality(
                    plan=plan,
                    markdown=markdown,
                    evidence_pack=evidence_pack_input,
                    contract=self._contract,
                    revision_rounds=revision_rounds,
                )
            if not quality.passed:
                raise RuntimeError(
                    "section_quality_gate_failed:"
                    f"{plan.section_id}:"
                    + ",".join(quality.failures)
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
            drafts.append(
                SectionDraft(
                    plan=plan,
                    markdown=markdown,
                    artifact_ref=artifact_ref,
                    quality=quality.to_metadata(),
                )
            )

        final_markdown = await self._polish_final(brief=brief, drafts=drafts)
        final_ref = await self._record_final(
            final_markdown,
            workflow_id=workflow_id,
            thread_id=thread_id,
            agent_id=agent_id,
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
                "section_count": len(outline),
                "created_count": len(artifact_refs),
                "artifact_refs": artifact_refs,
            },
        )

    async def _build_outline(
        self,
        *,
        brief: Mapping[str, Any],
        upstream: Mapping[str, Any],
    ) -> tuple[str, tuple[SectionPlan, ...]]:
        prompt = (
            "Design a concise longform document outline.\n"
            f"Coverage model: {self._contract.coverage_model}.\n"
            f"Length profile hint: {self._contract.length_profile}.\n"
            "Choose the length_profile from the original task and evidence. "
            "If the hint is adaptive, infer the appropriate profile from the "
            "task semantics. If the user asks for a complete document and does "
            "not constrain length, prefer enough sections for full coverage. "
            f"Return {self._contract.min_outline_sections}-"
            f"{self._contract.max_outline_sections} sections. "
            "Each section must have a focused evidence query and an appropriate "
            "minimum information-unit target.\n\n"
            f"Original task:\n{_json_block(brief, limit=8000)}\n\n"
            f"Available upstream/workpad refs:\n{_json_block(upstream, limit=12000)}"
        )
        result = await self._llm.structured(
            messages=[{"role": "user", "content": prompt}],
            output_schema=_outline_schema(self._contract),
            temperature=0.2,
        )
        structured = result.get("structured") if isinstance(result, Mapping) else None
        raw_sections = structured.get("sections") if isinstance(structured, Mapping) else None
        length_profile = (
            str(structured.get("length_profile") or self._contract.length_profile).strip()
            if isinstance(structured, Mapping)
            else self._contract.length_profile
        )
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
        if plans:
            return length_profile or "adaptive", tuple(plans[: self._contract.max_outline_sections])
        return length_profile or "adaptive", (
            SectionPlan(
                section_id="executive-summary",
                title="Executive Summary",
                objective="Summarize the decision-relevant findings.",
                evidence_query=str(brief.get("title") or brief.get("goal") or "executive summary"),
                min_words=self._contract.default_section_words,
            ),
            SectionPlan(
                section_id="evidence-analysis",
                title="Evidence Analysis",
                objective="Synthesize the strongest upstream evidence.",
                evidence_query="evidence analysis",
                min_words=self._contract.default_section_words,
            ),
            SectionPlan(
                section_id="recommendations",
                title="Recommendations",
                objective="Translate findings into practical recommendations.",
                evidence_query="recommendations implications",
                min_words=self._contract.default_section_words,
            ),
        )

    async def _draft_section(
        self,
        *,
        brief: Mapping[str, Any],
        plan: SectionPlan,
        previous: list[SectionDraft],
        evidence_pack: Mapping[str, Any],
        revision_feedback: SectionQualityGateResult | None = None,
    ) -> str:
        previous_index = [
            {
                "section_id": draft.plan.section_id,
                "title": draft.plan.title,
                "artifact_ref": draft.artifact_ref.get("artifact_ref"),
            }
            for draft in previous
        ]
        prompt = (
            "Write exactly this document section in Markdown.\n"
            "Use the original task, the prior section ledger, and the bounded evidence pack. "
            "Do not include process notes. Cite artifact refs or source refs when evidence is used. "
            f"Write at least {plan.min_words} substantive information units.\n\n"
            f"Original task:\n{_json_block(brief, limit=8000)}\n\n"
            f"Section plan:\n{_json_block(asdict(plan), limit=4000)}\n\n"
            f"Prior section refs:\n{_json_block(previous_index, limit=4000)}\n\n"
            f"Evidence pack:\n{_json_block(evidence_pack, limit=24000)}"
        )
        if revision_feedback is not None:
            prompt += (
                "\n\nSection quality gate failed. Rewrite the same section only, "
                "fixing these deterministic failures without dropping evidence:\n"
                f"{_json_block(revision_feedback.to_metadata(), limit=4000)}"
            )
        result = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
        )
        content = str(result.get("content") or "").strip()
        if not content:
            content = f"## {plan.title}\n\nNo section draft was returned."
        if not content.lstrip().startswith("#"):
            content = f"## {plan.title}\n\n{content}"
        return content

    async def _polish_final(
        self,
        *,
        brief: Mapping[str, Any],
        drafts: list[SectionDraft],
    ) -> str:
        combined = "\n\n".join(draft.markdown.strip() for draft in drafts if draft.markdown.strip())
        refs = [
            {
                "section_id": draft.plan.section_id,
                "title": draft.plan.title,
                "artifact_ref": draft.artifact_ref.get("artifact_ref"),
            }
            for draft in drafts
        ]
        prompt = (
            "Polish the concatenated sections into one coherent final Markdown deliverable. "
            "Preserve factual claims, source refs, and section substance. "
            "Improve transitions and remove repetition without shortening materially.\n\n"
            f"Original task:\n{_json_block(brief, limit=8000)}\n\n"
            f"Section artifact refs:\n{_json_block(refs, limit=6000)}\n\n"
            f"Draft sections:\n{combined[:48000]}"
        )
        result = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        polished = str(result.get("content") or "").strip()
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
        result = await _create_and_record_strict(
            self._ledger,
            artifact_type=self._contract.outline_artifact_type
            or f"{self._artifact_type}.outline",
            content=content,
            kind="outline",
            title=f"{self._artifact_type} outline",
            content_type="application/json",
            summary=f"{self._artifact_type} outline with {len(outline)} sections",
            workflow_id=workflow_id,
            thread_id=thread_id,
            step_id=self._step_id,
            agent_id=agent_id,
            capability_id=self._capability_id,
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
        result = await _create_and_record_strict(
            self._ledger,
            artifact_type=self._contract.section_artifact_type
            or f"{self._artifact_type}.section",
            content=markdown,
            kind="section_draft",
            title=plan.title,
            content_type="text/markdown",
            summary=_preview(markdown),
            workflow_id=workflow_id,
            thread_id=thread_id,
            step_id=self._step_id,
            agent_id=agent_id,
            capability_id=self._capability_id,
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
        result = await _create_and_record_strict(
            self._ledger,
            artifact_type=self._contract.final_artifact_type or self._artifact_type,
            content=markdown,
            kind="final_deliverable",
            title=f"{self._artifact_type} final deliverable",
            content_type="text/markdown",
            summary=_preview(markdown),
            workflow_id=workflow_id,
            thread_id=thread_id,
            step_id=self._step_id,
            agent_id=agent_id,
            capability_id=self._capability_id,
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
    workpad = result.get("workpad")
    if isinstance(artifact, Mapping) and artifact.get("available", True) is False:
        raise RuntimeError(
            "artifact_create_failed:"
            f"{artifact.get('error') or 'artifact_create_unavailable'}"
        )
    if not isinstance(artifact, Mapping) or not artifact.get("artifact_ref"):
        raise RuntimeError("artifact_create_failed:artifact_ref_missing")
    if isinstance(workpad, Mapping) and workpad.get("available", True) is False:
        raise RuntimeError(
            "workpad_record_failed:"
            f"{workpad.get('error') or 'workpad_record_unavailable'}"
        )
    if not isinstance(workpad, Mapping):
        raise RuntimeError("workpad_record_failed:workpad_result_missing")


def _evaluate_section_quality(
    *,
    plan: SectionPlan,
    markdown: str,
    evidence_pack: Mapping[str, Any],
    contract: SectionedAuthoringContract,
    revision_rounds: int,
) -> SectionQualityGateResult:
    text = str(markdown or "").strip()
    failures: list[str] = []
    information_units = _information_units(text)
    evidence_items = _evidence_items(evidence_pack)
    citation_count = _citation_count(text, evidence_items)

    if not text:
        failures.append("empty_section")
    if not _section_has_heading(text, plan.title):
        failures.append("missing_section_heading")
    if information_units < plan.min_words:
        failures.append("insufficient_section_depth")
    if _INTERNAL_PROCESS_RE.search(text):
        failures.append("internal_process_language")
    if contract.require_evidence_refs and evidence_items and citation_count == 0:
        failures.append("missing_evidence_reference")

    return SectionQualityGateResult(
        failures=tuple(dict.fromkeys(failures)),
        information_units=information_units,
        citation_count=citation_count,
        evidence_item_count=len(evidence_items),
        revision_rounds=revision_rounds,
    )


def _section_has_heading(markdown: str, title: str) -> bool:
    wanted = _normalise_heading(title)
    for match in _HEADING_RE.finditer(str(markdown or "")):
        if _normalise_heading(match.group(1)) == wanted:
            return True
    return False


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


def _outline_schema(contract: SectionedAuthoringContract) -> dict[str, Any]:
    return {
        # LangChain uses the title as the structured-output function name;
        # a title-less dict schema is rejected by with_structured_output.
        "title": "report_outline_plan",
        "type": "object",
        "additionalProperties": False,
        "required": ["length_profile", "sections"],
        "properties": {
            "length_profile": {
                "type": "string",
                "enum": ["short", "medium", "long"],
            },
            "sections": {
                "type": "array",
                "minItems": contract.min_outline_sections,
                "maxItems": contract.max_outline_sections,
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
