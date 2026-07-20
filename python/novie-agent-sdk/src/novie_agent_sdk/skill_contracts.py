"""Runtime contract loading for skill-driven agents.

Skills are LLM-facing instruction bundles. Some skills also need a small,
machine-readable runtime contract so agent code can choose a generic execution
loop without hardcoding business policy in Python. This module reads those
contracts from skill directories and returns typed, agent-agnostic objects.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class RuntimeContract:
    """Generic runtime strategy declared by a skill."""

    strategy: str = ""
    preparation: str = "direct"
    context_policy: str = ""
    finalization: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskProfileContract:
    """LLM-selectable profile schema declared by a skill."""

    selected_by: str = ""
    schema: Mapping[str, Any] = field(default_factory=dict)
    defaults: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DocumentOutlineContract:
    """Generic outline bounds for document-style runtimes."""

    min_sections: int = 0
    max_sections: int = 0


@dataclass(frozen=True, slots=True)
class DocumentSectionContract:
    """Generic per-section bounds for document-style runtimes."""

    min_units: int = 0
    default_units: int = 0
    max_units: int = 0
    max_revision_rounds: int = 0


@dataclass(frozen=True, slots=True)
class DocumentFinalContract:
    """Generic final assembly/polish bounds for document-style runtimes."""

    min_retention_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class DocumentLengthProfileContract:
    """Length-profile override for document-style runtimes."""

    name: str
    strategy: str = ""
    finalization: str = ""
    evidence_depth: str = ""
    min_sections: int = 0
    max_sections: int = 0
    min_units: int = 0
    default_units: int = 0
    max_units: int = 0
    max_revision_rounds: int = 0
    max_document_output_tokens: int = 0
    final_retention_ratio: float = 0.0
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DocumentRuntimeContract:
    """Document-oriented runtime settings with no business-domain semantics."""

    outline: DocumentOutlineContract = field(default_factory=DocumentOutlineContract)
    section: DocumentSectionContract = field(default_factory=DocumentSectionContract)
    final: DocumentFinalContract = field(default_factory=DocumentFinalContract)
    length_profiles: Mapping[str, DocumentLengthProfileContract] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ArtifactPolicy:
    """Artifact type mapping declared by a skill contract."""

    outline_type: str = ""
    section_type: str = ""
    final_type: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkpadPolicy:
    """Workpad recording policy declared by a skill contract."""

    record_outline_ref: bool = False
    record_section_refs: bool = False
    record_final_deliverable_ref: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SubagentContract:
    """Subagent declaration from a skill contract.

    The SDK keeps this generic. Agent runtimes decide how to map ``role`` to
    concrete prompts, tools, and model choices.
    """

    name: str
    role: str = ""
    description: str = ""
    system_prompt: str = ""
    tools: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)

    def to_deepagents_spec(self) -> dict[str, Any]:
        """Return fields accepted by DeepAgents before runtime model/tool binding."""
        spec = dict(self.raw)
        spec["name"] = self.name
        if self.description:
            spec["description"] = self.description
        if self.system_prompt:
            spec["system_prompt"] = self.system_prompt
        if self.tools:
            spec["tools"] = list(self.tools)
        if self.skills:
            spec["skills"] = list(self.skills)
        return spec


@dataclass(frozen=True, slots=True)
class QualityGatePolicy:
    """Machine-readable quality gate thresholds declared by a skill."""

    require_evidence_refs: bool = False
    require_confidence_layer: bool = False
    forbid_step_artifact_only_citations: bool = False
    min_unique_sources_per_core_section: int = 0
    max_reuse_per_evidence_ref: int = 0
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SkillRuntimeContract:
    """Merged runtime contract resolved from one or more skill sources."""

    version: int = 1
    name: str = ""
    runtime: RuntimeContract = field(default_factory=RuntimeContract)
    task_profile: TaskProfileContract = field(default_factory=TaskProfileContract)
    document: DocumentRuntimeContract = field(default_factory=DocumentRuntimeContract)
    artifacts: ArtifactPolicy = field(default_factory=ArtifactPolicy)
    workpad: WorkpadPolicy = field(default_factory=WorkpadPolicy)
    quality_gates: QualityGatePolicy = field(default_factory=QualityGatePolicy)
    subagents: tuple[SubagentContract, ...] = ()
    required_tools: tuple[str, ...] = ()
    strict_runtime: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict)
    sources: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.raw

    @property
    def strategy(self) -> str:
        return self.runtime.strategy

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "strategy": self.runtime.strategy,
            "preparation": self.runtime.preparation,
            "context_policy": self.runtime.context_policy,
            "length_profiles": sorted(self.document.length_profiles),
            "sources": list(self.sources),
        }


class SkillContractError(RuntimeError):
    """Raised when a required skill runtime contract cannot be loaded."""


class SkillContractResolver:
    """Resolve structured runtime contracts from skill directories.

    ``skill_sources`` may be normal filesystem paths or virtual paths such as
    ``/skills/analyst/report_synthesis/``. Virtual paths are resolved relative
    to ``root_dir``.
    """

    def __init__(
        self,
        *,
        root_dir: str | Path | None = None,
    ) -> None:
        self._root_dir = Path(root_dir).resolve() if root_dir is not None else None

    def resolve(
        self,
        skill_sources: Sequence[str | Path],
        *,
        required: bool = False,
    ) -> SkillRuntimeContract:
        merged: dict[str, Any] = {}
        sources: list[str] = []
        warnings: list[str] = []

        for source in skill_sources:
            skill_dir = self._resolve_skill_dir(source)
            loaded = self._load_contract(skill_dir)
            if loaded is None:
                warnings.append(f"{skill_dir}:contract_not_found")
                continue
            raw, path = loaded
            if not isinstance(raw, Mapping):
                warnings.append(f"{path}:contract_not_mapping")
                continue
            merged = _deep_merge(merged, dict(raw))
            sources.append(str(path))

        if not merged and required:
            raise SkillContractError(
                "No skill runtime contract found for sources: "
                + ", ".join(str(item) for item in skill_sources)
            )
        return _contract_from_mapping(merged, sources=tuple(sources), warnings=tuple(warnings))

    def _resolve_skill_dir(self, source: str | Path) -> Path:
        raw = Path(str(source))
        if raw.is_absolute() and self._root_dir is None:
            path = raw
        elif str(source).startswith("/") and self._root_dir is not None:
            path = self._root_dir / str(source).lstrip("/")
        elif self._root_dir is not None:
            path = self._root_dir / raw
        else:
            path = raw

        if path.name == "contract.yaml":
            raise SkillContractError(
                "contract.yaml is no longer supported; declare "
                "metadata.novie.runtime_contract in SKILL.md"
            )
        if path.name == "SKILL.md":
            return path.parent
        return path

    def _load_contract(self, skill_dir: Path) -> tuple[Mapping[str, Any], Path] | None:
        skill_path = skill_dir / "SKILL.md"
        contract_path = skill_dir / "contract.yaml"
        if contract_path.exists():
            raise SkillContractError(
                "contract.yaml is no longer supported; declare "
                f"metadata.novie.runtime_contract in {skill_path}"
            )
        if skill_path.exists():
            frontmatter = _load_skill_frontmatter(skill_path)
            skill_raw = _frontmatter_runtime_contract(frontmatter)
            if skill_raw is not None:
                return skill_raw, skill_path
        return None


def _contract_from_mapping(
    raw: Mapping[str, Any],
    *,
    sources: tuple[str, ...],
    warnings: tuple[str, ...],
) -> SkillRuntimeContract:
    runtime = _mapping(raw.get("runtime"))
    task_profile = _mapping(raw.get("task_profile"))
    document = _mapping(raw.get("document"))
    outline = _mapping(document.get("outline"))
    section = _mapping(document.get("section"))
    final = _mapping(document.get("final"))
    length_profiles = _length_profiles_from_raw(document.get("length_profiles"))
    artifacts = _mapping(raw.get("artifacts"))
    workpad = _mapping(raw.get("workpad"))
    quality_gates = _mapping(raw.get("quality_gates"))

    return SkillRuntimeContract(
        version=_positive_int(raw.get("version"), 1),
        name=str(raw.get("name") or ""),
        runtime=RuntimeContract(
            strategy=str(runtime.get("strategy") or ""),
            preparation=str(runtime.get("preparation") or "direct"),
            context_policy=str(runtime.get("context_policy") or ""),
            finalization=str(runtime.get("finalization") or ""),
            raw=runtime,
        ),
        task_profile=TaskProfileContract(
            selected_by=str(task_profile.get("selected_by") or ""),
            schema=_mapping(task_profile.get("schema")),
            defaults=_mapping(task_profile.get("defaults")),
            raw=task_profile,
        ),
        document=DocumentRuntimeContract(
            outline=DocumentOutlineContract(
                min_sections=_non_negative_int(outline.get("min_sections"), 0),
                max_sections=_non_negative_int(outline.get("max_sections"), 0),
            ),
            section=DocumentSectionContract(
                min_units=_non_negative_int(section.get("min_units"), 0),
                default_units=_non_negative_int(section.get("default_units"), 0),
                max_units=_non_negative_int(section.get("max_units"), 0),
                max_revision_rounds=_non_negative_int(section.get("max_revision_rounds"), 0),
            ),
            final=DocumentFinalContract(
                min_retention_ratio=_ratio(final.get("min_retention_ratio"), 0.0),
            ),
            length_profiles=length_profiles,
            raw=document,
        ),
        artifacts=ArtifactPolicy(
            outline_type=str(artifacts.get("outline_type") or ""),
            section_type=str(artifacts.get("section_type") or ""),
            final_type=str(artifacts.get("final_type") or ""),
            raw=artifacts,
        ),
        workpad=WorkpadPolicy(
            record_outline_ref=_bool(workpad.get("record_outline_ref"), False),
            record_section_refs=_bool(workpad.get("record_section_refs"), False),
            record_final_deliverable_ref=_bool(
                workpad.get("record_final_deliverable_ref"),
                False,
            ),
            raw=workpad,
        ),
        quality_gates=QualityGatePolicy(
            require_evidence_refs=_bool(
                quality_gates.get("require_evidence_refs"),
                False,
            ),
            require_confidence_layer=_bool(
                quality_gates.get("require_confidence_layer"),
                False,
            ),
            forbid_step_artifact_only_citations=_bool(
                quality_gates.get("forbid_step_artifact_only_citations"),
                False,
            ),
            min_unique_sources_per_core_section=_non_negative_int(
                quality_gates.get("min_unique_sources_per_core_section"),
                0,
            ),
            max_reuse_per_evidence_ref=_non_negative_int(
                quality_gates.get("max_reuse_per_evidence_ref"),
                0,
            ),
            raw=quality_gates,
        ),
        subagents=tuple(_subagents_from_raw(raw.get("subagents"))),
        required_tools=tuple(_strings(raw.get("required_tools"))),
        strict_runtime=_bool(raw.get("strict_runtime"), False),
        raw=dict(raw),
        sources=sources,
        warnings=warnings,
    )


def _subagents_from_raw(value: Any) -> list[SubagentContract]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    out: list[SubagentContract] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        raw = dict(item)
        out.append(
            SubagentContract(
                name=name,
                role=str(raw.get("role") or ""),
                description=str(raw.get("description") or ""),
                system_prompt=str(raw.get("system_prompt") or ""),
                tools=tuple(_strings(raw.get("tools"))),
                skills=tuple(_strings(raw.get("skills"))),
                raw=raw,
            )
        )
    return out


def _length_profiles_from_raw(value: Any) -> dict[str, DocumentLengthProfileContract]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, DocumentLengthProfileContract] = {}
    for key, raw_value in value.items():
        name = str(key or "").strip().lower()
        if not name or not isinstance(raw_value, Mapping):
            continue
        raw = dict(raw_value)
        out[name] = DocumentLengthProfileContract(
            name=name,
            strategy=str(raw.get("strategy") or ""),
            finalization=str(raw.get("finalization") or ""),
            evidence_depth=str(raw.get("evidence_depth") or ""),
            min_sections=_non_negative_int(raw.get("min_sections"), 0),
            max_sections=_non_negative_int(raw.get("max_sections"), 0),
            min_units=_non_negative_int(raw.get("min_units"), 0),
            default_units=_non_negative_int(raw.get("default_units"), 0),
            max_units=_non_negative_int(raw.get("max_units"), 0),
            max_revision_rounds=_non_negative_int(raw.get("max_revision_rounds"), 0),
            max_document_output_tokens=_non_negative_int(
                raw.get("max_document_output_tokens"),
                0,
            ),
            final_retention_ratio=_ratio(raw.get("final_retention_ratio"), 0.0),
            raw=raw,
        )
    return out


def _load_skill_frontmatter(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end < 0:
        return {}
    loaded = yaml.safe_load(text[4:end]) or {}
    return loaded if isinstance(loaded, Mapping) else {}


def _frontmatter_runtime_contract(frontmatter: Mapping[str, Any]) -> Mapping[str, Any] | None:
    metadata = _mapping(frontmatter.get("metadata"))
    novie = _mapping(metadata.get("novie"))
    raw = novie.get("runtime_contract")
    if isinstance(raw, Mapping):
        return raw
    return None


def _deep_merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        elif key == "subagents":
            merged[key] = _merge_subagents(merged.get(key), value)
        else:
            merged[key] = value
    return merged


def _merge_subagents(left: Any, right: Any) -> list[Any]:
    items: list[Any] = []
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        items.extend(left)
    if isinstance(right, Sequence) and not isinstance(right, (str, bytes)):
        by_name: dict[str, Any] = {}
        order: list[str] = []
        for item in items:
            if isinstance(item, Mapping):
                name = str(item.get("name") or "")
                if name:
                    by_name[name] = dict(item)
                    order.append(name)
        for item in right:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name") or "")
            if not name:
                continue
            if name not in by_name:
                order.append(name)
            by_name[name] = _deep_merge(_mapping(by_name.get(name)), item)
        return [by_name[name] for name in order if name in by_name]
    return items


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _ratio(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < 0.0:
        return default
    return min(parsed, 1.0)


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    return default
