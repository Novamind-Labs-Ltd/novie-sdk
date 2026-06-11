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
class DocumentRuntimeContract:
    """Document-oriented runtime settings with no business-domain semantics."""

    outline: DocumentOutlineContract = field(default_factory=DocumentOutlineContract)
    section: DocumentSectionContract = field(default_factory=DocumentSectionContract)
    final: DocumentFinalContract = field(default_factory=DocumentFinalContract)
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
    raw: Mapping[str, Any] = field(default_factory=dict)

    def to_deepagents_spec(self) -> dict[str, Any]:
        """Return fields accepted by DeepAgents before runtime model/tool binding."""
        spec = dict(self.raw)
        spec["name"] = self.name
        if self.description:
            spec["description"] = self.description
        if self.system_prompt:
            spec["system_prompt"] = self.system_prompt
        return spec


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
    subagents: tuple[SubagentContract, ...] = ()
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
            "context_policy": self.runtime.context_policy,
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
        contract_filename: str = "contract.yaml",
    ) -> None:
        self._root_dir = Path(root_dir).resolve() if root_dir is not None else None
        self._contract_filename = contract_filename

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

        if path.name in {self._contract_filename, "SKILL.md"}:
            return path.parent
        return path

    def _load_contract(self, skill_dir: Path) -> tuple[Mapping[str, Any], Path] | None:
        contract_path = skill_dir / self._contract_filename
        if contract_path.exists():
            return _load_yaml_mapping(contract_path), contract_path

        skill_path = skill_dir / "SKILL.md"
        if not skill_path.exists():
            return None
        frontmatter = _load_skill_frontmatter(skill_path)
        raw = frontmatter.get("runtime_contract")
        if isinstance(raw, Mapping):
            return raw, skill_path
        raw = frontmatter.get("contract")
        if isinstance(raw, Mapping):
            return raw, skill_path
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
    artifacts = _mapping(raw.get("artifacts"))
    workpad = _mapping(raw.get("workpad"))

    return SkillRuntimeContract(
        version=_positive_int(raw.get("version"), 1),
        name=str(raw.get("name") or ""),
        runtime=RuntimeContract(
            strategy=str(runtime.get("strategy") or ""),
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
        subagents=tuple(_subagents_from_raw(raw.get("subagents"))),
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
                raw=raw,
            )
        )
    return out


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, Mapping):
        raise SkillContractError(f"Skill contract must be a mapping: {path}")
    return loaded


def _load_skill_frontmatter(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end < 0:
        return {}
    loaded = yaml.safe_load(text[4:end]) or {}
    return loaded if isinstance(loaded, Mapping) else {}


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
