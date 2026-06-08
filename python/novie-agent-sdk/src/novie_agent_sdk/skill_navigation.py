"""Bounded skill-scope compilation for document agents."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DOCUMENT_SKILL_SECTIONS: tuple[str, ...] = (
    "When to Use",
    "Research Focus",
    "Analysis Framework",
    "Synthesis Process",
    "Section Depth Contract",
    "Artifact Fetch Discipline",
    "Section Evidence Index",
    "Output Contract",
    "Output Expectations",
    "Evidence Cards to Include",
    "Finalization Requirements",
    "Quality Bar",
)


@dataclass(frozen=True)
class SkillMetadata:
    """Parsed SKILL.md metadata used for bounded navigation hints."""

    source: str
    name: str
    description: str
    allowed_tools: tuple[str, ...]
    instruction_digest: str = ""
    finalization_requirements: str = ""


@dataclass(frozen=True)
class SkillScope:
    """DeepAgents skill scope plus prompt-facing navigation hints."""

    skill_sources: list[str]
    prompt_hint: str
    skills: tuple[SkillMetadata, ...] = ()
    allowed_tools: tuple[str, ...] = ()


def _strip_scalar(value: str) -> str:
    out = value.strip()
    if len(out) >= 2 and out[0] == out[-1] and out[0] in {"'", '"'}:
        return out[1:-1]
    return out


def _frontmatter_block(text: str) -> str:
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md must start with YAML frontmatter")
    end = text.find("\n---", 4)
    if end < 0:
        raise ValueError("SKILL.md frontmatter is not closed")
    return text[4:end]


def parse_skill_frontmatter(text: str) -> dict[str, Any]:
    """Parse the simple SKILL.md frontmatter shape used by Novie skills."""
    parsed: dict[str, Any] = {}
    current_map: str | None = None
    for raw_line in _frontmatter_block(text).splitlines():
        if not raw_line.strip():
            continue
        if raw_line.startswith((" ", "\t")):
            if current_map is None:
                continue
            key, sep, value = raw_line.strip().partition(":")
            if sep:
                target = parsed.setdefault(current_map, {})
                if isinstance(target, dict):
                    target[key.strip()] = _strip_scalar(value)
            continue
        current_map = None
        key, sep, value = raw_line.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if value:
            parsed[key] = _strip_scalar(value)
        else:
            parsed[key] = {}
            current_map = key
    return parsed


def skill_source_to_file(package_root: Path, source: str) -> Path:
    """Resolve a canonical virtual skill source to its bundled SKILL.md."""
    normalized = "/" + source.strip().strip("/")
    prefix = "/skills/"
    if not normalized.startswith(prefix):
        raise ValueError(f"skill source must live under /skills: {source!r}")
    relative = normalized[len(prefix):]
    return package_root / "skills" / relative / "SKILL.md"


def extract_markdown_section(text: str, heading: str) -> str:
    target = heading.strip().lower()
    lines = str(text or "").splitlines()
    start: int | None = None
    start_level = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        marker, _, title = stripped.partition(" ")
        if not marker or set(marker) != {"#"}:
            continue
        if title.strip().lower() == target:
            start = index + 1
            start_level = len(marker)
            break
    if start is None:
        return ""
    section: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("#"):
            marker, _, _title = stripped.partition(" ")
            if marker and set(marker) == {"#"} and len(marker) <= start_level:
                break
        section.append(line)
    return "\n".join(section).strip()


def bounded_text(text: str, *, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 14)].rstrip() + "\n[condensed]"


def skill_instruction_digest(
    raw: str,
    *,
    section_names: Sequence[str] = DEFAULT_DOCUMENT_SKILL_SECTIONS,
    max_chars: int = 2200,
) -> str:
    """Compile prompt-needed skill guidance without runtime file-tool reads."""
    sections: list[str] = []
    for name in section_names:
        body = extract_markdown_section(raw, name)
        if body:
            sections.append(f"## {name}\n{body}")
    if not sections:
        body = raw.split("\n---", 1)[-1] if raw.startswith("---\n") else raw
        sections.append(body)
    return bounded_text("\n\n".join(sections), max_chars=max_chars)


def load_skill_metadata(
    spec: Any,
    source: str,
    *,
    section_names: Sequence[str] = DEFAULT_DOCUMENT_SKILL_SECTIONS,
    max_digest_chars: int = 2200,
) -> SkillMetadata:
    """Load metadata from a canonical virtual skill source."""
    package_root = Path(getattr(spec, "package_root"))
    skill_file = skill_source_to_file(package_root, source)
    raw = skill_file.read_text(encoding="utf-8")
    frontmatter = parse_skill_frontmatter(raw)
    name = str(frontmatter.get("name") or "").strip()
    description = str(frontmatter.get("description") or "").strip()
    allowed_tools_raw = str(frontmatter.get("allowed-tools") or "").strip()
    allowed_tools = tuple(
        item.strip()
        for item in allowed_tools_raw.split(",")
        if item.strip()
    )
    if not name:
        raise ValueError(f"skill metadata missing name: {source!r}")
    if not description:
        raise ValueError(f"skill metadata missing description: {source!r}")
    return SkillMetadata(
        source=source,
        name=name,
        description=description,
        allowed_tools=allowed_tools,
        instruction_digest=skill_instruction_digest(
            raw,
            section_names=section_names,
            max_chars=max_digest_chars,
        ),
        finalization_requirements=extract_markdown_section(
            raw,
            "Finalization Requirements",
        ),
    )


def load_skill_metadata_for_sources(
    spec: Any,
    sources: list[str],
    *,
    section_names: Sequence[str] = DEFAULT_DOCUMENT_SKILL_SECTIONS,
    max_digest_chars: int = 2200,
) -> tuple[SkillMetadata, ...]:
    return tuple(
        load_skill_metadata(
            spec,
            source,
            section_names=section_names,
            max_digest_chars=max_digest_chars,
        )
        for source in sources
    )


def compile_skill_scope(
    spec: Any | None,
    *,
    upstream: dict[str, Any] | None = None,
    source_resolver: Callable[[list[str], dict[str, Any]], list[str] | tuple[str, ...]] | None = None,
    section_names: Sequence[str] = DEFAULT_DOCUMENT_SKILL_SECTIONS,
    max_digest_chars: int = 2200,
    synthesis_allowed_tools: tuple[str, ...] = ("fetch_artifact",),
    no_file_tool_warning: bool = True,
) -> SkillScope:
    """Compile the deterministic boundary within which DeepAgents may choose."""
    if spec is None:
        return SkillScope(skill_sources=[], prompt_hint="")

    base_sources = list(getattr(spec, "skill_sources", []) or [])
    synthesis_path = bool(getattr(spec, "synthesis_path", False))
    if synthesis_path and source_resolver is not None:
        sources = list(source_resolver(base_sources, upstream or {}))
    else:
        sources = base_sources

    skills = load_skill_metadata_for_sources(
        spec,
        sources,
        section_names=section_names,
        max_digest_chars=max_digest_chars,
    )
    if synthesis_path:
        effective_allowed_tools = synthesis_allowed_tools
    else:
        effective_allowed_tools = tuple(
            sorted({tool for skill in skills for tool in skill.allowed_tools})
        )

    hint_parts = [
        f"Current capability: {getattr(spec, 'capability_id', '')}",
        f"Target artifact: {getattr(spec, 'artifact_type', '')}",
        f"Artifact access: {getattr(spec, 'artifact_access', 'summary_then_fetch')}",
    ]
    research_track = getattr(spec, "research_track", None)
    if research_track:
        hint_parts.append(f"Preferred research track: {research_track}")
    if synthesis_path:
        hint_parts.append(
            "Synthesis mode: use upstream handoff summaries and artifact reads; "
            "do not start fresh research unless explicitly allowed."
        )
    if skills:
        hint_parts.append(
            "Available bounded skills:\n"
            + "\n".join(
                f"- {skill.name}: {skill.description}" for skill in skills
            )
        )
        if effective_allowed_tools:
            hint_parts.append(
                "Allowed tools in scope: " + ", ".join(effective_allowed_tools)
            )
        instruction_blocks = [
            f"### {skill.name}\n{skill.instruction_digest}"
            for skill in skills
            if skill.instruction_digest
        ]
        if instruction_blocks:
            warning = (
                "These bounded skill instructions are already loaded into this "
                "prompt. Do not call read_file to inspect /skills or write_todos "
                "to restate the skill plan; use these instructions directly.\n\n"
                if no_file_tool_warning
                else ""
            )
            hint_parts.append(
                "Loaded skill instructions:\n"
                + warning
                + "\n\n".join(instruction_blocks)
            )
        finalization_blocks = [
            f"- {skill.name}:\n{skill.finalization_requirements}"
            for skill in skills
            if skill.finalization_requirements
        ]
        if finalization_blocks:
            hint_parts.append(
                "Final editor contract:\n"
                "Use the finalization requirements from the matching loaded "
                "document skill. Do not add generic recommendation, action-plan, "
                "validation-plan, PRD, or architecture sections unless that "
                "matched skill/task calls for them.\n"
                + "\n\n".join(finalization_blocks)
            )

    return SkillScope(
        skill_sources=sources,
        prompt_hint="\n".join(hint_parts),
        skills=skills,
        allowed_tools=effective_allowed_tools,
    )


__all__ = [
    "DEFAULT_DOCUMENT_SKILL_SECTIONS",
    "SkillMetadata",
    "SkillScope",
    "bounded_text",
    "compile_skill_scope",
    "extract_markdown_section",
    "load_skill_metadata",
    "load_skill_metadata_for_sources",
    "parse_skill_frontmatter",
    "skill_instruction_digest",
    "skill_source_to_file",
]
