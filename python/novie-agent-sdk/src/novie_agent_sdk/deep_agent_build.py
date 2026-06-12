"""DeepAgents assembly helpers for bounded document-agent skill scopes."""
from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .skill_contracts import SkillContractError, SkillRuntimeContract

_log = logging.getLogger(__name__)

_MATERIALIZED_SKILL_SOURCE = "/skills/"


def canonical_skill_path(package_root: Path, source: str) -> Path:
    """Resolve a canonical virtual skill directory into an on-disk directory."""
    normalized = source.strip().strip("/")
    if not normalized:
        raise ValueError("skill source path must not be empty")

    path = (package_root / normalized).resolve()
    root = package_root.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"skill source escapes package root: {source!r}")
    if not (path / "SKILL.md").is_file():
        raise FileNotFoundError(f"skill source does not contain SKILL.md: {source!r}")
    return path


def materialized_skill_dir_name(source_path: Path) -> str:
    """Return the Agent Skills compliant directory name for a canonical skill."""
    return source_path.name.replace("_", "-")


def materialize_skill_collection(
    package_root: Path,
    skill_sources: list[str],
    *,
    cache_namespace: str = "novie-document-agent",
) -> tuple[Path, list[str]]:
    """Create a DeepAgents collection directory for selected canonical skills."""
    resolved = [canonical_skill_path(package_root, source) for source in skill_sources]
    names = [materialized_skill_dir_name(path) for path in resolved]
    duplicate_names = {name for name in names if names.count(name) > 1}
    if duplicate_names:
        duplicates = ", ".join(sorted(duplicate_names))
        raise ValueError(f"duplicate skill directory names in capability spec: {duplicates}")

    digest_input = "\n".join(
        [
            str(package_root.resolve()),
            *[
                f"{source}->{name}"
                for source, name in zip(skill_sources, names, strict=True)
            ],
        ]
    )
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16]
    root = Path(tempfile.gettempdir()) / f"{cache_namespace}-skill-collections" / digest
    collection_root = root / _MATERIALIZED_SKILL_SOURCE.strip("/")
    collection_root.mkdir(parents=True, exist_ok=True)

    for source_path in resolved:
        shutil.copytree(
            source_path,
            collection_root / materialized_skill_dir_name(source_path),
            dirs_exist_ok=True,
        )

    return root, [_MATERIALIZED_SKILL_SOURCE]


def build_deep_agent_executor(
    *,
    package_root: Path,
    model: Any,
    tools: Sequence[Any],
    system_prompt: Any,
    skill_sources: list[str] | None,
    middleware: Sequence[Any] = (),
    subagents: Sequence[Any] | None = None,
    native_skill_loading: bool = False,
    strict_skills: bool = False,
    skill_contract: SkillRuntimeContract | None = None,
    cache_namespace: str = "novie-document-agent",
) -> Any:
    """Build a DeepAgents executor for one bounded document capability scope."""
    from deepagents import create_deep_agent  # type: ignore[import-untyped]
    from deepagents.backends.filesystem import FilesystemBackend  # type: ignore[import-untyped]

    if strict_skills:
        if not skill_sources:
            raise SkillContractError("strict DeepAgents runtime requires skill_sources")
        if skill_contract is None or skill_contract.is_empty:
            raise SkillContractError("strict DeepAgents runtime requires a skill contract")
        native_skill_loading = True

    if skill_contract is not None and not skill_contract.is_empty:
        _validate_required_tools(
            tools,
            required_tools=skill_contract.required_tools,
            subagents=skill_contract.subagents,
        )

    if native_skill_loading and skill_sources:
        collection_root, deepagent_skill_sources = materialize_skill_collection(
            package_root,
            skill_sources,
            cache_namespace=cache_namespace,
        )
    else:
        collection_root = Path(tempfile.mkdtemp(prefix=f"{cache_namespace}-deepagents-"))
        deepagent_skill_sources = []
    backend = FilesystemBackend(root_dir=str(collection_root), virtual_mode=True)
    if middleware:
        _log.debug(
            "building document deep agent with %d custom middleware entries",
            len(middleware),
        )
    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        backend=backend,
        skills=deepagent_skill_sources,
        subagents=list(subagents or ()),
        middleware=tuple(middleware),
    )


def _validate_required_tools(
    tools: Sequence[Any],
    *,
    required_tools: Sequence[str],
    subagents: Sequence[Any],
) -> None:
    available = {str(getattr(tool, "name", "") or "") for tool in tools}
    missing = [name for name in required_tools if name and name not in available]
    for subagent in subagents:
        raw_tools = getattr(subagent, "tools", ())
        for name in raw_tools:
            if name and name not in available:
                missing.append(f"{getattr(subagent, 'name', 'subagent')}:{name}")
        raw_skills = getattr(subagent, "skills", ())
        for source in raw_skills:
            if str(source or "").strip():
                continue
            missing.append(f"{getattr(subagent, 'name', 'subagent')}:empty_skill")
    if missing:
        raise SkillContractError(
            "strict DeepAgents runtime missing required tools/skills: "
            + ", ".join(dict.fromkeys(missing))
        )


__all__ = [
    "build_deep_agent_executor",
    "canonical_skill_path",
    "materialize_skill_collection",
    "materialized_skill_dir_name",
]
