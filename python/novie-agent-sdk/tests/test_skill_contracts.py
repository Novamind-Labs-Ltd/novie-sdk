from __future__ import annotations

from pathlib import Path

import pytest

from novie_agent_sdk import (
    ContextPackBuilder,
    EvidencePackBuilder,
    SkillContractError,
    SkillContractResolver,
)


def test_skill_contract_resolver_merges_contract_yaml_sources(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    shared = root / "skills" / "shared"
    report = root / "skills" / "report"
    shared.mkdir(parents=True)
    report.mkdir(parents=True)
    (shared / "contract.yaml").write_text(
        """
version: 1
name: shared-defaults
runtime:
  strategy: sectioned_document
  context_policy: artifact_ref_context_pack
document:
  outline:
    min_sections: 1
    max_sections: 4
  section:
    min_units: 50
    default_units: 100
    max_units: 200
    max_revision_rounds: 1
workpad:
  record_outline_ref: true
subagents:
  - name: writer
    role: section_writer
    description: Write sections.
""",
        encoding="utf-8",
    )
    (report / "contract.yaml").write_text(
        """
name: report-synthesis
runtime:
  finalization: bounded_polish
task_profile:
  selected_by: llm_structured
  schema:
    length_profile:
      enum: [short, medium, long]
document:
  outline:
    max_sections: 9
  final:
    min_retention_ratio: 0.8
artifacts:
  outline_type: management_report.outline
  section_type: management_report.section
  final_type: management_report
workpad:
  record_section_refs: true
  record_final_deliverable_ref: true
subagents:
  - name: writer
    description: Write one section from bounded context.
  - name: editor
    role: final_editor
""",
        encoding="utf-8",
    )

    contract = SkillContractResolver(root_dir=root).resolve(
        ["/skills/shared/", "/skills/report/"],
        required=True,
    )

    assert contract.name == "report-synthesis"
    assert contract.strategy == "sectioned_document"
    assert contract.runtime.context_policy == "artifact_ref_context_pack"
    assert contract.runtime.finalization == "bounded_polish"
    assert contract.task_profile.selected_by == "llm_structured"
    assert contract.task_profile.schema["length_profile"]["enum"] == [
        "short",
        "medium",
        "long",
    ]
    assert contract.document.outline.min_sections == 1
    assert contract.document.outline.max_sections == 9
    assert contract.document.section.default_units == 100
    assert contract.document.final.min_retention_ratio == 0.8
    assert contract.artifacts.final_type == "management_report"
    assert contract.workpad.record_outline_ref is True
    assert contract.workpad.record_section_refs is True
    assert contract.workpad.record_final_deliverable_ref is True
    assert [item.name for item in contract.subagents] == ["writer", "editor"]
    assert contract.subagents[0].role == "section_writer"
    assert contract.subagents[0].description == "Write one section from bounded context."
    assert len(contract.sources) == 2


def test_skill_contract_resolver_reads_skill_frontmatter_fallback(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "memo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        """---
name: memo
runtime_contract:
  version: 1
  runtime:
    strategy: sectioned_document
  artifacts:
    final_type: legal_memo
---

# Memo Skill
""",
        encoding="utf-8",
    )

    contract = SkillContractResolver(root_dir=tmp_path).resolve(["skills/memo"])

    assert contract.runtime.strategy == "sectioned_document"
    assert contract.artifacts.final_type == "legal_memo"
    assert contract.sources == (str(skill / "SKILL.md"),)


def test_skill_contract_resolver_required_raises_when_missing(tmp_path: Path) -> None:
    (tmp_path / "skills" / "empty").mkdir(parents=True)

    with pytest.raises(SkillContractError, match="No skill runtime contract"):
        SkillContractResolver(root_dir=tmp_path).resolve(["skills/empty"], required=True)


def test_context_pack_builder_is_generic_alias() -> None:
    assert issubclass(ContextPackBuilder, EvidencePackBuilder)
