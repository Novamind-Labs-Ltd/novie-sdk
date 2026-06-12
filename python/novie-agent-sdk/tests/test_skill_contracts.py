from __future__ import annotations

from pathlib import Path

import pytest

from novie_agent_sdk import (
    ContextPackBuilder,
    EvidencePackBuilder,
    SkillContractError,
    SkillContractResolver,
)


def test_skill_contract_resolver_merges_skill_frontmatter_sources(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    shared = root / "skills" / "shared"
    report = root / "skills" / "report"
    shared.mkdir(parents=True)
    report.mkdir(parents=True)
    (shared / "SKILL.md").write_text(
        """---
name: shared
metadata:
  novie:
    runtime_contract:
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
---

# Shared Skill
""",
        encoding="utf-8",
    )
    (report / "SKILL.md").write_text(
        """---
name: report
metadata:
  novie:
    runtime_contract:
      name: document-authoring
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
        outline_type: document.outline
        section_type: document.section
        final_type: document
      workpad:
        record_section_refs: true
        record_final_deliverable_ref: true
      subagents:
        - name: writer
          description: Write one section from bounded context.
        - name: editor
          role: final_editor
---

# Report Skill
""",
        encoding="utf-8",
    )

    contract = SkillContractResolver(root_dir=root).resolve(
        ["/skills/shared/", "/skills/report/"],
        required=True,
    )

    assert contract.name == "document-authoring"
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
    assert contract.artifacts.final_type == "document"
    assert contract.workpad.record_outline_ref is True
    assert contract.workpad.record_section_refs is True
    assert contract.workpad.record_final_deliverable_ref is True
    assert [item.name for item in contract.subagents] == ["writer", "editor"]
    assert contract.subagents[0].role == "section_writer"
    assert contract.subagents[0].description == "Write one section from bounded context."
    assert len(contract.sources) == 2


def test_skill_contract_resolver_rejects_top_level_frontmatter_contract(tmp_path: Path) -> None:
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

    with pytest.raises(SkillContractError, match="No skill runtime contract"):
        SkillContractResolver(root_dir=tmp_path).resolve(["skills/memo"], required=True)


def test_skill_contract_resolver_reads_novie_metadata_contract(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "report"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        """---
name: report
metadata:
  novie:
    runtime_contract:
      version: 1
      runtime:
        strategy: sectioned_longform
      quality_gates:
        require_evidence_refs: true
        require_confidence_layer: true
        min_unique_sources_per_core_section: 2
      document:
        length_profiles:
          long:
            strategy: sectioned_longform
            min_sections: 8
            max_sections: 16
            min_units: 260
            default_units: 520
            max_units: 900
            max_revision_rounds: 2
            finalization: progressive_section_merge
            evidence_depth: deep
      subagents:
        - name: researcher
          tools: [web_research, fetch_artifact]
          skills:
            - /skills/analyst/research/
---

# Report Skill
""",
        encoding="utf-8",
    )

    contract = SkillContractResolver(root_dir=tmp_path).resolve(
        ["skills/report"],
        required=True,
    )

    assert contract.strategy == "sectioned_longform"
    assert contract.quality_gates.require_evidence_refs is True
    assert contract.quality_gates.require_confidence_layer is True
    assert contract.quality_gates.min_unique_sources_per_core_section == 2
    assert contract.document.length_profiles["long"].max_sections == 16
    assert contract.document.length_profiles["long"].default_units == 520
    assert contract.document.length_profiles["long"].finalization == "progressive_section_merge"
    assert contract.subagents[0].tools == ("web_research", "fetch_artifact")
    assert contract.subagents[0].skills == ("/skills/analyst/research/",)
    assert contract.sources == (str(skill / "SKILL.md"),)


def test_skill_contract_resolver_rejects_contract_yaml_source(
    tmp_path: Path,
) -> None:
    skill = tmp_path / "skills" / "report"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        """---
name: report
metadata:
  novie:
    runtime_contract:
      version: 1
      runtime:
        strategy: sectioned_longform
---

# Report Skill
""",
        encoding="utf-8",
    )
    (skill / "contract.yaml").write_text(
        """
version: 1
runtime:
  strategy: other
""",
        encoding="utf-8",
    )

    with pytest.raises(SkillContractError, match="contract.yaml is no longer supported"):
        SkillContractResolver(root_dir=tmp_path).resolve(["skills/report"])


def test_skill_contract_resolver_rejects_direct_contract_yaml_path(tmp_path: Path) -> None:
    contract_path = tmp_path / "skills" / "report" / "contract.yaml"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text("runtime:\n  strategy: sectioned_longform\n", encoding="utf-8")

    with pytest.raises(SkillContractError, match="contract.yaml is no longer supported"):
        SkillContractResolver(root_dir=tmp_path).resolve([contract_path])


def test_skill_contract_resolver_required_raises_when_missing(tmp_path: Path) -> None:
    (tmp_path / "skills" / "empty").mkdir(parents=True)

    with pytest.raises(SkillContractError, match="No skill runtime contract"):
        SkillContractResolver(root_dir=tmp_path).resolve(["skills/empty"], required=True)


def test_context_pack_builder_is_generic_alias() -> None:
    assert issubclass(ContextPackBuilder, EvidencePackBuilder)
