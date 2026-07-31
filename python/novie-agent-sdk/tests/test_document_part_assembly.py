from __future__ import annotations

import pytest

from novie_agent_sdk.document_authoring_identity import part_identity
from novie_agent_sdk.document_authoring_plan import PlannedPart
from novie_agent_sdk.document_part_assembly import (
    PlannedPartRejected,
    accept_part,
    information_units,
)


def _plan(maximum: int) -> PlannedPart:
    return PlannedPart(
        section_id="s1",
        ordinal=1,
        total=1,
        objective="Explain.",
        objective_digest="objective",
        evidence_digest="evidence",
        part_identity=part_identity(
            scope={},
            section_id="s1",
            objective_digest="objective",
            evidence_digest="evidence",
        ),
        target_information_units=maximum,
        max_information_units=maximum,
    )


def test_information_units_counts_cjk_characters_and_latin_words() -> None:
    assert information_units("这是中文 content with two words.") == 8


def test_part_hard_cap_rejects_output_over_exact_planned_maximum() -> None:
    with pytest.raises(PlannedPartRejected, match="part_over_maximum"):
        accept_part(
            _plan(10),
            "one two three four five six seven eight nine ten eleven twelve",
            section_title="Section",
            truncated=False,
        )
