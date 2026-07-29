from __future__ import annotations

from typing import Any

import pytest

from novie_agent_sdk.document_completeness import (
    DocumentCompletenessReviewError,
    markdown_structure_violation,
    review_section_completeness,
)


class _StructuredLlm:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    async def structured(self, **_kwargs: Any) -> dict[str, Any]:
        return {"structured": self.payload}


@pytest.mark.asyncio
async def test_review_section_completeness_accepts_valid_strict_shape() -> None:
    review = await review_section_completeness(
        _StructuredLlm(
            {
                "complete": False,
                "issue_code": "cut_off",
                "reason": "The option ends mid-thought.",
            }
        ),
        section_title="Open questions",
        section_objective="Document all unresolved choices.",
        markdown="## Open questions\n\nOption B requires provider support",
    )

    assert review.complete is False
    assert review.issue_code == "cut_off"


@pytest.mark.asyncio
async def test_review_section_completeness_rejects_contradictory_shape() -> None:
    with pytest.raises(DocumentCompletenessReviewError):
        await review_section_completeness(
            _StructuredLlm(
                {
                    "complete": True,
                    "issue_code": "cut_off",
                    "reason": "Contradictory.",
                }
            ),
            section_title="Open questions",
            section_objective="Document all unresolved choices.",
            markdown="## Open questions\n\nBody.",
        )


def test_markdown_structure_violation_detects_unclosed_fence() -> None:
    assert markdown_structure_violation("## Example\n\n```python\nprint('x')") == (
        "markdown_structure_unclosed"
    )
    assert markdown_structure_violation("## Example\n\n```python\nprint('x')\n```") is None
    assert markdown_structure_violation("## Example\n\n```python\n~~~\n```") is None
    assert markdown_structure_violation("## Example\n\n```python\n``") == (
        "markdown_structure_unclosed"
    )
