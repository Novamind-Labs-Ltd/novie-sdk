from __future__ import annotations

import pytest

from novie_agent_sdk import (
    DocumentLengthProfileContract,
    DocumentRuntimeContract,
    SkillRuntimeContract,
    TaskProfileContract,
    select_document_length_profile,
)


def _contract(*, default: str = "adaptive") -> SkillRuntimeContract:
    return SkillRuntimeContract(
        task_profile=TaskProfileContract(defaults={"length_profile": default}),
        document=DocumentRuntimeContract(
            length_profiles={
                name: DocumentLengthProfileContract(name=name)
                for name in ("short", "medium", "long")
            }
        ),
    )


@pytest.mark.asyncio
async def test_explicit_length_profile_skips_llm_selection() -> None:
    class _Llm:
        async def structured(self, **_kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("explicit profile should not call the LLM")

    result = await select_document_length_profile(
        inputs={"length_profile": "long"},
        brief={},
        contract=_contract(),
        llm_facade=_Llm(),
    )

    assert result == {
        "profile": "long",
        "source": "user_input",
        "confidence": "confirmed",
    }


@pytest.mark.asyncio
async def test_adaptive_profile_uses_structured_llm_selection() -> None:
    class _Llm:
        async def structured(self, **kwargs):  # type: ignore[no-untyped-def]
            assert kwargs["output_schema"]["properties"]["length_profile"]["enum"] == [
                "long",
                "medium",
                "short",
            ]
            return {
                "structured": {
                    "length_profile": "short",
                    "confidence": "inferred",
                    "reason": "concise decision brief",
                }
            }

    result = await select_document_length_profile(
        inputs={},
        brief={"user_goal": "Summarise the decision."},
        contract=_contract(),
        llm_facade=_Llm(),
    )

    assert result == {
        "profile": "short",
        "source": "inferred",
        "confidence": "inferred",
    }


@pytest.mark.asyncio
async def test_adaptive_profile_uses_deterministic_fallback_without_structured_llm() -> None:
    result = await select_document_length_profile(
        inputs={},
        brief={"user_goal": "Write a document."},
        contract=_contract(),
        llm_facade=object(),
    )

    assert result == {
        "profile": "medium",
        "source": "runtime_fallback",
        "confidence": "inferred",
    }
