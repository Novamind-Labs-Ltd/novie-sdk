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
        "reason": "Explicit length profile in capability input.",
        "requested_min_information_units": 0,
        "requested_max_information_units": 0,
    }


@pytest.mark.asyncio
async def test_platform_document_length_in_brief_metadata_skips_llm_selection() -> None:
    class _Llm:
        async def structured(self, **_kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("declared platform profile should not call the LLM")

    result = await select_document_length_profile(
        inputs={},
        brief={
            "raw_metadata": {
                "length_or_detail_level": "deep",
            }
        },
        contract=_contract(),
        llm_facade=_Llm(),
    )

    assert result["profile"] == "long"
    assert result["source"] == "brief"
    assert result["confidence"] == "confirmed"


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
                    "request_basis": "explicit_length_request",
                    "requested_min_information_units": 0,
                    "requested_max_information_units": 0,
                }
            }

    result = await select_document_length_profile(
        inputs={},
        brief={
            "reception_resolution": {
                "raw_query": "Keep this decision brief concise."
            }
        },
        contract=_contract(),
        llm_facade=_Llm(),
    )

    assert result == {
        "profile": "short",
        "source": "explicit_description",
        "confidence": "confirmed",
        "reason": "concise decision brief",
        "requested_min_information_units": 0,
        "requested_max_information_units": 0,
    }


@pytest.mark.asyncio
async def test_adaptive_profile_does_not_turn_completeness_into_long_length() -> None:
    class _Llm:
        async def structured(self, **_kwargs):  # type: ignore[no-untyped-def]
            return {
                "structured": {
                    "length_profile": "long",
                    "confidence": "inferred",
                    "reason": "Many complete sections were requested.",
                    "request_basis": "unspecified",
                    "requested_min_information_units": 4000,
                    "requested_max_information_units": 7000,
                }
            }

    result = await select_document_length_profile(
        inputs={},
        brief={
            "user_goal": (
                "Write a reviewable report with complete sections, risks, "
                "implementation steps, and acceptance criteria."
            ),
            "reception_resolution": {
                "raw_query": (
                    "Write a reviewable report with complete sections, risks, "
                    "implementation steps, and acceptance criteria."
                )
            },
        },
        contract=_contract(),
        llm_facade=_Llm(),
    )

    assert result == {
        "profile": "medium",
        "source": "adaptive_default",
        "confidence": "inferred",
        "reason": "Many complete sections were requested.",
        "requested_min_information_units": 0,
        "requested_max_information_units": 0,
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
        "source": "adaptive_default",
        "confidence": "inferred",
        "reason": (
            "No authoritative original user request was available for "
            "descriptive length classification."
        ),
        "requested_min_information_units": 0,
        "requested_max_information_units": 0,
    }


@pytest.mark.asyncio
async def test_normalized_brief_cannot_invent_explicit_long_request() -> None:
    class _Llm:
        async def structured(self, **kwargs):  # type: ignore[no-untyped-def]
            prompt = kwargs["messages"][0]["content"]
            assert "2,500–4,000" not in prompt
            assert "篇幅约" not in prompt
            return {
                "structured": {
                    "length_profile": "medium",
                    "confidence": "inferred",
                    "reason": "No explicit length request in the raw query.",
                    "request_basis": "unspecified",
                    "requested_min_information_units": 0,
                    "requested_max_information_units": 0,
                }
            }

    result = await select_document_length_profile(
        inputs={},
        brief={
            "summary": "篇幅约 2,500–4,000 字。",
            "constraints": ["篇幅约 2,500–4,000 字"],
            "reception_resolution": {
                "raw_query": "请写一份章节完整、可直接评审的管理报告。"
            },
        },
        contract=_contract(),
        llm_facade=_Llm(),
    )

    assert result["profile"] == "medium"
    assert result["source"] == "adaptive_default"


@pytest.mark.asyncio
async def test_explicit_numeric_range_is_preserved_over_llm_inference() -> None:
    class _Llm:
        async def structured(self, **_kwargs):  # type: ignore[no-untyped-def]
            return {
                "structured": {
                    "length_profile": "long",
                    "confidence": "inferred",
                    "reason": "The requested range requires depth.",
                    "request_basis": "explicit_length_request",
                    "requested_min_information_units": 1,
                    "requested_max_information_units": 2,
                }
            }

    result = await select_document_length_profile(
        inputs={
            "requested_min_information_units": 7000,
            "requested_max_information_units": 9000,
        },
        brief={"user_goal": "Write the requested document."},
        contract=_contract(),
        llm_facade=_Llm(),
    )

    assert result["profile"] == "long"
    assert result["source"] == "user_input"
    assert result["confidence"] == "confirmed"
    assert result["requested_min_information_units"] == 7000
    assert result["requested_max_information_units"] == 9000
