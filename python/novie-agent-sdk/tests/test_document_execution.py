from __future__ import annotations

import pytest

from novie_agent_sdk import (
    DIRECT_AUTHORING,
    EVIDENCE_GRAPH,
    GRAPH_HANDOFF,
    RuntimeContract,
    SkillRuntimeContract,
    build_document_handoff_event,
    resolve_document_execution_plan,
    step_run_policy,
)


def _contract(preparation: str) -> SkillRuntimeContract:
    return SkillRuntimeContract(runtime=RuntimeContract(preparation=preparation))


def test_upstream_handoff_uses_graph_once_regardless_of_skill_preparation() -> None:
    plan = resolve_document_execution_plan(
        run_policy=step_run_policy({"output_contract": {"kind": "bounded_handoff"}}),
        skill_contract=_contract(DIRECT_AUTHORING),
    )

    assert plan.preparation == GRAPH_HANDOFF
    assert plan.run_graph is True
    assert plan.graph_output == "artifact"
    assert plan.run_sectioned_authoring is False


@pytest.mark.parametrize(
    ("preparation", "run_graph", "graph_output"),
    [
        (DIRECT_AUTHORING, False, ""),
        (EVIDENCE_GRAPH, True, "evidence_dossier"),
    ],
)
def test_terminal_plan_has_exactly_one_sectioned_author(
    preparation: str,
    run_graph: bool,
    graph_output: str,
) -> None:
    plan = resolve_document_execution_plan(
        run_policy=step_run_policy({"output_contract": {"kind": "final_deliverable"}}),
        skill_contract=_contract(preparation),
    )

    assert plan.preparation == preparation
    assert plan.run_graph is run_graph
    assert plan.graph_output == graph_output
    assert plan.run_sectioned_authoring is True


def test_terminal_plan_rejects_unknown_skill_preparation() -> None:
    with pytest.raises(RuntimeError, match="invalid_document_runtime_preparation"):
        resolve_document_execution_plan(
            run_policy=step_run_policy({}),
            skill_contract=_contract("two_full_authors"),
        )


def test_document_handoff_event_is_bounded_and_internal() -> None:
    policy = step_run_policy(
        {"output_contract": {"kind": "bounded_handoff", "max_bytes": 1024}}
    )

    event = build_document_handoff_event(
        handoff_markdown="# Evidence\n\n" + ("detail " * 1000),
        artifact_type="research_report",
        artifact_family="research",
        capability_id="agent.demo.research",
        run_policy=policy,
        provided_artifact_names=("research_report",),
        metadata={"agent": "demo"},
    )

    assert event.output["kind"] == "bounded_handoff"
    assert event.output["output_visibility"] == "internal"
    assert len(event.output["final_markdown"].encode("utf-8")) <= 1024
