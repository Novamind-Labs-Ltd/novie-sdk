from __future__ import annotations

from types import SimpleNamespace

from novie_agent_sdk import provided_artifacts_for_capability


def test_provided_artifacts_for_capability_projects_manifest_provides() -> None:
    manifest = {
        "capability_manifest": [
            {"capability_id": "cap-1", "provides": ["prd", "requirements"]},
            {"capability_id": "cap-2", "provides": ["architecture"]},
        ]
    }

    out = provided_artifacts_for_capability(
        manifest,
        capability_id="cap-1",
        artifact_type="prd",
        structured_output={"title": "Checkout"},
    )

    assert set(out) == {"prd", "requirements"}
    assert out["prd"]["structured_output"] == {"title": "Checkout"}
    assert out["requirements"]["structured_output"] == {"title": "Checkout"}


def test_provided_artifacts_for_capability_accepts_object_manifest() -> None:
    manifest = SimpleNamespace(
        capability_manifest=[
            SimpleNamespace(capability_id="cap-1", provides=("design",)),
        ]
    )

    out = provided_artifacts_for_capability(
        manifest,
        capability_id="cap-1",
        artifact_type="",
        structured_output={"ok": True},
    )

    assert out == {"design": {"structured_output": {"ok": True}}}


def test_provided_artifacts_for_capability_includes_artifact_type_without_match() -> None:
    out = provided_artifacts_for_capability(
        {"capability_manifest": []},
        capability_id="missing",
        artifact_type="fallback",
        structured_output={"ok": True},
    )

    assert out == {"fallback": {"structured_output": {"ok": True}}}
