"""SDK compatibility checks for platform-owned dev seed manifests.

The platform's dev seed manifests (analyst / task_splitter / novie-cortex)
are vendored into this test tree at tests/fixtures/dev-seed-manifests/.
When the platform updates a manifest, refresh this fixture directory by
copying the JSON files from
``novie/apps/agentic-beta/deploy/dev/manifests/``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from novie_protocol.contracts.agent_sdk_v2 import AgentManifestV2

MANIFEST_DIR = Path(__file__).resolve().parent / "fixtures" / "dev-seed-manifests"


def _load_manifest(name: str) -> dict[str, Any]:
    return json.loads((MANIFEST_DIR / name).read_text(encoding="utf-8"))


def _capability_ids(manifest: dict[str, Any]) -> list[str]:
    return [
        entry["capability_id"]
        for entry in manifest.get("capability_manifest", [])
    ]


def test_dev_seed_manifests_pass_platform_validation() -> None:
    for path in sorted(MANIFEST_DIR.glob("*.agent.json")):
        manifest = AgentManifestV2.from_dict(json.loads(path.read_text(encoding="utf-8")))
        assert manifest.validate() == [], path


def test_dev_seed_preserves_core_agent_capability_ids() -> None:
    analyst = _load_manifest("analyst.agent.json")
    task_splitter = _load_manifest("task_splitter.agent.json")
    cortex = _load_manifest("novie-cortex.agent.json")

    assert "agent.analyst.report_synthesis" in _capability_ids(analyst)
    assert "agent.task_splitter.from_requirements" in _capability_ids(task_splitter)
    assert "agent.novie-cortex.execute_task_bundle" in _capability_ids(cortex)
