"""Universal provider folder authoring tests."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from novie_agent_sdk.provider_authoring import (
    FileProviderRegistryWriter,
    load_provider_folder,
    register_provider_folder,
)


def _write_provider_folder(
    root: Path,
    *,
    capability_body: str | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "provider.yaml").write_text(
        dedent(
            """
            provider:
              id: integration.github
              type: openapi
              display_name: GitHub Adapter
              version: 0.1.0
              transport:
                kind: openapi
                spec_url: https://api.github.com/openapi.json
                base_url: https://api.github.com
              health:
                kind: http_get
                url: https://api.github.com/rate_limit
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "capabilities.yaml").write_text(
        (
            capability_body
            or dedent(
                """
                capabilities:
                  - capability_id: integration.github.repo.read
                    kind: query
                    risk_level: read
                    side_effect: none
                    input_schema:
                      type: object
                    output_schema:
                      type: object
                    consumes_resources:
                      - repository
                    routing_hints:
                      when_to_use: Read GitHub repository metadata.
                """
            ).strip()
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "resources.yaml").write_text(
        dedent(
            """
            resource_types:
              - repository
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return root


def test_load_provider_folder_accepts_read_only_provider(tmp_path: Path) -> None:
    outcome = load_provider_folder(_write_provider_folder(tmp_path / "github"))

    assert outcome.result.is_valid
    assert outcome.provider is not None
    assert outcome.provider.provider_id == "integration.github"
    assert outcome.provider.provider_type == "openapi"
    assert outcome.provider.resource_types == ("repository",)
    assert outcome.provider.capabilities[0].capability_id == "integration.github.repo.read"
    assert outcome.provider.capabilities[0].provider_id == "integration.github"


def test_load_provider_folder_requires_three_standard_files(tmp_path: Path) -> None:
    outcome = load_provider_folder(tmp_path / "missing")

    assert not outcome.result.is_valid
    assert {issue.code for issue in outcome.result.errors} == {"provider_file_missing"}
    assert {issue.field_path for issue in outcome.result.errors} == {
        "provider.yaml",
        "capabilities.yaml",
        "resources.yaml",
    }


def test_load_provider_folder_rejects_write_without_preview_or_gate(tmp_path: Path) -> None:
    capability_body = dedent(
        """
        capabilities:
          - capability_id: integration.github.repo.write
            kind: command
            risk_level: write
            side_effect: external
            dry_run_support: none
            confirmation_default: auto
            input_schema:
              type: object
            output_schema:
              type: object
        """
    ).strip()

    outcome = load_provider_folder(
        _write_provider_folder(tmp_path / "github", capability_body=capability_body)
    )

    assert not outcome.result.is_valid
    codes = {issue.code for issue in outcome.result.errors}
    assert "provider_contract_validation_failed" in codes
    assert "write_capability_missing_preview_or_gate" in codes


def test_load_provider_folder_accepts_write_with_gate(tmp_path: Path) -> None:
    capability_body = dedent(
        """
        capabilities:
          - capability_id: integration.github.repo.write
            kind: command
            risk_level: write
            side_effect: external
            confirmation_default: gated
            gate_policy:
              - repo_write_approval
            input_schema:
              type: object
            output_schema:
              type: object
        """
    ).strip()

    outcome = load_provider_folder(
        _write_provider_folder(tmp_path / "github", capability_body=capability_body)
    )

    assert outcome.result.is_valid
    assert outcome.provider is not None
    assert outcome.provider.capabilities[0].confirmation_default == "gated"


def test_register_provider_folder_writes_deterministic_provider_json(tmp_path: Path) -> None:
    provider_dir = _write_provider_folder(tmp_path / "github")
    registry_dir = tmp_path / "registry"

    first = register_provider_folder(provider_dir, registry_dir=registry_dir)
    second = register_provider_folder(provider_dir, registry_dir=registry_dir)

    assert first.ok
    assert second.ok
    assert first.registry_ref == second.registry_ref
    target = Path(first.registry_ref)
    assert target.name == "integration.github.json"
    assert target.read_text(encoding="utf-8") == Path(second.registry_ref).read_text(
        encoding="utf-8"
    )
    assert '"provider_id": "integration.github"' in target.read_text(encoding="utf-8")
    assert '"registry_source": "novie_providers_register"' in target.read_text(
        encoding="utf-8"
    )


def test_register_provider_folder_does_not_write_invalid_provider(tmp_path: Path) -> None:
    provider_dir = _write_provider_folder(
        tmp_path / "github",
        capability_body=dedent(
            """
            capabilities:
              - capability_id: integration.github.repo.write
                kind: command
                risk_level: write
                side_effect: external
                dry_run_support: none
                confirmation_default: auto
                input_schema: {}
                output_schema: {}
            """
        ).strip(),
    )
    registry_dir = tmp_path / "registry"

    outcome = register_provider_folder(provider_dir, registry_dir=registry_dir)

    assert outcome.category == "provider_validation"
    assert not outcome.ok
    assert not registry_dir.exists()


def test_register_provider_folder_surfaces_writer_failures(tmp_path: Path) -> None:
    class BrokenWriter:
        def register_provider(self, provider, *, source="test"):  # type: ignore[no-untyped-def]
            raise RuntimeError(f"cannot write {provider.provider_id}")

    provider_dir = _write_provider_folder(tmp_path / "github")

    outcome = register_provider_folder(provider_dir, writer=BrokenWriter())

    assert outcome.category == "registry_storage"
    assert outcome.provider_id == "integration.github"
    assert "cannot write integration.github" in outcome.errors[0]


def test_file_provider_registry_writer_sanitizes_provider_filename(tmp_path: Path) -> None:
    provider_dir = _write_provider_folder(tmp_path / "github")
    loaded = load_provider_folder(provider_dir)
    assert loaded.provider is not None

    writer = FileProviderRegistryWriter(tmp_path / "registry")
    registered = writer.register_provider(
        loaded.provider.__class__(
            provider_id="integration/github",
            provider_type=loaded.provider.provider_type,
            display_name=loaded.provider.display_name,
            version=loaded.provider.version,
            transport=loaded.provider.transport,
            capabilities=loaded.provider.capabilities,
        )
    )

    assert Path(registered).name == "integration_github.json"
