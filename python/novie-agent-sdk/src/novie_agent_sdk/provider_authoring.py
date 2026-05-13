"""Universal provider folder authoring helpers.

W8 of the universal capability backlog standardizes the folder shape used by
internal platform providers, third-party adapters, MCP/OpenAPI bridges, and
future non-agent providers. These helpers are intentionally file-based and
side-effect free: they load YAML, project it onto the frozen
``CapabilityProvider`` contract, and return structured diagnostics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from novie_protocol.contracts.universal_capability import (
    CapabilityProvider,
    validate_capability_provider,
)


ProviderIssueSeverity = Literal["error", "warning", "info"]
ProviderRegistrationCategory = Literal[
    "ok",
    "provider_validation",
    "registry_storage",
]


@dataclass(frozen=True, slots=True)
class ProviderValidationIssue:
    severity: ProviderIssueSeverity
    field_path: str
    code: str
    author_message: str
    platform_message: str


@dataclass(frozen=True, slots=True)
class ProviderValidationResult:
    issues: tuple[ProviderValidationIssue, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[ProviderValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[ProviderValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "warning")

    @property
    def infos(self) -> tuple[ProviderValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "info")

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class ProviderFolderOutcome:
    provider: CapabilityProvider | None
    result: ProviderValidationResult
    provider_path: Path
    capabilities_path: Path
    resources_path: Path


@dataclass(frozen=True, slots=True)
class ProviderRegistrationOutcome:
    category: ProviderRegistrationCategory
    provider_id: str = ""
    registry_ref: str = ""
    errors: tuple[str, ...] = field(default_factory=tuple)
    detail: str = ""
    validation: ProviderValidationResult = field(default_factory=ProviderValidationResult)

    @property
    def ok(self) -> bool:
        return self.category == "ok"


class ProviderRegistryWriter(Protocol):
    """Minimal persistence interface for universal providers.

    The SDK helper does not assume a platform backend. Local files, future PG
    storage, or a gateway HTTP API can all implement this protocol.
    """

    def register_provider(
        self,
        provider: CapabilityProvider,
        *,
        source: str = ...,
    ) -> str: ...


class FileProviderRegistryWriter:
    """Deterministic file-backed writer for local authoring and CI.

    Writes ``<provider_id>.json`` under ``registry_dir``. This is a staging
    format for W8: platform-backed registration can replace this writer without
    changing authoring validation or CLI behavior.
    """

    def __init__(self, registry_dir: Path) -> None:
        self._registry_dir = registry_dir

    def register_provider(
        self,
        provider: CapabilityProvider,
        *,
        source: str = "novie_providers_register",
    ) -> str:
        self._registry_dir.mkdir(parents=True, exist_ok=True)
        target = (
            self._registry_dir
            / f"{_safe_provider_filename(provider.provider_id)}.json"
        )
        payload = provider.to_dict()
        metadata = dict(payload.get("metadata") or {})
        metadata["registry_source"] = source
        payload["metadata"] = metadata
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return str(target)


def load_provider_folder(path: Path) -> ProviderFolderOutcome:
    """Load ``provider.yaml`` + ``capabilities.yaml`` + ``resources.yaml``.

    The projected result is the universal ``CapabilityProvider`` contract. A
    green result is therefore immediately consumable by the W2 discovery source
    and W4 invocation pipeline without Reception/Planner code changes.
    """
    root = path.resolve()
    provider_path = root / "provider.yaml"
    capabilities_path = root / "capabilities.yaml"
    resources_path = root / "resources.yaml"
    issues: list[ProviderValidationIssue] = []

    provider_doc = _load_yaml_object(provider_path, issues)
    capabilities_doc = _load_yaml_object(capabilities_path, issues)
    resources_doc = _load_yaml_object(resources_path, issues)
    if issues:
        return ProviderFolderOutcome(
            provider=None,
            result=ProviderValidationResult(tuple(issues)),
            provider_path=provider_path,
            capabilities_path=capabilities_path,
            resources_path=resources_path,
        )

    provider_payload = _project_provider_payload(
        provider_doc or {},
        capabilities_doc or {},
        resources_doc or {},
        issues,
    )
    if issues:
        return ProviderFolderOutcome(
            provider=None,
            result=ProviderValidationResult(tuple(issues)),
            provider_path=provider_path,
            capabilities_path=capabilities_path,
            resources_path=resources_path,
        )

    try:
        provider = CapabilityProvider.from_dict(provider_payload)
    except Exception as exc:  # noqa: BLE001 - surface contract parse failures
        issues.append(
            ProviderValidationIssue(
                severity="error",
                field_path="<provider>",
                code="provider_contract_parse_failed",
                author_message=f"provider folder cannot be parsed: {exc}",
                platform_message=f"CapabilityProvider.from_dict raised {exc!r}",
            )
        )
        return ProviderFolderOutcome(
            provider=None,
            result=ProviderValidationResult(tuple(issues)),
            provider_path=provider_path,
            capabilities_path=capabilities_path,
            resources_path=resources_path,
        )

    for error in validate_capability_provider(provider):
        issues.append(
            ProviderValidationIssue(
                severity="error",
                field_path=error.pointer or "<provider>",
                code="provider_contract_validation_failed",
                author_message=error.message,
                platform_message=str(error),
            )
        )
    issues.extend(_validate_provider_authoring_rules(provider))
    return ProviderFolderOutcome(
        provider=provider,
        result=ProviderValidationResult(tuple(issues)),
        provider_path=provider_path,
        capabilities_path=capabilities_path,
        resources_path=resources_path,
    )


def register_provider_folder(
    path: Path,
    *,
    writer: ProviderRegistryWriter | None = None,
    registry_dir: Path | None = None,
    source: str = "novie_providers_register",
) -> ProviderRegistrationOutcome:
    """Validate and register a provider folder through a writer.

    Defaults to a deterministic local file registry when no writer is supplied.
    Future platform/PG registration should implement ``ProviderRegistryWriter``
    and reuse this helper.
    """
    loaded = load_provider_folder(path)
    if not loaded.result.is_valid or loaded.provider is None:
        return ProviderRegistrationOutcome(
            category="provider_validation",
            errors=tuple(issue.author_message for issue in loaded.result.errors),
            validation=loaded.result,
        )

    effective_writer = writer or FileProviderRegistryWriter(
        registry_dir or Path(".novie/providers")
    )
    try:
        registry_ref = effective_writer.register_provider(
            loaded.provider,
            source=source,
        )
    except Exception as exc:  # noqa: BLE001 - keep CLI diagnostics stable
        return ProviderRegistrationOutcome(
            category="registry_storage",
            provider_id=loaded.provider.provider_id,
            errors=(f"{type(exc).__name__}: {exc}",),
            detail=f"provider registry writer failed for {loaded.provider.provider_id!r}",
            validation=loaded.result,
        )
    return ProviderRegistrationOutcome(
        category="ok",
        provider_id=loaded.provider.provider_id,
        registry_ref=registry_ref,
        validation=loaded.result,
    )


def _load_yaml_object(
    path: Path,
    issues: list[ProviderValidationIssue],
) -> dict[str, Any] | None:
    if not path.is_file():
        issues.append(
            ProviderValidationIssue(
                severity="error",
                field_path=str(path.name),
                code="provider_file_missing",
                author_message=f"required provider folder file is missing: {path.name}",
                platform_message=f"{path} is required by provider folder contract",
            )
        )
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        issues.append(
            ProviderValidationIssue(
                severity="error",
                field_path=str(path.name),
                code="provider_yaml_unparseable",
                author_message=f"{path.name} is not valid YAML: {exc}",
                platform_message=f"yaml.safe_load failed for {path}: {exc!r}",
            )
        )
        return None
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        issues.append(
            ProviderValidationIssue(
                severity="error",
                field_path=str(path.name),
                code="provider_yaml_not_object",
                author_message=f"{path.name} must be a YAML object",
                platform_message=f"{path} root must be mapping, got {type(raw)!r}",
            )
        )
        return None
    return dict(raw)


def _project_provider_payload(
    provider_doc: dict[str, Any],
    capabilities_doc: dict[str, Any],
    resources_doc: dict[str, Any],
    issues: list[ProviderValidationIssue],
) -> dict[str, Any]:
    provider = provider_doc.get("provider", provider_doc)
    if not isinstance(provider, dict):
        issues.append(_schema_issue("provider.yaml.provider", "must be an object"))
        return {}

    capabilities = capabilities_doc.get("capabilities", capabilities_doc)
    if not isinstance(capabilities, list):
        issues.append(_schema_issue("capabilities.yaml.capabilities", "must be a list"))
        capabilities = []
    resource_types = resources_doc.get("resource_types", resources_doc.get("resources", []))
    if not isinstance(resource_types, list):
        issues.append(_schema_issue("resources.yaml.resource_types", "must be a list"))
        resource_types = []

    provider_id = str(provider.get("provider_id") or provider.get("id") or "")
    provider_type = str(provider.get("provider_type") or provider.get("type") or "")
    payload = {
        "provider_id": provider_id,
        "provider_type": provider_type,
        "display_name": str(provider.get("display_name") or provider.get("name") or provider_id),
        "version": str(provider.get("version") or ""),
        "conformance_version": str(
            provider.get("conformance_version") or "universal-capability-v1"
        ),
        "health": dict(provider.get("health") or {}),
        "resource_types": [str(item) for item in resource_types],
        "capabilities": [_project_capability(item, provider_id, issues) for item in capabilities],
        "metadata": dict(provider.get("metadata") or {}),
    }
    transport = provider.get("transport")
    if transport:
        if isinstance(transport, dict):
            payload["transport"] = dict(transport)
        else:
            issues.append(_schema_issue("provider.yaml.transport", "must be an object"))
    return payload


def _project_capability(
    raw: Any,
    provider_id: str,
    issues: list[ProviderValidationIssue],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        issues.append(_schema_issue("capabilities.yaml.capabilities[]", "must be an object"))
        return {}
    capability = dict(raw)
    capability.setdefault("provider_id", provider_id)
    capability.setdefault("kind", "query")
    capability.setdefault("risk_level", "read")
    capability.setdefault("side_effect", "none")
    capability.setdefault("status", "stable")
    capability.setdefault("input_schema", {})
    capability.setdefault("output_schema", {})
    capability.setdefault("caller_types", ["reception", "planner", "executor"])
    capability.setdefault("caller_modes", ["plan", "execute"])
    capability.setdefault("routing_hints", {})
    capability.setdefault("metadata", {})
    return capability


def _validate_provider_authoring_rules(
    provider: CapabilityProvider,
) -> list[ProviderValidationIssue]:
    issues: list[ProviderValidationIssue] = []
    if not provider.capabilities:
        issues.append(
            ProviderValidationIssue(
                severity="error",
                field_path="capabilities.yaml.capabilities",
                code="provider_has_no_capabilities",
                author_message="provider must declare at least one capability",
                platform_message="CapabilityProvider.capabilities is empty",
            )
        )
    for index, capability in enumerate(provider.capabilities):
        if capability.risk_level not in ("write", "dangerous"):
            continue
        has_preview = capability.dry_run_support != "none"
        has_gate = capability.confirmation_default in ("required", "gated") or bool(
            capability.gate_policy
        )
        if not (has_preview or has_gate):
            issues.append(
                ProviderValidationIssue(
                    severity="error",
                    field_path=f"capabilities.yaml.capabilities[{index}]",
                    code="write_capability_missing_preview_or_gate",
                    author_message=(
                        "write/dangerous capabilities must declare dry-run support "
                        "or explicit confirmation/gate behavior"
                    ),
                    platform_message=(
                        f"{capability.capability_id} risk_level={capability.risk_level!r} "
                        "has no dry_run_support, confirmation_default, or gate_policy"
                    ),
                )
            )
    return issues


def _schema_issue(field_path: str, message: str) -> ProviderValidationIssue:
    return ProviderValidationIssue(
        severity="error",
        field_path=field_path,
        code="provider_folder_schema_invalid",
        author_message=message,
        platform_message=f"{field_path}: {message}",
    )


def _safe_provider_filename(provider_id: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in provider_id
    )


__all__ = [
    "FileProviderRegistryWriter",
    "ProviderFolderOutcome",
    "ProviderIssueSeverity",
    "ProviderRegistrationCategory",
    "ProviderRegistrationOutcome",
    "ProviderRegistryWriter",
    "ProviderValidationIssue",
    "ProviderValidationResult",
    "load_provider_folder",
    "register_provider_folder",
]
