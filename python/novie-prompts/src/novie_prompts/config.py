"""Typed config for novie-prompts. Values are INJECTED by the consumer repo
(via set_config) — the package reads no env vars itself, so secret indirection
stays in the consumer (Hard Discipline #7)."""
from __future__ import annotations
from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class PromptConfig:
    enabled: bool = False
    host: str | None = None
    cache_ttl_seconds: int = 60
    fetch_timeout_seconds: int = 2          # SDK passes this as httpx timeout_in_seconds (int|None)
    public_key: str | None = None
    secret_key: str | None = None


_config = PromptConfig()


def set_config(**overrides) -> None:
    """Replace the process config. No args → defaults. Keyword overrides only."""
    global _config
    _config = replace(PromptConfig(), **overrides)


def current() -> PromptConfig:
    return _config


def is_enabled() -> bool:
    return _config.enabled


def host() -> str | None:
    return _config.host


def cache_ttl_seconds() -> int:
    return max(1, _config.cache_ttl_seconds)  # 0 disables SDK caching; floor it


def fetch_timeout_seconds() -> int:
    return _config.fetch_timeout_seconds
