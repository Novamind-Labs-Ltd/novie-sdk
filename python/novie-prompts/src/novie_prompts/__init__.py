"""novie-prompts — fail-soft Langfuse-managed prompt fetch with in-repo fallback (ADR-075)."""
from .config import configure
from .registry import get_managed_prompt
from .telemetry import Recorder, has_recorder, set_recorder

__all__ = ["get_managed_prompt", "configure", "set_recorder", "has_recorder", "Recorder"]
