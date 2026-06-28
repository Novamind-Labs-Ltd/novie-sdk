"""Cross-agent prompt management — fail-soft Langfuse fetch with in-repo fallback."""
from .registry import get_managed_prompt, resolve_prompt

__all__ = ["get_managed_prompt", "resolve_prompt"]
