"""Generic bounded context packs for SDK agents.

The original names in ``artifact_ledger`` use "evidence" because the first
consumer was an analyst research agent. These aliases expose the same runtime
primitive without tying future agents to research/report terminology.
"""
from __future__ import annotations

from .artifact_ledger import (
    ContextBudget,
    EvidencePack,
    EvidencePackBuilder,
    EvidencePackItem,
)


ContextPackBudget = ContextBudget
ContextPack = EvidencePack
ContextPackItem = EvidencePackItem


class ContextPackBuilder(EvidencePackBuilder):
    """Build bounded prompt context from workpad and upstream artifact refs."""


__all__ = [
    "ContextPack",
    "ContextPackBudget",
    "ContextPackBuilder",
    "ContextPackItem",
]
