"""Deterministic ADR-115 Planned Part acceptance and section assembly."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from .document_authoring_plan import PlannedPart

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class PlannedPartRejected(RuntimeError):
    code = "document_planned_part_rejected"


@dataclass(frozen=True, slots=True)
class AcceptedPart:
    plan: PlannedPart
    markdown: str
    information_units: int
    artifact_ref: str = ""
    content_digest: str = ""


def information_units(markdown: str) -> int:
    text = re.sub(
        r"!?\[[^\]]*\]\([^)]*\)|\[(?:artifact|source)://[^\]]+\]",
        "",
        str(markdown or ""),
        flags=re.IGNORECASE,
    )
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def normalize_part_markdown(markdown: str, *, section_title: str) -> str:
    text = str(markdown or "").strip()
    if not text:
        return ""
    # Repair the narrow model failure where an ATX heading is welded to the
    # previous prose character.  This is deterministic typography, not a
    # semantic rewrite; ordinary C#, inline hash mentions, and legal line-start
    # headings do not match.
    text = re.sub(
        r"(?<=[^#\s])(?=#{2,6}[ \t]+\S)",
        "\n\n",
        text,
    )
    lines = text.splitlines()
    normalized: list[str] = []
    section_key = section_title.strip().casefold()
    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            normalized.append(line.rstrip())
            continue
        title = match.group(2).strip()
        if title.casefold() == section_key:
            continue
        level = max(3, len(match.group(1)))
        normalized.append(f"{'#' * level} {title}")
    return "\n".join(normalized).strip()


def accept_part(
    plan: PlannedPart,
    markdown: str,
    *,
    section_title: str,
    truncated: bool,
) -> AcceptedPart:
    if truncated:
        raise PlannedPartRejected("document_planned_part_rejected: output_truncated")
    normalized = normalize_part_markdown(markdown, section_title=section_title)
    units = information_units(normalized)
    if not normalized or units <= 0:
        raise PlannedPartRejected("document_planned_part_rejected: empty_part")
    # ``max_information_units`` is the prompt/planning bound, not the provider
    # token stop.  Natural-language models often need extra prose to close a
    # paragraph, citation, list, or table.  Keep that completion headroom
    # bounded by an independent hard ceiling; materially runaway output still
    # enters repartition recovery, and the document-level output budget remains
    # the aggregate safety limit.
    hard_maximum = max(
        plan.max_information_units + 8,
        math.ceil(plan.max_information_units * 12),
    )
    if units > hard_maximum:
        raise PlannedPartRejected("document_planned_part_rejected: part_over_maximum")
    return AcceptedPart(plan=plan, markdown=normalized, information_units=units)


def assemble_section(
    *,
    section_title: str,
    parts: Sequence[AcceptedPart],
) -> str:
    if not parts:
        raise PlannedPartRejected("document_planned_part_rejected: no_accepted_parts")
    ordered = sorted(parts, key=lambda item: item.plan.ordinal)
    actual = [item.plan.ordinal for item in ordered]
    expected = list(range(1, len(ordered) + 1))
    if actual != expected:
        raise PlannedPartRejected("document_planned_part_rejected: non_contiguous_parts")
    bodies = [
        normalize_part_markdown(item.markdown, section_title=section_title)
        for item in ordered
    ]
    body = "\n\n".join(item for item in bodies if item).strip()
    section = f"## {section_title.strip()}\n\n{body}".strip()
    h2_titles = [
        title.strip()
        for hashes, title in _HEADING_RE.findall(section)
        if len(hashes) == 2
    ]
    if h2_titles != [section_title.strip()]:
        raise PlannedPartRejected(
            "document_planned_part_rejected: invalid_section_heading"
        )
    return section


__all__ = [
    "AcceptedPart",
    "PlannedPartRejected",
    "accept_part",
    "assemble_section",
    "information_units",
    "normalize_part_markdown",
]
