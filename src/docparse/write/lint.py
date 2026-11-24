"""Output guards (S7) for the format traps that silently score zero. M0 reports findings; it replaces nothing."""

from __future__ import annotations

import re
from dataclasses import dataclass

_COMMENT = re.compile(r"<!--")
_TOKEN = re.compile(r"\S+")
LOOP_WINDOW = 40  # tokens
LOOP_REPEATS = 5


@dataclass(frozen=True)
class LintFinding:
    check: str
    detail: str = ""


def _looped(markdown: str) -> bool:
    """The same 40-token window five times: a model stuck in a loop."""
    tokens = _TOKEN.findall(markdown)
    if len(tokens) < LOOP_WINDOW * LOOP_REPEATS:
        return False
    seen: dict[int, int] = {}
    for i in range(0, len(tokens) - LOOP_WINDOW + 1):
        h = hash(tuple(tokens[i : i + LOOP_WINDOW]))
        seen[h] = seen.get(h, 0) + 1
        if seen[h] >= LOOP_REPEATS:
            return True
    return False


def lint_page(markdown: str, layout: dict | None, page_number: int = 1) -> list[LintFinding]:
    findings = []
    if _COMMENT.search(markdown):
        findings.append(LintFinding("html_comment", "HTML comments are invisible to every scorer"))
    if _looped(markdown):
        findings.append(LintFinding("looped_text"))
    if layout is not None:
        if layout.get("page_number") != page_number:
            findings.append(LintFinding("page_number", f"{layout.get('page_number')} != {page_number}"))
        for i, item in enumerate(layout.get("items") or []):
            if not item.get("bbox") or not item.get("layout_segments"):
                findings.append(LintFinding("item_without_boxes", f"item {i}"))
            elif any(not seg.get("label") for seg in item["layout_segments"]):
                findings.append(LintFinding("segment_without_label", f"item {i}"))
    return findings
