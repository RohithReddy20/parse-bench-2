"""Layout has no severity cap: a page's score is weighted element credit over (element weight + extra-box
penalty). This splits a page's lost score (1 - score) across the scorer's own outcomes.

Each reference element loses weight * (1 - credit); each extra predicted box adds its penalty to the
denominator. The page's loss is shared out in proportion to those masses, so the rows of a page always sum to
exactly 1 - score.
"""

from __future__ import annotations

import re
from collections import defaultdict

ELEMENT_REASONS = {
    "no_localization": "element not located (no box matches)",
    "token_f1_below_threshold": "box found, text does not match (token F1 < 0.8)",
    "group_token_f1_below_threshold": "box found, text does not match (token F1 < 0.8)",
    "table_word_coverage": "table box misses its words (< 90% of word centres)",
    "group": "box looser or tighter than the reference (IoU ramp)",
    "pass": "box looser or tighter than the reference (IoU ramp)",
}
CLASS_GROUPS = {
    "Text": "text", "List-item": "text", "Caption": "text", "Footnote": "text", "Code": "text", "Formula": "text",
    "Section": "heading", "Section-header": "heading", "Title": "heading",
    "Picture": "picture", "Table": "table", "Page-header": "header/footer", "Page-footer": "header/footer",
    "Form": "form", "Key-Value Region": "form",
}  # fmt: skip
_ID = re.compile(r"\s*(?:#\d+\s*)?\(?[0-9a-f]{16}\)?|\s*#\d+")


def extra_reason(reason: str) -> str:
    """'holds words locked to #1 (4842cd87a1fe9424)' -> 'extra box: holds words locked to another element'."""
    text = _ID.sub("", reason or "").strip()
    renames = {
        "holds words locked to": "holds words locked to another element",
        "inside": "inside another element",
        "garbage child of table": "garbage child of a table",
        "tight child of table": "tight child of a table",
    }
    text = renames.get(text, text)
    return f"extra box: {text or 'unspecified'}"


def page_reasons(metadata: dict, score: float) -> dict[str, float]:
    """reason -> share of this page's score lost to it; the values sum to 1 - score."""
    lost = 1.0 - score
    if lost <= 0:
        return {}
    elements = metadata.get("element_results") or []
    if not elements:
        return {"page failed (no usable layout)": lost}
    masses: dict[str, float] = defaultdict(float)
    weight = credit = 0.0
    for e in elements:
        w, s = float(e.get("weight") or 0.0), float(e.get("score") or 0.0)
        weight += w
        credit += w * s
        if w * (1.0 - s) > 0:
            group = CLASS_GROUPS.get(e.get("gt_class"), "other")
            masses[f"{ELEMENT_REASONS.get(e.get('why'), e.get('why') or 'unknown')}: {group}"] += w * (1.0 - s)
    base = credit / weight if weight else 0.0  # the score before extra boxes
    for p in metadata.get("prediction_results") or []:
        penalty = float(p.get("penalty") or 0.0)
        if penalty > 0:
            masses[extra_reason(p.get("reason", ""))] += penalty * base
    total = sum(masses.values())
    if total <= 0:
        return {"unattributed": lost}
    return {reason: lost * mass / total for reason, mass in masses.items()}
