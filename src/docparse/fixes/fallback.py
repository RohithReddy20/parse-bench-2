"""When the page read is degenerate (a loop, an empty grid, a page cut short) and the PDF has a text layer,
the text layer is the better reading. `reading_recall` measures the read against the text layer;
`looped` spots a read that repeats itself; `text_dump` writes the text layer as plain lines.
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from collections import Counter

from docparse.fixes.layout_refine import PtWord, lines_of

MIN_TOKENS = 50  # shorter text layers say too little about the reading
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_TAG = re.compile(r"<[^>]+>")
_MATH_DISPLAY = re.compile(r"\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)")
_MATH_CMDS = re.compile(
    r"\\(?:frac|sum|int|prod|equiv|pmod|bmod|leq|geq|neq|approx|subset|subseteq|in|notin"
    r"|to|rightarrow|longrightarrow|alpha|beta|gamma|delta|epsilon|varepsilon|theta|lambda"
    r"|mu|nu|pi|rho|sigma|phi|psi|omega|mathbb|mathcal|mathrm|text|begin|end)\b"
)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).lower())


def has_math(markdown: str) -> bool:
    """True when the reading carries real LaTeX math (which a plain PDF text dump would destroy)."""
    return bool(_MATH_DISPLAY.search(markdown)) or len(set(_MATH_CMDS.findall(markdown))) >= 2


def _close_dangling_html(text: str) -> str:
    """Close an unclosed HTML table cut off when a repeating loop tail was stripped."""
    low = text.lower()
    if low.count("<table") <= low.count("</table>"):
        return text
    # Drop a trailing partial tag (e.g. '<td' or '</tr') if cut mid-tag
    cut = re.sub(r"<[^>]*$", "", text)
    # Trim back to the last closed </tr> if there is an incomplete trailing row
    last_tr_end = cut.lower().rfind("</tr>")
    last_table_start = cut.lower().rfind("<table")
    if last_tr_end > last_table_start:
        cut = cut[: last_tr_end + 5] + "</table>"
    else:
        cut = cut + "</table>"
    return cut


def trim_loop_tail(markdown: str, min_chars: int = 1200) -> str:
    """Strip a repeating decoding-loop suffix (e.g. `\\quad `, `<td></td>`, `*[___]*`, repeated `<tr>...</tr>`)
    from the end of a page reading, preserving the valid content written before the loop began.
    Works on both digital and scanned pages (no PDF text layer needed)."""
    if len(markdown) < min_chars:
        return markdown
    # Ignore a short truncated tail fragment (up to 300 chars) when searching for the repeating unit
    best_cut = None
    n = len(markdown)
    # Candidate anchor offsets near the end (handles cut-off mid-token)
    anchors = [n]
    for back in (1, 8, 32, 96, 200):
        if n - back > 600:
            anchors.append(n - back)
    for end in anchors:
        # Look for period p in 1..400 repeating at markdown[:end]
        # Fast search: check occurrences of the last 8-char (or short) signature within 400 chars
        tail_probe_len = 16
        if end < 600:
            continue
        candidate_periods: list[int] = list(range(1, 25))
        probe = markdown[end - tail_probe_len : end]
        search_start = max(0, end - 450)
        pos = markdown.rfind(probe, search_start, end - 1)
        while pos != -1:
            p = end - pos
            if p >= 25:
                candidate_periods.append(p)
            pos = markdown.rfind(probe, search_start, pos)
        for p in candidate_periods:
            if end < p * 6:
                continue
            unit = markdown[end - p : end]
            # Count consecutive repetitions of `unit` ending at `end`
            idx = end - p
            reps = 1
            while idx >= p and markdown[idx - p : idx] == unit:
                idx -= p
                reps += 1
            span = end - idx
            min_reps = 12 if p <= 6 else 6
            min_span = 200 if p <= 6 else 400
            if reps >= min_reps and span >= min_span:
                # Keep one copy unless the repeating unit is pure filler (`<td></td>`, `\quad`, `[___]`, spaces)
                stripped_unit = re.sub(r"</?t[hd]\s*>|\\quad|\\qquad|\*+|\[_+\]|\s+", "", unit, flags=re.I)
                cut_at = idx if not stripped_unit else idx + p
                if best_cut is None or cut_at < best_cut:
                    best_cut = cut_at
                break
    if best_cut is None:
        return markdown
    return _close_dangling_html(markdown[:best_cut].rstrip())


def reading_recall(markdown: str, words: list[PtWord]) -> float:
    """Share of the text layer's word tokens that the reading contains (each as often as the page shows it)."""
    page = Counter(t for w in words for t in _tokens(w.text))
    total = sum(page.values())
    if not total:
        return 1.0
    read = Counter(_tokens(_TAG.sub(" ", markdown)))
    return sum(min(n, read[t]) for t, n in page.items()) / total


def looped(markdown: str, ratio: float = 30.0, min_chars: int = 3000) -> bool:
    """A read that repeats itself compresses far better than any real page (real pages: 3x to 15x)."""
    data = markdown.encode("utf-8")
    return len(data) >= min_chars and len(data) / len(zlib.compress(data)) >= ratio


def text_dump(words: list[PtWord]) -> str:
    """The text layer as lines, top to bottom; a vertical gap over one line height starts a new paragraph."""
    out: list[str] = []
    previous = None
    for line in lines_of(words):
        top, bottom = min(w.y0 for w in line), max(w.y1 for w in line)
        if previous is not None and top - previous > bottom - top:
            out.append("")
        out.append(" ".join(w.text for w in line))
        previous = bottom
    return "\n".join(out)
