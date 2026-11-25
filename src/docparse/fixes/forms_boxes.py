"""Check-box states from the tick marks a fill tool typed on the page (form option `boxes`).

Every box mark of the read (`[x]`, `[ ]`, `☐`, ...) has an option label beside it on its line or cell: after the
box when the line starts with a box, before it otherwise. The label's markdown tokens are aligned with printed PDF
words, so the label has a place on the page. Each check box drawn on the page (`docparse.pdf.boxes`)
belongs to the option label on its line whose near edge is closest to it (a box hugs its own label more than the
next one, whichever side it is on), and so does a tick glyph drawn outside any box. A mark whose label owns a
box holding a tick, or a tick, is written checked; a mark whose label owns only empty boxes is written unchecked.
A mark whose label was not found, or owns no box and no tick, keeps the read's state. Tick glyphs the read
wrote as stray text (`Yes [ ] **X** No [ ]`) are removed from lines whose marks were decided.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from docparse.fixes.forms import TICKS, FormPage, _same_line
from docparse.fixes.layout_refine import PtWord

_MARK = re.compile(
    r"\[[ \t]*(?:[xX✓✔✗✘☒☑*]|checked|selected|unchecked|not[ \t]?checked)?[ \t]*\]|[☐☑☒□■]", re.IGNORECASE
)
_UNIT_END = re.compile(r"\n|<br[ \t]*/?>|</?t[dhr][^>\n]*>|(?<!\\)\|")
BOX_REACH = 4.0  # a tick at most this many label heights beside the label's near edge
LABEL_TOKENS = 6  # tokens of a label next to its box


@dataclass
class Mark:
    start: int
    end: int
    checked: bool
    label: list[int]  # tokens of its option label, the one next to the box first
    after: bool  # the label follows the box


def _is_checked(text: str) -> bool:
    inner = text.strip("[]").strip().lower()
    return text in "☑☒■" or (inner != "" and not inner.startswith(("un", "not")) and text not in "☐□")


def _written(original: str, checked: bool) -> str:
    """The new state in the read's own spelling of a box."""
    if original in "☐☑☒□■":
        return "☒" if checked else "☐"
    inner = original.strip("[]").strip().lower()
    if inner.isalpha() and len(inner) > 1:
        return "[checked]" if checked else "[unchecked]"
    return "[x]" if checked else "[ ]"


def find_marks(page: FormPage) -> tuple[list[Mark], list[tuple[int, int]]]:
    """Box marks with their labels, and the (start, end) of every line or cell holding a mark."""
    md, tokens = page.markdown, page.al.tokens
    bounds = [0] + [m.end() for m in _UNIT_END.finditer(md)] + [len(md)]
    marks: list[Mark] = []
    units: list[tuple[int, int]] = []
    t = 0
    for u0, u1 in zip(bounds, bounds[1:], strict=False):
        found = [m for m in _MARK.finditer(md, u0, u1)]
        while t < len(tokens) and tokens[t].start < u0:
            t += 1
        if not found:
            continue
        units.append((u0, u1))
        mine = []
        k = t
        while k < len(tokens) and tokens[k].end <= u1:
            mine.append(k)
            k += 1
        # "[ ] Yes [ ] No" ends with a label, "Yes [ ] No [ ]" with a box
        after = any(tokens[i].start >= found[-1].end() and tokens[i].text not in TICKS for i in mine)
        edges = [u0] + [m.end() for m in found]
        nexts = [m.start() for m in found[1:]] + [u1]
        for i, m in enumerate(found):
            if after:
                lo, hi = m.end(), nexts[i]
                stop = md.find(":", lo, hi)
                hi = stop if stop >= 0 else hi
                label = [x for x in mine if lo <= tokens[x].start and tokens[x].end <= hi][:LABEL_TOKENS]
            else:
                lo, hi = edges[i], m.start()
                inside = [x for x in mine if lo <= tokens[x].start and tokens[x].end <= hi]
                colons = [c for c in range(lo, hi) if md[c] == ":" and any(tokens[x].start > c for x in inside)]
                if colons:  # "Question?: No [ ]": the label is what follows the question's colon
                    inside = [x for x in inside if tokens[x].start > colons[-1]]
                label = inside[-LABEL_TOKENS:][::-1]
            marks.append(Mark(m.start(), m.end(), _is_checked(m.group(0)), label, after))
    return marks, units


def _label_words(page: FormPage, mark: Mark) -> list[PtWord]:
    """The printed words of a mark's label on the line of the word next to the box; none when the token next
    to the box was not found on the page."""
    al = page.al
    label = [t for t in mark.label if not (al.tokens[t].text in TICKS and not al.token_words(t))]
    words = []
    for k, t in enumerate(label):
        mine = [al.words[w] for w in al.token_words(t) if page.owner[w] < 0]
        if k == 0 and not mine:
            return []
        words += mine
    return [w for w in words if _same_line(w, words[0])] if words else []


Rect = tuple[float, float, float, float]


def _gap(box: Rect, span: list[PtWord], after: bool) -> float | None:
    """Horizontal gap between a box (or tick) and a label on the same line, the box on the side the read puts
    it (before the label when `after`); None when not on that line or that side."""
    x0, x1 = min(w.x0 for w in span), max(w.x1 for w in span)
    y0, y1 = min(w.y0 for w in span), max(w.y1 for w in span)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if not (y0 - 0.5 * (y1 - y0) <= cy <= y1 + 0.5 * (y1 - y0)):
        return None
    if after:
        return max(x0 - box[2], 0.0) if cx <= x0 else None
    return max(box[0] - x1, 0.0) if cx >= x1 else None


def _nearest(rect: Rect, spans: list[list[PtWord]], marks: list[Mark]) -> int | None:
    """The label a box or tick belongs to: the closest on its line, on the side the read writes boxes."""
    best = None
    for i, span in enumerate(spans):
        if not span:
            continue
        gap = _gap(rect, span, marks[i].after)
        if gap is not None and gap <= BOX_REACH * (span[0].y1 - span[0].y0) and (best is None or gap < best[0]):
            best = (gap, i)
    return None if best is None else best[1]


def _inside(point: tuple[float, float], box: Rect, pad: float = 2.0) -> bool:
    return box[0] - pad <= point[0] <= box[2] + pad and box[1] - pad <= point[1] <= box[3] + pad


def box_edits(page: FormPage, drawn: list[Rect]) -> list[tuple[int, int, str]]:
    """New states for the read's box marks: checked when the label's box (or the label itself) holds a tick,
    unchecked when the label's drawn box is empty; marks with neither keep their state."""
    marks, units = find_marks(page)
    spans = [_label_words(page, m) for m in marks]
    state: dict[int, bool] = {}
    loose = list(page.ticks)
    for box in drawn:
        i = _nearest(box, spans, marks)
        ticked = [t for t in page.ticks if _inside((t.cx, t.cy), box)]
        loose = [t for t in loose if t not in ticked]
        if i is not None:
            state[i] = state.get(i, False) or bool(ticked)
    for tick in loose:  # a tick with no drawn box around it
        i = _nearest((tick.x0, tick.y0, tick.x1, tick.y1), spans, marks)
        if i is not None:
            state[i] = True
    edits: list[tuple[int, int, str]] = []
    decided: set[int] = set()
    for i, checked in state.items():
        mark = marks[i]
        decided.update(u for u, (u0, u1) in enumerate(units) if u0 <= mark.start < u1)
        if checked != mark.checked:
            edits.append((mark.start, mark.end, _written(page.markdown[mark.start : mark.end], checked)))
    edits += _stray_ticks(page, [units[u] for u in sorted(decided)], marks)
    return edits


_BESIDE = re.compile(r"[ \t*_]*")


def _stray_ticks(page: FormPage, units: list[tuple[int, int]], marks: list[Mark]) -> list[tuple[int, int, str]]:
    """Tick glyphs written as text (`**X**`) right beside a box mark on a decided line, with their emphasis
    and one space. (A tick written beside a label with no box is the read's only record of that state.)"""
    md, al = page.markdown, page.al
    out = []
    for t, tok in enumerate(al.tokens):
        if tok.text not in TICKS or al.token_words(t):
            continue
        if not any(u0 <= tok.start < u1 for u0, u1 in units):
            continue
        start, end = tok.start, tok.end
        lead = re.search(r"(\*{1,2}|_{1,2})$", md[:start])
        trail = re.match(r"\*{1,2}|_{1,2}", md[end:])
        if lead and trail and lead.group(0) == trail.group(0):
            start, end = start - len(lead.group(0)), end + len(trail.group(0))
        if not any(
            (m.end <= start and _BESIDE.fullmatch(md[m.end : start]))
            or (m.start >= end and _BESIDE.fullmatch(md[end : m.start]))
            for m in marks
        ):
            continue
        if md[end : end + 1] == " ":
            end += 1
        elif md[start - 1 : start] == " ":
            start -= 1
        out.append((start, end, ""))
    return out
