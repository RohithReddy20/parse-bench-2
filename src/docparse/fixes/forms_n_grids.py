"""Checkbox rating-grid and bare-tick repairs on filled forms (options `box_grids`, `bare_ticks`).

- `box_grids`: in HTML rating/option grids whose `<th>` header names short option labels and body rows
  have checkbox-only cells (`<td>[ ]</td><td>[x]</td>`), writes the column's option label beside each
  checkbox (`[ ] Excellent`, `[x] Fair`) so each option carries its label.
- `bare_ticks`: when a page read emits a bare overlay tick (`X`, `x`, `✓`) next to a printed option label
  without brackets, replaces it with `[x]` once confirmed against the PDF's overlay tick coordinates, and
  adds `[ ]` beside sibling options on the same line that have matching empty drawn check boxes in the PDF.
"""

from __future__ import annotations

import html
import re

from docparse.fixes.forms import FormPage, _same_line
from docparse.fixes.tables_html import parse_table

_HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.S | re.I)
_BOX_ONLY = re.compile(r"^\s*(\[[ \t]*(?:[xX✓✔✗✘☒☑*])?[ \t]*\]|[☐☑☒□■○◯●◉])\s*$")
_HAS_MARK = re.compile(r"\[[ \t]*(?:[xX✓✔])?[ \t]*\]|[☐☑☒□■]")
_UNIT_END = re.compile(r"\n|<br[ \t]*/?>|</?t[dhr][^>\n]*>|(?<!\\)\|")
_BARE_TICKS = frozenset({"X", "x", "✓", "✔"})
_DUP_BOX_LINE = re.compile(
    r"^([ \t]*(?:[-*+][ \t]+)?)\[[ \t]*\][ \t]+([^\n\[\]:]{1,35}?)[ \t]+(\[[xX✓✔]\])[ \t]*$",
    re.M,
)
_CAP_THEN_BOXES = re.compile(
    r"^(?P<head>[ \t]*(?:[-*+][ \t]+)?[^\n\[\]]{2,80}?:\*{0,2}[ \t]+)"
    r"(?P<opts>\[[ \t]*(?:[xX✓✔])?[ \t]*\][ \t]+[^\n\[\]|]{1,40}"
    r"(?:[ \t]+\[[ \t]*(?:[xX✓✔])?[ \t]*\][ \t]+[^\n\[\]|]{1,40})+)",
    re.M,
)


def box_grid_edits(markdown: str) -> list[tuple[int, int, str]]:
    """Repair checkbox layouts on filled forms:
    - label checkbox-only cells in rating/option HTML tables from their column headers (`[x]` -> `[x] Fair`);
    - drop an orphan leading `[ ]` when a lone option has `[ ] Label [x]` (`[x] Label`);
    - split a group caption ending with `:` from inline leading-box options (`Sexe: [ ] M [x] F`)."""
    edits: list[tuple[int, int, str]] = []
    tranges: list[tuple[int, int]] = []
    for m in _HTML_TABLE.finditer(markdown):
        ts, te = m.start(), m.end()
        tranges.append((ts, te))
        table = parse_table(m.group(0))
        if table is None or len(table.rows) < 3:
            continue
        h_end = 0
        while (
            h_end < len(table.rows) and table.rows[h_end].cells and all(c.tag == "th" for c in table.rows[h_end].cells)
        ):
            h_end += 1
        if h_end == 0:
            continue
        col_label: dict[int, str] = {}
        for r in range(h_end):
            for c in table.rows[r].cells:
                if c.colspan == 1 and c.text.strip():
                    col_label[c.col] = c.text.strip()
        body = table.rows[h_end:]
        col_box_rows: dict[int, int] = {}
        col_nonempty_rows: dict[int, int] = {}
        for row in body:
            for c in row.cells:
                if c.colspan != 1:
                    continue
                txt = c.text.strip()
                if not txt:
                    continue
                col_nonempty_rows[c.col] = col_nonempty_rows.get(c.col, 0) + 1
                if _BOX_ONLY.fullmatch(txt):
                    col_box_rows[c.col] = col_box_rows.get(c.col, 0) + 1
        opt_cols = {
            col
            for col, cnt in col_box_rows.items()
            if cnt >= 2
            and cnt >= 0.8 * col_nonempty_rows.get(col, 0)
            and col in col_label
            and 1 <= len(col_label[col].split()) <= 4
            and len(col_label[col]) <= 35
        }
        if len(opt_cols) < 2:
            continue
        changed = False
        for row in body:
            row_opt = [
                c for c in row.cells if c.colspan == 1 and c.col in opt_cols and _BOX_ONLY.fullmatch(c.text.strip())
            ]
            if len(row_opt) < 2:
                continue
            for c in row_opt:
                c.inner = f"{c.text.strip()} {html.escape(col_label[c.col])}"
                c.raw = ""
                row.edited = True
                changed = True
        if changed:
            table.changed = True
            edits.append((ts, te, table.render()))
    for m in _DUP_BOX_LINE.finditer(markdown):
        if any(a <= m.start() < b for a, b in tranges):
            continue
        edits.append((m.start(), m.end(), f"{m.group(1)}{m.group(3)} {m.group(2)}"))
    for m in _CAP_THEN_BOXES.finditer(markdown):
        if any(a <= m.start() < b for a, b in tranges):
            continue
        edits.append((m.start("opts"), m.start("opts"), "\n"))
    return edits


def _inside(pt: tuple[float, float], box: tuple[float, float, float, float], pad: float = 2.0) -> bool:
    return box[0] - pad <= pt[0] <= box[2] + pad and box[1] - pad <= pt[1] <= box[3] + pad


def bare_tick_edits(page: FormPage, drawn: list[tuple[float, float, float, float]]) -> list[tuple[int, int, str]]:
    """Replace unaligned bare overlay ticks (`X`, `x`, `✓`) with `[x]` and mark empty drawn sibling boxes `[ ]`."""
    if not page.ticks:
        return []
    md, al = page.markdown, page.al
    edits: list[tuple[int, int, str]] = []

    bounds = [0] + [m.end() for m in _UNIT_END.finditer(md)] + [len(md)]
    t_ptr = 0
    for u0, u1 in zip(bounds, bounds[1:], strict=False):
        while t_ptr < len(al.tokens) and al.tokens[t_ptr].start < u0:
            t_ptr += 1
        mine: list[int] = []
        k = t_ptr
        while k < len(al.tokens) and al.tokens[k].end <= u1:
            if al.tokens[k].start >= u0:
                mine.append(k)
            k += 1
        if not mine or _HAS_MARK.search(md[u0:u1]):
            continue

        bare = [t for t in mine if al.tokens[t].text in _BARE_TICKS and not al.token_words(t)]
        if not bare:
            continue
        bare_set = set(bare)
        added_empty: set[int] = set()

        for bt in bare:
            bt_idx = mine.index(bt)
            after_w = []
            for j in mine[bt_idx + 1 : bt_idx + 4]:
                ws = [al.words[w] for w in al.token_words(j) if page.owner[w] < 0]
                if ws:
                    after_w = ws
                    break
            before_w = []
            for j in reversed(mine[max(0, bt_idx - 3) : bt_idx]):
                ws = [al.words[w] for w in al.token_words(j) if page.owner[w] < 0]
                if ws:
                    before_w = ws
                    break

            matched_tick = None
            side = None
            best_gap = 1e9
            if after_w:
                w0 = after_w[0]
                h = w0.y1 - w0.y0
                for tk in page.ticks:
                    if _same_line(w0, tk) and tk.cx <= w0.x0 + 2.0:
                        g = max(0.0, w0.x0 - tk.x1)
                        if g <= 4.0 * h and g < best_gap:
                            best_gap = g
                            matched_tick = tk
                            side = "after"
            if before_w:
                w0 = before_w[-1]
                h = w0.y1 - w0.y0
                for tk in page.ticks:
                    if _same_line(w0, tk) and tk.cx >= w0.x1 - 2.0:
                        g = max(0.0, tk.x0 - w0.x1)
                        if g <= 4.0 * h and g < best_gap:
                            best_gap = g
                            matched_tick = tk
                            side = "before"

            if matched_tick is None:
                continue

            tok = al.tokens[bt]
            edits.append((tok.start, tok.end, "[x]"))
            if not drawn:
                continue

            tbox = next((b for b in drawn if _inside((matched_tick.cx, matched_tick.cy), b)), None)
            for j in mine:
                if j in bare_set or j in added_empty:
                    continue
                ws = [al.words[w] for w in al.token_words(j) if page.owner[w] < 0]
                if not ws:
                    continue
                w0 = ws[0] if side == "after" else ws[-1]
                if not _same_line(w0, matched_tick):
                    continue
                h = w0.y1 - w0.y0
                for b in drawn:
                    if b == tbox:
                        continue
                    bw, bh = b[2] - b[0], b[3] - b[1]
                    if tbox is not None:
                        tw, th = tbox[2] - tbox[0], tbox[3] - tbox[1]
                        if not (0.7 * tw <= bw <= 1.3 * tw and 0.7 * th <= bh <= 1.3 * th):
                            continue
                    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    if not (w0.y0 - 0.6 * h <= bcy <= w0.y1 + 0.6 * h):
                        continue
                    if any(_inside((tk.cx, tk.cy), b) for tk in page.ticks):
                        continue
                    if any(b[0] <= p.cx <= b[2] and b[1] <= p.cy <= b[3] for p in page.printed):
                        continue
                    if side == "after" and bcx <= w0.x0 and 0 <= w0.x0 - b[2] <= 2.5 * h:
                        if not any(b[2] < p.cx < w0.x0 and _same_line(p, w0) for p in page.printed):
                            edits.append((al.tokens[j].start, al.tokens[j].start, "[ ] "))
                            added_empty.add(j)
                            break
                    elif side == "before" and bcx >= w0.x1 and 0 <= b[0] - w0.x1 <= 2.5 * h:
                        if not any(w0.x1 < p.cx < b[0] and _same_line(p, w0) for p in page.printed):
                            edits.append((al.tokens[j].end, al.tokens[j].end, " [ ]"))
                            added_empty.add(j)
                            break

    return edits
