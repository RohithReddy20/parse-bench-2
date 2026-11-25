"""Structural table cleanups on filled forms (option `table_clean`).

1. Merges a dedicated leading printed row-ID column (`<td>5.7</td><td>Caption</td>`) into its caption cell
   (`<td colspan="2">5.7 Caption</td>`) when >=2 rows in the table have that shape.
2. Merges 3-cell `Caption | Overlay | PrintedCode` rows (`<td>Heat</td><td>724.48</td><td>7A</td>`) into
   `7A Heat: 724.48`.
3. Strips a trailing printed-only `<br/>...` line (bilingual label or note) from a 2-column table's
   overlay value cell (`<td>Sharjah<br/>الإمارة</td>` -> `<td>Sharjah</td>`).
4. Rewrites horizontal checkbox strip tables (`<th>RA</th><th>RB</th>...` over `<td></td><td>X</td>...`)
   into `[x] Label` / `[ ] Label` cells decided strictly from PDF drawn boxes and all overlay tick glyphs
   (including adjacent ticks grouped together in `page.fills`).
"""

from __future__ import annotations

import html
import re

from docparse.fixes.forms import TICKS, FormPage
from docparse.fixes.forms_tables import _cell_spans, _cell_toks, _classify_cell, _clean_code, _token_flags
from docparse.fixes.tables_html import HCell, parse_table

_HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.S | re.I)
_ID_CELL = re.compile(r"^(?:[A-Za-z]{0,3}[0-9]{1,4}[A-Za-z]?\.?|[0-9]{1,2}\.[0-9]{1,3}\.?|[A-Za-z][0-9]{1,2})$")
_BR_SPLIT = re.compile(r"(?:<br\s*/?>\s*)+", re.I)
_HAS_BOX = re.compile(r"\[[ \t]*(?:[xX✓✔])?[ \t]*\]|[☐☑☒□■]")
_TICK_OR_EMPTY = re.compile(
    r"^\s*(?:<(?:strong|b)>)?(?:[xX✓✔✗✘☒☑]|\[[ \t]*(?:[xX✓✔])?[ \t]*\]|[☐☑☒])?(?:</(?:strong|b)>)?\s*$"
)
_TAG = re.compile(r"<[^>]+>")


def _plain(s: str) -> str:
    return " ".join(html.unescape(_TAG.sub(" ", s)).split())


def _all_tick_words(page: FormPage):
    out = list(page.ticks)
    for f in page.fills:
        if len(f.words) >= 2 and all(w.text in TICKS for w in f.words):
            out.extend(f.words)
    return out


def _inside(pt: tuple[float, float], box: tuple[float, float, float, float], pad: float = 2.5) -> bool:
    return box[0] - pad <= pt[0] <= box[2] + pad and box[1] - pad <= pt[1] <= box[3] + pad


def table_clean_edits(
    page: FormPage, drawn: list[tuple[float, float, float, float]] | None = None
) -> list[tuple[int, int, str]]:
    """Apply overlay-backed structural table cleanups on filled form pages."""
    md, al = page.markdown, page.al
    tok_ov, _ = _token_flags(page)
    all_ticks = _all_tick_words(page) if drawn else []
    edits: list[tuple[int, int, str]] = []

    for m in _HTML_TABLE.finditer(md):
        ts, te = m.start(), m.end()
        raw_t = m.group(0)
        table = parse_table(raw_t)
        if table is None:
            continue
        spans = _cell_spans(raw_t, ts, table)
        kinds: dict[int, str] = {}
        for row in table.rows:
            for c in row.cells:
                sp = spans.get(id(c))
                ct = _cell_toks(al, sp[0], sp[1]) if sp else []
                kinds[id(c)] = _classify_cell(c, ct, tok_ov)

        changed = False

        # 1. Horizontal checkbox strip tables (`<th>RA</th><th>RB</th>...` over `<td></td><td>X</td>...`)
        if drawn and all_ticks and len(table.rows) >= 2:
            new_rows = []
            strip_changed = False
            r = 0
            while r < len(table.rows):
                if r + 1 < len(table.rows):
                    r0, r1 = table.rows[r], table.rows[r + 1]
                    if len(r0.cells) >= 4 and all(_TICK_OR_EMPTY.fullmatch(c.inner) for c in r1.cells):
                        matched: dict[int, tuple[str, bool]] = {}
                        for ci, c0 in enumerate(r0.cells):
                            lbl = _plain(c0.inner)
                            if not lbl or len(lbl) > 80:
                                continue
                            sp = spans.get(id(c0))
                            if not sp:
                                continue
                            ctoks = _cell_toks(al, sp[0], sp[1])
                            pws = [al.words[w] for t in ctoks for w in al.token_words(t) if page.owner[w] < 0]
                            if not pws:
                                continue
                            wx0, wx1 = min(w.x0 for w in pws), max(w.x1 for w in pws)
                            wy0, wy1 = min(w.y0 for w in pws), max(w.y1 for w in pws)
                            wcx = (wx0 + wx1) / 2
                            h = max(6.0, wy1 - wy0)
                            cands = []
                            for b in drawn:
                                bcx = (b[0] + b[2]) / 2
                                if wx0 - 6.0 <= bcx <= wx1 + 6.0 and wy0 - 2.0 <= b[1] <= wy1 + 4.5 * h:
                                    cands.append((abs(bcx - wcx) + max(0.0, b[1] - wy1), b))
                            if not cands:
                                continue
                            _, box = min(cands)
                            is_ticked = any(_inside((tk.cx, tk.cy), box) for tk in all_ticks)
                            matched[ci] = (lbl, is_ticked)
                        if len(matched) >= 4 and (any(chk for _, chk in matched.values()) or strip_changed):
                            new_cells = []
                            for ci, c0 in enumerate(r0.cells):
                                if ci in matched:
                                    lbl, chk = matched[ci]
                                    mark = "[x]" if chk else "[ ]"
                                    new_cells.append(
                                        HCell(
                                            tag="td",
                                            attrs=c0.attrs,
                                            inner=f"{mark} {html.escape(lbl)}",
                                            raw="",
                                            colspan=c0.colspan,
                                        )
                                    )
                                else:
                                    new_cells.append(c0)
                            r0.cells = new_cells
                            r0.edited = True
                            new_rows.append(r0)
                            strip_changed = True
                            r += 2
                            continue
                new_rows.append(table.rows[r])
                r += 1
            if strip_changed:
                table.rows = new_rows
                table.seps = [table.seps[0]] + [""] * (len(new_rows) - 1) + [table.seps[-1]]
                table.changed = True
                edits.append((ts, te, table.render()))
                continue

        # 2. Trailing printed-only `<br/>...` line inside 2-column overlay value cells (`<td>Sharjah<br/>الإمارة</td>`)
        for row in table.rows:
            if len(row.cells) != 2:
                continue
            c0, c1 = row.cells
            if c0.tag != "td" or c1.tag != "td" or "<br" not in c1.inner.lower() or _HAS_BOX.search(c1.text):
                continue
            sp0, sp1 = spans.get(id(c0)), spans.get(id(c1))
            if not sp0 or not sp1:
                continue
            t0 = _cell_toks(al, sp0[0], sp0[1])
            if not t0 or any(tok_ov[t] for t in t0):
                continue
            bm = _BR_SPLIT.search(c1.inner)
            if not bm:
                continue
            inner_s = raw_t.find(c1.inner, sp1[0] - ts)
            if inner_s < 0:
                continue
            abs_in = ts + inner_s
            ht = _cell_toks(al, abs_in, abs_in + bm.start())
            tt = _cell_toks(al, abs_in + bm.end(), abs_in + len(c1.inner))
            if ht and tt and all(tok_ov[t] for t in ht) and not any(tok_ov[t] for t in tt):
                c1.inner = c1.inner[: bm.start()].strip()
                c1.raw = ""
                row.edited = True
                changed = True

        # 3. 3-cell `Caption | Overlay | PrintedCode` rows (`<td>Heat</td><td>724.48</td><td>7A</td>`)
        has_th_header = (
            bool(table.rows)
            and all(c.tag == "th" for c in table.rows[0].cells)
            and sum(bool(c.text.strip()) for c in table.rows[0].cells) >= 2
        )
        if not has_th_header:
            for r_idx, row in enumerate(table.rows):
                if len(row.cells) != 3 or any(c.rowspan > 1 for c in row.cells) or table.spanned_into(r_idx):
                    continue
                c0, c1, c2 = row.cells
                if kinds[id(c0)] == "CAPTION" and kinds[id(c1)] == "OVERLAY" and kinds[id(c2)] == "CODE":
                    code = _clean_code(c2.text)
                    cap = c0.text.strip()
                    val = c1.text.strip()
                    if code and not cap.lower().startswith(code.lower()):
                        sep = " " if cap.endswith(":") else ": "
                        span_sum = c0.colspan + c1.colspan + c2.colspan
                        row.cells = [
                            HCell(
                                tag="td",
                                attrs=f' colspan="{span_sum}"',
                                inner=f"{html.escape(code)} {html.escape(cap)}{sep}{html.escape(val)}",
                                raw="",
                                colspan=span_sum,
                            )
                        ]
                        row.edited = True
                        changed = True

        # 4. Merge leading dedicated printed row-ID column (`<td>5.7</td><td>Caption</td>`) into its caption cell
        cand_rows: list[int] = []
        for r_idx, row in enumerate(table.rows):
            if len(row.cells) < 3 or any(c.rowspan > 1 for c in row.cells[:2]) or table.spanned_into(r_idx):
                continue
            c0, c1 = row.cells[0], row.cells[1]
            if c0.tag != "td" or c1.tag != "td" or c0.col != 0 or c0.colspan != 1:
                continue
            t0 = c0.text.strip().lstrip("*")
            if kinds[id(c0)] in ("CODE", "CAPTION") and _ID_CELL.fullmatch(t0) and kinds[id(c1)] == "CAPTION":
                if not c1.text.strip().lower().startswith(t0.lower()):
                    cand_rows.append(r_idx)
        if len(cand_rows) >= 2:
            for r_idx in cand_rows:
                row = table.rows[r_idx]
                c0, c1 = row.cells[0], row.cells[1]
                t0 = c0.text.strip().lstrip("*")
                span_sum = c0.colspan + c1.colspan
                merged = HCell(
                    tag="td",
                    attrs=f' colspan="{span_sum}"',
                    inner=f"{html.escape(t0)} {c1.inner.strip()}",
                    raw="",
                    colspan=span_sum,
                )
                row.cells = [merged] + row.cells[2:]
                row.edited = True
                changed = True

        if changed:
            table.changed = True
            edits.append((ts, te, table.render()))

    return edits
