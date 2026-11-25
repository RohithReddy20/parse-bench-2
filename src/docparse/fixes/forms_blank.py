"""Blank-form structural repairs on pages without a typed-in overlay (option `n_blank_forms`).

Repairs unmistakable blank-form structures when no typed overlay is present:
1. Tables with >= 3 bold box-numbered blank cells (`<td><strong>141.</strong><br/>zł</td>`,
   `<td><strong>67.</strong><br/>,</td>`) -> `<td>141. <row_title>: [___]</td>`.
2. Tables with >= 3 rows of `<td>CAPTION</td><td><strong>CODE</strong></td><td>[___]</td>`
   (including dual-column code/slot pairs per row) -> `<td>CODE CAPTION: [___]</td>`.
3. Dense blank-field pages (>= 10 `[___]` slots): split adjacent inline `**Field:** [___] **NextField**`
   onto separate lines and strip trailing `Years` / `Months` / `Days` unit hints right after `: [___]`.
"""

from __future__ import annotations

import html
import re

from docparse.fixes.tables_html import HCell, parse_table

_HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_FIRST_STRONG = re.compile(r"<(?:strong|b)>(.*?)</(?:strong|b)>", re.S | re.I)
_BOX_NUM_BLANK = re.compile(
    r"^\s*<(?:strong|b)>([0-9]{1,4}\.)</(?:strong|b)>\s*(?:<br\s*/?>\s*)+([,.\-_/—–\s]+|zł|zł,\s*gr)\s*$",
    re.S | re.I,
)
_CODE_BOLD = re.compile(r"^\s*<(?:strong|b)>([0-9]{2,5}[A-Za-z]?)</(?:strong|b)>\s*$", re.I)
_INLINE_NEXT_FIELD = re.compile(r"(\[_{2,}\])[ \t]+(\*\*[A-Za-zÀ-ÿ][^*\n]{1,50}\*\*)")
_UNIT_AFTER_BLANK = re.compile(r"(:\*{0,2}[ \t]*\[_{2,}\])[ \t]+(?:Years|Months|Days)\b")
_FMT_TAG = re.compile(r"</?(?:u|s|del)>|~~", re.I)


def _plain(s: str) -> str:
    return " ".join(html.unescape(_TAG.sub(" ", s)).split())


def _first_title(cell: HCell) -> str:
    m = _FIRST_STRONG.search(cell.inner)
    if m:
        t = _plain(m.group(1))
        if len(t) >= 4:
            return t
    for part in re.split(r"<br\s*/?>", cell.inner, flags=re.I):
        t = _plain(part)
        if t:
            return t
    return cell.text.strip()


def repair_blank_form(markdown: str) -> str:
    """Apply blank-form structural repairs on a page without an overlay."""
    if not markdown:
        return markdown

    def _fix_table(m: re.Match[str]) -> str:
        t = parse_table(m.group(0))
        if t is None or len(t.rows) < 2:
            return m.group(0)

        # 1. Tables with >= 3 bold box-numbered blank cells (`<strong>141.</strong><br/>,`)
        row_hits = []
        for r_idx, row in enumerate(t.rows):
            if (
                not row.cells
                or all(c.tag == "th" for c in row.cells)
                or t.spanned_into(r_idx)
                or any(c.rowspan > 1 for c in row.cells)
            ):
                continue
            c0 = row.cells[0]
            if not c0.text.strip():
                continue
            title = _first_title(c0)
            bcs = [(c, mb.group(1)) for c in row.cells[1:] if (mb := _BOX_NUM_BLANK.match(c.inner))]
            if bcs:
                row_hits.append((row, c0, title, bcs))
        if sum(len(bcs) for *_, bcs in row_hits) >= 3:
            for row, c0, title, bcs in row_hits:
                for c, code_s in bcs:
                    c.inner = f"{html.escape(code_s)} {html.escape(title)}: [___]"
                    c.raw = ""
                if "<br" in c0.inner.lower():
                    c0.inner = ""
                    c0.raw = ""
                row.edited = True
            t.changed = True
            return t.render()

        # 2. Tables with >= 3 rows of `<td>CAPTION</td><td><strong>CODE</strong></td><td>[___]</td>`
        code_rows = sum(
            1
            for row in t.rows
            if len(row.cells) >= 3
            and row.cells[0].tag == "td"
            and _CODE_BOLD.match(row.cells[1].inner)
            and "[___]" in row.cells[2].text
        )
        if code_rows >= 3:
            for row in t.rows:
                if len(row.cells) < 3 or row.cells[0].tag != "td" or any(c.rowspan > 1 for c in row.cells):
                    continue
                cap = row.cells[0].text.strip()
                if not cap:
                    continue
                pairs: list[tuple[int, str]] = []
                i = 1
                while i < len(row.cells):
                    mc = _CODE_BOLD.match(row.cells[i].inner)
                    if (
                        mc
                        and i + 1 < len(row.cells)
                        and ("[___]" in row.cells[i + 1].text or not row.cells[i + 1].text.strip())
                    ):
                        pairs.append((i, mc.group(1)))
                        i += 2
                        continue
                    i += 1
                if pairs:
                    for idx, code in pairs:
                        row.cells[idx].inner = f"{html.escape(code)} {html.escape(cap)}: [___]"
                        row.cells[idx].raw = ""
                        row.cells[idx + 1].inner = ""
                        row.cells[idx + 1].raw = ""
                    row.cells[0].inner = ""
                    row.cells[0].raw = ""
                    row.edited = True
            t.changed = True
            return t.render()

        return m.group(0)

    out = _HTML_TABLE.sub(_fix_table, markdown)
    if out.count("[___]") >= 10 and not _FMT_TAG.search(out):
        parts = re.split(r"(<table\b[^>]*>.*?</table>)", out, flags=re.S | re.I)
        for i in range(0, len(parts), 2):
            new_lines = []
            for ln in parts[i].splitlines(keepends=True):
                if "[___]" in ln:
                    ln = _UNIT_AFTER_BLANK.sub(r"\1", ln)
                    ln = _INLINE_NEXT_FIELD.sub(r"\1\n\2", ln)
                new_lines.append(ln)
            parts[i] = "".join(new_lines)
        out = "".join(parts)
    return out
