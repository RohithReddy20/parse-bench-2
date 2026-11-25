"""Table corrections from the PDF text layer (design §6, tables). Each is one option, measured on its own.

    regrid   a row narrower than its table (a header written without its blank corner cell, a body row missing
             an empty cell) gets empty cells inserted so that each of its cells found on the page sits in the
             column its words lie in. Columns are located by the cells of full-width rows.
    regrid_wide  with regrid, also a row wider than its table that holds empty cells: its empty cells are laid
             out again so that at least two found cells sit in their columns and the row fits the table.
    complete a table the reading left unclosed (cut off by the token limit) gets the rows the text layer shows
             below its last row: each line whose words all fall in the table's columns and fill half of them.
    merge    two tables the reading wrote one after the other but that stand side by side on the same text lines
             become one table, when the right one has no row labels of its own and does not repeat the left
             one's row-label header (repeated column groups stay separate tables).
    health   share of a table's text cells that must be found word for word on the page before anything in the
             table is changed (default 0.5). A broken font map or a stale OCR layer finds few cells, and the
             model's table stays as it is.

A table the options do not change is written back byte for byte, and a page without a text layer is untouched.
The corrected table is a new candidate of its region (reader "fix.tables.pdf"); OCRia-1's reading stays in the IR.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from docparse.core.ir import Candidate, Page, Provenance, Region
from docparse.fixes.tables_align import Alignment, TWord, align_table, find_on_lines, fold, lines_of, page_words
from docparse.fixes.tables_html import HCell, HTable, parse_table
from docparse.pdf.render import RenderedPage

READER = "fix.tables.pdf"
HEADER_LOOKUP = 8  # lines above a table's first matched row searched for header cells


@dataclass
class _Found:
    """One table region of the page, its table and where its cells sit on the page."""

    region: Region
    table: HTable
    al: Alignment
    healthy: bool


def fix_tables(page: Page, rendered: RenderedPage, options: dict) -> None:
    """Apply the enabled options to every HTML table region of the page, in reading order."""
    if not any(options.get(k) for k in ("regrid", "complete", "merge")):
        return
    words = page_words(rendered)
    if not words:
        return
    lines = lines_of(words)
    health = float(options.get("health", 0.5))
    used: set[int] = set()
    found: list[_Found] = []
    new: dict[str, tuple[str, str]] = {}  # region id -> (raw, what changed it)
    for region in page.regions:
        base = region.chosen()
        if region.kind == "table":
            table = parse_table(base.raw)
            if table is None:
                continue
            al = align_table(table, lines, used)
            for ws in al.words.values():
                used.update(w.index for w in ws)
            healthy = bool(al.cells) and al.found >= health
            found.append(_Found(region, table, al, healthy))
            if healthy and options.get("regrid") and regrid(table, al, lines, used, bool(options.get("regrid_wide"))):
                table.changed = True
                new[region.id] = (table.render(), "regrid")
        elif region.kind == "opaque" and options.get("complete") and base.raw.lower().startswith("<table"):
            raw = complete_truncated(base.raw, lines, used, health)
            if raw is not None:
                new[region.id] = (raw, "complete")
    if options.get("merge"):
        for left, right in zip(found, found[1:], strict=False):
            if left.region.id in new and new[left.region.id][1] == "merge":
                continue  # already merged with the table before it
            raw = merge_side_by_side(left, right)
            if raw is not None:
                new[left.region.id] = (raw, "merge")
                new[right.region.id] = ("", "merge")
    for region in page.regions:
        if region.id in new:
            raw, detail = new[region.id]
            cand = Candidate(
                id=f"{region.id}c{len(region.candidates)}",
                reader=READER,
                raw=raw,
                prov=[Provenance(source="pdf_text", ref="text_layer", detail=detail)],
            )
            region.candidates.append(cand)
            region.selected = cand.id


Bands = dict[int, list[tuple[float, float]]]  # grid column -> x-spans of the words of its cells


def _bands(table: HTable, al: Alignment, width: int) -> Bands:
    """Per grid column, the x-spans of the found single-column cells of full-width rows."""
    out: Bands = {}
    for row in table.rows:
        for cell in row.cells:
            span = al.span(cell)
            if span and cell.colspan == 1 and table.row_width(cell.row) == width:
                out.setdefault(cell.col, []).append(span)
    return out


def column_of(x0: float, x1: float, colspan: int, bands: Bands, width: int) -> int | None:
    """The first column of a cell whose words span x0..x1: for one column, the band it overlaps most; for a
    spanning cell, the run of `colspan` columns whose extent holds the words (or lies inside them) and whose
    centre is nearest the words' centre."""
    lo = {c: min(a for a, _ in v) for c, v in bands.items()}
    hi = {c: max(b for _, b in v) for c, v in bands.items()}
    best, best_score = None, None
    for c in range(width - colspan + 1):
        cols = [k for k in range(c, c + colspan) if k in bands]
        if not cols:
            continue
        if colspan == 1:
            overlap = min(x1, hi[c]) - max(x0, lo[c])
            if overlap <= 0:
                continue
            score = -overlap
        else:
            left, right = min(lo[k] for k in cols), max(hi[k] for k in cols)
            if not (left - 2 <= x0 and x1 <= right + 2) and not (x0 <= left and right <= x1):
                continue
            score = abs((x0 + x1) / 2 - (left + right) / 2)
        if best_score is None or score < best_score:
            best, best_score = c, score
    return best


def _header_spans(row_cells: list[HCell], lines: list[list[TWord]], top: int, used: set[int]) -> dict[int, tuple]:
    """For a header row no line was matched to: each cell whose every written line occurs exactly once on the
    lines just above the table body, with the x-span of those words."""
    out = {}
    for cell in row_cells:
        parts = [fold(p) for p in cell.lines]
        found: list[TWord] = []
        for key in parts:
            hits = find_on_lines(key, lines, top - HEADER_LOOKUP, top, used)
            if len(hits) == 1:
                found += hits[0]
        if found:
            out[id(cell)] = (min(w.x0 for w in found), max(w.x1 for w in found))
    return out


def regrid(table: HTable, al: Alignment, lines: list[list[TWord]], used: set[int], wide: bool = False) -> bool:
    width = table.width()
    bands = _bands(table, al, width)
    if len(bands) < max(2, width - 2) or not al.row_line:
        return False
    top = min(al.row_line.values())
    changed = False
    for r, row in enumerate(table.rows):
        own = table.row_width(r)
        if not row.cells or own == width:
            continue
        if own > width and not (wide and any(c.plain_empty for c in row.cells)):
            continue
        if r in al.row_line:
            spans = {id(c): al.span(c) for c in row.cells if al.span(c)}
        elif all(c.tag == "th" for c in row.cells):
            spans = _header_spans(row.cells, lines, top, used)
        else:
            continue
        target = {}
        for cell in row.cells:
            if id(cell) in spans:
                col = column_of(*spans[id(cell)], cell.colspan, bands, width)
                if col is not None:
                    target[id(cell)] = col
        if own < width:
            cells = _insert_empties(row.cells, target, table.spanned_into(r))
            if cells is None or own + len(cells) - len(row.cells) > width:
                continue
        else:  # a row with a stray empty cell: the empty cells are laid out again around the found cells
            if len(target) < 2 or all(c.col == target[id(c)] for c in row.cells if id(c) in target):
                continue
            cells = _relayout(row.cells, target, table.spanned_into(r), width)
            if cells is None:
                continue
        row.cells, row.edited = cells, True
        table.place()
        changed = True
    return changed


_NUMBER = re.compile(r"[\s$€£¥(+\-–]*[0-9][0-9.,]*%?\)?")


def _header_rows(table: HTable) -> int:
    n = 0
    while n < len(table.rows) and table.rows[n].cells and all(c.tag == "th" for c in table.rows[n].cells):
        n += 1
    return n


def merge_side_by_side(left: _Found, right: _Found) -> str | None:
    """`left` and `right` as one table, when the reading wrote one table as two: every body row of `right` lies
    on the text line of a body row of `left`, to its right; `right` has no row labels (its first column holds
    numbers) and does not repeat `left`'s row-label header (a repeated column group is a separate table); both
    have the same number of header rows. Rows of `left` with nothing beside them get empty cells."""
    if not (left.healthy and right.healthy):
        return None
    t1, t2 = left.table, right.table
    h1, h2 = _header_rows(t1), _header_rows(t2)
    if h1 != h2 or h1 == 0 or len(t2.rows) - h2 < 3:
        return None
    corner = next((fold(c.text) for c in t1.rows[0].cells if c.col == 0 and c.text), "")
    heads = [[fold(c.text) for row in t.rows[:h] for c in row.cells] for t, h in ((t1, h1), (t2, h2))]
    if heads[0] == heads[1] or (corner and corner in heads[1]):
        return None  # the same headers again, row-label header included: a repeated column group
    firsts = [row.cells[0].text for row in t2.rows[h2:] if row.cells and row.cells[0].text]
    if not firsts or sum(1 for f in firsts if _NUMBER.fullmatch(f)) < 0.8 * len(firsts):
        return None
    left_rows = {li: r for r, li in left.al.row_line.items() if r >= h1}
    beside: dict[int, int] = {}  # left row -> right row on the same line
    for r in range(h2, len(t2.rows)):
        li = right.al.row_line.get(r)
        if li is None or li not in left_rows:
            return None
        r1 = left_rows[li]
        ends = [w.x1 for c in t1.rows[r1].cells for w in left.al.words.get(id(c), [])]
        starts = [w.x0 for c in t2.rows[r].cells for w in right.al.words.get(id(c), [])]
        if not ends or not starts or min(starts) <= max(ends):
            return None
        beside[r1] = r
    blank = "<td></td>" * t2.width()
    rows = []
    for r, row in enumerate(t1.rows):
        cells = "".join(c.render() for c in row.cells)
        if r < h1:
            cells += "".join(c.render() for c in t2.rows[r].cells)
        else:
            cells += "".join(c.render() for c in t2.rows[beside[r]].cells) if r in beside else blank
        rows.append(row.head + cells + row.tail)
    return t1.seps[0] + "".join(rows) + t1.seps[-1]


_ROW_END = re.compile(r"</tr\s*>", re.I)


def complete_truncated(raw: str, lines: list[list[TWord]], used: set[int], health: float) -> str | None:
    """A table the reading left unclosed (cut off by the token limit), completed from the text layer: its rows
    up to the last complete one, then one row per text line below the last row found on the page whose words
    each fall in a column of the table and fill at least half of its columns (two at least). The first line
    that does not fit ends the table. None when nothing is added."""
    ends = [m.end() for m in _ROW_END.finditer(raw)]
    if not ends:
        return None
    body = raw[: ends[-1]]
    table = parse_table(body + "</table>")
    if table is None:
        return None
    al = align_table(table, lines, used)
    for ws in al.words.values():
        used.update(w.index for w in ws)
    width = table.width()
    bands = _bands(table, al, width)
    if not al.cells or al.found < health or not al.row_line or len(bands) < max(2, width - 2):
        return None
    rows = []
    for line in lines[max(al.row_line.values()) + 1 :]:
        free = [w for w in line if w.index not in used]
        if not free:
            continue
        cols = _columns_of_line(free, bands)
        if cols is None or len(cols) < max(2, width / 2):
            break
        used.update(w.index for w in free)
        cells = (html.escape(" ".join(w.text for w in cols.get(c, [])), quote=False) for c in range(width))
        rows.append("<tr>" + "".join(f"<td>{text}</td>" for text in cells) + "</tr>")
    return body + "".join(rows) + "</table>" if rows else None


def _columns_of_line(line: list[TWord], bands: Bands, slack: float = 2.0) -> dict[int, list[TWord]] | None:
    """The line's words by the column band each overlaps most; None when a word overlaps no band."""
    lo = {c: min(a for a, _ in v) - slack for c, v in bands.items()}
    hi = {c: max(b for _, b in v) + slack for c, v in bands.items()}
    out: dict[int, list[TWord]] = {}
    for w in line:
        best, overlap = None, 0.0
        for c in sorted(bands):
            o = min(w.x1, hi[c]) - max(w.x0, lo[c])
            if o > overlap:
                best, overlap = c, o
        if best is None:
            return None
        out.setdefault(best, []).append(w)
    return out


def _relayout(cells: list[HCell], target: dict[int, int], taken: set[int], width: int) -> list[HCell] | None:
    """The row's empty cells dropped, then empty cells put back so each targeted cell starts on its column and
    the row is `width` columns wide; None when a cell would have to move left or the row stays too wide."""
    out: list[HCell] = []
    pos = 0
    tag = cells[0].tag
    for cell in (c for c in cells if not c.plain_empty):
        tag = cell.tag
        while pos in taken:
            pos += 1
        if id(cell) in target:
            if target[id(cell)] < pos:
                return None
            while pos < target[id(cell)]:
                if pos not in taken:
                    out.append(HCell.empty(cell.tag))
                pos += 1
        out.append(cell)
        pos += cell.colspan
    while True:
        while pos in taken:
            pos += 1
        if pos >= width:
            break
        out.append(HCell.empty(tag))
        pos += 1
    return out if pos <= width else None


def _insert_empties(cells: list[HCell], target: dict[int, int], taken: set[int]) -> list[HCell] | None:
    """The row with empty cells inserted so each targeted cell starts on its column; None when that is
    impossible (a cell would have to move left) or nothing needs to move."""
    out: list[HCell] = []
    pos, inserted = 0, 0
    for cell in cells:
        while pos in taken:
            pos += 1
        if id(cell) in target:
            if target[id(cell)] < pos:
                return None
            while pos < target[id(cell)]:
                if pos not in taken:
                    out.append(HCell.empty(cell.tag))
                    inserted += 1
                pos += 1
        out.append(cell)
        pos += cell.colspan
    return out if inserted else None
