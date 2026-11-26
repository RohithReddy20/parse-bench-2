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
    extend   a header row still narrower than its table whose last cell spans columns and ends the row: that cell
             spans to the table's last column. This one needs no text layer (it also runs on scans).
    health   share of a table's text cells that must be found word for word on the page before anything in the
             table is changed (default 0.5). A broken font map or a stale OCR layer finds few cells, and the
             model's table stays as it is.

Round two (docs/results/2026-11-25-tables-fix.md, round-two section):

    leaf     (kept) on a table no page geometry places (a scan, or a text layer that did not match), a last header
             row short of its table is moved right over the last columns: leaf headers name the value columns,
             and the reading left out the blank corner over the row labels. Not when a one-column header ends the
             row above (then the gap may be under it). `leaf_mode` 3 is the measured rule.
    dedupe   copies of a table that a looping read wrote again and again (only on a read `fallback.looped` flags).
    complete_extent  with complete, a cut-off table is completed from the words inside its own horizontal extent,
             so a table printed in column groups side by side is completed down its own group.
    upright  the text layer of a page printed sideways or upside down is turned to read upright before matching.
    absorb   a unit printed under a header that spans down ("(DKK 1,000)" under "Assets") joins that header.
    split    two numbers in one cell, on either side of a gutter running through the column, become two columns
             (tables_split.py).
    unspan   a body value spanning columns whose words lie in one of them moves to that column (`unspan_mode` 2:
             its centre lies inside that column).

A table the options do not change is written back byte for byte, and a page without a text layer is untouched by
every option but `extend`.
The corrected table is a new candidate of its region (reader "fix.tables.pdf"); OCRia-1's reading stays in the IR.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from docparse.core.ir import Candidate, Page, Provenance, Region
from docparse.fixes.tables_align import (
    Alignment,
    TWord,
    align_table,
    find_on_lines,
    fold,
    lines_of,
    page_words,
    turned,
    upright_turn,
)
from docparse.fixes.tables_html import HCell, HTable, parse_table
from docparse.fixes.tables_split import split_columns
from docparse.pdf.render import RenderedPage

READER = "fix.tables.pdf"
HEADER_LOOKUP = 8  # lines above a table's first matched row searched for header cells
EXTENT_SLACK = 6.0  # points beyond a table's outer columns where a completed row's words may still sit


@dataclass
class _Found:
    """One table region of the page, its table and where its cells sit on the page."""

    region: Region
    table: HTable
    al: Alignment
    healthy: bool


def fix_tables(page: Page, rendered: RenderedPage, options: dict) -> None:
    """Apply the enabled options to every HTML table region of the page, in reading order."""
    if not any(options.get(k) for k in ("regrid", "complete", "merge", "extend", "leaf", "dedupe")):
        return
    words = page_words(rendered)
    if words and options.get("upright"):
        angle = upright_turn(words, rendered.width_pt, rendered.height_pt)
        if angle:
            words = turned(words, rendered.width_pt, rendered.height_pt, angle)
    lines = lines_of(words) if words else []
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
            what = []
            healthy = False
            if lines:
                al = align_table(table, lines, used)
                for ws in al.words.values():
                    used.update(w.index for w in ws)
                healthy = bool(al.cells) and al.found >= health
                found.append(_Found(region, table, al, healthy))
                wide = bool(options.get("regrid_wide"))
                if healthy and options.get("regrid") and regrid(table, al, lines, used, wide):
                    what.append("regrid")
                if healthy and options.get("absorb") and absorb_header(table, al):
                    what.append("absorb")
                if healthy and options.get("split") and split_columns(table, al):
                    what.append("split")
                if healthy and options.get("unspan") and unspan(table, al, float(options.get("unspan_mode", 0))):
                    what.append("unspan")
            placed = bool(lines) and healthy
            if options.get("leaf") and not placed and right_align_leaf(table, float(options.get("leaf_mode", 0))):
                what.append("leaf")
            if options.get("extend") and extend_header(table):
                what.append("extend")
            if what:
                table.changed = True
                new[region.id] = (table.render(), "+".join(what))
        elif region.kind == "opaque" and lines and options.get("complete") and base.raw.lower().startswith("<table"):
            raw = complete_truncated(base.raw, lines, used, health, bool(options.get("complete_extent")))
            if raw is not None:
                new[region.id] = (raw, "complete")
    if options.get("dedupe"):
        for region_id in looped_copies(page):
            new[region_id] = ("", "dedupe")
    if options.get("merge"):
        for left, right in zip(found, found[1:], strict=False):
            if left.region.id in new and new[left.region.id][1] in ("merge", "dedupe"):
                continue  # already merged with the table before it, or a looped copy
            if right.region.id in new and new[right.region.id][1] == "dedupe":
                continue
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
    """For a header row no line was matched to: each cell with a written line that occurs exactly once on the
    lines just above the table body, with the x-span of the words of those lines."""
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
    """Rows narrower than the table (and with `wide`, wider ones holding empty cells) laid out so that each cell
    found on the page starts in the column its words lie in. True when any row changed."""
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
            cells = _insert_empties(
                row.cells,
                target,
                table.spanned_into(r),
                _leaf_header_spans(table) if r >= _header_rows(table) else None,
            )
            if cells is None or own + sum(c.colspan for c in cells) - sum(c.colspan for c in row.cells) > width:
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


_COLSPAN = re.compile(r"colspan\s*=\s*[\"']?\d+[\"']?", re.I)
_COLSPAN_ATTR = re.compile(r"\s*colspan\s*=\s*[\"']?\d+[\"']?", re.I)  # with the space before it


def right_align_leaf(table: HTable, mode: float = 0) -> bool:
    """With no page geometry to place it (a scan, or a text layer that did not match): the last header row of a
    multi-row header (`h >= 2`), one or more cells short of its table, is moved right so that its cells stand over
    the last columns. Leaf headers name the value columns, which follow the row-label columns; a short leaf row
    left out the blank corner. Never shifts a flat 1-row header (`h < 2`), where column 0 is usually already the
    row-label header.
    mode 0: only when every body row is as wide as the table; mode 2: whatever the body rows; mode 3 (the one
    kept): as mode 2, but not when a one-column header ends the row above. True when the row changed."""
    h = _header_rows(table)
    if h < 2 or h == len(table.rows):
        return False
    width = table.width()
    r = h - 1
    row = table.rows[r]
    own = table.row_width(r)
    if own >= width or not row.cells or any(c.colspan > 1 for c in row.cells):
        return False
    if mode < 2 and any(
        table.row_width(k) != width
        for k in range(h, len(table.rows))
        if table.rows[k].cells and len(table.rows[k].cells) > 1
    ):
        return False
    if mode == 3 and r > 0:
        last = max(table.rows[r - 1].cells, key=lambda c: c.col, default=None)
        if last is not None and last.colspan == 1 and last.rowspan == 1:
            return False  # a one-column header ends the row above: the leaf row's gap is under it, at the end
    missing = width - own
    taken = table.spanned_into(r)
    free = [k for k in range(width) if k not in taken]
    if len(free) - missing != len(row.cells):
        return False
    first = free[missing]  # where the row's first cell goes
    tag = row.cells[0].tag
    row.cells = [HCell.empty(tag) for k in free if k < first] + row.cells
    row.edited = True
    table.place()
    return True


def absorb_header(table: HTable, al: Alignment) -> bool:
    """A header row wider than its table whose extra cells lie, on the page, under a header that spans down into
    the row from above (a unit "(DKK 1,000)" printed under "Assets"): those cells' text joins the spanning header
    and the cells go, when that makes the row fit. True when any row changed."""
    width = table.width()
    bands = _bands(table, al, width)
    if len(bands) < max(2, width - 2):
        return False
    changed = False
    for r, row in enumerate(table.rows):
        if table.row_width(r) <= width or not row.cells or not all(c.tag == "th" for c in row.cells):
            continue
        above = {}
        for prev in table.rows[:r]:
            for cell in prev.cells:
                if cell.row < r < cell.row + cell.rowspan and cell.tag == "th":
                    for k in range(cell.col, cell.col + cell.colspan):
                        above[k] = cell
        moves = []
        for cell in row.cells:
            span = al.span(cell)
            col = column_of(*span, 1, bands, width) if span and cell.colspan == 1 and cell.text else None
            if col is not None and col in above:
                moves.append((cell, above[col]))
        extra = sum(c.colspan for c, _ in moves)
        if not moves or table.row_width(r) - extra != width:
            continue
        for cell, host in moves:
            host.inner = f"{host.inner} {cell.inner}".strip()
            host.raw = ""
        row.cells = [c for c in row.cells if all(c is not m for m, _ in moves)]
        row.edited = True
        table.place()
        changed = True
    return changed


LOOP_COPIES = 3  # a table written this many times over on one page is a looped read


def _cells_key(raw: str) -> str:
    """A table's text, cell by cell (markup, bold style and spacing aside)."""
    cells = re.findall(r"<t[hd]\b[^>]*>(.*?)(?:</t[hd]\s*>|$)", raw, re.S | re.I)
    texts = (fold(html.unescape(re.sub(r"<[^>]+>", " ", c)).replace("*", "")) for c in cells)
    return "|".join(texts)


def looped_copies(page: Page) -> list[str]:
    """Region ids of the copies of a table the read wrote again and again (a decoding loop): every table region
    after the first whose cells say the same as a table written at least LOOP_COPIES times on the page, and a last
    cut-off copy (an unclosed table whose cells begin the same way). Only on a page whose read repeats itself
    (`fallback.looped`): a form printing the same blank table several times is not a loop."""
    from docparse.fixes.fallback import looped
    from docparse.write.markdown import write_page

    if not looped(write_page(page)):
        return []
    keys = {}
    for region in page.regions:
        raw = region.chosen().raw
        if region.kind in ("table", "opaque") and raw.lstrip().lower().startswith("<table"):
            keys[region.id] = (_cells_key(raw), region.kind == "table")
    counts: dict[str, int] = {}
    for key, closed in keys.values():
        if closed and key:
            counts[key] = counts.get(key, 0) + 1
    looped = {k for k, n in counts.items() if n >= LOOP_COPIES}
    if not looped:
        return []
    out, seen = [], set()
    for region_id, (key, closed) in keys.items():
        if closed and key in looped:
            if key in seen:
                out.append(region_id)
            seen.add(key)
        elif not closed and key and any(k.startswith(key) for k in looped):
            out.append(region_id)
    return out


def unspan(table: HTable, al: Alignment, mode: float = 0) -> bool:
    """A body cell spanning columns whose words on the page lie in one of those columns only becomes a one-column
    cell in that column, with empty cells in the others. True when any cell changed."""
    width = table.width()
    bands = _bands(table, al, width)
    if len(bands) < max(2, width - 2):
        return False
    lo = {c: min(a for a, _ in v) for c, v in bands.items()}
    hi = {c: max(b for _, b in v) for c, v in bands.items()}
    changed = False
    for r, row in enumerate(table.rows):
        if len(row.cells) < 2 or r not in al.row_line or all(c.tag == "th" for c in row.cells):
            continue
        out: list[HCell] = []
        for cell in row.cells:
            span = al.span(cell)
            cols = [k for k in range(cell.col, cell.col + cell.colspan) if k in bands]
            if cell.colspan < 2 or cell.rowspan != 1 or not span or cell.colspan >= width or len(cols) < cell.colspan:
                out.append(cell)
                continue
            x0, x1 = span
            overlaps = {k: min(x1, hi[k]) - max(x0, lo[k]) for k in cols}
            hits = [k for k, o in overlaps.items() if o > 0]
            if mode in (0, 2) and len(hits) != 1:
                out.append(cell)
                continue
            if mode == 1 and not hits:
                out.append(cell)
                continue
            mid = (x0 + x1) / 2
            if mode == 2 and not any(lo[k] <= mid <= hi[k] for k in hits):
                out.append(cell)
                continue
            target = max(hits, key=lambda k: overlaps[k])
            cell.attrs = _COLSPAN_ATTR.sub("", cell.attrs)
            cell.colspan, cell.raw = 1, ""
            out += [HCell.empty(cell.tag) for _ in range(cell.col, target)]
            out.append(cell)
            out += [HCell.empty(cell.tag) for _ in range(target + 1, cell.col + len(cols))]
            changed = True
        if len(out) != len(row.cells):
            row.cells, row.edited = out, True
            table.place()
    return changed


def extend_header(table: HTable) -> bool:
    """A header row narrower than its table whose last cell is a group header (it spans columns and nothing
    follows it) gets that cell spanned to the table's last column: the label stands over the columns the reading
    left without a header. True when any row changed."""
    width, changed = table.width(), False
    for r, row in enumerate(table.rows):
        own = table.row_width(r)
        if not row.cells or own >= width or not all(c.tag == "th" for c in row.cells):
            continue
        last = row.cells[-1]
        if last.colspan < 2 or not last.text or last.col + last.colspan != own:
            continue
        span = last.colspan + width - own
        last.attrs = _COLSPAN.sub(f'colspan="{span}"', last.attrs)
        last.colspan, last.raw = span, ""
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
    numbers) and does not repeat `left`'s row-label header (a repeated column group is a separate table); `left`
    has at least as many header rows (its extra top rows, a title, get a blank cell over `right`). Rows of `left`
    with nothing beside them get empty cells."""
    if not (left.healthy and right.healthy):
        return None
    t1, t2 = left.table, right.table
    h1, h2 = _header_rows(t1), _header_rows(t2)
    if h2 == 0 or h1 < h2 or len(t2.rows) - h2 < 3:
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
    w2 = t2.width()
    blank, over = "<td></td>" * w2, f'<th colspan="{w2}"></th>' if w2 > 1 else "<th></th>"
    rows = []
    for r, row in enumerate(t1.rows):
        cells = "".join(c.render() for c in row.cells)
        if r < h1 - h2:  # a header row of `left` above all of `right`'s (a title row): blank over `right`
            cells += over
        elif r < h1:
            cells += "".join(c.render() for c in t2.rows[r - h1 + h2].cells)
        else:
            cells += "".join(c.render() for c in t2.rows[beside[r]].cells) if r in beside else blank
        rows.append(row.head + cells + row.tail)
    return t1.seps[0] + "".join(rows) + t1.seps[-1]


_ROW_END = re.compile(r"</tr\s*>", re.I)


def complete_truncated(
    raw: str, lines: list[list[TWord]], used: set[int], health: float, extent: bool = False
) -> str | None:
    """A table the reading left unclosed (cut off by the token limit), completed from the text layer: its rows
    up to the last complete one, then one row per text line below the last row found on the page whose words
    each fall in a column of the table and fill at least half of its columns (two at least). The first line
    that does not fit ends the table. None when nothing is added.
    With `extent`, only the words of a line within the table's horizontal extent count: a table printed in
    column groups side by side shares its lines with the groups beside it."""
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
        if extent:
            left = min(a for v in bands.values() for a, _ in v) - EXTENT_SLACK
            right = max(b for v in bands.values() for _, b in v) + EXTENT_SLACK
            free = [w for w in free if w.x1 > left and w.x0 < right]
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


def _leaf_header_spans(table: HTable) -> dict[int, tuple[int, int]]:
    """For each column covered by a multi-column header cell (`colspan >= 2`) in the table's bottom-most header
    row covering that column, map column -> `(start_col, end_col_exclusive)`."""
    h = _header_rows(table)
    if h == 0:
        return {}
    spans: dict[int, tuple[int, int]] = {}
    for r in range(h):
        for cell in table.rows[r].cells:
            if cell.colspan >= 2:
                rng = (cell.col, cell.col + cell.colspan)
                for k in range(cell.col, cell.col + cell.colspan):
                    spans[k] = rng
            else:
                spans.pop(cell.col, None)
    return spans


def _widen_cell(cell: HCell, delta: int) -> HCell:
    """Return a copy of `cell` with its colspan widened by `delta`."""
    new_span = cell.colspan + delta
    if _COLSPAN.search(cell.attrs):
        attrs = _COLSPAN.sub(f'colspan="{new_span}"', cell.attrs)
    else:
        attrs = f'{cell.attrs} colspan="{new_span}"'
    return HCell(
        tag=cell.tag,
        attrs=attrs,
        inner=cell.inner,
        raw="",
        rowspan=cell.rowspan,
        colspan=new_span,
        row=cell.row,
        col=cell.col,
    )


def _insert_empties(
    cells: list[HCell],
    target: dict[int, int],
    taken: set[int],
    header_spans: dict[int, tuple[int, int]] | None = None,
) -> list[HCell] | None:
    """The row with empty cells inserted (or a preceding cell's colspan widened when the header spans both columns)
    so each targeted cell starts on its column; None when a cell would have to move left or nothing needs to move."""
    out: list[HCell] = []
    pos, added = 0, 0
    prev_start: int | None = None
    for cell in cells:
        while pos in taken:
            pos += 1
        if id(cell) in target:
            if target[id(cell)] < pos:
                return None
            if pos < target[id(cell)]:
                gap_end = target[id(cell)]
                # If the preceding cell sits under a multi-column header spanning [prev_start .. gap_end],
                # widen that cell's colspan instead of inserting a phantom empty <td></td> inside the column.
                if (
                    header_spans
                    and out
                    and prev_start is not None
                    and not out[-1].plain_empty
                    and out[-1].rowspan == 1
                    and prev_start in header_spans
                    and header_spans[prev_start][0] <= prev_start
                    and header_spans[prev_start][1] >= gap_end
                    and not any(k in taken for k in range(pos, gap_end))
                ):
                    delta = gap_end - pos
                    out[-1] = _widen_cell(out[-1], delta)
                    added += delta
                    pos = gap_end
                else:
                    while pos < gap_end:
                        if pos not in taken:
                            out.append(HCell.empty(cell.tag))
                            added += 1
                        pos += 1
        prev_start = pos
        out.append(cell)
        pos += cell.colspan
    return out if added else None
