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

Round three (the Chandra readings: cells and columns the page does not print):

    squeeze  a row wider than its table sheds the cells the page does not print: a value written once more than
             its line shows it, cells past the table's last column whose text is not on the row's line, a group
             header spanning one column too many. When the found cells of two grid columns lie in one printed
             column (every row holds one cell too many), the two become one column, in every row or not at all.
    sign     a body column that holds nothing but one currency sign ("R", "$", "Rs"), next to its amounts, for
             which no header row has a cell: sign and amount become one cell ("R 52,333.13").
    head_fill  a header row matched to a text line and still short of its table gets the cells of the columns
             after its last one from the free words of that line that lie in those columns.
    last_column  a body one column short of its header, whose rows' lines print one more value after the row's
             last cell, gets that column from the text layer (and a row that ends in it already, the value it
             left out before).
    merge_apart  with merge, also two tables with other tables written between them, when the right one stands
             beside at least APART_COVER of the left one's body rows.
    merge_strict  with merge, two tables that each start with a title of their own are not merged.

Words taken from the text layer into a table (head_fill, last_column) must show ink on the page image.

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
    main = ("regrid", "complete", "merge", "extend", "leaf", "dedupe", "squeeze", "sign", "head_fill", "last_column")
    if not any(options.get(k) for k in main):
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
    # round three, once every table has its words: a word no table claimed is then a word outside the tables
    for f in found if any(options.get(k) for k in ("squeeze", "head_fill", "last_column", "sign")) else ():
        if not f.healthy:
            continue
        what = [new[f.region.id][1]] if f.region.id in new else []
        if options.get("squeeze") and squeeze(f.table, f.al, lines, used):
            what.append("squeeze")
        if options.get("head_fill") and fill_header(f.table, f.al, lines, used, rendered):
            what.append("head_fill")
        if options.get("last_column") and add_last_column(f.table, f.al, lines, used, rendered):
            what.append("last_column")
        if options.get("sign") and join_signs(f.table, f.al):
            what.append("sign")
        if len(what) > (f.region.id in new):
            f.table.changed = True
            new[f.region.id] = (f.table.render(), "+".join(what))
    if options.get("dedupe"):
        for region_id in looped_copies(page):
            new[region_id] = ("", "dedupe")
    if options.get("merge"):
        joined: dict[str, _Found | None] = {}  # left region id -> the merged table, placed on the page again
        strict = bool(options.get("merge_strict"))
        for left, right in zip(found, found[1:], strict=False):
            if left.region.id in new and new[left.region.id][1] in ("merge", "dedupe"):
                continue  # already merged with the table before it, or a looped copy
            if right.region.id in new and new[right.region.id][1] == "dedupe":
                continue
            raw = merge_side_by_side(left, right, strict=strict)
            if raw is not None:
                new[left.region.id] = (raw, "merge")
                new[right.region.id] = ("", "merge")
                joined[left.region.id] = _joined(left, right, raw, lines, used)
        if options.get("merge_apart"):  # the same, for two tables the reading wrote with other blocks between
            gone = {k for k, (raw, what) in new.items() if what == "dedupe" or (what == "merge" and not raw)}
            gone |= {k for k, f in joined.items() if f is None}
            tabs = [joined.get(f.region.id) or f for f in found if f.region.id not in gone]
            done: set[str] = set()
            for i, left in enumerate(tabs):
                for right in tabs[i + 1 :]:
                    if left.region.id in done or right.region.id in done:
                        continue
                    raw = merge_side_by_side(left, right, APART_COVER, strict)
                    if raw is not None:
                        new[left.region.id] = (raw, "merge_apart")
                        new[right.region.id] = ("", "merge_apart")
                        done.update((left.region.id, right.region.id))
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


def _mostly_inside(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    """Most x-spans of `a` lie (by more than half their width) inside the extent of `b`."""
    lo, hi = min(x for x, _ in b), max(y for _, y in b)
    return sum(1 for x, y in a if min(y, hi) - max(x, lo) > 0.5 * (y - x)) >= 0.6 * len(a)


def _same_column(bands: Bands, width: int) -> list[int]:
    """Columns k whose found cells lie in the x-range of column k+1's and the other way round: one printed column
    the reading wrote as two (in a row, only one of the two cells is on the page)."""
    out: list[int] = []
    for k in range(width - 1):
        a, b = bands.get(k), bands.get(k + 1)
        if a and b and (not out or out[-1] != k - 1) and _mostly_inside(a, b) and _mostly_inside(b, a):
            out.append(k)
    return out


def _shed(
    cells: list[HCell], idx: list[int], excess: int, al: Alignment, bare: bool, narrow: bool
) -> list[HCell] | None:
    """`cells[idx]` (the cells between two cells found in their columns) made `excess` columns narrower, or None
    when the page does not say how. `bare`: the row's line prints nothing between those two cells, so a cell
    with text among `cells[idx]` that was not found is not on the page. With no column left for them, they all go
    if each is empty or not on the page. Otherwise exactly one of: the cells not on the page that say the same
    as a neighbour (a value written once too often), when any `excess` of them are the same; and with `narrow`
    (the table is known to be narrower than written; without it, stray empty cells are `regrid_wide`'s to move)
    the empty cells, when there are as many as columns too many (or nothing but empty cells), or the one
    spanning cell among them, which then spans less."""
    keys = [fold(c.text) for c in cells]
    plain = [j for j in idx if cells[j].colspan == 1 and cells[j].rowspan == 1]
    empty = [j for j in plain if not keys[j]]
    absent = [j for j in plain if keys[j] and id(cells[j]) not in al.words] if bare else []
    if excess == sum(cells[j].colspan for j in idx):
        return [] if len(empty) + len(absent) == len(idx) and (narrow or absent) else None
    spans = [j for j in idx if cells[j].colspan > 1]
    twice = [
        j for j in absent
        if (j > 0 and keys[j - 1] == keys[j]) or (j + 1 < len(cells) and keys[j + 1] == keys[j])
    ]  # fmt: skip
    ways: list[tuple[set[int], int | None]] = []  # (cells to drop, cell to narrow)
    if narrow and len(spans) == 1 and cells[spans[0]].colspan > excess and cells[spans[0]].rowspan == 1:
        ways.append((set(), spans[0]))
    if narrow and empty and (len(empty) == excess or (len(empty) > excess and len(empty) == len(idx))):
        ways.append((set(empty[:excess]), None))
    if twice and (len(twice) == excess or (len(twice) > excess and len({keys[j] for j in twice}) == 1)):
        ways.append((set(twice[:excess]), None))
    if len(ways) != 1:
        return None
    drop, slim = ways[0]
    return [_widen_cell(cells[j], -excess) if j == slim else cells[j] for j in idx if j not in drop]


def _squeezed(
    table: HTable, r: int, al: Alignment, lines: list[list[TWord]], used: set[int], bands: Bands, width: int,
    narrow: bool,
) -> list[HCell] | None:  # fmt: skip
    """Row r, wider than `width`, with the cells the page does not print taken out so that every one-column cell
    found on the page stands in the column its words lie in; None when that cannot be told. "Does not print":
    between the two found cells around it, the row's line and the lines next to it (a cell may wrap) hold no word
    that no table cell was matched to."""
    row, taken = table.rows[r], table.spanned_into(r)
    li = al.row_line.get(r)
    near = None  # the words around the row that no table cell was matched to
    if li is not None:
        near = [w for line in lines[max(0, li - 1) : li + 2] for w in line if w.index not in used]
    out: list[HCell] = []
    pending: list[int] = []
    pos, left = 0, float("-inf")
    for i, cell in enumerate([*row.cells, None]):
        if cell is None:
            target, right = width, float("inf")
        else:
            span = al.span(cell) if cell.colspan == 1 else None
            target = column_of(*span, 1, bands, width) if span else None
            if target is None:
                pending.append(i)
                continue
            right = span[0]
        if target < pos or target in taken:
            return None
        free = sum(1 for k in range(pos, target) if k not in taken)
        need = sum(row.cells[j].colspan for j in pending)
        if need < free:
            return None  # a gap as well: not a row with cells too many only
        if need > free:
            bare = near is not None and not any(left < (w.x0 + w.x1) / 2 < right for w in near)
            kept = _shed(row.cells, pending, need - free, al, bare, narrow)
            if kept is None:
                return None
        else:
            kept = [row.cells[j] for j in pending]
        out += kept
        pending = []
        if cell is not None:
            out.append(cell)
            pos, left = target + 1, span[1]
    return out


def squeeze(table: HTable, al: Alignment, lines: list[list[TWord]], used: set[int]) -> bool:
    """Rows wider than the table lose the cells the page does not print (see `_shed`). Where the found cells of
    two neighbouring grid columns lie in one printed column, the table is one column narrower than the reading
    wrote it: then every row must be brought to that width, or the table stays as it is. True when rows changed."""
    width = table.width()
    bands = _bands(table, al, width)
    if len(bands) < max(2, width - 2) or not al.row_line:
        return False
    same = _same_column(bands, width)
    goal = width - len(same)
    merged: Bands = {}
    for c, spans in bands.items():
        merged.setdefault(c - sum(1 for k in same if k < c), []).extend(spans)
    saved = [(row, row.cells, row.edited) for row in table.rows]
    changed = False
    for r, row in enumerate(table.rows):
        if not row.cells or table.row_width(r) <= goal:
            continue
        cells = _squeezed(table, r, al, lines, used, merged, goal, bool(same))
        if cells is None:
            if same:
                break
            continue
        row.cells, row.edited = cells, True
        table.place()
        changed = True
    wide = any(table.row_width(r) > goal for r in range(len(table.rows)))
    if same and (not changed or wide or table.width() != goal):
        for row, cells, edited in saved:
            row.cells, row.edited = cells, edited
        table.place()
        return False
    return changed


_SIGN = re.compile(r"[$€£¥₹₩₽₺₫₪฿]|[A-Z]{1,3}\$?|Rs\.?|kr\.?")  # a currency sign or code standing alone
_AMOUNT = re.compile(r"\(?[+\-–—]?\s*[0-9][0-9.,\s]*\)?|[\-–—]+|n/?a\.?|nil", re.I)
_DASH = re.compile(r"[\-–—]+")
MIN_SIGNS = 3  # body cells that must hold the sign


def _sign_columns(table: HTable, al: Alignment, width: int, head: int) -> list[int]:
    """Columns of the body holding one and the same currency sign and nothing else (a dash aside), each found on
    the page as a word of its own, with amounts in the column to their right."""
    cols: dict[int, list[HCell]] = {}
    for row in table.rows[head:]:
        for cell in row.cells:
            if cell.colspan == 1 and table.row_width(cell.row) == width:
                cols.setdefault(cell.col, []).append(cell)
    out: list[int] = []
    for k in range(width - 1):
        filled = [c for c in cols.get(k, []) if c.text and not _DASH.fullmatch(c.text)]  # a dash: no amount
        signs = [c for c in filled if c.text == filled[0].text]
        if len(signs) < MIN_SIGNS or len(signs) != len(filled) or not _SIGN.fullmatch(signs[0].text):
            continue
        if sum(1 for c in signs if id(c) in al.words) < 0.8 * len(signs) or (out and out[-1] == k - 1):
            continue
        beside = {c.row: c for c in cols.get(k + 1, [])}
        amounts = [beside[c.row].text for c in signs if c.row in beside and beside[c.row].text]
        if len(amounts) >= 0.8 * len(signs) and sum(1 for a in amounts if _AMOUNT.fullmatch(a)) >= 0.9 * len(amounts):
            out.append(k)
    return out


def join_signs(table: HTable, al: Alignment) -> bool:
    """Currency signs the reading wrote as columns of their own join the amounts to their right, when the header
    has no cell for them: every header row is as many columns short of the body as there are sign columns, or
    covers sign and amount with one spanning cell. All sign columns of the table, or none. True when it changed."""
    width, head = table.width(), _header_rows(table)
    if head == 0 or head == len(table.rows):
        return False
    signs = _sign_columns(table, al, width, head)
    goal = width - len(signs)
    if not signs:
        return False
    rows: list[list[HCell] | None] = []
    for r, row in enumerate(table.rows):
        own = table.row_width(r)
        if own == goal and r < head:
            rows.append(None)  # a header row without cells for the signs
            continue
        if own != width:
            return False
        cells: list[HCell] = []
        carry: tuple[int, str] | None = None  # the sign waiting for its amount: (its column, its text)
        for cell in row.cells:
            inside = [k for k in signs if cell.col <= k < cell.col + cell.colspan]
            if cell.colspan == 1 and inside:
                if cell.rowspan != 1 or carry or (r < head and cell.text):
                    return False  # a header of its own over the signs: a column of its own
                carry = (cell.col, cell.inner.strip())
                continue
            if carry and (cell.col != carry[0] + 1 or cell.colspan != 1 or cell.rowspan != 1):
                return False
            if cell.colspan > 1 and any(not cell.col <= k + 1 < cell.col + cell.colspan for k in inside):
                return False  # a cell ending on a sign column: sign and amount do not stand under one header
            if inside:
                cell = _widen_cell(cell, -len(inside))
            elif carry and carry[1]:
                inner = f"{carry[1]} {cell.inner.strip()}".strip()
                cell = HCell(cell.tag, cell.attrs, inner, "", cell.rowspan, cell.colspan)
            carry = None
            cells.append(cell)
        if carry:
            return False
        rows.append(cells)
    for row, cells in zip(table.rows, rows, strict=True):
        if cells is not None:
            row.cells, row.edited = cells, True
    table.place()
    return True


INK_SHARE = 0.02  # share of a word box that must differ from its background for the word to count as printed


def printed(words: list[TWord], rendered: RenderedPage | None) -> bool:
    """Every word shows ink on the page image: a text layer also holds words nobody sees (white on white, text
    under a picture), and those are not put into a table. True without an image to look at."""
    image = getattr(rendered, "image", None)
    if image is None or not hasattr(image, "crop"):
        return True
    s = rendered.scale
    for w in words:
        box = (int(w.x0 / s), int(w.y0 / s), int(w.x1 / s) + 1, int(w.y1 / s) + 1)
        grey = image.crop(box).convert("L")
        counts = grey.histogram()
        ground = max(range(256), key=counts.__getitem__)
        ink = sum(n for v, n in enumerate(counts) if abs(v - ground) > 60)
        if ink < INK_SHARE * max(1, sum(counts)):
            return False
    return True


def fill_header(
    table: HTable, al: Alignment, lines: list[list[TWord]], used: set[int], rendered: RenderedPage | None = None
) -> bool:
    """A header row short of its table, every cell of which was found on one text line: the columns after its
    last cell get header cells holding the free words of that line that lie in them, when every such column has
    words, the words show ink, and no header cell the page did not show already says the same (the reading then
    wrote them in another row). Words under a header spanning down from above are that header's. True when it
    changed."""
    width, head = table.width(), _header_rows(table)
    bands = _bands(table, al, width)
    if len(bands) < max(2, width - 2):
        return False
    lost = {fold(c.text) for row in table.rows[:head] for c in row.cells if c.text and id(c) not in al.words}
    changed = False
    for r in range(head):
        row = table.rows[r]
        own = table.row_width(r)
        if own >= width or r not in al.row_line or sum(1 for c in row.cells if id(c) in al.words) < 2:
            continue
        if any(c.text and id(c) not in al.words for c in row.cells):
            continue  # the line's free words may be those of a cell that was not found
        taken = table.spanned_into(r)
        last = max(c.col + c.colspan for c in row.cells)
        missing = [k for k in range(last, width) if k not in taken]
        if len(missing) != width - own or any(k not in bands for k in missing):
            continue
        edge = max(w.x1 for c in row.cells for w in al.words.get(id(c), []))
        free = [w for w in lines[al.row_line[r]] if w.index not in used and w.x0 > edge]
        cols = _columns_of_line(free, bands) if free else None
        if cols is None or any(k not in cols for k in missing) or any(k not in taken for k in set(cols) - set(missing)):
            continue
        texts = {k: " ".join(w.text for w in cols[k]) for k in missing}
        if any(fold(t) in lost for t in texts.values()) or not printed([w for k in missing for w in cols[k]], rendered):
            continue
        for k in missing:
            row.cells.append(HCell("th", "", html.escape(texts[k], quote=False), ""))
            used.update(w.index for w in cols[k])
        row.edited = True
        table.place()
        changed = True
    return changed


TAIL_GAP = 8.0  # points: wider gaps between the words after a row's last cell are column gaps
TAIL_ROWS = 0.8  # share of the body rows that must show the column


def _aligned(xs: list[float], slack: float = 6.0) -> bool:
    """The middle 80% of `xs` lie within `slack` points."""
    xs = sorted(xs)
    k = len(xs) // 10
    return xs[len(xs) - 1 - k] - xs[k] <= slack


def add_last_column(
    table: HTable, al: Alignment, lines: list[list[TWord]], used: set[int], rendered: RenderedPage | None = None
) -> bool:
    """The header is one column wider than the body, and the lines of the body rows print one more value after
    the row's last cell: the body gets that column from the text layer. The words after the last cell must be
    one group per row, the groups one above the other (aligned left, right or centre: a column, not notes beside
    single rows), no farther from the last column than two column widths, on at least TAIL_ROWS of the body rows.
    True when it changed."""
    width, head = table.width(), _header_rows(table)
    if head == 0 or max(table.row_width(r) for r in range(head)) != width + 1:
        return False
    body = [r for r in range(head, len(table.rows)) if table.rows[r].cells]
    rows = [r for r in body if table.row_width(r) == width and r in al.row_line]
    bands = _bands(table, al, width)
    if len(rows) < max(3, TAIL_ROWS * len(body)) or len(bands) < max(2, width - 2):
        return False
    if any(table.row_width(r) != width and table.rows[r].cells[0].colspan != width for r in body):
        return False  # a body row of another width that is not one cell across the table
    tails: dict[int, list[TWord]] = {}
    edges: list[float] = []  # where the rows that print one more value end as the reading wrote them
    for r in rows:
        ws = [w for c in table.rows[r].cells for w in al.words.get(id(c), [])]
        if not ws:
            continue
        edge = max(w.x1 for w in ws)
        free = [w for w in lines[al.row_line[r]] if w.index not in used and w.x0 > edge]
        if any(b.x0 - a.x1 > TAIL_GAP for a, b in zip(free, free[1:], strict=False)):
            return False  # more than one group of words after the table: not one more column
        if free:
            tails[r] = free
            edges.append(edge)
    if len(tails) < TAIL_ROWS * len(rows):
        return False
    spans = [(t[0].x0, t[-1].x1) for t in tails.values()]
    if not any(_aligned([f(a, b) for a, b in spans]) for f in (min, max, lambda a, b: (a + b) / 2)):
        return False  # the groups do not stand one above the other (left, right or centre)
    centres = sorted((min(a for a, _ in v) + max(b for _, b in v)) / 2 for v in bands.values())
    pitch = sorted(b - a for a, b in zip(centres, centres[1:], strict=False))[len(centres) // 2 - 1]
    if sorted(a for a, _ in spans)[len(spans) // 2] - sorted(edges)[len(edges) // 2] > 2 * pitch:
        return False  # not beside the last column
    if not printed([w for t in tails.values() for w in t], rendered):
        return False
    grid: Bands = {width: spans}  # the columns as the rows that print the value show them
    for r in tails:
        for cell in table.rows[r].cells:
            if cell.colspan == 1 and al.span(cell):
                grid.setdefault(cell.col, []).append(al.span(cell))
    for r in body:
        row = table.rows[r]
        if table.row_width(r) != width:
            row.cells = [_widen_cell(row.cells[0], 1), *row.cells[1:]]
        elif r in tails:
            text = html.escape(" ".join(w.text for w in tails[r]), quote=False)
            row.cells = [*row.cells, HCell("td", "", text, "")]
            used.update(w.index for w in tails[r])
        else:
            gapped = _with_gap(table, r, al, lines, used, grid, width + 1, rendered)
            row.cells = gapped or [*row.cells, HCell.empty("td")]
        row.edited = True
    table.place()
    return True


def _with_gap(
    table: HTable, r: int, al: Alignment, lines: list[list[TWord]], used: set[int], bands: Bands, width: int,
    rendered: RenderedPage | None,
) -> list[HCell] | None:  # fmt: skip
    """Row r, one cell short of `width` and ending in the last column already: the row with a cell put in where
    its found cells leave a column out, holding the free words its line prints in that column (if any, and
    if they show ink). None when the found cells do not say where the gap is."""
    row = table.rows[r]
    if r not in al.row_line or table.spanned_into(r):
        return None
    target = {}
    for cell in row.cells:
        span = al.span(cell) if cell.colspan == 1 else None
        col = column_of(*span, 1, bands, width) if span else None
        if col is not None:
            target[id(cell)] = col
    cells = _insert_empties(row.cells, target, table.spanned_into(r))
    if cells is None or sum(c.colspan for c in cells) != width:
        return None
    ends = [w.x1 for c in row.cells for w in al.words.get(id(c), [])]
    free = [w for w in lines[al.row_line[r]] if w.index not in used and w.x0 < max(ends)]
    cols = _columns_of_line(free, bands) if free else None
    pos = 0
    for i, cell in enumerate(cells):  # no cell spans down into the row: pos is the column
        if all(cell is not c for c in row.cells) and cols and pos in cols and printed(cols[pos], rendered):
            cells[i] = HCell("td", "", html.escape(" ".join(w.text for w in cols[pos]), quote=False), "")
            used.update(w.index for w in cols[pos])
        pos += cell.colspan
    return cells


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


APART_COVER = 0.8  # share of the left table's body rows a table written apart from it must stand beside


def _joined(left: _Found, right: _Found, raw: str, lines: list[list[TWord]], used: set[int]) -> _Found | None:
    """The table merged from `left` and `right`, found on the page again with the words of the two."""
    table = parse_table(raw)
    if table is None:
        return None
    mine = {w.index for f in (left, right) for ws in f.al.words.values() for w in ws}
    return _Found(left.region, table, align_table(table, lines, used - mine), True)


def merge_side_by_side(left: _Found, right: _Found, cover: float = 0.0, strict: bool = False) -> str | None:
    """`left` and `right` as one table, when the reading wrote one table as two: every body row of `right` lies
    on the text line of a body row of `left`, to its right; `right` has no row labels (its first column holds
    numbers) and does not repeat `left`'s row-label header (a repeated column group is a separate table); `left`
    has at least as many header rows (its extra top rows, a title, get a blank cell over `right`). Rows of `left`
    with nothing beside them get empty cells. With `cover`, `right` must stand beside that share of `left`'s body
    rows: a short table next to a long one is a table of its own. With `strict`, so are two tables that each
    start with a title of their own (a first header row of one cell across all the table's columns)."""
    if not (left.healthy and right.healthy):
        return None
    t1, t2 = left.table, right.table
    h1, h2 = _header_rows(t1), _header_rows(t2)
    if h2 == 0 or h1 < h2 or len(t2.rows) - h2 < 3:
        return None
    if strict and all(
        t.width() > 1 and len(t.rows[0].cells) == 1 and t.rows[0].cells[0].colspan >= t.width() for t in (t1, t2)
    ):
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
    if len(beside) < cover * (len(t1.rows) - h1):
        return None
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
    """Return a copy of `cell` with its colspan widened by `delta` (narrowed by a negative one)."""
    new_span = cell.colspan + delta
    if new_span == 1:
        attrs = _COLSPAN_ATTR.sub("", cell.attrs)
    elif _COLSPAN.search(cell.attrs):
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
