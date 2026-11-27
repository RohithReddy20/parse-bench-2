"""Choosing between independent readings of one page.

Two or three models read every page (Chandra-OCR-2, OCRia-1, a general VLM). They fail on different pages, so the
page keeps the reading the evidence supports best:

- on a born-digital page (a healthy, visible PDF text layer) the evidence is the text layer itself: how much of it a
  reading holds (recall), how much of the reading it confirms (precision), the same for its numbers alone, whether
  neighbouring words stay neighbours, and how well the reading's tables line up with the layer's word grid;
- on a scan there is no layer, and the evidence is agreement between the readings: the one closest to the others is
  kept (a reading that dropped half the page, looped, or invented text stands apart from the rest).

Nothing here looks at a file name, a document type or a benchmark's wording.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass

from docparse.fixes.align_text import TOKEN, key
from docparse.fixes.layout_refine import PtWord, lines_of, words_in_points
from docparse.pdf.render import RenderedPage

_TAG = re.compile(r"<[^>]+>")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_TABLE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)
_ROW = re.compile(r"<tr\b", re.IGNORECASE)
_DIGIT = re.compile(r"\d")


def reading_tokens(markdown: str) -> list[str]:
    """Word tokens of a reading, without markup, image descriptions and comments."""
    text = _COMMENT.sub(" ", markdown)
    text = _IMAGE.sub(" ", text)
    text = html.unescape(_TAG.sub(" ", text))
    return [key(m.group()) for m in TOKEN.finditer(text)]


def layer_tokens(words: list[PtWord]) -> list[str]:
    return [key(m.group()) for w in words for m in TOKEN.finditer(w.text)]


def _overlap(a: Counter, b: Counter) -> int:
    return sum(min(n, b[k]) for k, n in a.items())


def _bigrams(lines: list[list[str]]) -> Counter:
    out: Counter = Counter()
    for toks in lines:
        out.update(zip(toks, toks[1:], strict=False))
    return out


@dataclass(frozen=True)
class LayerFit:
    """How a reading agrees with the page's text layer (all shares are 0-1; 1.0 when there is nothing to check)."""

    n_layer: int  # tokens in the layer
    n_read: int  # tokens in the reading
    recall: float  # share of the layer's tokens the reading holds
    precision: float  # share of the reading's tokens the layer holds
    num_recall: float  # the same over tokens with a digit
    num_precision: float
    n_layer_num: int
    n_read_num: int
    pair_recall: float  # share of the layer's same-line neighbour pairs that are neighbours in the reading
    tables: int  # HTML tables in the reading
    rows: int  # rows in them
    table_cells: int  # cells of tables checked against the layer's word grid
    table_found: float  # share of those cells found in place


def layer_fit(markdown: str, rendered: RenderedPage, with_tables: bool = True) -> LayerFit:
    words = words_in_points(rendered)
    lay = layer_tokens(words)
    read = reading_tokens(markdown)
    lc, rc = Counter(lay), Counter(read)
    hit = _overlap(lc, rc)
    lnum = Counter({k: n for k, n in lc.items() if _DIGIT.search(k)})
    rnum = Counter({k: n for k, n in rc.items() if _DIGIT.search(k)})
    nhit = _overlap(lnum, rnum)
    lay_pairs = _bigrams([[key(m.group()) for w in line for m in TOKEN.finditer(w.text)] for line in lines_of(words)])
    read_pairs = Counter(zip(read, read[1:], strict=False))
    n_pairs = sum(lay_pairs.values())
    tables = _TABLE.findall(markdown)
    cells, found = 0, 1.0
    if with_tables and tables and words:
        from docparse.fixes.verify import table_alignment_stats

        try:
            _, _matched, cells, found = table_alignment_stats(markdown, rendered)
        except Exception:  # an odd table must not cost the comparison
            cells, found = 0, 1.0
    return LayerFit(
        n_layer=len(lay),
        n_read=len(read),
        recall=hit / len(lay) if lay else 1.0,
        precision=hit / len(read) if read else 1.0,
        num_recall=nhit / sum(lnum.values()) if lnum else 1.0,
        num_precision=nhit / sum(rnum.values()) if rnum else 1.0,
        n_layer_num=sum(lnum.values()),
        n_read_num=sum(rnum.values()),
        pair_recall=_overlap(lay_pairs, read_pairs) / n_pairs if n_pairs else 1.0,
        tables=len(tables),
        rows=sum(len(_ROW.findall(t)) for t in tables),
        table_cells=cells,
        table_found=found,
    )


def agreement(a: str, b: str) -> float:
    """Token F1 of two readings (multiset): 1.0 for the same words, 0.0 for none in common."""
    ta, tb = Counter(reading_tokens(a)), Counter(reading_tokens(b))
    na, nb = sum(ta.values()), sum(tb.values())
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    hit = _overlap(ta, tb)
    return 2 * hit / (na + nb)


def fit_score(fit: LayerFit) -> float:
    """One number for a reading's agreement with the text layer: the mean of its word F1 and its number F1 (a
    reading with no numbers on a page without numbers scores its word F1 twice)."""

    def f1(p: float, r: float) -> float:
        return 2 * p * r / (p + r) if p + r else 0.0

    return 0.5 * f1(fit.precision, fit.recall) + 0.5 * f1(fit.num_precision, fit.num_recall)


def prefer_alternative(primary_md: str, alt_md: str, rendered: RenderedPage, options: dict) -> bool:
    """On a page with a healthy text layer: does the other reading agree with the layer clearly better (by
    `layer_margin`, default 0.03) than the primary one? Never on a page without such a layer."""
    from docparse.fixes.verify import has_healthy_text_layer

    if not alt_md.strip() or rendered.hidden is None or not has_healthy_text_layer(rendered):
        return False
    margin = float(options.get("layer_margin", 0.03))
    a = fit_score(layer_fit(primary_md, rendered, with_tables=False))
    b = fit_score(layer_fit(alt_md, rendered, with_tables=False))
    return b > a + margin


# --------------------------------------------------------------------------- fusing the two readings

_PIECE = re.compile(r"\s+|\S+")
_MARKUP = re.compile(r"[<>|*_#~`$\\\[\]]")


def _word_pieces(markdown: str) -> tuple[list[str], list[int]]:
    """The reading split into alternating blank / non-blank pieces, and the indexes of the non-blank ones."""
    parts = _PIECE.findall(markdown)
    return parts, [i for i, t in enumerate(parts) if not t.isspace()]


def fuse_witness_words(kept_md: str, other_md: str, witness: Counter) -> tuple[str, int]:
    """Where the two readings disagree on plain words and a third witness (the text layer's tokens, or a third
    reader's) sides with the other reading, the kept reading takes the other one's words: every word of the
    other's variant is known to the witness and some word of the kept variant is not. Two witnesses against one;
    markup, table cells with tags and formulas are never touched, and nothing is inserted or deleted.
    Returns the reading and the number of swaps."""
    import difflib

    if not kept_md.strip() or not other_md.strip() or not witness:
        return kept_md, 0
    parts, widx = _word_pieces(kept_md)
    a = [parts[i] for i in widx]
    b = [t for t in _PIECE.findall(other_md) if not t.isspace()]
    ka, kb = [reading_tokens(w) for w in a], [reading_tokens(w) for w in b]
    sa, sb = [" ".join(k) for k in ka], [" ".join(k) for k in kb]
    swaps = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, sa, sb, autojunk=False).get_opcodes():
        if op != "replace":
            continue
        ak = [k for ks in ka[i1:i2] for k in ks]
        bk = [k for ks in kb[j1:j2] for k in ks]
        if not ak or not bk:
            continue
        if any(_MARKUP.search(w) for w in a[i1:i2]) or any(_MARKUP.search(w) for w in b[j1:j2]):
            continue
        if all(witness[k] for k in bk) and any(not witness[k] for k in ak):
            swaps.append((i1, i2, j1, j2))
    for i1, i2, j1, j2 in sorted(swaps, reverse=True):
        parts[widx[i1] : widx[i2 - 1] + 1] = [" ".join(b[j1:j2])]
    return "".join(parts), len(swaps)


def fuse_layer_words(kept_md: str, other_md: str, rendered: RenderedPage) -> tuple[str, int]:
    """`fuse_witness_words` with the page's healthy text layer as the witness (no layer, no change)."""
    from docparse.fixes.verify import has_healthy_text_layer

    if rendered.hidden is None or not has_healthy_text_layer(rendered):
        return kept_md, 0
    return fuse_witness_words(kept_md, other_md, Counter(layer_tokens(words_in_points(rendered))))


# --------------------------------------------------------------------------- a jury of other readers


def _counter_f1(a: Counter, b: Counter) -> float | None:
    na, nb = sum(a.values()), sum(b.values())
    if not na and not nb:
        return None
    if not na or not nb:
        return 0.0
    return 2 * _overlap(a, b) / (na + nb)


def reading_similarity(a: str, b: str) -> float:
    """How alike two readings of a page are, in [0, 1]: the mean of the F1 of their words, of their tokens with a
    digit, and of their neighbouring word pairs (a part neither reading has is left out)."""
    ta, tb = reading_tokens(a), reading_tokens(b)
    parts = [
        _counter_f1(Counter(ta), Counter(tb)),
        _counter_f1(Counter(t for t in ta if _DIGIT.search(t)), Counter(t for t in tb if _DIGIT.search(t))),
        _counter_f1(Counter(zip(ta, ta[1:], strict=False)), Counter(zip(tb, tb[1:], strict=False))),
    ]
    known = [p for p in parts if p is not None]
    return sum(known) / len(known) if known else 1.0


_PIPE_ROW = re.compile(r"^[ \t]*\|(.+)\|[ \t]*$", re.MULTILINE)
_PIPE_RULE = re.compile(r"^[\s|:\-]+$")


def _norm_cell(text: str) -> str:
    return " ".join(reading_tokens(text))


def _grids(markdown: str) -> list[tuple[list[list[str]], int]]:
    """Every table of the reading as (rows of cell texts with spans filled in, number of header rows)."""
    from docparse.fixes.tables_html import parse_table

    out: list[tuple[list[list[str]], int]] = []
    for raw in _TABLE.findall(markdown):
        table = parse_table(raw)
        if table is None:
            continue
        cells: dict[tuple[int, int], str] = {}
        for row in table.rows:
            for cell in row.cells:
                text = _norm_cell(cell.text)
                for i in range(cell.row, min(cell.row + cell.rowspan, len(table.rows))):
                    for j in range(cell.col, cell.col + cell.colspan):
                        cells.setdefault((i, j), text)
        width = max((c for _, c in cells), default=-1) + 1
        grid = [[cells.get((r, c), "") for c in range(width)] for r in range(len(table.rows))]
        head = 0
        for row in table.rows:
            if row.cells and all(c.tag == "th" for c in row.cells):
                head += 1
            else:
                break
        out.append((grid, max(1, head) if len(grid) > 1 else 0))
    rest = _TABLE.sub("\n", markdown)
    block: list[list[str]] = []
    for line in [*rest.splitlines(), ""]:  # Markdown pipe tables
        m = _PIPE_ROW.match(line)
        if m:
            if not _PIPE_RULE.match(line):
                block.append([_norm_cell(c) for c in m.group(1).split("|")])
            continue
        if len(block) > 1:
            width = max(len(r) for r in block)
            out.append(([r + [""] * (width - len(r)) for r in block], 1))
        block = []
    return out


def table_records(markdown: str) -> tuple[Counter, Counter, Counter]:
    """What the reading's tables say: every filled body cell as (row label, column heading, value), and the same
    cells as (row label, value) and (column heading, value). A value under the wrong heading or beside the wrong
    label changes its records; a table split, merged or read as text loses them."""
    full: Counter = Counter()
    by_row: Counter = Counter()
    by_col: Counter = Counter()
    for grid, head in _grids(markdown):
        width = len(grid[0]) if grid else 0
        heads = []
        for c in range(width):
            parts = list(dict.fromkeys(grid[r][c] for r in range(head) if grid[r][c]))
            heads.append(" / ".join(parts))
        for row in grid[head:]:
            label = row[0] if row else ""
            for c in range(1, width):
                if not row[c]:
                    continue
                full[(label, heads[c], row[c])] += 1
                by_row[(label, row[c])] += 1
                by_col[(heads[c], row[c])] += 1
    return full, by_row, by_col


def table_similarity(a: str, b: str) -> float | None:
    """How alike the tables of two readings are, in [0, 1] (None when neither has a table): the mean F1 of their
    cell records, with partial credit for a value that keeps its row label or its column heading."""
    ra, rb = table_records(a), table_records(b)
    parts = [_counter_f1(x, y) for x, y in zip(ra, rb, strict=True)]
    known = [p for p in parts if p is not None]
    return sum(known) / len(known) if known else None


def page_similarity(a: str, b: str) -> float:
    """`reading_similarity`, averaged with `table_similarity` when either reading has a table."""
    words = reading_similarity(a, b)
    tables = table_similarity(a, b)
    return words if tables is None else 0.5 * (words + tables)


def jury_choice(primary_md: str, alt_md: str, jurors: list[str], margin: float = 0.0) -> bool | None:
    """Which of the two readings the other readers' readings resemble more: True for the alternative, False for
    the primary, None when there is no juror or the difference is within `margin`. The reading that stands apart
    from independent readers of the same page is the one more likely to have dropped, invented or misread text."""
    jurors = [j for j in jurors if j.strip()]
    if not jurors or not alt_md.strip() or not primary_md.strip():
        return None
    a = sum(reading_similarity(primary_md, j) for j in jurors) / len(jurors)
    b = sum(reading_similarity(alt_md, j) for j in jurors) / len(jurors)
    if b > a + margin:
        return True
    if a > b + margin:
        return False
    return None
