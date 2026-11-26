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
