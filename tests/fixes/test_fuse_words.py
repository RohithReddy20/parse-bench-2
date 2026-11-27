from collections import Counter

from docparse.fixes.formatting import _ruled_rows, _visible_tokens
from docparse.fixes.select import fuse_witness_words, reading_tokens


def _witness(text: str) -> Counter:
    return Counter(reading_tokens(text))


def test_word_the_witness_confirms_replaces_the_kept_one():
    kept = "The branch was sold to Samuel Pinsky in 1954."
    other = "The branch was sold to Samuel Pinsly in 1954."
    fused, n = fuse_witness_words(kept, other, _witness("The branch was sold to Samuel Pinsly in 1954."))
    assert n == 1
    assert fused == other


def test_nothing_changes_when_the_witness_knows_both_or_neither():
    kept, other = "total 1,250 units", "total 1,280 units"
    assert fuse_witness_words(kept, other, _witness("1,250 and 1,280")) == (kept, 0)
    assert fuse_witness_words(kept, other, _witness("total units")) == (kept, 0)


def test_markup_and_table_cells_are_left_alone():
    kept = "<table><tr><td>Pinsky</td><td>12</td></tr></table> and **Pinsky** again"
    other = "<table><tr><td>Pinsly</td><td>12</td></tr></table> and **Pinsly** again"
    assert fuse_witness_words(kept, other, _witness("Pinsly 12 and again")) == (kept, 0)


def test_nothing_is_inserted_or_deleted():
    kept = "alpha beta gamma"
    assert fuse_witness_words(kept, "alpha beta extra gamma", _witness("alpha beta extra gamma")) == (kept, 0)
    assert fuse_witness_words(kept, "alpha gamma", _witness("alpha gamma")) == (kept, 0)


def _under(raw: str, marked: set[str]):
    toks = _visible_tokens(raw)
    return toks, [t[2] in marked for t in toks]


def test_lines_under_several_cells_of_most_rows_are_a_ruling():
    raw = "<table>" + "".join(f"<tr><td>label{i}</td><td>v{i}</td></tr>" for i in range(4)) + "</table>"
    toks, under = _under(raw, {f"label{i}" for i in range(4)} | {f"v{i}" for i in range(4)})
    assert _ruled_rows(raw, toks, under)


def test_one_underlined_cell_is_not_a_ruling():
    raw = "<table>" + "".join(f"<tr><td>label{i}</td><td>v{i}</td></tr>" for i in range(4)) + "</table>"
    toks, under = _under(raw, {"label2"})
    assert not _ruled_rows(raw, toks, under)
