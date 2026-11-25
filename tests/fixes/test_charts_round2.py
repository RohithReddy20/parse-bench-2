import pytest

from docparse.fixes.charts_geometry import find_axes
from docparse.fixes.layout_refine import PtWord

KEPT = {"signs": False, "upright": False}  # the kept configuration's axis rules


def test_tick_labels_with_lost_leading_digits_take_their_value_from_the_sequence():
    # "80 0 0 0 0 30 20 10 0" down a column: the 7, 6, 5 and 4 glyphs have no text mapping
    texts = ("80", "0", "0", "0", "0", "30", "20", "10", "0")
    col = [PtWord(t, 40 + 3 * (len(t) == 1), 300 + 10 * i, 50, 305 + 10 * i) for i, t in enumerate(texts)]
    assert all(len(a.values) <= 3 for a in find_axes(col, **KEPT))
    (axis,) = find_axes(col, lost=True, **KEPT)
    assert axis.values == [80, 70, 60, 50, 40, 30, 20, 10, 0]
    assert axis.value(302.5 + 10 * 2) == pytest.approx(60)


def test_lost_minus_signs_and_other_numbers_are_not_lost_digits():
    # "20 10 0 10 20": a minus sign alone is not a lost digit (population pyramids print both sides positive)
    col = [PtWord(t, 40, 300 + 13 * i, 50, 306 + 13 * i) for i, t in enumerate(("20", "10", "0", "10", "20"))]
    assert all(min(a.values) >= 0 for a in find_axes(col, lost=True, **KEPT))
    # a label that is not the tail of its value in the sequence leaves the column alone
    odd = [PtWord(t, 40, 300 + 10 * i, 50, 305 + 10 * i) for i, t in enumerate(("80", "5", "60", "50", "40"))]
    assert all(70 not in a.values for a in find_axes(odd, lost=True, **KEPT))


def test_rotated_category_labels_split_into_pieces_place_the_categories():
    from docparse.fixes.charts_cats import rotated_positions

    box = (90, 50, 400, 200)  # the plot; rotated labels just under it, read bottom to top
    words = []
    for x, pieces in ((100, ["Swit", "erland"]), (150, ["United", "States"]), (200, ["at", "ia"]), (250, ["Spain"])):
        y = 260.0
        for p in pieces:  # each piece a narrow, tall box; the next piece above it
            words.append(PtWord(p, x, y - 4 * len(p), x + 5, y))
            y -= 4 * len(p) + 2
    words.append(PtWord("Source:", 90, 270, 120, 276))  # horizontal text under the labels
    cats = ["Switzerland", "United States", "Latvia", "Spain"]
    assert rotated_positions(cats, words, box) == [102.5, 152.5, 202.5, 252.5]
    assert rotated_positions(["Spain", "Latvia", "United States", "Switzerland"], words, box) is None
    assert rotated_positions(["Spain", "Latvia", "United States", "Switzerland"], words, box, ordered=False) == [
        252.5,
        202.5,
        152.5,
        102.5,
    ]
