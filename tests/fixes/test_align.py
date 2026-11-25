from docparse.fixes.align import align, visible_tokens
from docparse.fixes.layout_refine import PtWord


def _words(*texts: str) -> list[PtWord]:
    return [PtWord(t, 10.0 * i, 0.0, 10.0 * i + 8, 10.0) for i, t in enumerate(texts)]


def test_tags_entities_and_box_marks_are_not_words():
    md = "<td>**Family name**: TO &amp; [x] Yes [ ] No ![img](a.png)</td>"
    assert [t.text for t in visible_tokens(md)] == ["Family", "name", "TO", "Yes", "No"]
    tok = visible_tokens(md)[2]
    assert md[tok.start : tok.end] == "TO"


def test_tokens_align_with_the_pdf_words_they_spell():
    words = _words("Family", "name", "Given", "names")
    al = align("**Family name**: SMITH\n**Given names**: HARRY", words)
    assert [al.token_words(t) for t in range(len(al.tokens))] == [[0], [1], [], [2], [3], []]
    assert al.word_share() == [1.0, 1.0, 1.0, 1.0]
    assert al.token_share() == [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]


def test_different_word_splits_still_align():
    al = align("Postcode: B 3 7 P W", _words("Post", "code:", "B37PW"))
    assert al.token_words(0) == [0, 1]
    assert [al.token_words(t) for t in range(1, 6)] == [[2]] * 5


def test_columns_stored_in_another_order_align_out_of_order():
    words = _words("Second", "column", "text", "here", "First", "column", "paragraph")
    al = align("First column paragraph\n\nSecond column text here", words)
    assert al.word_share() == [1.0] * 7
    assert al.token_words(0) == [4] and al.token_words(3) == [0]


def test_a_short_stray_match_inside_other_words_is_dropped():
    al = align("No or Ok", _words("Nominal", "Okay"))  # one- and two-letter blocks count only as whole words
    assert al.word_share() == [0.0, 0.0]
    assert align("No or Ok", _words("No", "Ok")).word_share() == [1.0, 1.0]
