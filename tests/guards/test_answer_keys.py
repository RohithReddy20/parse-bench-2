import pytest

from evalkit.guards import AnswerKeyAccess, forbid_answer_keys


def test_opening_an_answer_key_inside_the_context_raises(tmp_path):
    key = tmp_path / "case.test.json"
    key.write_text("{}")
    other = tmp_path / "case.pdf"
    other.write_bytes(b"x")
    assert key.read_text() == "{}"  # allowed outside
    with forbid_answer_keys():
        assert other.read_bytes() == b"x"
        with pytest.raises(AnswerKeyAccess):
            key.read_text()
        with forbid_answer_keys():  # re-entrant
            with pytest.raises(AnswerKeyAccess):
                open(str(key).encode(), "rb")
        with pytest.raises(AnswerKeyAccess):
            open(key)
    assert key.read_text() == "{}"  # and allowed again afterwards
