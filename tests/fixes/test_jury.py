from docparse.fixes.select import jury_choice, page_similarity, reading_similarity, table_similarity
from docparse.models.adapters import jury

TABLE = (
    "<table><tr><th>Item</th><th>2023</th><th>2024</th></tr>"
    "<tr><td>Revenue</td><td>10</td><td>12</td></tr><tr><td>Cost</td><td>4</td><td>5</td></tr></table>"
)


def test_identical_readings_are_fully_alike():
    text = "Total due 1,250 on 3 March"
    assert reading_similarity(text, text) == 1.0
    assert reading_similarity(text, "nothing in common here") < 0.2


def test_a_changed_number_costs_more_than_a_changed_word():
    base = "The invoice total is 1250 euros payable in 30 days"
    assert reading_similarity(base, base.replace("1250", "1280")) < reading_similarity(base, base.replace("payable", "due"))


def test_values_under_the_wrong_heading_lower_table_similarity():
    swapped = TABLE.replace("<td>10</td><td>12</td>", "<td>12</td><td>10</td>")
    assert table_similarity(TABLE, TABLE) == 1.0
    assert 0.5 < table_similarity(TABLE, swapped) < 1.0
    assert table_similarity("no table", "none here") is None


def test_pipe_and_html_tables_compare_equal():
    pipe = "| Item | 2023 | 2024 |\n|---|---|---|\n| Revenue | 10 | 12 |\n| Cost | 4 | 5 |\n"
    assert table_similarity(TABLE, pipe) == 1.0
    assert page_similarity(TABLE, pipe) == 1.0


def test_jury_picks_the_reading_the_others_resemble():
    primary = "Samuel Pinsky bought the line in 1954 and ended passenger service"
    alt = "Samuel Pinsly bought the line in 1954 and ended passenger service a year later"
    jurors = ["Samuel Pinsly bought the line in 1954 and ended passenger service a year later.", "Samuel Pinsly bought the line in 1954"]
    assert jury_choice(primary, alt, jurors) is True
    assert jury_choice(alt, primary, jurors) is False


def test_jury_abstains_without_jurors_or_within_the_margin():
    assert jury_choice("a b c", "a b d", []) is None
    assert jury_choice("a b c", "a b d", ["", "  "]) is None
    assert jury_choice("a b c", "a b d", ["a b"], margin=0.5) is None


def test_stand_in_calls_are_keyed_by_juror_and_page():
    a, b = jury.published_call("one", "hash1"), jury.published_call("two", "hash1")
    assert a.key() != b.key()
    assert a.key() == jury.published_call("one", "hash1").key()
    assert a.key() != jury.published_call("one", "hash2").key()
