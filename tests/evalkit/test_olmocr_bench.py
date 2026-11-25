import pytest

from evalkit.external.olmocr_bench import (
    category_dirs,
    changed_pages,
    failed_tests,
    md_name,
    overall,
    parse_summary,
    side_by_side,
)

LOG = """
Candidate: docparse_plain
  [FAIL] Test tables_1_a on tables/x page 1 average pass ratio: 0.000 (0/1 repeats passed). Ex: No tables found
  [FAIL] Test hf_3 on headers_footers/y page 1 average pass ratio: 0.000 (0/1 repeats passed). Ex: Expected absence
  Average Score: 50.0% (95% CI: [40.0%, 60.0%]) over 4 tests.

============================================================
Final Summary with 95% Confidence Intervals:
docparse_plain       : Average Score: 58.3% ± 10.0% (average of per-JSONL scores)
    absent  : 0.0% average pass rate over 1 tests
    table   : 50.0% average pass rate over 2 tests

    Results by JSONL file:
        baseline                      : 100.0% (1/1 tests)
        headers_footers.jsonl         : 0.0% (0/1 tests)
        table_tests.jsonl             : 75.0% (3/4 tests)

"""


def test_md_name_is_the_scorers_page_file():
    assert md_name("tables/x.pdf", 1) == "tables/x_pg1_repeat1.md"
    assert md_name("old_scans/a.b.pdf", 3, 2) == "old_scans/a.b_pg3_repeat2.md"
    with pytest.raises(ValueError):
        md_name("tables/x.png", 1)


def test_category_dirs():
    assert category_dirs([{"pdf": "tables/a.pdf"}, {"pdf": "tables/b.pdf"}, {"pdf": "arxiv_math/c.pdf"}]) == {
        "tables",
        "arxiv_math",
    }


def test_parse_summary_reads_the_final_block_only():
    out = parse_summary(LOG)
    assert list(out) == ["docparse_plain"]
    assert out["docparse_plain"]["files"] == {
        "baseline": (1, 1),
        "headers_footers.jsonl": (0, 1),
        "table_tests.jsonl": (3, 4),
    }
    assert out["docparse_plain"]["printed"] == "58.3% ± 10.0%"
    with pytest.raises(ValueError):
        parse_summary("no summary here")


def test_overall_is_the_mean_of_file_pass_rates():
    files = parse_summary(LOG)["docparse_plain"]["files"]
    assert overall(files) == pytest.approx(100 * (1 + 0 + 0.75) / 3)
    assert overall({}) == 0.0


def test_failed_tests():
    assert failed_tests(LOG) == {"tables_1_a", "hf_3"}


def test_side_by_side_shows_the_change():
    a = {"table_tests.jsonl": (3, 4), "baseline": (1, 1)}
    b = {"table_tests.jsonl": (4, 4), "baseline": (1, 1)}
    text = side_by_side({"plain": a, "corrected": b})
    rows = {line.split()[0]: line.split() for line in text.splitlines()[1:]}
    assert rows["overall"][-1] == "+12.50"
    assert rows["table_tests"][-1] == "+25.00"
    assert "change" not in side_by_side({"only": a}).splitlines()[0]


def test_changed_pages(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root, text in ((a, "x"), (b, "y")):
        (root / "t").mkdir(parents=True)
        (root / "t" / "p_pg1_repeat1.md").write_text("same")
        (root / "t" / "q_pg1_repeat1.md").write_text(text)
    (a / "t" / "only_pg1_repeat1.md").write_text("z")
    assert changed_pages(a, b) == (1, 2)
