import pytest

from evalkit.ledger import CaseRow
from evalkit.losses import CAP, CONTENT, FAILED, check_sums, dimension_scores, loss_table, parsing_bound, recompute


def table_case(example_id, subset, proportional, events, cells=40, score=None):
    meta = {"proportional": proportional, "reference_cells": cells,
            "policy_inputs": [{"label": label, "severity": sev, "count": n} for label, sev, n in events]}  # fmt: skip
    return CaseRow(
        example_id, subset, "table_next", recompute("table_next", meta) if score is None else score, True, None, meta
    )


def by_label(rows, subset=None):
    return {r.label: r for r in rows if subset is None or r.subset == subset}


def test_cap_and_counterfactuals_for_a_table_page():
    case = table_case("tables/a", "tables", 0.9, [("row missing", "critical", 1), ("title missing", "major", 1),
                                                   ("column out of order", "major", 1)])  # fmt: skip
    assert case.score == pytest.approx(0.8)  # cap = 1 - 0.10 - 2 * 0.05
    rows, problems = loss_table([case])
    assert problems == []
    got = by_label(rows)
    # one page is the whole `tables` slot (weight 1/2), so w = 50 dimension points
    assert got[CONTENT].dim_points == pytest.approx(50 * 0.1)
    assert got[CAP].dim_points == pytest.approx(50 * 0.1)
    assert got["row missing"].dim_points == pytest.approx(50 * 0.1)  # without it the cap (0.9) no longer binds
    assert got["title missing"].dim_points == pytest.approx(50 * 0.05)
    assert got["row missing"].severity == "critical" and got["row missing"].kind == "event"
    assert got[CAP].overall_points == pytest.approx(50 * 0.1 / 6)
    assert got["row missing"].examples == [(pytest.approx(5.0), "tables/a")]


def test_no_cap_loss_when_the_proportional_score_binds():
    case = table_case("tables/a", "tables", 0.5, [("row missing", "critical", 1)])
    assert case.score == pytest.approx(0.5)
    got = by_label(loss_table([case])[0])
    assert CAP not in got and "row missing" not in got
    assert got[CONTENT].dim_points == pytest.approx(25.0)


def test_slot_weights_and_page_counts():
    cases = [table_case("tables/a", "tables", 1.0, [("row missing", "critical", 1)]),
             table_case("tables/b", "tables", 1.0, []),
             table_case("dense_tables/c", "dense_tables", 1.0, [("row missing", "critical", 1)])]  # fmt: skip
    rows, _ = loss_table(cases)
    assert by_label(rows, "tables")[CAP].dim_points == pytest.approx(0.5 / 2 * 100 * 0.1)
    assert by_label(rows, "dense_tables")[CAP].dim_points == pytest.approx(0.125 / 1 * 100 * 0.1)
    assert rows[0].subset == "tables"  # sorted by points lost


def test_repeated_errors_scale_with_table_size():
    big = table_case("tables/a", "tables", 1.0, [("wrong or misplaced number in a row", "critical", 5)], cells=800)
    # first critical costs 0.10; the other four are scaled by 80 / 800
    assert big.score == pytest.approx(1 - 0.1 * (1 + 4 * 0.1))


def test_forms_coverage_bound_and_unscaled_minors():
    meta = {"proportional": 0.9, "coverage": 0.3, "atoms": 50,
            "policy_inputs": [{"label": "extra field", "severity": "minor", "grouped": False, "count": 3}]}  # fmt: skip
    assert parsing_bound("forms_v2", meta) == pytest.approx(0.6)
    meta["coverage"] = 1.0
    assert recompute("forms_v2", meta) == pytest.approx(0.9)  # 1 - 0.03 = 0.97 does not bind
    meta["proportional"] = 1.0
    assert recompute("forms_v2", meta) == pytest.approx(0.97)


def test_failed_page_and_mismatch_reporting():
    failed = CaseRow("tables/x", "tables", "table_next", 0.0, True, "No inference result found", {})
    wrong = table_case("tables/y", "tables", 0.9, [("row missing", "critical", 1)], score=0.123)
    rows, problems = loss_table([failed, wrong])
    assert by_label(rows)[FAILED].dim_points == pytest.approx(25.0)
    assert len(problems) == 1 and "tables/y" in problems[0]


def test_content_plus_cap_sums_to_the_lost_score():
    cases = [table_case(f"{s}/{i}", s, 0.95 - 0.1 * i, [("row missing", "critical", i)])
             for s in ("tables", "complex_table_headers", "financial_tables", "insurance_filing_tables", "dense_tables")
             for i in range(3)]  # fmt: skip
    rows, _ = loss_table(cases)
    dims = dimension_scores(cases)
    assert set(dims) == {"tables"}
    assert check_sums(rows, dims)["tables"] == pytest.approx(0.0, abs=1e-9)
