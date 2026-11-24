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


def test_text_class_gain_is_shared_among_its_kinds():
    from evalkit.losses import kind_classes

    def text_case(example_id, events, kinds):
        meta = {"proportional": 1.0, "n_ref_tokens": 300, "events": {"critical": 0, "major": 0, "minor": 0, "moved": 0, "broken": 0, **events},
                "event_kinds": kinds}  # fmt: skip
        return CaseRow(example_id, "text", "text_next", recompute("text_next", meta), True, None, meta)

    cases = [
        text_case("text/a", {"critical": 1}, {"wrong number or identifier": 1}),  # teaches: that kind is critical
        text_case("text/b", {"major": 2}, {"omitted": 1, "wrong words": 1}),
        text_case(
            "text/c", {"critical": 2, "major": 1}, {"wrong number or identifier": 1, "omitted passage": 1, "omitted": 1}
        ),
    ]
    classes = kind_classes(cases, "text_next")
    assert classes["wrong number or identifier"] == "critical" and classes["omitted"] == "major"
    rows, problems = loss_table(cases)
    assert problems == []
    kinds = {r.label: r for r in rows if r.kind == "kind"}
    events = {r.label: r for r in rows if r.kind == "event"}
    assert kinds["omitted"].severity == "major" and kinds["wrong number or identifier"].pages == 2
    share = sum(r.dim_points for r in kinds.values() if r.severity == "critical")
    # every critical gain is accounted for by a kind row or by the "kind not recorded" remainder
    assert share == pytest.approx(events["all critical errors"].dim_points)


def test_layout_page_reasons_sum_to_the_lost_score():
    from evalkit.layout_losses import extra_reason, page_reasons

    meta = {
        "element_results": [
            {"gt_class": "Text", "why": "pass", "score": 1.0, "weight": 10.0},
            {"gt_class": "Text", "why": "pass", "score": 0.5, "weight": 10.0},
            {"gt_class": "Picture", "why": "no_localization", "score": 0.0, "weight": 20.0},
            {"gt_class": "Table", "why": "token_f1_below_threshold", "score": 0.0, "weight": 10.0},
        ],
        "prediction_results": [
            {"kind": "extra", "reason": "holds words locked to #1 (4842cd87a1fe9424)", "penalty": 5.0},
            {"kind": "neutral", "reason": "on ignore/optional element", "penalty": 0.0},
        ],
    }
    score = 15.0 / 55.0  # credit 15 over weight 50 + extra 5
    reasons = page_reasons(meta, score)
    assert sum(reasons.values()) == pytest.approx(1 - score)
    assert set(reasons) == {
        "box looser or tighter than the reference (IoU ramp): text",
        "element not located (no box matches): picture",
        "box found, text does not match (token F1 < 0.8): table",
        "extra box: holds words locked to another element",
    }
    assert (
        reasons["element not located (no box matches): picture"]
        > reasons["extra box: holds words locked to another element"]
    )
    assert extra_reason("blank area") == "extra box: blank area"
    assert extra_reason("garbage child of table #13") == "extra box: garbage child of a table"
    assert page_reasons({}, 0.0) == {"page failed (no usable layout)": 1.0}
    assert page_reasons(meta, 1.0) == {}


@pytest.mark.data
def test_baseline_run_invariants():
    """On the real OCRia-1 baseline: saved counts rebuild every page score, and the rows add up to the lost score."""
    from evalkit import paths
    from evalkit.ledger import Ledger
    from evalkit.losses import table_for_run

    ledger = Ledger(paths.ledger_path())
    if "published-ocria_1-baseline" not in ledger.runs():
        pytest.skip("baseline run not in the ledger (python -m evalkit score-published)")
    rows, problems = table_for_run(ledger, "published-ocria_1-baseline")
    assert problems == [], problems[:5]
    run = ledger.run("published-ocria_1-baseline")
    for dim, off in check_sums(rows, run.dimensions).items():
        assert abs(off) < 0.02, (dim, off)
    assert dimension_scores(ledger.cases(run.run_id))["overall"] == pytest.approx(run.overall, abs=1e-6)


def test_content_parts_share_the_content_row():
    from evalkit.losses import content_parts, work_queue

    assert content_parts("chart_next", {"recall": 0.7, "precision": 0.9}) == pytest.approx(
        {"points missing (recall)": 0.75, "points extra or wrong (precision)": 0.25})  # fmt: skip
    forms = {
        "proportional": 0.8,
        "coverage": 0.9,
        "lanes": {"filled": 0.6, "empty": 1.0, "checkboxes": None, "signatures": 0.8},
    }
    assert content_parts("forms_v2", forms) == pytest.approx({"filled values": 2 / 3, "signatures": 1 / 3})
    forms["coverage"] = 0.3
    assert content_parts("forms_v2", forms) == {"under half of the items found (coverage cap)": 1.0}
    assert content_parts("table_next", {"content": 1.0, "row_order": 1.0}) == {"unattributed": 1.0}

    meta = {"proportional": 0.6, "reference_cells": 40, "content": 0.6, "row_order": 1.0, "policy_inputs": []}
    case = CaseRow("tables/a", "tables", "table_next", 0.6, True, None, meta)
    rows, _ = loss_table([case])
    parts = [r for r in rows if r.kind == "content_part"]
    assert [r.label for r in parts] == ["content: cell content"]
    assert parts[0].dim_points == pytest.approx(by_label(rows)[CONTENT].dim_points)
    assert [r.kind for r in work_queue(rows)] == ["content_part"]  # the queue shows the parts, not the total
