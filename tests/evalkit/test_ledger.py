import json

import pytest

from evalkit.ledger import CaseRow, Ledger, RunRow, case_rows, ingest, slot_average, slot_metrics
from evalkit.score import ScoreResult


def _run(run_id="r1"):
    return RunRow(run_id=run_id, kind="published", pipeline="ocria_1", harness_commit="h" * 40,
                  data_revision="d" * 40, overall=85.2, dimensions={"text": 87.9})  # fmt: skip


def _example(example_id, metrics, success=True, error=None):
    return {
        "test_id": example_id, "example_id": example_id, "pipeline_name": "p", "product_type": "parse",
        "success": success, "error": error,
        "metrics": [{"metric_name": n, "value": v, "metadata": {"proportional": v}} for n, v in metrics.items()],
    }  # fmt: skip


def test_round_trip(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    led.add_run(_run())
    n = led.add_cases("r1", [CaseRow("text/a", "text", "text_next", 0.5, True, None, {"events": {"critical": 1}})])
    assert n == 1
    got = led.run("r1")
    assert got.overall == 85.2 and got.dimensions == {"text": 87.9} and got.created_at
    (case,) = led.cases("r1", subset="text", metric="text_next")
    assert case.score == 0.5 and case.counted and case.metadata == {"events": {"critical": 1}}
    assert led.cases("r1", subset="charts") == []
    assert led.runs() == ["r1"]


def test_duplicate_run_raises(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    led.add_run(_run())
    with pytest.raises(ValueError, match="already"):
        led.add_run(_run())
    with pytest.raises(KeyError):
        led.run("nope")


def test_slot_metrics():
    m = slot_metrics()
    assert m["text"] == ["text_next", "formatting_v2"]
    assert m["forms"] == ["forms_v2"] and m["layout"] == ["layout_grounding_pass_rate"]
    assert len(m) == 13


def test_case_rows_counting_rules():
    report = {
        "per_example_results": [
            _example("text/a", {"text_next": 0.8, "formatting_v2": 0.6, "other": 1.0}),
            _example("text/b", {"text_next": 0.4}),  # formatting not produced: left out of that average
            _example("text/c", {}, success=False, error="No inference result found"),  # provider failure: 0
            _example("text/a#q1", {"text_next": 0.0}),  # per-question result: ignored
        ]
    }
    rows = case_rows("text", report)
    text = [r for r in rows if r.metric == "text_next"]
    fmt = [r for r in rows if r.metric == "formatting_v2"]
    assert [(r.example_id, r.score, r.counted) for r in text] == [
        ("text/a", 0.8, True), ("text/b", 0.4, True), ("text/c", 0.0, True)]  # fmt: skip
    assert slot_average(text) == pytest.approx(0.4)
    assert [(r.score, r.counted) for r in fmt] == [(0.6, True), (None, False), (0.0, True)]
    assert slot_average(fmt) == pytest.approx(0.3)
    assert text[0].metadata == {"proportional": 0.8}


def _score(tmp_path, slot_score):
    path = tmp_path / "headings_report.json"
    path.write_text(json.dumps({"per_example_results": [
        _example("headings/a", {"formatting_v2": 1.0}), _example("headings/b", {"formatting_v2": 0.5})]}))  # fmt: skip
    slots = [{"dimension": "formatting", "subset": "headings", "metric": "formatting_v2", "weight": 0.125,
              "score": slot_score, "documents": 2}]  # fmt: skip
    return ScoreResult(overall=75.0, dimensions={"formatting": 75.0}, subset_reports={"headings": path},
                       score_path=path, slots=slots)  # fmt: skip


def test_ingest_stores_rows_and_checks_slot_average(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    assert ingest(_score(tmp_path, 0.75), _run(), led) == 2
    assert len(led.cases("r1")) == 2 and led.run("r1").slots[0]["subset"] == "headings"


def test_ingest_rejects_rows_that_do_not_match_the_harness(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    with pytest.raises(ValueError, match="do not reproduce"):
        ingest(_score(tmp_path, 0.9), _run(), led)
    assert led.runs() == []
