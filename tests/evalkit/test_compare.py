import pytest

from evalkit.compare import compare_runs, gt_view, markdown_similarity
from evalkit.ledger import CaseRow, Ledger, RunRow


def test_gt_view_links_only_the_chosen_cases(tmp_path):
    gt = tmp_path / "gt"
    (gt / "text").mkdir(parents=True)
    for name in ("a.pdf", "a.test.json", "ab.pdf", "ab.test.json", "b.pdf", "b.test.json", "a.notes.txt"):
        (gt / "text" / name).write_text(name)
    view = gt_view(gt, ["text/a"], tmp_path / "view")
    assert sorted(p.name for p in (view / "text").iterdir()) == ["a.pdf", "a.test.json"]  # not ab.*, not b.*
    assert (view / "text" / "a.pdf").is_symlink() and (view / "text" / "a.pdf").read_text() == "a.pdf"
    gt_view(gt, ["text/a"], tmp_path / "view")  # idempotent


def test_compare_runs_on_shared_pages(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    for run_id, scores in (("ours", {"a": 0.9, "b": 0.5}), ("base", {"a": 0.8, "b": 0.7, "c": 1.0})):
        led.add_run(RunRow(run_id=run_id, kind="pipeline", pipeline="p", harness_commit="h", data_revision="d",
                           overall=0.0, dimensions={}))  # fmt: skip
        led.add_cases(run_id, [CaseRow(f"charts/{k}", "charts", "chart_next", v, True) for k, v in scores.items()])
    out = compare_runs(led, "ours", "base", ["charts/a", "charts/b", "charts/c"])
    (slot,) = out["slots"]
    assert slot["pages"] == 2 and slot["ours"] == pytest.approx(70.0) and slot["baseline"] == pytest.approx(75.0)
    assert out["dimensions"]["charts"]["ours"] == pytest.approx(70.0)
    assert [p["example_id"] for p in out["pages"]] == ["charts/b", "charts/a"]  # worst delta first
    assert out["only_baseline"] == 1 and out["only_ours"] == 0


def test_markdown_similarity():
    assert markdown_similarity("a  b\n\nc", "a b c") == 1.0
    assert markdown_similarity("", "") == 1.0
    assert 0.5 < markdown_similarity("one two three four", "one two three five") < 1.0
    assert markdown_similarity("x", "") == 0.0
