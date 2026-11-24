import json
import sys

import pytest

from evalkit import score as S


def _layout(tmp_path, subsets):
    out, gt = tmp_path / "out", tmp_path / "gt"
    for s in subsets:
        (out / s / "p").mkdir(parents=True)
        (gt / s).mkdir(parents=True)
    return out, gt


def _combined(scores: dict[str, float | None]):
    from parse_bench.evaluation.next_score import NEXT_DIMENSIONS

    dims = {}
    for name, slots in NEXT_DIMENSIONS.items():
        dims[name] = {
            "score": scores.get(name),
            "missing_subsets": [],
            "slots": [
                {"subset": s.subset, "metric": s.metric, "weight": s.weight, "score": scores.get(name), "documents": 1}
                for s in slots
            ],
        }
    return {"next_score": None, "missing_dimensions": [], "dimensions": dims}


def test_command_matches_the_leaderboard_rescore(tmp_path):
    out, gt = _layout(tmp_path, ["text", "charts"])
    sources = S.sources_for(out, "p", gt)
    assert set(sources) == {"text", "charts"}
    assert sources["text"] == {
        "output_dir": str((out / "text" / "p").resolve()),
        "test_cases_dir": str((gt / "text").resolve()),
    }
    cmd = S.build_command(out, sources, tmp_path / "rep")
    assert cmd[:6] == [sys.executable, "-I", "-m", "parse_bench.cli", "evaluation", "run"]
    assert cmd[6] == str(out.resolve())
    assert cmd[7] == "--next" and json.loads(cmd[8]) == sources
    assert cmd[9] == "--next_only"
    assert cmd[10:12] == ["--report_dir", str((tmp_path / "rep").resolve())]
    assert cmd[12:14] == ["--force", "True"]
    assert cmd[14:] == [
        "--export_csv", "False", "--export_rule_csv", "False",
        "--export_markdown", "False", "--export_html", "False",
    ]  # fmt: skip


def test_read_score_scales_and_averages(tmp_path):
    from evalkit.explode import SUBSETS

    out, gt = _layout(tmp_path, SUBSETS)
    sources = S.sources_for(out, "p", gt)
    for src in sources.values():
        (tmp_path / "x").mkdir(exist_ok=True)
        with open(src["output_dir"] + "/_evaluation_report.json", "w") as fh:
            json.dump({"per_example_results": []}, fh)
    vals = {"text": 0.8787, "formatting": 0.9158, "tables": 0.8628, "charts": 0.7766, "layout": 0.8596, "forms": 0.821}
    path = tmp_path / "_next_score.json"
    path.write_text(json.dumps(_combined(vals)))
    res = S.read_score(path, sources)
    assert res.dimensions["text"] == pytest.approx(87.87)
    assert res.overall == pytest.approx(85.24, abs=0.01)
    assert set(res.subset_reports) == set(SUBSETS)
    assert len(res.slots) == 14


def test_missing_subset_raises(tmp_path):
    out, gt = _layout(tmp_path, ["text"])
    sources = S.sources_for(out, "p", gt)
    path = tmp_path / "_next_score.json"
    path.write_text(json.dumps(_combined({"text": 0.9})))
    with pytest.raises(S.ScoreError, match="not published"):
        S.read_score(path, sources)


def test_no_sources_raises(tmp_path):
    with pytest.raises(S.ScoreError):
        S.score_output(tmp_path, "p", tmp_path, tmp_path / "rep")
