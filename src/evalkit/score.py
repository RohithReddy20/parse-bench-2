"""Drive the harness's own scorer, exactly as its `scripts/leaderboard/rescore.py` does for ParseBench 2.

One `parse-bench evaluation run --next {...} --next_only` over every subset folder; the six dimensions of
`_next_score.json` are the leaderboard columns and the overall score is their mean.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from parse_bench.evaluation.next_score import NEXT_DIMENSIONS

DIMENSIONS: tuple[str, ...] = tuple(NEXT_DIMENSIONS)


class ScoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScoreResult:
    overall: float  # 0-100
    dimensions: dict[str, float]  # 0-100
    subset_reports: dict[str, Path]  # subset -> _evaluation_report.json
    score_path: Path
    slots: list[dict] = field(default_factory=list)  # every (subset, metric, weight, score 0-1, documents)


def sources_for(output_root: Path, pipeline: str, gt_dir: Path) -> dict[str, dict[str, str]]:
    subsets = list(dict.fromkeys(s.subset for slots in NEXT_DIMENSIONS.values() for s in slots))
    return {
        subset: {
            "output_dir": str((output_root / subset / pipeline).resolve()),
            "test_cases_dir": str((gt_dir / subset).resolve()),
        }
        for subset in subsets
        if (output_root / subset / pipeline).is_dir()
    }


def build_command(output_root: Path, sources: dict[str, dict[str, str]], report_dir: Path) -> list[str]:
    flags = []
    for name in ("--export_csv", "--export_rule_csv", "--export_markdown", "--export_html"):
        flags += [name, "False"]
    return [
        sys.executable,
        "-I",  # isolated: no PYTHONPATH, no user site, no script directory on sys.path
        "-m",
        "parse_bench.cli",
        "evaluation",
        "run",
        str(output_root.resolve()),
        "--next",
        json.dumps(sources),
        "--next_only",
        "--report_dir",
        str(report_dir.resolve()),
        "--force",
        "True",
        *flags,
    ]


def _dropped(report: dict) -> Counter:
    """Pages the evaluator left out of its average (harness crashes), by error."""
    from parse_bench.evaluation.runner import _is_infra_failure
    from parse_bench.schemas.evaluation import EvaluationResult

    return Counter(
        (r.error or "")[:200]
        for r in (EvaluationResult.model_validate(x) for x in report.get("per_example_results") or [])
        if not r.success and _is_infra_failure(r)
    )


def read_score(
    score_path: Path, sources: dict[str, dict[str, str]], check_dropped: bool = True, partial: bool = False
) -> ScoreResult:
    """`partial`: score only the dimensions whose subsets were all run (an experiment on some subsets)."""
    combined = json.loads(score_path.read_text(encoding="utf-8"))
    reports: dict[str, Path] = {}
    for subset, src in sources.items():
        path = Path(src["output_dir"]) / "_evaluation_report.json"
        if path.exists():
            reports[subset] = path
    dims: dict[str, float] = {}
    slots: list[dict] = []
    problems: list[str] = []
    for dimension in DIMENSIONS:
        subsets = list(dict.fromkeys(slot.subset for slot in NEXT_DIMENSIONS[dimension]))
        if partial and any(s not in sources for s in subsets):
            continue
        problems += [f"{dimension}: {s} not published" for s in subsets if s not in sources]
        problems += [f"{dimension}: {s} not evaluated" for s in subsets if s in sources and s not in reports]
        entry = combined.get("dimensions", {}).get(dimension) or {}
        value = entry.get("score")
        if value is None:
            problems.append(f"{dimension}: missing from the score")
        else:
            dims[dimension] = 100.0 * float(value)
        slots += [{"dimension": dimension, **row} for row in entry.get("slots") or []]
    if check_dropped:
        for subset, path in reports.items():
            dropped = _dropped(json.loads(path.read_text(encoding="utf-8")))
            if dropped:
                n = sum(dropped.values())
                problems.append(f"{subset}: {n} page(s) left out of the average: {dropped.most_common(3)}")
    if problems:
        raise ScoreError("; ".join(dict.fromkeys(problems)))
    return ScoreResult(
        overall=sum(dims.values()) / len(dims),
        dimensions=dims,
        subset_reports=reports,
        score_path=score_path,
        slots=slots,
    )


def score_output(
    output_root: Path, pipeline: str, gt_dir: Path, report_dir: Path, partial: bool = False
) -> ScoreResult:
    sources = sources_for(output_root, pipeline, gt_dir)
    if not sources:
        raise ScoreError(f"no <subset>/{pipeline}/ folders under {output_root}")
    report_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(output_root, sources, report_dir)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=report_dir)
    (report_dir / "_evaluation_stdout.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    score_path = report_dir.resolve() / "_next_score.json"
    if not score_path.exists():
        tail = (proc.stdout + proc.stderr)[-3000:]
        raise ScoreError(f"evaluation failed (exit {proc.returncode}):\n{tail}")
    return read_score(score_path, sources, partial=partial)
