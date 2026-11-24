"""The score ledger: one row per scored run, and one row per example per scored metric.

Case rows keep the metric's full metadata (`proportional`, `coverage`, event counts, `policy_inputs`), which
is what the loss table recomputes page caps from. `ingest` proves the rows are complete by rebuilding every
slot average of `_next_score.json` from them.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from parse_bench.evaluation.next_score import NEXT_DIMENSIONS

from evalkit.score import ScoreResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, kind TEXT NOT NULL, pipeline TEXT NOT NULL,
    harness_commit TEXT NOT NULL, data_revision TEXT NOT NULL, code_sha TEXT NOT NULL,
    config_hash TEXT NOT NULL, cache_snapshot TEXT NOT NULL, overall REAL NOT NULL,
    dimensions TEXT NOT NULL, slots TEXT NOT NULL, notes TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS case_scores (
    run_id TEXT NOT NULL, example_id TEXT NOT NULL, subset TEXT NOT NULL, metric TEXT NOT NULL,
    score REAL, counted INTEGER NOT NULL, error TEXT, metadata TEXT NOT NULL,
    PRIMARY KEY (run_id, example_id, metric)
);
CREATE INDEX IF NOT EXISTS case_scores_by_subset ON case_scores (run_id, subset, metric);
"""


@dataclass
class RunRow:
    run_id: str
    kind: str  # "published" | "pipeline"
    pipeline: str
    harness_commit: str
    data_revision: str
    overall: float
    dimensions: dict[str, float]
    slots: list[dict] = field(default_factory=list)
    code_sha: str = ""
    config_hash: str = ""
    cache_snapshot: str = ""
    notes: str = ""
    created_at: str = ""


@dataclass
class CaseRow:
    example_id: str
    subset: str
    metric: str
    score: float | None  # None: the example is not part of this metric's average
    counted: bool  # True when the example enters the slot average (a provider failure counts as 0)
    error: str | None = None
    metadata: dict = field(default_factory=dict)


def code_sha(repo: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo or Path(__file__).parent
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=repo or Path(__file__).parent
        )
        return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except OSError:
        return ""


class Ledger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    def add_run(self, run: RunRow) -> None:
        if not run.created_at:
            run.created_at = datetime.now(UTC).isoformat(timespec="seconds")
        try:
            with self._db:
                self._db.execute(
                    "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run.run_id, run.created_at, run.kind, run.pipeline, run.harness_commit, run.data_revision,
                        run.code_sha, run.config_hash, run.cache_snapshot, run.overall,
                        json.dumps(run.dimensions), json.dumps(run.slots), run.notes,
                    ),
                )  # fmt: skip
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"run {run.run_id!r} is already in the ledger") from exc

    def add_cases(self, run_id: str, rows: Iterable[CaseRow]) -> int:
        data = [
            (run_id, r.example_id, r.subset, r.metric, r.score, int(r.counted), r.error, json.dumps(r.metadata))
            for r in rows
        ]
        with self._db:
            self._db.executemany("INSERT INTO case_scores VALUES (?,?,?,?,?,?,?,?)", data)
        return len(data)

    def run(self, run_id: str) -> RunRow:
        row = self._db.execute(
            "SELECT run_id, created_at, kind, pipeline, harness_commit, data_revision, code_sha, config_hash,"
            " cache_snapshot, overall, dimensions, slots, notes FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return RunRow(
            run_id=row[0], created_at=row[1], kind=row[2], pipeline=row[3], harness_commit=row[4],
            data_revision=row[5], code_sha=row[6], config_hash=row[7], cache_snapshot=row[8], overall=row[9],
            dimensions=json.loads(row[10]), slots=json.loads(row[11]), notes=row[12],
        )  # fmt: skip

    def runs(self) -> list[str]:
        return [r[0] for r in self._db.execute("SELECT run_id FROM runs ORDER BY created_at, run_id")]

    def cases(self, run_id: str, subset: str | None = None, metric: str | None = None) -> list[CaseRow]:
        sql = "SELECT example_id, subset, metric, score, counted, error, metadata FROM case_scores WHERE run_id = ?"
        args: list = [run_id]
        if subset is not None:
            sql += " AND subset = ?"
            args.append(subset)
        if metric is not None:
            sql += " AND metric = ?"
            args.append(metric)
        return [
            CaseRow(
                example_id=r[0],
                subset=r[1],
                metric=r[2],
                score=r[3],
                counted=bool(r[4]),
                error=r[5],
                metadata=json.loads(r[6]),
            )  # fmt: skip
            for r in self._db.execute(sql + " ORDER BY example_id, metric", args)
        ]


def slot_metrics() -> dict[str, list[str]]:
    """subset -> the metrics that feed a dimension slot for it."""
    out: dict[str, list[str]] = {}
    for slots in NEXT_DIMENSIONS.values():
        for slot in slots:
            if slot.metric not in out.setdefault(slot.subset, []):
                out[slot.subset].append(slot.metric)
    return out


def case_rows(subset: str, report: dict) -> list[CaseRow]:
    """One row per example per slot metric of `subset`, from an `_evaluation_report.json`.

    The harness averages a metric over the successful examples that carry it and counts every provider failure
    (no output, no layout) as 0; harness crashes are left out. The same rule decides `counted` here.
    """
    from parse_bench.evaluation.runner import _is_infra_failure
    from parse_bench.schemas.evaluation import EvaluationResult

    rows: list[CaseRow] = []
    for raw in report.get("per_example_results") or []:
        result = EvaluationResult.model_validate(raw)
        if "#q" in result.test_id:
            continue  # per-question results never join a slot
        by_name = {m.metric_name: m for m in result.metrics}
        for metric in slot_metrics().get(subset, []):
            if result.success:
                m = by_name.get(metric)
                if m is None:
                    rows.append(CaseRow(result.example_id, subset, metric, None, False, "metric not produced"))
                else:
                    rows.append(CaseRow(result.example_id, subset, metric, float(m.value), True, None, m.metadata))
            elif _is_infra_failure(result):
                rows.append(CaseRow(result.example_id, subset, metric, None, False, result.error))
            else:
                rows.append(CaseRow(result.example_id, subset, metric, 0.0, True, result.error))
    return rows


def slot_average(rows: Iterable[CaseRow]) -> float | None:
    values = [r.score for r in rows if r.counted]
    return sum(values) / len(values) if values else None


def ingest(score: ScoreResult, run: RunRow, ledger: Ledger, tolerance: float = 1e-6) -> int:
    """Store the run and its case rows; raise when the rows do not rebuild the harness's slot averages."""
    all_rows: list[CaseRow] = []
    for subset, path in sorted(score.subset_reports.items()):
        all_rows += case_rows(subset, json.loads(Path(path).read_text(encoding="utf-8")))
    problems = []
    for slot in score.slots:
        if slot["subset"] not in score.subset_reports:
            continue  # a partial run: this subset was not run
        ours = slot_average(r for r in all_rows if r.subset == slot["subset"] and r.metric == slot["metric"])
        if slot["score"] is None or ours is None or abs(ours - slot["score"]) > tolerance:
            problems.append(f"{slot['subset']}/{slot['metric']}: harness {slot['score']} vs ledger {ours}")
    if problems:
        raise ValueError("case rows do not reproduce the slot averages: " + "; ".join(problems))
    run.slots = score.slots
    ledger.add_run(run)
    return ledger.add_cases(run.run_id, all_rows)


def run_as_dict(run: RunRow) -> dict:
    return asdict(run)
