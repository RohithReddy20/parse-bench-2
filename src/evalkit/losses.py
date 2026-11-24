"""The loss table: points lost per error type per dimension (design §8.1). It is the work queue.

All numbers are on the 0-100 scale of a dimension; `overall_points = dim_points / 6`.

For a capped metric (text, formatting, tables, charts, forms) a page's score is min(parsing, cap):
  content (not recovered)   sum of w * (1 - parsing)        what the output never contained or got wrong
  severity cap              sum of w * (parsing - score)    what the error events took on top of that
These two rows add up to exactly 100 - dimension score. Under the cap row, one row per event label gives the
counterfactual gain if that label never occurred: the page's cap is recomputed from the saved event counts
with the harness's own `metric_policy.score_metric` under a policy that ignores the label. Those rows explain
the cap; they overlap (the first error of a severity costs most), so they do not add up.

Layout has no cap. Its rows are 100 - score split by the scorer's own per-element outcome.

`w` is the page weight: slot weight / pages counted in the subset * 100.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field

from parse_bench.evaluation.metric_policy import METRIC_KEYS, PAPER, MetricPolicy, score_metric
from parse_bench.evaluation.next_score import NEXT_DIMENSIONS

from evalkit.ledger import CaseRow, Ledger
from evalkit.splits import fold_of

CONTENT = "content (not recovered)"
CAP = "severity cap"
FAILED = "page failed (no usable output)"
TEXT_LABELS = (
    ("all critical errors", "critical"),
    ("all major errors", "major"),
    ("all minor errors", "minor"),
    ("moved block (reading order)", "major"),
    ("cut sentence (reading order)", "major"),
)


@dataclass
class LossRow:
    dimension: str
    subset: str
    metric: str
    label: str
    severity: str  # "critical" | "major" | "minor" | "" for content, cap and layout rows
    kind: str  # "content" | "cap" | "event" | "layout" | "failed"
    pages: int = 0
    dim_points: float = 0.0
    examples: list[tuple[float, str]] = field(default_factory=list)  # (points lost on the page, example id)

    @property
    def overall_points(self) -> float:
        return self.dim_points / 6


def paper_policy() -> MetricPolicy:
    return MetricPolicy.model_validate(copy.deepcopy(PAPER))


def policy_without(label: str, key: str) -> MetricPolicy:
    data = copy.deepcopy(PAPER)
    data["error_classes"][key][label] = "ignore"
    return MetricPolicy.model_validate(data)


def parsing_bound(metric: str, metadata: dict) -> float:
    """The score a page would get with no severity events at all (the proportional or coverage bound)."""
    parsing = metadata["proportional"]
    if METRIC_KEYS[metric] in ("fmt", "forms"):
        parsing = min(parsing, metadata["coverage"] / 0.5)
    return float(parsing)


def case_labels(metric: str, metadata: dict) -> list[tuple[str, str]]:
    """(label, severity) of every event class charged on this page, as the policy names them."""
    key = METRIC_KEYS[metric]
    if key == "text":
        events = metadata.get("events") or {}
        fields = {"all critical errors": "critical", "all major errors": "major", "all minor errors": "minor",
                  "moved block (reading order)": "moved", "cut sentence (reading order)": "broken"}  # fmt: skip
        return [(label, sev) for label, sev in TEXT_LABELS if events.get(fields[label])]
    inputs = metadata.get("policy_inputs") or []
    if key == "charts":
        seen: dict[tuple[str, str], None] = {}
        from parse_bench.evaluation.metric_policy import event_label

        for group in inputs:
            for kind, severity, _count in group["events"]:
                seen[(event_label("charts", kind, severity), severity)] = None
        return [k for k in seen if k[1] != "ignore"]
    return list(dict.fromkeys((e["label"], e["severity"]) for e in inputs if e["severity"] != "ignore"))


def recompute(metric: str, metadata: dict, policy: MetricPolicy | None = None) -> float:
    return score_metric(metric, metadata, policy or paper_policy())


def slot_weights() -> dict[tuple[str, str], tuple[str, float]]:
    """(subset, metric) -> (dimension, slot weight)."""
    return {(s.subset, s.metric): (dim, s.weight) for dim, slots in NEXT_DIMENSIONS.items() for s in slots}


def dimension_scores(cases: list[CaseRow], fold: str = "all") -> dict[str, float]:
    """Dimension scores (0-100) rebuilt from case rows, optionally for one fold of the split."""
    sums: dict[tuple[str, str], list[float]] = defaultdict(list)
    for c in cases:
        if c.counted and (fold == "all" or fold_of(c.example_id) == fold):
            sums[(c.subset, c.metric)].append(c.score or 0.0)
    out: dict[str, float] = {}
    for dim, slots in NEXT_DIMENSIONS.items():
        total = 0.0
        for s in slots:
            values = sums.get((s.subset, s.metric))
            if not values:
                break
            total += s.weight * sum(values) / len(values)
        else:
            out[dim] = 100.0 * total
    if len(out) == len(NEXT_DIMENSIONS):
        out["overall"] = sum(out.values()) / len(out)
    return out


def layout_rows(case: CaseRow) -> dict[str, float]:
    """Share of a layout page's lost score per failure reason. Filled in from the metric's own metadata."""
    from evalkit.layout_losses import page_reasons

    return page_reasons(case.metadata, case.score or 0.0)


def loss_table(cases: list[CaseRow], fold: str = "all", top_examples: int = 10) -> tuple[list[LossRow], list[str]]:
    """(rows, problems). `problems` lists every case whose saved counts do not rebuild its score."""
    weights = slot_weights()
    selected = [c for c in cases if c.counted and (fold == "all" or fold_of(c.example_id) == fold)]
    per_slot: dict[tuple[str, str], int] = defaultdict(int)
    for c in selected:
        per_slot[(c.subset, c.metric)] += 1
    rows: dict[tuple, LossRow] = {}
    problems: list[str] = []
    policies: dict[tuple[str, str], MetricPolicy] = {}
    base = paper_policy()

    def add(c: CaseRow, dim: str, label: str, severity: str, kind: str, points: float) -> None:
        if points <= 1e-12:
            return
        row = rows.setdefault((dim, c.subset, c.metric, label, severity, kind),
                              LossRow(dim, c.subset, c.metric, label, severity, kind))  # fmt: skip
        row.pages += 1
        row.dim_points += points
        row.examples.append((points, c.example_id))

    for c in selected:
        dim, slot_weight = weights[(c.subset, c.metric)]
        w = slot_weight / per_slot[(c.subset, c.metric)] * 100.0
        score = c.score or 0.0
        if c.metric not in METRIC_KEYS:  # layout
            for reason, share in layout_rows(c).items():
                add(c, dim, reason, "", "layout", w * share)
            continue
        if c.error or "proportional" not in c.metadata:
            add(c, dim, FAILED, "", "failed", w * (1.0 - score))
            continue
        parsing = parsing_bound(c.metric, c.metadata)
        again = recompute(c.metric, c.metadata, base)
        if abs(again - score) > 1e-6:
            problems.append(f"{c.example_id} {c.metric}: harness {score:.6f} vs recomputed {again:.6f}")
        add(c, dim, CONTENT, "", "content", w * (1.0 - parsing))
        add(c, dim, CAP, "", "cap", w * (parsing - score))
        key = METRIC_KEYS[c.metric]
        for label, severity in case_labels(c.metric, c.metadata):
            policy = policies.get((key, label))
            if policy is None:
                policy = policies[(key, label)] = policy_without(label, key)
            add(c, dim, label, severity, "event", w * (recompute(c.metric, c.metadata, policy) - again))
    out = sorted(rows.values(), key=lambda r: -r.dim_points)
    for row in out:
        row.examples = sorted(row.examples, reverse=True)[:top_examples]
    return out, problems


def table_for_run(ledger: Ledger, run_id: str, fold: str = "all") -> tuple[list[LossRow], list[str]]:
    return loss_table(ledger.cases(run_id), fold)


def check_sums(rows: list[LossRow], dimensions: dict[str, float]) -> dict[str, float]:
    """Per dimension: (content + cap + failed + layout rows) - (100 - score). Should be ~0."""
    lost: dict[str, float] = defaultdict(float)
    for r in rows:
        if r.kind != "event":
            lost[r.dimension] += r.dim_points
    return {d: lost[d] - (100.0 - s) for d, s in dimensions.items() if d != "overall"}


def render_markdown(rows: list[LossRow], limit: int = 60) -> str:
    lines = ["| # | Dimension | Subset | Error type | Severity | Pages | Dimension pts | Overall pts |",
             "|---:|---|---|---|---|---:|---:|---:|"]  # fmt: skip
    for i, r in enumerate(rows[:limit], start=1):
        lines.append(f"| {i} | {r.dimension} | {r.subset} | {r.label} | {r.severity or '—'} | {r.pages} | "
                     f"{r.dim_points:.2f} | {r.overall_points:.3f} |")  # fmt: skip
    return "\n".join(lines)
