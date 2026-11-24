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
import re
from collections import Counter, defaultdict
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
    # "content" | "cap" | "failed" | "layout": these add up to the lost score
    # "event": counterfactual gain of one policy label; "kind": a class's gain shared among its event kinds
    # "content_part": the content row shared among the metric's sub-scores
    kind: str
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


_KIND_NOISE = re.compile(r" \(systematic, \d+x\)$| \(\d+% of the reference tokens not found\)$")
TEXT_CLASS_LABEL = {"critical": "all critical errors", "major": "all major errors", "minor": "all minor errors",
                    "moved": "moved block (reading order)", "broken": "cut sentence (reading order)"}  # fmt: skip
# text kinds that never occur alone on a page, so the data cannot vote on their class
TEXT_KIND_FALLBACK = {"blocks out of order": "moved", "reading order": "moved", "sentences cut": "broken",
                      "negator in changed text": "critical", "invented passage": "critical",
                      "numbers swapped": "critical", "checkbox flipped": "critical"}  # fmt: skip


def clean_kind(kind: str) -> str:
    return _KIND_NOISE.sub("", kind)


def kind_classes(cases: list[CaseRow], metric: str) -> dict[str, str]:
    """Event kind -> its severity class, for the metrics whose policy counts are per class only (text, formatting).

    Taken from the saved examples when they carry a severity, else voted by the pages on which only one class
    occurs; a few rare text kinds fall back to a fixed table.
    """
    votes: dict[str, Counter] = defaultdict(Counter)
    for c in cases:
        if c.metric != metric or "events" not in c.metadata:
            continue
        kinds = c.metadata.get("event_kinds") or {}
        for ex in c.metadata.get("event_examples") or []:
            if ex.get("severity") in ("critical", "major", "minor") and ex.get("charged", True):
                votes[clean_kind(ex["kind"])][ex["severity"]] += 3
        nonzero = [k for k, v in c.metadata["events"].items() if v]
        if len(nonzero) == 1:
            for kind, n in kinds.items():
                votes[clean_kind(kind)][nonzero[0]] += n
    out = {kind: counter.most_common(1)[0][0] for kind, counter in votes.items()}
    if metric == "text_next":
        for kind, cls in TEXT_KIND_FALLBACK.items():
            out[kind] = cls if kind in ("blocks out of order", "sentences cut") else out.get(kind, cls)
    return out


LANE_NAMES = {
    "filled": "filled values", "empty": "blank fields", "checkboxes": "check boxes", "signatures": "signatures",
    "ids": "printed ids", "extra": "extra fields", "inline": "inline styles (bold, italic, ...)",
    "headings": "headings", "lists": "lists", "strikeout": "strikeout", "underline": "underline",
}  # fmt: skip


def _num(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 1.0


def content_parts(metric: str, metadata: dict) -> dict[str, float]:
    """Shares (summing to 1) of a page's content loss, from the metric's own sub-scores.

    The harness does not publish how sub-scores combine, so the content loss is shared in proportion to each
    sub-score's shortfall. It says which kind of content is missing, not an exact attribution.
    """
    key = METRIC_KEYS[metric]
    short: dict[str, float] = {}
    if key in ("fmt", "forms"):
        if metadata["coverage"] / 0.5 < metadata["proportional"]:
            return {"under half of the items found (coverage cap)": 1.0}
        for lane, value in (metadata.get("lanes") or {}).items():
            if value is not None and value < 1:
                short[LANE_NAMES.get(lane, lane)] = 1 - value
    elif key in ("charts", "text"):
        what = "points" if key == "charts" else "words"
        short = {f"{what} missing (recall)": 1 - _num(metadata.get("recall")),
                 f"{what} extra or wrong (precision)": 1 - _num(metadata.get("precision"))}  # fmt: skip
    elif key == "tables":
        short = {"cell content": 1 - _num(metadata.get("content")), "row order": 1 - _num(metadata.get("row_order"))}
    short = {k: v for k, v in short.items() if v > 0}
    total = sum(short.values())
    return {k: v / total for k, v in short.items()} if total else {"unattributed": 1.0}


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
    classes = {m: kind_classes(selected, m) for m in ("text_next", "formatting_v2")}

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
        if parsing < 1:
            for part, share in content_parts(c.metric, c.metadata).items():
                add(c, dim, f"content: {part}", "", "content_part", w * (1.0 - parsing) * share)
        add(c, dim, CAP, "", "cap", w * (parsing - score))
        key = METRIC_KEYS[c.metric]
        for label, severity in case_labels(c.metric, c.metadata):
            policy = policies.get((key, label))
            if policy is None:
                policy = policies[(key, label)] = policy_without(label, key)
            gain = w * (recompute(c.metric, c.metadata, policy) - again)
            add(c, dim, label, severity, "event", gain)
            if c.metric in classes and gain > 0:  # share the class's gain among the kinds that make it up
                wanted = (
                    next((k for k, v in TEXT_CLASS_LABEL.items() if v == label), severity)
                    if key == "text"
                    else severity
                )
                kinds = {clean_kind(k): n for k, n in (c.metadata.get("event_kinds") or {}).items()
                         if classes[c.metric].get(clean_kind(k)) == wanted}  # fmt: skip
                total = sum(kinds.values())
                for kind, n in kinds.items():
                    add(c, dim, kind, severity, "kind", gain * n / total)
                if not total:
                    add(c, dim, f"{label} (kind not recorded)", severity, "kind", gain)
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
        if r.kind in ("content", "cap", "failed", "layout"):
            lost[r.dimension] += r.dim_points
    return {d: lost[d] - (100.0 - s) for d, s in dimensions.items() if d != "overall"}


def work_queue(rows: list[LossRow]) -> list[LossRow]:
    """The rows to act on: per-label gains (per-kind for text and formatting), content and layout rows."""
    out = []
    for r in rows:
        per_class_only = r.metric in ("text_next", "formatting_v2")
        if r.kind in ("content_part", "layout", "failed") or r.kind == ("kind" if per_class_only else "event"):
            out.append(r)
    return sorted(out, key=lambda r: -r.dim_points)


def merge_subsets(rows: list[LossRow]) -> list[LossRow]:
    """One row per (dimension, label): the same error type summed over a dimension's subsets."""
    merged: dict[tuple, LossRow] = {}
    for r in rows:
        m = merged.setdefault((r.dimension, r.label, r.severity, r.kind),
                              LossRow(r.dimension, "all", r.metric, r.label, r.severity, r.kind))  # fmt: skip
        m.pages += r.pages
        m.dim_points += r.dim_points
        m.examples = sorted(m.examples + r.examples, reverse=True)[:10]
    return sorted(merged.values(), key=lambda r: -r.dim_points)


def render_markdown(rows: list[LossRow], limit: int = 60) -> str:
    lines = ["| # | Dimension | Subset | Error type | Severity | Pages | Dimension pts | Overall pts |",
             "|---:|---|---|---|---|---:|---:|---:|"]  # fmt: skip
    for i, r in enumerate(rows[:limit], start=1):
        lines.append(f"| {i} | {r.dimension} | {r.subset} | {r.label} | {r.severity or '—'} | {r.pages} | "
                     f"{r.dim_points:.2f} | {r.overall_points:.3f} |")  # fmt: skip
    return "\n".join(lines)
