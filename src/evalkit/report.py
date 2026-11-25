"""Markdown reports from the ledger: run scores per dimension, subset and fold, and the loss table."""

from __future__ import annotations

from collections import defaultdict

from evalkit.ledger import Ledger
from evalkit.losses import LossRow, dimension_scores, loss_table, merge_subsets, work_queue

DIMS = ("text", "formatting", "tables", "charts", "layout", "forms")


def scores_markdown(
    ledger: Ledger, run_id: str, reference: dict[str, float] | None = None, ref_name: str = "Board"
) -> str:
    run = ledger.run(run_id)
    cases = ledger.cases(run_id)
    dev, check = dimension_scores(cases, "dev"), dimension_scores(cases, "check")
    ours = {"overall": run.overall, **run.dimensions}
    head = "| | Overall | " + " | ".join(d.capitalize() for d in DIMS) + " |"
    lines = [head, "|---|" + "---:|" * 7]

    def row(name: str, values: dict[str, float]) -> str:  # a partial run has only some dimensions
        cells = (f"{values[k]:.2f}" if k in values else "—" for k in ("overall", *DIMS))
        return f"| {name} | " + " | ".join(cells) + " |"

    if reference:
        lines.append(row(ref_name, reference))
    lines.append(row("This run, all pages", ours))
    lines.append(row("Development part (80%)", dev))
    lines.append(row("Check slice (20%)", check))
    return "\n".join(lines)


def subsets_markdown(ledger: Ledger, run_id: str) -> str:
    run = ledger.run(run_id)
    cases = ledger.cases(run_id)
    zero: dict[tuple[str, str], int] = defaultdict(int)
    perfect: dict[tuple[str, str], int] = defaultdict(int)
    for c in cases:
        if c.counted:
            zero[(c.subset, c.metric)] += (c.score or 0.0) == 0.0
            perfect[(c.subset, c.metric)] += (c.score or 0.0) >= 0.9999
    lines = ["| Dimension | Subset | Metric | Weight | Pages | Score | Pages at 0 | Pages at 1 |",
             "|---|---|---|---:|---:|---:|---:|---:|"]  # fmt: skip
    for s in run.slots:
        key = (s["subset"], s["metric"])
        lines.append(f"| {s['dimension']} | {s['subset']} | {s['metric']} | {s['weight']:.4g} | {s['documents']} | "
                     f"{100 * s['score']:.2f} | {zero[key]} | {perfect[key]} |")  # fmt: skip
    return "\n".join(lines)


def summary_markdown(rows: list[LossRow], dimensions: dict[str, float]) -> str:
    """Per dimension: where the lost points are, by the rows that add up."""
    parts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in rows:
        if r.kind in ("content", "cap", "failed", "layout"):
            parts[r.dimension]["layout" if r.kind == "layout" else r.kind] += r.dim_points
    lines = [
        "| Dimension | Score | Lost | Content not recovered | Severity cap | Failed pages | Layout outcomes "
        "| Overall pts lost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in sorted(dimensions, key=lambda d: dimensions[d]):
        if d == "overall":
            continue
        p = parts[d]
        lost = 100 - dimensions[d]
        cell = lambda k, p=p: f"{p[k]:.2f}" if p.get(k) else "—"  # noqa: E731
        cells = " | ".join(cell(k) for k in ("content", "cap", "failed", "layout"))
        lines.append(f"| {d} | {dimensions[d]:.2f} | {lost:.2f} | {cells} | {lost / 6:.2f} |")
    return "\n".join(lines)


def queue_markdown(rows: list[LossRow], limit: int = 60, examples: int = 0) -> str:
    q = merge_subsets(work_queue(rows))
    lines = [
        "| # | Dimension | Error type | Severity | Pages | Dimension pts | Overall pts |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for i, r in enumerate(q[:limit], start=1):
        lines.append(f"| {i} | {r.dimension} | {r.label} | {r.severity or '—'} | {r.pages} | {r.dim_points:.2f} | "
                     f"{r.overall_points:.3f} |")  # fmt: skip
    if examples:
        lines.append("")
        for i, r in enumerate(q[:examples], start=1):
            pages = ", ".join(f"`{eid}` ({pts:.3f})" for pts, eid in r.examples)
            lines.append(f"{i}. **{r.dimension}: {r.label}** — {pages}")
    return "\n".join(lines)


def by_subset_markdown(rows: list[LossRow], dimension: str, limit: int = 12) -> str:
    q = [r for r in work_queue(rows) if r.dimension == dimension][:limit]
    lines = ["| Subset | Error type | Severity | Pages | Dimension pts |", "|---|---|---|---:|---:|"]
    lines += [f"| {r.subset} | {r.label} | {r.severity or '—'} | {r.pages} | {r.dim_points:.2f} |" for r in q]
    return "\n".join(lines)


def loss_report(ledger: Ledger, run_id: str, fold: str = "all") -> tuple[str, list[str]]:
    rows, problems = loss_table(ledger.cases(run_id), fold)
    dims = dimension_scores(ledger.cases(run_id), fold)
    body = [
        "## Where the points are\n",
        summary_markdown(rows, dims),
        "\n## Work queue\n",
        queue_markdown(rows, 70, 15),
    ]
    for d in DIMS:
        body += [f"\n### {d.capitalize()} by subset\n", by_subset_markdown(rows, d)]
    return "\n".join(body), problems
