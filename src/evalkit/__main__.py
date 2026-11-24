"""`python -m evalkit <command>`."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from evalkit import paths, pins


def _cmd_fetch(args: argparse.Namespace) -> int:
    from evalkit import fetch

    if args.what == "gt":
        print(fetch.fetch_ground_truth())
    elif args.what == "published":
        print(fetch.fetch_published(args.pipeline))
    else:
        print(fetch.fetch_ocria_code())
    return 0


def print_scores(score, baseline: dict[str, float] | None, baseline_name: str = "board") -> None:
    names = ["overall", *score.dimensions]
    ours = {"overall": score.overall, **score.dimensions}
    print(f"{'':12}{'ours':>8}" + (f"{baseline_name:>9}{'delta':>8}" if baseline else ""))
    for name in names:
        line = f"{name:12}{ours[name]:8.2f}"
        if baseline and name in baseline:
            line += f"{baseline[name]:9.2f}{ours[name] - baseline[name]:+8.2f}"
        print(line)


def _cmd_score_published(args: argparse.Namespace) -> int:
    from evalkit import fetch
    from evalkit.explode import explode
    from evalkit.ledger import Ledger, RunRow, code_sha, ingest
    from evalkit.score import score_output

    gt = fetch.fetch_ground_truth()
    results = fetch.fetch_published(args.pipeline, args.revision)
    run_id = args.run_id or f"published-{args.pipeline}-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    run_dir = paths.runs_dir() / run_id
    output_root = run_dir / "output" / "parsebench-2"
    written = explode(results, output_root, args.pipeline)
    print(f"exploded {sum(written.values())} records: {written}", flush=True)
    score = score_output(output_root, args.pipeline, gt, run_dir / "reports")
    run = RunRow(
        run_id=run_id,
        kind="published",
        pipeline=args.pipeline,
        harness_commit=pins.HARNESS_COMMIT,
        data_revision=pins.DATA_REVISION,
        overall=score.overall,
        dimensions=score.dimensions,
        code_sha=code_sha(),
        notes=f"{pins.RESULTS_REPO}@{args.revision}",
    )
    ledger = Ledger(paths.ledger_path())
    cases = ingest(score, run, ledger)
    print(f"run {run_id}: {cases} case rows in {paths.ledger_path()}")
    print_scores(score, pins.BOARD_OCRIA if args.pipeline == pins.OCRIA_PIPELINE else None)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalkit")
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download pinned data into $DOCPARSE_DATA")
    f.add_argument("what", choices=["gt", "published", "ocria"])
    f.add_argument("--pipeline", default=pins.OCRIA_PIPELINE)
    f.set_defaults(func=_cmd_fetch)
    s = sub.add_parser("score-published", help="score a published pipeline's outputs with the harness")
    s.add_argument("--pipeline", default=pins.OCRIA_PIPELINE)
    s.add_argument("--revision", default=pins.RESULTS_REVISION)
    s.add_argument("--run-id", default="")
    s.set_defaults(func=_cmd_score_published)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
