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


def _published_path(pipeline: str = pins.OCRIA_PIPELINE):
    return paths.data_root() / "published" / pins.RESULTS_REVISION / pipeline / "results" / pipeline


def _cmd_standin(args: argparse.Namespace) -> int:
    from docparse.models.cache import Cache
    from evalkit import fetch
    from evalkit.standin import import_published, report_json

    gt = fetch.fetch_ground_truth()
    fetch.fetch_published(pins.OCRIA_PIPELINE)
    report = import_published(_published_path(), gt, Cache(paths.cache_dir()), workers=args.workers)
    print(report_json(report))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from docparse.core.config import Config
    from evalkit import fetch
    from evalkit.run import list_examples, run_pipeline

    gt = fetch.fetch_ground_truth()
    config = Config.load(args.config)
    run_dir = paths.runs_dir() / args.run_id
    subsets = tuple(args.subsets.split(",")) if args.subsets else None
    ids = list_examples(gt, subsets)
    if args.pilot:
        from evalkit.pilot import pilot_ids

        ids = pilot_ids(ids)
    if args.limit:
        ids = ids[:: max(1, len(ids) // args.limit)][: args.limit]
    manifest = run_dir / "manifest" if args.plan else None
    summary = run_pipeline(config, args.name, run_dir / "output" / "parsebench-2", gt, paths.cache_dir(), ids,
                           manifest_dir=manifest, workers=args.workers)  # fmt: skip
    print(f"run {args.run_id} ({args.name}, config {config.config_hash()}): {summary.written} written, "
          f"{summary.pending} pending ({summary.new_calls} new calls), {len(summary.errors)} errors")  # fmt: skip
    for example_id, error in summary.errors[:20]:
        print(f"  error {example_id}: {error}")
    if summary.lint:
        print(f"  lint: {dict(summary.lint)}")
    if manifest and summary.new_calls:
        print(f"  manifest: {manifest}  (python -m docparse cache plan --manifest ... before any GPU run)")
    return 1 if summary.errors else 0


def _cmd_score(args: argparse.Namespace) -> int:
    from docparse.core.config import Config
    from docparse.models.cache import Cache
    from evalkit import fetch
    from evalkit.ledger import Ledger, RunRow, code_sha, ingest
    from evalkit.score import score_output

    gt = fetch.fetch_ground_truth()
    run_dir = paths.runs_dir() / args.run_id
    ids = None
    if args.pilot:  # score only the pilot pages: the harness sees a ground-truth folder holding just those cases
        from evalkit.compare import gt_view
        from evalkit.pilot import pilot_ids
        from evalkit.run import list_examples

        ids = pilot_ids(list_examples(gt))
        gt = gt_view(gt, ids, run_dir / "gt_view")
    score = score_output(run_dir / "output" / "parsebench-2", args.name, gt, run_dir / "reports")
    ledger = Ledger(paths.ledger_path())
    config_hash = Config.load(args.config).config_hash() if args.config else ""
    run = RunRow(
        run_id=args.run_id, kind="pipeline", pipeline=args.name, harness_commit=pins.HARNESS_COMMIT,
        data_revision=pins.DATA_REVISION, overall=score.overall, dimensions=score.dimensions, code_sha=code_sha(),
        config_hash=config_hash, cache_snapshot=Cache(paths.cache_dir()).snapshot_hash(), notes=args.notes,
    )  # fmt: skip
    cases = ingest(score, run, ledger)
    print(f"run {args.run_id}: {cases} case rows in {paths.ledger_path()}")
    baseline = None
    if args.baseline:
        base = ledger.run(args.baseline)
        baseline = {"overall": base.overall, **base.dimensions}
    if ids is None:
        print_scores(score, baseline, "baseline")
        return 0
    from evalkit.compare import compare_runs

    out = compare_runs(ledger, args.run_id, args.baseline, ids)
    print(f"{'dimension':12}{'ours':>8}{'baseline':>10}{'delta':>8}   (the same {len(ids)} pages in both runs)")
    for dim, v in out["dimensions"].items():
        print(f"{dim:12}{v['ours']:8.2f}{v['baseline']:10.2f}{v['ours'] - v['baseline']:+8.2f}")
    for row in out["slots"]:
        print(f"  {row['subset']:26}{row['metric']:28}{row['pages']:3d}{row['ours']:8.2f}{row['baseline']:8.2f}"
              f"{row['delta']:+8.2f}")  # fmt: skip
    return 0


def _cmd_losses(args: argparse.Namespace) -> int:
    from evalkit.ledger import Ledger
    from evalkit.report import loss_report, scores_markdown, subsets_markdown

    ledger = Ledger(paths.ledger_path())
    body, problems = loss_report(ledger, args.run_id, args.fold)
    ref = pins.BOARD_OCRIA if ledger.run(args.run_id).pipeline == pins.OCRIA_PIPELINE else None
    text = "\n\n".join([scores_markdown(ledger, args.run_id, ref), subsets_markdown(ledger, args.run_id), body])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    for line in problems[:20]:
        print("MISMATCH", line)
    return 1 if problems else 0


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
    r = sub.add_parser("run", help="parse benchmark cases with the pipeline and write result records")
    r.add_argument("--config", required=True)
    r.add_argument("--name", required=True, help="pipeline name of the output folders")
    r.add_argument("--run-id", required=True)
    r.add_argument("--subsets", default="")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--pilot", action="store_true", help="only the fixed pilot sample of development-part pages")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--plan", action="store_true", help="turn cache misses into a manifest instead of errors")
    r.set_defaults(func=_cmd_run)
    sc = sub.add_parser("score", help="score a run with the harness and add it to the ledger")
    sc.add_argument("--run-id", required=True)
    sc.add_argument("--name", required=True)
    sc.add_argument("--config", default="")
    sc.add_argument("--baseline", default="")
    sc.add_argument("--notes", default="")
    sc.add_argument("--pilot", action="store_true", help="score only the pilot pages (needs --baseline)")
    sc.set_defaults(func=_cmd_score)
    lo = sub.add_parser("losses", help="loss table of a run in the ledger")
    lo.add_argument("--run-id", required=True)
    lo.add_argument("--fold", choices=["all", "dev", "check"], default="all")
    lo.add_argument("--out", default="")
    lo.set_defaults(func=_cmd_losses)
    st = sub.add_parser("standin", help="import OCRia-1's published outputs into the cache as stand-in readings")
    st.add_argument("--workers", type=int, default=8)
    st.set_defaults(func=_cmd_standin)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
