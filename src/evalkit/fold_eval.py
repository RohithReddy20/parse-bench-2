"""Plan and score any config on a ParseBench 2 fold (dev / check / all, or an id list), in its own run folder.

    uv run python -m evalkit.fold_eval plan  --config configs/x.yaml --fold dev --name x-dev [--subsets charts,forms]
    uv run python -m evalkit.fold_eval score --config configs/x.yaml --fold dev --name x-dev [--subsets ...]

`plan` writes the calls the cache lacks to `<runs>/<space>/<name>/manifest` (one manifest per model repo under
`manifest_<alias>`); `score` replays the cache, writes result records and runs the harness's own scorer, then
prints the six dimensions (those whose subsets were all run) and saves `scores.json` with per-example values.
Layout cases run with the config as is; every other subset with `layout=none`, as the leaderboard run does.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from docparse.core.config import Config
from docparse.models.manifest import ManifestWriter, read_manifest
from evalkit import paths
from evalkit.explode import split_example_id
from evalkit.run import list_examples, run_pipeline
from evalkit.score import build_command

ROOT = Path(__file__).resolve().parents[2]
SPACE = "fable90"
METRICS = {
    "text": ("text_next", "formatting_v2"),
    "headings": ("formatting_v2",),
    "strikeout": ("formatting_v2",),
    "underline": ("formatting_v2",),
    "tables": ("table_next",),
    "complex_table_headers": ("table_next",),
    "financial_tables": ("table_next",),
    "insurance_filing_tables": ("table_next",),
    "dense_tables": ("table_next",),
    "charts": ("chart_next",),
    "layout": ("layout_grounding_pass_rate",),
    "forms": ("forms_v2",),
    "fillable_forms": ("forms_v2",),
}


def fold_ids(fold: str, subsets: list[str] | None = None, ids_file: Path | None = None) -> list[str]:
    if ids_file is not None:
        ids = [ln.strip() for ln in ids_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    elif fold == "all":
        ids = list_examples(paths.gt_dir())
    else:
        folds = json.loads((ROOT / "src" / "evalkit" / "splits_data" / "v1.json").read_text(encoding="utf-8"))["folds"]
        ids = sorted(eid for eid, f in folds.items() if f == fold)
    if subsets:
        ids = [e for e in ids if split_example_id(e)[0] in subsets]
    return ids


def run_dir(name: str) -> Path:
    return paths.runs_dir() / SPACE / name


def _configs(config: Path, overrides: list[str]) -> tuple[Config, Config]:
    return Config.load(config, [*overrides, "layout=none"]), Config.load(config, overrides)


def _gt_view(base: Path, ids: list[str]) -> Path:
    view = base / "gt"
    if view.exists():
        shutil.rmtree(view)
    for eid in ids:
        sub, stem = split_example_id(eid)
        (view / sub).mkdir(parents=True, exist_ok=True)
        for src in (paths.gt_dir() / sub).glob(f"{stem}.*"):
            (view / sub / src.name).symlink_to(src)
    return view


def cmd_plan(args: argparse.Namespace) -> int:
    ids = fold_ids(args.fold, args.subsets, args.ids)
    md_cfg, lay_cfg = _configs(args.config, args.set)
    base = run_dir(args.name)
    mdir, out = base / "manifest", base / "plan_out"
    for d in (mdir, out):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    for stale in base.glob("manifest_*"):  # per-model manifests of an earlier plan: results stay in the cache
        shutil.rmtree(stale)
    lay = [e for e in ids if e.startswith("layout/")]
    md = [e for e in ids if not e.startswith("layout/")]
    new = 0
    for cfg, part in ((md_cfg, md), (lay_cfg, lay)):
        if part:
            s = run_pipeline(cfg, args.name, out, paths.gt_dir(), paths.cache_dir(), part, mdir, workers=args.workers)
            new += s.new_calls
            print(f"written={s.written} pending={s.pending} new_calls={s.new_calls} errors={len(s.errors)}", flush=True)
            if s.errors:
                print("  sample errors:", s.errors[:3], flush=True)
    entries = list(read_manifest(mdir))
    by = Counter((c.model.repo, c.prompt_id) for c, _ in entries)
    for (repo, prompt), n in sorted(by.items()):
        print(f"  {n:6d}  {repo}  {prompt}")
    writers: dict[str, ManifestWriter] = {}
    for call, meta in entries:  # one manifest per model: a GPU session serves one model
        alias = call.model.repo.split("/")[-1]
        w = writers.setdefault(alias, ManifestWriter(base / f"manifest_{alias}"))
        w.add(call, {h: (mdir / "blobs" / h).read_bytes() for h in call.inputs}, meta)
    print(f"total new calls: {len(entries)} -> {base}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    ids = fold_ids(args.fold, args.subsets, args.ids)
    md_cfg, lay_cfg = _configs(args.config, args.set)
    base = run_dir(args.name)
    out, rep = base / "out", base / "rep"
    for d in (out, rep):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    gt_view = _gt_view(base, ids)
    lay = [e for e in ids if e.startswith("layout/")]
    md = [e for e in ids if not e.startswith("layout/")]
    written, errors = 0, []
    for cfg, part in ((md_cfg, md), (lay_cfg, lay)):
        if part:
            s = run_pipeline(cfg, args.name, out, paths.gt_dir(), paths.cache_dir(), part, workers=args.workers)
            written += s.written
            errors += s.errors
    print(f"written={written}/{len(ids)} errors={len(errors)}", flush=True)
    if errors:
        print("  sample errors:", errors[:5], flush=True)
    subsets = sorted({split_example_id(e)[0] for e in ids})
    sources = {
        sub: {"output_dir": str((out / sub / args.name).resolve()), "test_cases_dir": str((gt_view / sub).resolve())}
        for sub in subsets
    }
    proc = subprocess.run(build_command(out, sources, rep), capture_output=True, text=True)
    (rep / "_stdout.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    ns_path = rep / "_next_score.json"
    if not ns_path.exists():
        print((proc.stdout + proc.stderr)[-3000:])
        return 1
    ns = json.loads(ns_path.read_text())
    per_example: dict[str, dict[str, float]] = {}
    for sub in subsets:
        rpt = json.loads((out / sub / args.name / "_evaluation_report.json").read_text())
        for r in rpt["per_example_results"]:
            for m in r.get("metrics", []):
                if m["metric_name"] in METRICS.get(sub, ()) and m["value"] is not None:
                    per_example.setdefault(f"{sub}/{r['example_id'].split('/')[-1]}", {})[m["metric_name"]] = m["value"]
    dims = {}
    for d, v in ns["dimensions"].items():
        slots = v.get("slots") or []
        if slots and all(s["subset"] in subsets for s in slots) and v.get("score") is not None:
            dims[d] = 100.0 * float(v["score"])
        line = ", ".join(f"{s['subset']}={100.0 * float(s['score']):.2f}(n={s['documents']})" for s in slots
                         if s["subset"] in subsets and s.get("score") is not None)  # fmt: skip
        if line:
            print(f"  {d:11s} {dims.get(d, float('nan')):6.2f}  {line}", flush=True)
    if len(dims) == 6:
        print(f"  OVERALL     {sum(dims.values()) / 6:6.2f}", flush=True)
    (base / "scores.json").write_text(
        json.dumps({"dimensions": dims, "slots": ns["dimensions"], "per_example": per_example, "n": len(ids)})
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evalkit.fold_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "score"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--fold", default="check", choices=["dev", "check", "all"])
        p.add_argument("--name", required=True)
        p.add_argument("--subsets", type=lambda s: [x for x in s.split(",") if x])
        p.add_argument("--ids", type=Path)
        p.add_argument("--set", action="append", default=[])
        p.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)
    return cmd_plan(args) if args.cmd == "plan" else cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
