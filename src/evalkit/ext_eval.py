"""Any config on the two outside benchmarks (olmOCR-bench, RealDocBench): plan its model calls, write its outputs.

    uv run python -m evalkit.ext_eval paths --bench realdoc|olmocr                     # source files, one per line
    uv run python -m evalkit.ext_eval plan  --bench ... --config C --name N [--set k=v]  # -> runs/fable90/N/manifest_*
    uv run python -m evalkit.ext_eval write --bench ... --config C --name N [--set k=v]  # -> .../N/outputs.json
    uv run python -m evalkit.ext_eval score --bench ... --name N [--name M ...]          # on a machine with the scorers

`write` stores {document key: [page markdown, ...]}; `score` reads one or more of those files (so the outputs can be
produced on a GPU box and scored where the benchmarks' own scorers are installed). Every run uses `layout=none`, and
olmOCR-bench also `strip_furniture=true` (it grades the omission of running headers and footers).
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from docparse.core.config import Config
from docparse.core.errors import ParseError
from docparse.models.cache import Cache
from docparse.models.calls import CacheMiss, ModelCall
from docparse.models.client import ReplayClient
from docparse.models.manifest import ManifestWriter, read_manifest
from docparse.pipeline import parse
from docparse.write.markdown import write_page
from evalkit import paths
from evalkit.fold_eval import run_dir
from evalkit.run import _PlanningClient


def sources(bench: str) -> list[tuple[str, Path]]:
    """`(document key, path)` of the benchmark's documents."""
    if bench == "realdoc":
        from evalkit.external import forms_outside as fo

        return [(p.stem, p) for p in fo.chosen_documents()]
    from evalkit.external import olmocr_bench as ob

    data = ob.data_dir()
    return [(rel, data / "pdfs" / rel) for rel in ob.list_pdfs(data)]


def _config(args: argparse.Namespace) -> Config:
    extra = ["layout=none"] + (["strip_furniture=true"] if args.bench == "olmocr" else [])
    return Config.load(args.config, [*args.set, *extra])


def _plan_one(job: tuple[str, str, str, str, str]) -> dict:
    key, path, config_json, cache_root, manifest_dir = job
    client = _PlanningClient(Cache(Path(cache_root)), Path(manifest_dir))
    try:
        result = parse(Path(path).read_bytes(), Config.model_validate_json(config_json), client)
    except (ParseError, CacheMiss) as exc:
        return {"key": key, "error": f"{type(exc).__name__}: {exc}"}
    return {"key": key, "pending": result.document is None, "misses": client.misses}


def cmd_plan(args: argparse.Namespace) -> int:
    base = run_dir(args.name)
    mdir = base / "manifest"
    if mdir.exists():
        shutil.rmtree(mdir)
    (mdir / "blobs").mkdir(parents=True)
    for stale in base.glob("manifest_*"):
        shutil.rmtree(stale)
    cfg = _config(args).model_dump_json()
    jobs = [(k, str(p), cfg, str(paths.cache_dir()), str(mdir)) for k, p in sources(args.bench)]
    writer = ManifestWriter(mdir)
    done = pending = errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(_plan_one, jobs, chunksize=4):
            if "error" in res:
                errors += 1
            elif res["pending"]:
                pending += 1
                for call_json, meta in res["misses"]:
                    writer.add(ModelCall.model_validate(call_json), {}, meta)
            else:
                done += 1
    entries = list(read_manifest(mdir))
    for (repo, prompt), n in sorted(Counter((c.model.repo, c.prompt_id) for c, _ in entries).items()):
        print(f"  {n:6d}  {repo}  {prompt}")
    writers: dict[str, ManifestWriter] = {}
    for call, meta in entries:
        alias = call.model.repo.split("/")[-1]
        w = writers.setdefault(alias, ManifestWriter(base / f"manifest_{alias}"))
        w.add(call, {h: (mdir / "blobs" / h).read_bytes() for h in call.inputs}, meta)
    print(f"documents ready={done} pending={pending} errors={errors}; new calls={len(entries)} -> {base}")
    return 0


def _write_one(job: tuple[str, str, str, str]) -> tuple[str, list[str] | None]:
    key, path, config_json, cache_root = job
    try:
        doc = parse(
            Path(path).read_bytes(), Config.model_validate_json(config_json), ReplayClient(Cache(Path(cache_root)))
        )
    except (ParseError, CacheMiss):
        return key, None
    return key, [write_page(pg) for pg in doc.document.pages]


def cmd_write(args: argparse.Namespace) -> int:
    base = run_dir(args.name)
    base.mkdir(parents=True, exist_ok=True)
    cfg = _config(args).model_dump_json()
    jobs = [(k, str(p), cfg, str(paths.cache_dir())) for k, p in sources(args.bench)]
    out: dict[str, list[str]] = {}
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for key, pages in pool.map(_write_one, jobs, chunksize=4):
            if pages is None:
                failed += 1
            else:
                out[key] = pages
    (base / "outputs.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"written={len(out)} failed={failed} -> {base / 'outputs.json'}")
    return 0


def cmd_paths(args: argparse.Namespace) -> int:
    for _, p in sources(args.bench):
        print(p)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    outs = {n: json.loads((run_dir(n) / "outputs.json").read_text(encoding="utf-8")) for n in args.name}
    if args.bench == "realdoc":
        from evalkit.external import forms_outside as fo

        prof = json.loads((fo.out_dir() / "profile.json").read_text())
        items = [it for it in fo.load_items() if it["source_file"] in prof]
        md = {n: {k: "\n\n".join(v) for k, v in o.items()} for n, o in outs.items()}
        for n in md:  # a document the run could not write counts as empty
            for it in items:
                md[n].setdefault(it["source_file"], "")
        rows = fo.score_items(items, md)
        chart_docs = {it["source_file"] for it in items
                      if any(c.startswith(("chart_", "figure_")) for c in it.get("capabilities", []))}  # fmt: skip
        groups = {
            "all": lambda r: True,
            "scanned": lambda r: prof[r["doc"]]["text_pages"] == 0,
            "digital": lambda r: prof[r["doc"]]["text_pages"] > 0,
            "charts": lambda r: r["doc"] in chart_docs,
            "forms": lambda r: r["form"],
        }
        print(f"{'group':10s}" + "".join(f"{n[:22]:>24s}" for n in outs))
        for label, pred in groups.items():
            vals = [fo._summary([r for r in rows[n] if pred(r)])["values found, loose"] for n in outs]
            print(f"{label:10s}" + "".join(f"{v:24.2f}" for v in vals))
        names = list(outs)
        for other in names[1:]:
            per_doc: dict[str, list[int]] = {}
            for ra, rb in zip(rows[names[0]], rows[other], strict=True):
                acc = per_doc.setdefault(ra["doc"], [0, 0, 0])
                acc[0] += ra["loose"]
                acc[1] += rb["loose"]
                acc[2] += ra["fields"]
            gain, p_pos = fo.bootstrap_gain([tuple(v) for v in per_doc.values()])
            print(f"bootstrap loose gain ({other} vs {names[0]}): {gain:+.3f} pts, P(gain > 0) = {p_pos:.4f}")
        return 0
    from evalkit.external import olmocr_bench as ob

    data = ob.data_dir()
    bench = run_dir(args.name[0]) / "olm_bench"
    if bench.exists():
        shutil.rmtree(bench)
    bench.mkdir(parents=True)
    ob.build_view(data, bench, math=True)
    for n, o in outs.items():
        for rel in ob.list_pdfs(data):
            pages = o.get(rel) or [""]
            for i, text in enumerate(pages):
                target = bench / n / ob.md_name(rel, i + 1)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
    return ob.score(data, bench, tuple(outs), math=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evalkit.ext_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("paths", "plan", "write", "score"):
        p = sub.add_parser(name)
        p.add_argument("--bench", required=True, choices=["realdoc", "olmocr"])
        if name in ("plan", "write"):
            p.add_argument("--config", type=Path, required=True)
            p.add_argument("--set", action="append", default=[])
            p.add_argument("--workers", type=int, default=3)
        if name == "score":
            p.add_argument("--name", action="append", required=True)
        elif name != "paths":
            p.add_argument("--name", required=True)
    args = ap.parse_args(argv)
    return {"paths": cmd_paths, "plan": cmd_plan, "write": cmd_write, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
