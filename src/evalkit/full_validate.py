"""Full-benchmark planning, cache import, and official scoring for general Chandra-OCR-2 / Hybrid validation.

Usage:
    uv run python -m evalkit.full_validate plan --bench realdoc|olmocr|pb2_check|pb2_all
    uv run python -m evalkit.full_validate import --bench realdoc|olmocr|pb2_check|pb2_all --results results.jsonl.gz
    uv run python -m evalkit.full_validate score --bench realdoc|olmocr|pb2_check|pb2_all
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from docparse.core.config import Config
from docparse.core.errors import ParseError
from docparse.models.adapters import vlm_specialists
from docparse.models.cache import Cache
from docparse.models.calls import ModelCall, ModelRef
from docparse.models.client import ReplayClient, result_from_output
from docparse.pdf.render import render_pages
from docparse.pdf.upright import page_turn, turn_page
from docparse.pipeline import parse
from docparse.write.markdown import write_page
from evalkit import paths, pins
from evalkit.explode import split_example_id
from evalkit.external import forms_outside as fo
from evalkit.external import olmocr_bench as ob
from evalkit.standin import source_file

ROOT = Path(__file__).resolve().parents[2]
VAL_DIR = paths.runs_dir() / "full_validation"


def _plan_one(args: tuple[str, str, str, str, str, int, str, str, str]) -> dict[str, Any]:
    hf_repo, hf_rev, hf_pattern, rel, local_path, dpi, ch_model_json, runtime, cache_root = args
    ch_model = ModelRef.model_validate_json(ch_model_json)
    cache = Cache(Path(cache_root))
    try:
        data = Path(local_path).read_bytes()
        rps = render_pages(data, dpi)
    except Exception as exc:
        return {"rel": rel, "error": str(exc), "jobs": []}
    jobs: list[dict[str, Any]] = []
    for idx, stored in enumerate(rps):
        turn = page_turn(stored)
        rp = turn_page(stored, turn)
        call, _ = vlm_specialists.build_chandra_call(rp.image, model=ch_model, runtime=runtime)
        if cache.get(call.key()) is None:
            jobs.append(
                {
                    "key": call.key(),
                    "call": call.model_dump(mode="json"),
                    "hf_repo": hf_repo,
                    "hf_rev": hf_rev,
                    "hf_pattern": hf_pattern,
                    "rel": rel,
                    "page_idx": idx,
                    "turn": turn,
                    "dpi": dpi,
                }
            )
    return {"rel": rel, "error": None, "jobs": jobs}


def _pb2_check_ids() -> list[str]:
    splits = json.loads((ROOT / "src" / "evalkit" / "splits_data" / "v1.json").read_text(encoding="utf-8"))
    folds = splits["folds"]
    return sorted(eid for eid, fold in folds.items() if fold == "check")


def _pb2_all_ids() -> list[str]:
    from evalkit.run import list_examples

    return list_examples(paths.gt_dir())


def _tasks_for(bench: str, cfg: Config) -> list[tuple[str, str, str, str, str, int, str, str, str]]:
    ch_json = (cfg.chandra_model or vlm_specialists.CHANDRA2).model_dump_json()
    croot = str(paths.cache_dir())
    if bench == "realdoc":
        return [
            (fo.REPO, fo.REVISION, "docs/*.pdf", f"docs/{p.name}", str(p), cfg.dpi, ch_json, cfg.page_runtime, croot)
            for p in fo.chosen_documents()
        ]
    if bench == "olmocr":
        return [
            (
                ob.REPO,
                ob.REVISION,
                "bench_data/pdfs/**/*.pdf",
                f"bench_data/pdfs/{r}",
                str(ob.data_dir() / "pdfs" / r),
                cfg.dpi,
                ch_json,
                cfg.page_runtime,
                croot,
            )
            for r in ob.list_pdfs(ob.data_dir())
        ]
    if bench in ("pb2_check", "pb2_all"):
        ids = _pb2_check_ids() if bench == "pb2_check" else _pb2_all_ids()
        gt = paths.gt_dir()
        tasks = []
        for eid in ids:
            sub, stem = split_example_id(eid)
            src = source_file(gt, sub, stem)
            if src is None:
                continue
            rel = src.relative_to(gt).as_posix()
            tasks.append(
                (pins.DATA_REPO, pins.DATA_REVISION, "*", rel, str(src), cfg.dpi, ch_json, cfg.page_runtime, croot)
            )
        return tasks
    raise ValueError(f"Unknown bench: {bench}")


def cmd_plan(bench: str, workers: int) -> int:
    cfg = Config.load(ROOT / "configs" / "hybrid_v2.yaml", ["layout=none"])
    tasks = _tasks_for(bench, cfg)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_plan_one, tasks, chunksize=8))
    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    errs = 0
    for r in results:
        if r["error"]:
            errs += 1
        for j in r["jobs"]:
            if j["key"] not in seen:
                seen.add(j["key"])
                jobs.append(j)
    jf = VAL_DIR / f"{bench}_jobs.jsonl"
    jf.write_text("\n".join(json.dumps(j, ensure_ascii=False) for j in jobs) + ("\n" if jobs else ""), encoding="utf-8")
    kb = jf.stat().st_size / 1024
    print(f"[{bench}] docs={len(tasks)} new_chandra_calls={len(jobs)} errs={errs} -> {jf} ({kb:.1f} KB)")
    return 0


def cmd_import(bench: str, results_path: Path) -> int:
    jf = VAL_DIR / f"{bench}_jobs.jsonl"
    jobs = {
        row["key"]: row for ln in jf.read_text(encoding="utf-8").splitlines() if ln.strip() for row in [json.loads(ln)]
    }
    raw = (
        gzip.open(results_path, "rt", encoding="utf-8", errors="replace").read()
        if results_path.suffix == ".gz"
        else results_path.read_text(encoding="utf-8", errors="replace")
    )
    cache = Cache(paths.cache_dir())
    added = 0
    matched = 0
    for ln in raw.splitlines():
        if not ln.strip():
            continue
        try:
            row = json.loads(ln)
        except ValueError:
            continue
        key = row.get("key")
        if key in jobs and "output" in row and row.get("ok", True):
            matched += 1
            call = ModelCall.model_validate(jobs[key]["call"])
            meta = {"bench": bench, "rel": jobs[key]["rel"], "page": jobs[key]["page_idx"]}
            added += cache.put(result_from_output(call, row["output"]), call, meta)
    print(f"[{bench}] matched={matched}/{len(jobs)} added_to_cache={added}")
    return 0


def _write_olm_one(args: tuple[str, str, dict[str, str], str]) -> dict[str, Any]:
    path, rel, cfg_jsons, bench_dir = args
    data = Path(path).read_bytes()
    client = ReplayClient(Cache(paths.cache_dir()))
    out: dict[str, Any] = {"pdf": rel}
    for name, cjson in cfg_jsons.items():
        try:
            doc = parse(data, Config.model_validate_json(cjson), client).document
        except Exception as exc:
            out.setdefault("errors", []).append((name, str(exc)))
            t = Path(bench_dir) / name / ob.md_name(rel, 1)
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_text("", encoding="utf-8")
            continue
        for pg in doc.pages:
            t = Path(bench_dir) / name / ob.md_name(rel, pg.index + 1)
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_text(write_page(pg), encoding="utf-8")
    return out


def cmd_score(bench: str, workers: int) -> int:
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    if bench == "realdoc":
        pdfs = fo.chosen_documents()
        cache = Cache(paths.cache_dir())
        client = ReplayClient(cache)
        cfgs = {
            "plain": Config.load(ROOT / "configs" / "ocria_own.yaml", ["layout=none"]),
            "own_best": Config.load(ROOT / "configs" / "own_best.yaml", ["layout=none"]),
            "chandra_v2": Config.load(ROOT / "configs" / "chandra_v2.yaml", ["layout=none"]),
            "hybrid_v2": Config.load(ROOT / "configs" / "hybrid_v2.yaml", ["layout=none"]),
        }
        md_by_cand: dict[str, dict[str, str]] = {k: {} for k in cfgs}
        for name, cfg in cfgs.items():
            errs = 0
            for p in pdfs:
                try:
                    doc = parse(p.read_bytes(), cfg, client).document
                    md_by_cand[name][p.stem] = "\n\n".join(write_page(pg) for pg in doc.pages)
                except ParseError:
                    errs += 1
                    md_by_cand[name][p.stem] = ""
            print(f"[realdoc] parsed {name}: {len(pdfs) - errs}/{len(pdfs)} docs ({errs} errors)")

        prof = json.loads((fo.out_dir() / "profile.json").read_text(encoding="utf-8"))
        items = [it for it in fo.load_items() if it["source_file"] in prof]
        chart_docs = {
            it["source_file"]
            for it in items
            if any(c.startswith(("chart_", "figure_")) for c in it.get("capabilities", []))
        }
        rows = fo.score_items(items, md_by_cand)
        groups = {
            "all": lambda r: True,
            "scanned (no text layer)": lambda r: prof[r["doc"]]["text_pages"] == 0,
            "digital (has text layer)": lambda r: prof[r["doc"]]["text_pages"] > 0,
            "charts/figures": lambda r: r["doc"] in chart_docs,
            "form questions": lambda r: r["form"],
            **{f"domain {d}": (lambda r, d=d: r["domain"] == d) for d in sorted({r["domain"] for r in rows["plain"]})},
        }
        summary: dict[str, Any] = {}
        print("\n=== FULL REALDOCBENCH (581 docs, 1,356 Qs) — values found (loose) / questions complete (loose) ===")
        header = (
            f"{'Group':28s} {'Plain':>10s} {'own_best':>10s} {'chandra_v2':>11s} {'hybrid_v2':>11s} {'Δ vs plain':>11s}"
        )
        print(header)
        print("-" * len(header))
        for title, pred in groups.items():
            summary[title] = {n: fo._summary([r for r in rows[n] if pred(r)]) for n in cfgs}
            p_l = summary[title]["plain"]["values found, loose"]
            ob_l = summary[title]["own_best"]["values found, loose"]
            ch_l = summary[title]["chandra_v2"]["values found, loose"]
            hy_l = summary[title]["hybrid_v2"]["values found, loose"]
            print(f"{title:28s} {p_l:10.2f} {ob_l:10.2f} {ch_l:11.2f} {hy_l:11.2f} {hy_l - p_l:+11.2f}")

        # Bootstrap significance: plain vs hybrid_v2
        per_doc: dict[str, list[int]] = {}
        for ra, rb in zip(rows["plain"], rows["hybrid_v2"], strict=True):
            acc = per_doc.setdefault(ra["doc"], [0, 0, 0])
            acc[0] += ra["loose"]
            acc[1] += rb["loose"]
            acc[2] += ra["fields"]
        gain, p_pos = fo.bootstrap_gain([tuple(v) for v in per_doc.values()])
        print(f"\nBootstrap loose gain (hybrid_v2 vs plain): {gain:+.3f} pts, P(gain > 0) = {p_pos:.4f}")
        (VAL_DIR / "realdoc_full_summary.json").write_text(
            json.dumps({"groups": summary, "bootstrap_gain": gain, "p_gain": p_pos}, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0

    if bench == "olmocr":
        bench_dir = ob.out_dir() / "bench"
        cfgs = {
            "docparse_hybrid_v2": Config.load(ROOT / "configs" / "hybrid_v2.yaml", ["layout=none"]),
            "docparse_hybrid_v2_nofurn": Config.load(
                ROOT / "configs" / "hybrid_v2.yaml", ["layout=none", "strip_furniture=true"]
            ),
        }
        cfg_jsons = {k: v.model_dump_json() for k, v in cfgs.items()}
        data = ob.data_dir()
        pdfs = ob.list_pdfs(data)
        jobs = [(str(data / "pdfs" / r), r, cfg_jsons, str(bench_dir)) for r in pdfs]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            res = list(pool.map(_write_olm_one, jobs, chunksize=4))
        errs = [r for r in res if r.get("errors")]
        print(f"[olmocr] wrote {len(pdfs)} PDFs for {list(cfgs)} ({len(errs)} errors)")
        return ob.score(
            data,
            bench_dir,
            ("docparse_plain", "docparse_corrected", "docparse_hybrid_v2", "docparse_hybrid_v2_nofurn"),
            math=True,
        )

    if bench in ("pb2_check", "pb2_all"):
        import shutil
        import subprocess

        from evalkit.run import run_pipeline
        from evalkit.score import build_command

        ids = _pb2_check_ids() if bench == "pb2_check" else _pb2_all_ids()
        cfg = Config.load(ROOT / "configs" / "hybrid_v2.yaml")
        out_root = VAL_DIR / bench / "pb2_out"
        gt_view = VAL_DIR / bench / "pb2_gt"
        rep_dir = VAL_DIR / bench / "pb2_reports"
        for d in (out_root, gt_view, rep_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        gt_real = paths.gt_dir()
        sources: dict[str, dict[str, str]] = {}
        for eid in ids:
            sub, stem = split_example_id(eid)
            (gt_view / sub).mkdir(parents=True, exist_ok=True)
            for src in (gt_real / sub).glob(f"{stem}.*"):
                dst = gt_view / sub / src.name
                if not dst.exists():
                    dst.symlink_to(src)
            sources[sub] = {
                "output_dir": str((out_root / sub / "hybrid_v2").resolve()),
                "test_cases_dir": str((gt_view / sub).resolve()),
            }
        run_sum = run_pipeline(cfg, "hybrid_v2", out_root, gt_real, paths.cache_dir(), example_ids=ids, workers=workers)
        print(f"[{bench}] written={run_sum.written} pending={run_sum.pending} errors={len(run_sum.errors)}")
        if run_sum.errors:
            print("  sample errors:", run_sum.errors[:5])
        proc = subprocess.run(build_command(out_root, sources, rep_dir), capture_output=True, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr[-1000:], file=sys.stderr)
            return proc.returncode
        return 0

    raise ValueError(f"Unknown bench: {bench}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalkit.full_validate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "score"):
        p = sub.add_parser(name)
        p.add_argument("--bench", required=True, choices=["realdoc", "olmocr", "pb2_check", "pb2_all"])
        p.add_argument("--workers", type=int, default=8)
    p_imp = sub.add_parser("import")
    p_imp.add_argument("--bench", required=True, choices=["realdoc", "olmocr", "pb2_check", "pb2_all"])
    p_imp.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "plan":
        return cmd_plan(args.bench, args.workers)
    if args.cmd == "import":
        return cmd_import(args.bench, args.results)
    return cmd_score(args.bench, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
