"""Fast 150-page stratified multi-benchmark pilot slice across olmOCR-bench, RealDocBench, and ParseBench 2.

Every new perception upgrade (cropped high-res reading, scan restoration, 2nd model routing, vector-hinted
chart VLM) is tested here FIRST (~30s CPU / ~$0.30 Modal or $0 Colab) before ever touching full datasets.

    uv run python -m evalkit.pilot150 build
    uv run python -m evalkit.pilot150 plan  --config configs/clean_base.yaml --manifest /tmp/pilot-manifest
    uv run python -m evalkit.pilot150 eval  --config configs/clean_base.yaml --name clean_base
    uv run python -m evalkit.pilot150 compare plain clean_base
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from docparse.core.config import Config
from docparse.core.errors import ParseError
from docparse.models.cache import Cache
from docparse.models.calls import CacheMiss, ModelCall
from docparse.models.client import ReplayClient
from docparse.models.manifest import ManifestWriter
from docparse.pipeline import parse
from docparse.write.markdown import write_page
from evalkit import paths
from evalkit.explode import split_example_id
from evalkit.external import forms_outside as fo
from evalkit.external import olmocr_bench as ob
from evalkit.run import _PlanningClient, run_pipeline
from evalkit.score import build_command
from evalkit.splits import SPLIT_FILE

PILOT_FILE = Path(__file__).parent / "splits_data" / "pilot150.json"
PILOT_RUNS = paths.runs_dir() / "pilot150"
_FAIL_RE = re.compile(r"^\s+\[FAIL\] Test (.+?) on (.+?) page (\d+) average pass ratio: ([\d.]+)")


def _olm_plain_fails() -> set[str]:
    log = ob.out_dir() / "scores" / "docparse_plain.log"
    out = set()
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if (m := _FAIL_RE.match(line)) and float(m.group(4)) < 0.5:
            out.add(m.group(1))
    return out


def _pick_spaced(items: list[Any], n: int) -> list[Any]:
    """Deterministically pick `n` evenly spaced items from a sorted list."""
    if len(items) <= n:
        return list(items)
    if n == 1:
        return [items[0]]
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def build_pilot() -> dict[str, Any]:
    """Construct the fixed 150-page stratified multi-benchmark hard slice (PB2 strictly from dev split)."""
    olm_data = ob.data_dir()
    olm_fails = _olm_plain_fails()

    # 1. olm_old_scans (30 PDFs): stratified by fail rate among failing old_scans PDFs
    os_rows = ob.read_jsonl(olm_data / "old_scans.jsonl")
    by_pdf: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in os_rows:
        by_pdf[r["pdf"]][0] += int(r["id"] in olm_fails)
        by_pdf[r["pdf"]][1] += 1
    failing_os = sorted(
        [(f / t, f, pdf) for pdf, (f, t) in by_pdf.items() if f > 0],
        key=lambda x: (-x[0], -x[1], x[2]),
    )
    olm_old_scans = [x[2] for x in _pick_spaced(failing_os, 30)]

    # 2. olm_mc_tbl_math (30 PDFs): 15 multi_column + 10 table_tests + 5 arxiv_math
    def _top_failing_pdfs(cat: str, k: int, must_include: tuple[str, ...] = ()) -> list[str]:
        rows = ob.read_jsonl(olm_data / f"{cat}.jsonl")
        counts: Counter[str] = Counter(r["pdf"] for r in rows if r["id"] in olm_fails)
        chosen = [p for p in must_include if (olm_data / "pdfs" / p).exists()]
        rest = sorted([(c, p) for p, c in counts.items() if p not in chosen], key=lambda x: (-x[0], x[1]))
        chosen += [p for _, p in _pick_spaced(rest, k - len(chosen))]
        return sorted(chosen[:k])

    olm_mc = _top_failing_pdfs("multi_column", 15)
    olm_tbl = _top_failing_pdfs(
        "table_tests",
        10,
        must_include=(
            "tables/033c1efd2c0e15b829f9e9164f0e7bc1fecf_pg5_pg1.pdf",
            "tables/11e12a3ddca3b1743f6ec1c22a52cdbf9ffe_pg8_pg1.pdf",
        ),
    )
    olm_math = _top_failing_pdfs(
        "arxiv_math",
        5,
        must_include=(
            "arxiv_math/2503.06267_pg10.pdf",
            "arxiv_math/2503.07910_pg7.pdf",
            "arxiv_math/2503.04932_pg20.pdf",
        ),
    )
    olm_mc_tbl_math = olm_mc + olm_tbl + olm_math

    # 3. realdoc_hard (30 docs): 15 scanned + 10 chart/figure + 5 digital non-chart failing docs
    qa = fo.load_items()
    prof = json.loads((fo.out_dir() / "profile.json").read_text(encoding="utf-8"))
    stems = sorted(prof)
    plain_md = {s: "\n\n".join(fo._pages("plain", s)) for s in stems}
    rows = fo.score_items([it for it in qa if it["source_file"] in prof], {"plain": plain_md})["plain"]
    misses_by_doc: Counter[str] = Counter()
    chart_docs = {
        it["source_file"]
        for it in qa
        if it["source_file"] in prof and any(c.startswith(("chart_", "figure_")) for c in it.get("capabilities", []))
    }
    for r in rows:
        misses_by_doc[r["doc"]] += r["fields"] - r["loose"]

    scan_fail = sorted(
        [(m, s) for s, m in misses_by_doc.items() if m > 0 and prof[s]["text_pages"] == 0 and s not in chart_docs],
        key=lambda x: (-x[0], x[1]),
    )
    chart_fail = sorted(
        [(m, s) for s, m in misses_by_doc.items() if m > 0 and s in chart_docs],
        key=lambda x: (-x[0], x[1]),
    )
    digi_fail = sorted(
        [(m, s) for s, m in misses_by_doc.items() if m > 0 and prof[s]["text_pages"] > 0 and s not in chart_docs],
        key=lambda x: (-x[0], x[1]),
    )
    rd_must_scan = [
        s
        for s in ("finance_93", "finance_109", "finance_103", "finance_106", "finance_107")
        if s in prof and prof[s]["text_pages"] == 0 and s not in chart_docs
    ]
    rd_scan = rd_must_scan + [
        s for _, s in _pick_spaced([(m, s) for m, s in scan_fail if s not in rd_must_scan], 15 - len(rd_must_scan))
    ]
    rd_chart = [s for _, s in _pick_spaced(chart_fail, 10)]
    rd_digi = [s for _, s in _pick_spaced(digi_fail, 5)]
    realdoc_hard = sorted(set(rd_scan[:15] + rd_chart[:10] + rd_digi[:5]))

    # 4 & 5. ParseBench 2 (60 cases strictly from dev split, never check)
    split_raw = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    split = split_raw["folds"] if "folds" in split_raw else split_raw
    ps_files = sorted((paths.data_root() / "pagestats").glob("*.json"))
    ps = json.loads(ps_files[0].read_text(encoding="utf-8"))
    conn = sqlite3.connect(paths.data_root() / "ledger.sqlite")
    pb_rows = conn.execute(
        "SELECT subset, metric, example_id, score FROM case_scores WHERE run_id='own-best5' AND counted=1"
    ).fetchall()
    conn.close()

    def _pb_pick(pred, k: int) -> list[str]:
        cand = sorted(
            [
                (sc, f"{sub}/{eid}" if "/" not in eid else eid)
                for sub, met, eid, sc in pb_rows
                if pred(sub, met, eid, sc)
            ],
            key=lambda x: (x[0], x[1]),
        )
        return [eid for _, eid in _pick_spaced(cand, k)]

    def _is_dev(sub: str, eid: str) -> bool:
        full = eid if "/" in eid else f"{sub}/{eid}"
        return split.get(full) == "dev"

    def _is_img(sub: str, eid: str) -> bool:
        full = eid if "/" in eid else f"{sub}/{eid}"
        return int(ps.get(full, 0)) < 50

    pb_cth = _pb_pick(
        lambda s, m, e, sc: s == "complex_table_headers" and _is_dev(s, e) and _is_img(s, e) and sc < 0.85, 15
    )
    pb_dense = _pb_pick(
        lambda s, m, e, sc: s in ("dense_tables", "insurance_filing_tables") and _is_dev(s, e) and sc < 0.70, 10
    )
    pb_text = _pb_pick(
        lambda s, m, e, sc: s == "text" and m == "text_next" and _is_dev(s, e) and _is_img(s, e) and sc < 0.85, 5
    )
    pb_charts = _pb_pick(lambda s, m, e, sc: s == "charts" and _is_dev(s, e) and sc < 0.70, 30)

    pilot = {
        "version": 1,
        "counts": {
            "olm_old_scans": len(olm_old_scans),
            "olm_mc_tbl_math": len(olm_mc_tbl_math),
            "realdoc_hard": len(realdoc_hard),
            "pb2_hard_tables_text": len(pb_cth) + len(pb_dense) + len(pb_text),
            "pb2_hard_charts": len(pb_charts),
        },
        "olm_old_scans": olm_old_scans,
        "olm_mc_tbl_math": olm_mc_tbl_math,
        "realdoc_hard": realdoc_hard,
        "pb2_hard_tables_text": pb_cth + pb_dense + pb_text,
        "pb2_hard_charts": pb_charts,
    }
    pilot["total_pages"] = sum(pilot["counts"].values())
    PILOT_FILE.write_text(json.dumps(pilot, indent=2) + "\n", encoding="utf-8")
    return pilot


def load_pilot() -> dict[str, Any]:
    return json.loads(PILOT_FILE.read_text(encoding="utf-8"))


def _parse_md_job(job: tuple[str, str, str, str | None]) -> dict[str, Any]:
    path, config_json, cache_root, manifest_dir = job
    cfg = Config.model_validate_json(config_json)
    cache = Cache(Path(cache_root))
    client = _PlanningClient(cache, Path(manifest_dir)) if manifest_dir else ReplayClient(cache)
    try:
        res = parse(Path(path).read_bytes(), cfg, client)
    except (ParseError, CacheMiss) as exc:
        return {"path": path, "error": f"{type(exc).__name__}: {exc}"}
    if res.document is None:
        return {"path": path, "pending": True, "misses": client.misses}
    return {"path": path, "pages": [write_page(p) for p in res.document.pages]}


def cmd_plan(config_path: Path, manifest_dir: Path, overrides: list[str], workers: int) -> int:
    pilot = load_pilot()
    cfg = Config.load(config_path, overrides)
    cfg_json = cfg.model_dump_json()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    writer = ManifestWriter(manifest_dir)
    before = len(writer)

    # 1. olmOCR + RealDoc PDFs
    olm_pdfs = [str(ob.data_dir() / "pdfs" / p) for p in pilot["olm_old_scans"] + pilot["olm_mc_tbl_math"]]
    rd_pdfs = [str(fo.data_dir() / "docs" / f"{s}.pdf") for s in pilot["realdoc_hard"]]
    jobs = [(p, cfg_json, str(paths.cache_dir()), str(manifest_dir)) for p in olm_pdfs + rd_pdfs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_parse_md_job, jobs, chunksize=2))
    added = 0
    for res in results:
        for call_json, meta in res.get("misses", []):
            added += writer.add(ModelCall.model_validate(call_json), {}, meta)

    # 2. PB2 cases
    pb_ids = pilot["pb2_hard_tables_text"] + pilot["pb2_hard_charts"]
    tmp_out = manifest_dir / "_plan_pb2"
    pb_sum = run_pipeline(
        cfg,
        "pilot",
        tmp_out,
        paths.gt_dir(),
        paths.cache_dir(),
        example_ids=pb_ids,
        manifest_dir=manifest_dir,
        workers=workers,
    )
    shutil.rmtree(tmp_out, ignore_errors=True)
    print(
        f"pilot150 plan: {pilot['total_pages']} pages | new calls={len(writer) - before} "
        f"(total manifest={len(writer)}, pb2_pending={pb_sum.pending}) at {manifest_dir}"
    )
    return 0


def _score_olm_subset(
    run_dir: Path, name: str, pdf_rels: list[str], pages_by_pdf: dict[str, list[str]]
) -> dict[str, Any]:
    bench = run_dir / "olm_bench"
    if bench.exists():
        shutil.rmtree(bench)
    (bench / "pdfs").mkdir(parents=True, exist_ok=True)
    keep = set(pdf_rels)
    data = ob.data_dir()
    cat_files = ("old_scans", "multi_column", "table_tests", "arxiv_math")
    for cat in cat_files:
        rows = [r for r in ob.read_jsonl(data / f"{cat}.jsonl") if r["pdf"] in keep]
        if not rows:
            continue
        (bench / f"{cat}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        for d in ob.category_dirs(rows):
            dst_dir = bench / "pdfs" / d
            dst_dir.mkdir(parents=True, exist_ok=True)
            for rel in {r["pdf"] for r in rows}:
                target = bench / "pdfs" / rel
                if not target.exists():
                    target.symlink_to(data / "pdfs" / rel)
    for rel, mds in pages_by_pdf.items():
        for idx, md in enumerate(mds, 1):
            t = bench / name / ob.md_name(rel, idx)
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_text(md, encoding="utf-8")
    log_path = run_dir / f"olm_{name}.log"
    out = ob.run_scorer(bench, name, log_path)
    parsed = ob.parse_summary(out)[name]["files"]
    return {
        k.removesuffix(".jsonl"): {"passed": p, "total": t, "pct": round(100.0 * p / t, 2) if t else 0.0}
        for k, (p, t) in parsed.items()
    }


def _score_pb2_subset(run_dir: Path, name: str, cfg: Config, pb_ids: list[str], workers: int) -> dict[str, Any]:
    out_root = run_dir / "pb2_out"
    gt_view = run_dir / "pb2_gt"
    rep_dir = run_dir / "pb2_reports"
    for d in (out_root, gt_view, rep_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    gt_real = paths.gt_dir()
    sources: dict[str, dict[str, str]] = {}
    for eid in pb_ids:
        sub, stem = split_example_id(eid)
        (gt_view / sub).mkdir(parents=True, exist_ok=True)
        for src in (gt_real / sub).glob(f"{stem}.*"):
            dst = gt_view / sub / src.name
            if not dst.exists():
                dst.symlink_to(src)
        sources[sub] = {
            "output_dir": str((out_root / sub / name).resolve()),
            "test_cases_dir": str((gt_view / sub).resolve()),
        }
    run_sum = run_pipeline(cfg, name, out_root, gt_real, paths.cache_dir(), example_ids=pb_ids, workers=workers)
    if run_sum.errors:
        raise RuntimeError(f"PB2 pilot parse errors: {run_sum.errors[:5]}")
    cmd = build_command(out_root, sources, rep_dir)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"PB2 scorer failed ({proc.returncode}): {proc.stderr[-500:]}")
    by_sub: dict[str, Any] = {}
    primary = {"charts": "chart_next", "text": "text_next"}
    for sub, src in sources.items():
        rep = json.loads((Path(src["output_dir"]) / "_evaluation_report.json").read_text(encoding="utf-8"))
        want = primary.get(sub, "table_next")
        scores: list[float] = []
        for r in rep.get("per_example_results") or []:
            if not r.get("success"):
                continue
            for m in r.get("metrics") or []:
                if m.get("metric_name") == want and m.get("value") is not None:
                    scores.append(float(m["value"]))
                    break
        by_sub[sub] = {"n": len(scores), "mean": round(100.0 * sum(scores) / len(scores), 2) if scores else 0.0}
    return by_sub


def cmd_eval(config_path: Path, name: str, overrides: list[str], workers: int) -> int:
    pilot = load_pilot()
    cfg = Config.load(config_path, ["layout=none", *overrides])
    cfg_json = cfg.model_dump_json()
    run_dir = PILOT_RUNS / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. olmOCR-bench 60 pages
    olm_rels = pilot["olm_old_scans"] + pilot["olm_mc_tbl_math"]
    olm_jobs = [(str(ob.data_dir() / "pdfs" / p), cfg_json, str(paths.cache_dir()), None) for p in olm_rels]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        olm_res = list(pool.map(_parse_md_job, olm_jobs, chunksize=2))
    olm_pages: dict[str, list[str]] = {}
    for rel, res in zip(olm_rels, olm_res, strict=True):
        if "error" in res:
            raise RuntimeError(f"olmOCR pilot miss/error on {rel}: {res['error']}")
        olm_pages[rel] = res["pages"]
    olm_scores = _score_olm_subset(run_dir, name, olm_rels, olm_pages)

    # 2. RealDocBench 30 pages
    rd_stems = pilot["realdoc_hard"]
    rd_jobs = [(str(fo.data_dir() / "docs" / f"{s}.pdf"), cfg_json, str(paths.cache_dir()), None) for s in rd_stems]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rd_res = list(pool.map(_parse_md_job, rd_jobs, chunksize=2))
    rd_md: dict[str, str] = {}
    for s, res in zip(rd_stems, rd_res, strict=True):
        if "error" in res:
            raise RuntimeError(f"RealDoc pilot miss/error on {s}: {res['error']}")
        rd_md[s] = "\n\n".join(res["pages"])
    qa = [it for it in fo.load_items() if it["source_file"] in set(rd_stems)]
    rd_rows = fo.score_items(qa, {name: rd_md})[name]
    prof = json.loads((fo.out_dir() / "profile.json").read_text(encoding="utf-8"))
    chart_docs = {
        it["source_file"] for it in qa if any(c.startswith(("chart_", "figure_")) for c in it.get("capabilities", []))
    }
    rd_scores = {
        "all": fo._summary(rd_rows),
        "scanned": fo._summary(
            [r for r in rd_rows if prof[r["doc"]]["text_pages"] == 0 and r["doc"] not in chart_docs]
        ),
        "charts": fo._summary([r for r in rd_rows if r["doc"] in chart_docs]),
        "digital": fo._summary([r for r in rd_rows if prof[r["doc"]]["text_pages"] > 0 and r["doc"] not in chart_docs]),
    }

    # 3. PB2 60 pages
    pb_ids = pilot["pb2_hard_tables_text"] + pilot["pb2_hard_charts"]
    pb_scores = _score_pb2_subset(run_dir, name, cfg, pb_ids, workers)

    summary = {
        "name": name,
        "config": str(config_path),
        "overrides": overrides,
        "olmocr": olm_scores,
        "realdoc": rd_scores,
        "pb2": pb_scores,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _print_summary(summary)
    return 0


def _print_summary(s: dict[str, Any]) -> None:
    print(f"\n=== PILOT-150 SCOREBOARD: {s['name']} ({s['config']}) ===")
    print("  [olmOCR-bench 60 hard pages]")
    for k, v in sorted(s["olmocr"].items()):
        print(f"    {k:24s}: {v['pct']:6.2f}% ({v['passed']}/{v['total']})")
    print("  [RealDocBench 30 hard docs]")
    for k in ("all", "scanned", "charts", "digital"):
        v = s["realdoc"][k]
        ex, ls, nf = v["values found"], v["values found, loose"], v["fields"]
        print(f"    {k:24s}: exact={ex:6.2f}%  loose={ls:6.2f}%  (fields={nf})")
    print("  [ParseBench 2 60 hard dev pages]")
    for k, v in sorted(s["pb2"].items()):
        print(f"    {k:24s}: {v['mean']:6.2f}  (n={v['n']})")


def cmd_compare(name_a: str, name_b: str) -> int:
    sa = json.loads((PILOT_RUNS / name_a / "summary.json").read_text(encoding="utf-8"))
    sb = json.loads((PILOT_RUNS / name_b / "summary.json").read_text(encoding="utf-8"))
    print(f"\n=== PILOT-150 COMPARISON: {name_a} -> {name_b} ===")
    print(f"{'Metric':34s} {name_a:>12s} {name_b:>12s} {'Delta':>9s}")
    print("-" * 71)
    for k in sorted(set(sa["olmocr"]) | set(sb["olmocr"])):
        a, b = sa["olmocr"].get(k, {}).get("pct", 0.0), sb["olmocr"].get(k, {}).get("pct", 0.0)
        print(f"olmocr/{k:27s} {a:12.2f} {b:12.2f} {b - a:+9.2f}")
    for k in ("all", "scanned", "charts", "digital"):
        a = sa["realdoc"][k]["values found, loose"]
        b = sb["realdoc"][k]["values found, loose"]
        print(f"realdoc/{k:26s} {a:12.2f} {b:12.2f} {b - a:+9.2f}")
    for k in sorted(set(sa["pb2"]) | set(sb["pb2"])):
        a, b = sa["pb2"].get(k, {}).get("mean", 0.0), sb["pb2"].get(k, {}).get("mean", 0.0)
        print(f"pb2/{k:30s} {a:12.2f} {b:12.2f} {b - a:+9.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalkit.pilot150")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--config", type=Path, required=True)
    p_plan.add_argument("--manifest", type=Path, required=True)
    p_plan.add_argument("--set", dest="overrides", action="append", default=[])
    p_plan.add_argument("--workers", type=int, default=8)
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--config", type=Path, required=True)
    p_eval.add_argument("--name", required=True)
    p_eval.add_argument("--set", dest="overrides", action="append", default=[])
    p_eval.add_argument("--workers", type=int, default=8)
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("a")
    p_cmp.add_argument("b")
    args = parser.parse_args(argv)
    if args.cmd == "build":
        p = build_pilot()
        print(f"Built pilot150 ({p['total_pages']} pages) -> {PILOT_FILE}: {p['counts']}")
        return 0
    if args.cmd == "plan":
        return cmd_plan(args.config, args.manifest, args.overrides, args.workers)
    if args.cmd == "eval":
        return cmd_eval(args.config, args.name, args.overrides, args.workers)
    return cmd_compare(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
