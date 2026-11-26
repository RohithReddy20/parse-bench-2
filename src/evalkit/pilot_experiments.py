"""Plan and evaluate the 3 genuine perception-upgrade pilots on the fixed 150-page multi-benchmark slice.

uv run python -m evalkit.pilot_experiments plan
uv run python -m evalkit.pilot_experiments import --exp exp1_ocria_crops --results /path/to/results.jsonl
uv run python -m evalkit.pilot_experiments eval --exp exp1_ocria_crops
uv run python -m evalkit.pilot_experiments eval --exp exp2_qwen3vl_charts
uv run python -m evalkit.pilot_experiments eval --exp exp3_olmocr2_scans
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from docparse.base.ocria_parse import parse_page
from docparse.core.config import Config
from docparse.fixes.chart_hints import extract_chart_hint, format_chart_hint
from docparse.fixes.crop_reread import (
    chart_crops,
    column_crops,
    match_cropped_table,
    render_crop,
    stitch_bands,
    table_crops,
)
from docparse.fixes.fallback import looped, trim_loop_tail
from docparse.fixes.layout_refine import words_in_points
from docparse.models.adapters import heron, ocria, vlm_specialists
from docparse.models.cache import Cache
from docparse.models.client import ReplayClient
from docparse.models.manifest import ManifestWriter
from docparse.pdf.graphics import load_shapes
from docparse.pdf.render import RenderedPage, pixel_hash, render_pages
from docparse.pdf.scan_prep import prepare_scan, slice_horizontal_bands
from docparse.pdf.upright import page_turn, turn_page
from docparse.pipeline import parse
from docparse.write.markdown import write_page
from evalkit import paths
from evalkit.explode import split_example_id
from evalkit.external import forms_outside as fo
from evalkit.external import olmocr_bench as ob
from evalkit.pilot150 import PILOT_RUNS, _print_summary, _score_olm_subset, cmd_compare, load_pilot
from evalkit.score import build_command
from evalkit.standin import source_file

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS_DIR = PILOT_RUNS / "manifests"
CLEAN_CFG_PATH = ROOT / "configs" / "clean_base.yaml"


def _fit_long_side(image: Image.Image, max_side: int = 2600) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    s = max_side / max(w, h)
    return image.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)


def _pilot_items() -> dict[str, list[tuple[str, Path]]]:
    pilot = load_pilot()
    return {
        "olm_old_scans": [(p, ob.data_dir() / "pdfs" / p) for p in pilot["olm_old_scans"]],
        "olm_mc_tbl_math": [(p, ob.data_dir() / "pdfs" / p) for p in pilot["olm_mc_tbl_math"]],
        "realdoc_hard": [(s, fo.data_dir() / "docs" / f"{s}.pdf") for s in pilot["realdoc_hard"]],
        "pb2_hard_tables_text": [
            (e, source_file(paths.gt_dir(), *split_example_id(e))) for e in pilot["pb2_hard_tables_text"]
        ],
        "pb2_hard_charts": [(e, source_file(paths.gt_dir(), *split_example_id(e))) for e in pilot["pb2_hard_charts"]],
    }


def _realdoc_chart_set() -> set[str]:
    qa = fo.load_items()
    return {
        it["source_file"] for it in qa if any(c.startswith(("chart_", "figure_")) for c in it.get("capabilities", []))
    }


def _is_scan(rp: RenderedPage) -> bool:
    if rp.words is None:
        return True
    chars = sum(len(w.text) for w in rp.words)
    return chars < 50


def _heron_dets(cache: Cache, rp: RenderedPage) -> list[heron.Detection]:
    res = cache.get(heron.detect_call(pixel_hash(rp.image)).key())
    return heron.read_detections(res) if res is not None else []


def _base_read(cache: Cache, cfg: Config, rp: RenderedPage) -> ocria.PageRead | None:
    call, _ = ocria.build_page_call(rp.image, cfg.page_model, cfg.page_runtime, cfg.prompt_sha)
    res = cache.get(call.key())
    return ocria.read_page_result(res) if res is not None else None


def _select_chart_crops(
    data: bytes, rp: RenderedPage, dets: list[heron.Detection], max_crops: int = 3
) -> list[tuple[tuple[int, int, int, int], str]]:
    """Return up to `max_crops` `(pixel_box, formatted_vector_hint)` pairs for chart-like regions on a page."""
    w, h = rp.image.size
    crops = chart_crops(dets, w, h, min_area=0.05, min_score=0.40)
    is_pdf = data.lstrip()[:5].startswith(b"%PDF-")
    shapes = load_shapes(data, 0) if is_pdf else []
    pwords = words_in_points(rp) if rp.text_words else []
    scale = rp.scale  # px -> pt

    out: list[tuple[tuple[int, int, int, int], str]] = []
    for cb in sorted(crops, key=lambda c: -((c.box[2] - c.box[0]) * (c.box[3] - c.box[1])))[:max_crops]:
        bx = cb.box
        pt_box = (bx[0] * scale, bx[1] * scale, bx[2] * scale, bx[3] * scale)
        hint_str = ""
        if shapes and pwords:
            try:
                hint = extract_chart_hint(pwords, shapes, bbox=pt_box, page_size=(rp.width_pt, rp.height_pt))
                hint_str = format_chart_hint(hint, max_chars=2400)
            except Exception:
                hint_str = ""
        out.append((bx, hint_str))

    # Fallback: if heron missed a chart on a born-digital chart page that HAS vector axes, crop whole page!
    if not out and shapes and pwords:
        try:
            hint = extract_chart_hint(pwords, shapes, bbox=None, page_size=(rp.width_pt, rp.height_pt))
            hint_str = format_chart_hint(hint, max_chars=2400)
            if hint_str:
                out.append(((0, 0, w, h), hint_str))
        except Exception:
            pass
    return out


def cmd_plan() -> int:
    """Plan exact manifests for Exp 1 (OCRia-1 crops/prep), Exp 2 (Qwen3-VL vector charts), Exp 3 (olmOCR-2 scans)."""
    cfg = Config.load(CLEAN_CFG_PATH, ["layout=none"])
    cache = Cache(paths.cache_dir())
    items = _pilot_items()
    rd_charts = _realdoc_chart_set()
    rd_prof = json.loads((fo.out_dir() / "profile.json").read_text(encoding="utf-8"))

    m1_dir = MANIFESTS_DIR / "exp1_ocria_crops"
    m2_dir = MANIFESTS_DIR / "exp2_qwen3vl_charts"
    m2b_dir = MANIFESTS_DIR / "exp2b_chart_verify"
    m3_dir = MANIFESTS_DIR / "exp3_olmocr2_scans"
    m4_dir = MANIFESTS_DIR / "exp4_chandra2_scans"
    m4b_dir = MANIFESTS_DIR / "exp4b_chandra2_rest"
    for d in (m1_dir, m2_dir, m2b_dir, m3_dir, m4_dir, m4b_dir):
        d.mkdir(parents=True, exist_ok=True)
    w1, w2, w2b, w3, w4, w4b = (
        ManifestWriter(m1_dir),
        ManifestWriter(m2_dir),
        ManifestWriter(m2b_dir),
        ManifestWriter(m3_dir),
        ManifestWriter(m4_dir),
        ManifestWriter(m4b_dir),
    )
    client = ReplayClient(cache)

    stats: Counter[str] = Counter()

    def _queue(writer: ManifestWriter, call, blobs, meta: dict[str, Any], tag: str) -> None:
        if cache.get(call.key()) is not None:
            stats[f"{tag}_cached"] += 1
            return
        added = writer.add(call, blobs, meta)
        stats[f"{tag}_new"] += added

    for bucket, entries in items.items():
        for name, pdf_path in entries:
            data = pdf_path.read_bytes()
            rp = render_pages(data, cfg.dpi)[0]
            turn = page_turn(rp)
            if turn:
                rp = turn_page(rp, turn)
            dets = _heron_dets(cache, rp)
            base = _base_read(cache, cfg, rp)
            is_scan = _is_scan(rp)
            w, h = rp.image.size

            # --- EXP 1A: Loop-band recovery (digital & scanned looped/truncated pages) ---
            if base is not None and (base.finish_reason == "length" or looped(base.markdown)):
                base_img = _fit_long_side(rp.image, 2600)
                if is_scan:
                    base_img, _ = prepare_scan(base_img)
                for bi, (band_im, _bbox) in enumerate(slice_horizontal_bands(base_img, n_bands=2, overlap_frac=0.08)):
                    c, b = ocria.build_page_call(band_im, cfg.page_model, cfg.page_runtime, cfg.prompt_sha)
                    _queue(w1, c, b, {"exp": "1A_loop", "bucket": bucket, "item": name, "band": bi}, "exp1_loop")

            # --- EXP 1B: Tight high-res table crops (max 2 largest tables on table-bearing buckets) ---
            if bucket in ("pb2_hard_tables_text", "olm_mc_tbl_math", "realdoc_hard"):
                tcrops = table_crops(dets, w, h, min_area=0.05, max_area=0.92, min_score=0.50)
                tcrops = sorted(tcrops, key=lambda c: -((c.box[2] - c.box[0]) * (c.box[3] - c.box[1])))[:2]
                for ti, cb in enumerate(tcrops):
                    crop_im = render_crop(rp.image, cb.box, target_long_side=1800, max_upscale=2.0, border=24)
                    c, b = ocria.build_page_call(crop_im, cfg.page_model, cfg.page_runtime, cfg.prompt_sha)
                    _queue(w1, c, b, {"exp": "1B_table", "bucket": bucket, "item": name, "idx": ti}, "exp1_table")

            # --- EXP 1C: Degraded scan enhancement + zoomed bands (30 old_scans + 5 pb2_text scans + 15 rd scans) ---
            is_rd_scan = (
                bucket == "realdoc_hard" and rd_prof.get(name, {}).get("text_pages", 1) == 0 and name not in rd_charts
            )
            if (
                bucket == "olm_old_scans"
                or (bucket == "pb2_hard_tables_text" and name.startswith("text/"))
                or is_rd_scan
            ):
                clean_im, info = prepare_scan(_fit_long_side(rp.image, 2400))
                if info["enhanced"] or abs(float(info["skew_deg"])) >= 0.4 or max(w, h) > 2600:
                    c, b = ocria.build_page_call(clean_im, cfg.page_model, cfg.page_runtime, cfg.prompt_sha)
                    _queue(w1, c, b, {"exp": "1C_scan_prep", "bucket": bucket, "item": name}, "exp1_scan_prep")
                if bucket == "olm_old_scans":
                    for bi, (band_im, _bbox) in enumerate(
                        slice_horizontal_bands(clean_im, n_bands=2, overlap_frac=0.08)
                    ):
                        c, b = ocria.build_page_call(band_im, cfg.page_model, cfg.page_runtime, cfg.prompt_sha)
                        _queue(
                            w1,
                            c,
                            b,
                            {"exp": "1C_scan_band", "bucket": bucket, "item": name, "band": bi},
                            "exp1_scan_band",
                        )

            # --- EXP 1D: Multi-column clean 2-column crops (15 olm_mc pages) ---
            if bucket == "olm_mc_tbl_math" and name.startswith("multi_column/"):
                ccrops = column_crops(dets, w, h)
                for ci, cb in enumerate(ccrops):
                    crop_im = render_crop(rp.image, cb.box, target_long_side=1800, max_upscale=1.8, border=20)
                    c, b = ocria.build_page_call(crop_im, cfg.page_model, cfg.page_runtime, cfg.prompt_sha)
                    _queue(
                        w1,
                        c,
                        b,
                        {"exp": "1D_col", "bucket": bucket, "item": name, "label": cb.label, "idx": ci},
                        "exp1_col",
                    )

            # --- EXP 1E + EXP 2: Charts (30 pb2_hard_charts + 10 realdoc chart docs) ---
            if bucket == "pb2_hard_charts" or (bucket == "realdoc_hard" and name in rd_charts):
                ch_list = _select_chart_crops(data, rp, dets, max_crops=2)
                for chi, (box, hint_str) in enumerate(ch_list):
                    crop_im = render_crop(rp.image, box, target_long_side=1600, max_upscale=2.0, border=16)
                    # Exp 1E: OCRia-1 tight crop
                    c1, b1 = ocria.build_page_call(crop_im, cfg.page_model, cfg.page_runtime, cfg.prompt_sha)
                    _queue(
                        w1, c1, b1, {"exp": "1E_chart_ocria", "bucket": bucket, "item": name, "idx": chi}, "exp1_chart"
                    )
                    # Exp 2: Qwen3-VL-8B + Vector Hint
                    c2, b2 = vlm_specialists.build_chart_call(crop_im, hint_str)
                    _queue(
                        w2,
                        c2,
                        b2,
                        {
                            "exp": "2_chart_qwen3vl",
                            "bucket": bucket,
                            "item": name,
                            "idx": chi,
                            "has_hint": bool(hint_str),
                        },
                        "exp2_chart",
                    )

            # --- EXP 2B: Full-page multi-panel Vector-Guided Chart Verifier (40 chart pages) ---
            if bucket == "pb2_hard_charts" or (bucket == "realdoc_hard" and name in rd_charts):
                base_md = write_page(parse(data, cfg, client).document.pages[0])
                pg_ir = parse_page(base_md, 0, "draft")
                draft_tbls = "\n\n".join(reg.chosen().raw for reg in pg_ir.regions if reg.kind == "table")
                is_pdf = data.lstrip()[:5].startswith(b"%PDF-")
                full_hint = ""
                if is_pdf and rp.text_words:
                    try:
                        h_obj = extract_chart_hint(
                            words_in_points(rp), load_shapes(data, 0), bbox=None, page_size=(rp.width_pt, rp.height_pt)
                        )
                        full_hint = format_chart_hint(h_obj, max_chars=5500)
                    except Exception:
                        full_hint = ""
                c2b, b2b = vlm_specialists.build_chart_verify_call(rp.image, draft_tbls, full_hint)
                _queue(
                    w2b,
                    c2b,
                    b2b,
                    {"exp": "2b_chart_verify", "bucket": bucket, "item": name, "has_hint": bool(full_hint)},
                    "exp2b_chart_verify",
                )

            # --- EXP 3 & EXP 4: olmOCR-2-7B & Chandra-OCR-2 on the 65 scanned pilot pages ---
            is_pb2_cth = bucket == "pb2_hard_tables_text" and name.startswith(("complex_table_headers/", "text/"))
            if bucket == "olm_old_scans" or is_rd_scan or is_pb2_cth:
                c3, b3 = vlm_specialists.build_olmocr_call(rp.image)
                _queue(w3, c3, b3, {"exp": "3_olmocr2", "bucket": bucket, "item": name}, "exp3_olmocr2")
                c4, b4 = vlm_specialists.build_chandra_call(rp.image)
                _queue(w4, c4, b4, {"exp": "4_chandra2", "bucket": bucket, "item": name}, "exp4_chandra2")
            else:
                # --- EXP 4B: Chandra-OCR-2 on the remaining 85 Pilot-150 pages (full 150/150 head-to-head) ---
                c4b, b4b = vlm_specialists.build_chandra_call(rp.image)
                _queue(w4b, c4b, b4b, {"exp": "4b_chandra2_rest", "bucket": bucket, "item": name}, "exp4b_chandra2")

    print("\n=== PILOT-150 PERCEPTION EXPERIMENT MANIFESTS ===")
    print(f"  Exp 1  (OCRia-1 crops/loop/scan-prep): {len(w1):3d} calls -> {m1_dir}")
    print(f"  Exp 2  (Qwen3-VL-8B cropped charts)  : {len(w2):3d} calls -> {m2_dir}")
    print(f"  Exp 2B (Qwen3-VL-8B chart verifier)  : {len(w2b):3d} calls -> {m2b_dir}")
    print(f"  Exp 3  (olmOCR-2-7B scanned pages)   : {len(w3):3d} calls -> {m3_dir}")
    print(f"  Exp 4  (Chandra-OCR-2 scanned pages) : {len(w4):3d} calls -> {m4_dir}")
    print(f"  Exp 4B (Chandra-OCR-2 remaining 85)  : {len(w4b):3d} calls -> {m4b_dir}")
    print(f"  TOTAL MANIFEST CALLS                 : {len(w1) + len(w2) + len(w2b) + len(w3) + len(w4) + len(w4b):3d}")
    print("\nBreakdown by sub-experiment:")
    for k, v in sorted(stats.items()):
        print(f"  {k:24s}: {v:3d}")
    return 0


def _score_custom_md(
    name: str,
    olm_pages: dict[str, list[str]],
    rd_md: dict[str, str],
    pb_md: dict[str, str],
) -> dict[str, Any]:
    """Score custom per-page markdowns across all 3 benchmarks on pilot150."""
    import shutil
    import subprocess

    from docparse.bench.record import result_record
    from docparse.core.ir import Candidate, Document, Page, Region

    pilot = load_pilot()
    run_dir = PILOT_RUNS / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. olmOCR-bench (60 pages)
    olm_rels = pilot["olm_old_scans"] + pilot["olm_mc_tbl_math"]
    olm_scores = _score_olm_subset(run_dir, name, olm_rels, olm_pages)

    # 2. RealDocBench (30 docs)
    rd_stems = pilot["realdoc_hard"]
    qa = [it for it in fo.load_items() if it["source_file"] in set(rd_stems)]
    rd_rows = fo.score_items(qa, {name: rd_md})[name]
    prof = json.loads((fo.out_dir() / "profile.json").read_text(encoding="utf-8"))
    chart_docs = _realdoc_chart_set()
    rd_scores = {
        "all": fo._summary(rd_rows),
        "scanned": fo._summary(
            [r for r in rd_rows if prof[r["doc"]]["text_pages"] == 0 and r["doc"] not in chart_docs]
        ),
        "charts": fo._summary([r for r in rd_rows if r["doc"] in chart_docs]),
        "digital": fo._summary([r for r in rd_rows if prof[r["doc"]]["text_pages"] > 0 and r["doc"] not in chart_docs]),
    }

    # 3. ParseBench 2 (60 pages)
    pb_ids = pilot["pb2_hard_tables_text"] + pilot["pb2_hard_charts"]
    out_root, gt_view, rep_dir = run_dir / "pb2_out", run_dir / "pb2_gt", run_dir / "pb2_reports"
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
        out_sub = out_root / sub / name
        out_sub.mkdir(parents=True, exist_ok=True)
        sources[sub] = {"output_dir": str(out_sub.resolve()), "test_cases_dir": str((gt_view / sub).resolve())}
        pg = Page(
            index=0,
            regions=[
                Region(
                    id="p0r0",
                    kind="paragraph",
                    order=0,
                    candidates=[Candidate(id="p0r0c0", reader=name, raw=pb_md[eid])],
                    selected="p0r0c0",
                )
            ],
        )
        doc = Document(sha256="0" * 64, n_pages=1, config_hash="0" * 64, pages=[pg])
        rec = result_record(eid, name, doc)
        (out_sub / f"{stem}.result.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")

    proc = subprocess.run(build_command(out_root, sources, rep_dir), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"PB2 scorer failed: {proc.stderr[-500:]}")
    primary = {"charts": "chart_next", "text": "text_next"}
    pb_scores: dict[str, Any] = {}
    for sub, src in sources.items():
        rep = json.loads((Path(src["output_dir"]) / "_evaluation_report.json").read_text(encoding="utf-8"))
        want = primary.get(sub, "table_next")
        vals = [
            float(m["value"])
            for r in rep.get("per_example_results") or []
            if r.get("success")
            for m in r.get("metrics") or []
            if m.get("metric_name") == want and m.get("value") is not None
        ]
        pb_scores[sub] = {"n": len(vals), "mean": round(100.0 * sum(vals) / len(vals), 2) if vals else 0.0}

    summary = {
        "name": name,
        "config": f"experiment:{name}",
        "olmocr": olm_scores,
        "realdoc": rd_scores,
        "pb2": pb_scores,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _print_summary(summary)
    return summary


def cmd_eval(exp: str) -> int:
    """Evaluate a cached perception experiment on pilot150 and compare directly against `clean_base`."""
    cfg = Config.load(CLEAN_CFG_PATH, ["layout=none"])
    cache = Cache(paths.cache_dir())
    client = ReplayClient(cache)
    items = _pilot_items()
    rd_charts = _realdoc_chart_set()
    rd_prof = json.loads((fo.out_dir() / "profile.json").read_text(encoding="utf-8"))

    olm_pages: dict[str, list[str]] = {}
    rd_md: dict[str, str] = {}
    pb_md: dict[str, str] = {}
    applied: Counter[str] = Counter()

    for bucket, entries in items.items():
        for name, pdf_path in entries:
            data = pdf_path.read_bytes()
            base_doc = parse(data, cfg, client).document
            md = write_page(base_doc.pages[0])
            rp = render_pages(data, cfg.dpi)[0]
            turn = page_turn(rp)
            if turn:
                rp = turn_page(rp, turn)
            dets = _heron_dets(cache, rp)
            base_read = _base_read(cache, cfg, rp)
            is_scan = _is_scan(rp)
            w, h = rp.image.size
            is_rd_scan = (
                bucket == "realdoc_hard" and rd_prof.get(name, {}).get("text_pages", 1) == 0 and name not in rd_charts
            )
            is_pb2_cth = bucket == "pb2_hard_tables_text" and name.startswith(("complex_table_headers/", "text/"))

            if exp in ("exp1_loop", "exp1_all"):
                if base_read is not None and (base_read.finish_reason == "length" or looped(base_read.markdown)):
                    base_img = _fit_long_side(rp.image, 2600)
                    if is_scan:
                        base_img, _ = prepare_scan(base_img)
                    parts = []
                    for band_im, _ in slice_horizontal_bands(base_img, n_bands=2, overlap_frac=0.08):
                        c, _ = ocria.build_page_call(band_im, cfg.page_model, cfg.page_runtime, cfg.prompt_sha)
                        r = cache.get(c.key())
                        if r is not None:
                            pr = ocria.read_page_result(r)
                            parts.append(trim_loop_tail(pr.markdown))
                    if len(parts) == 2:
                        stitched = stitch_bands(parts)
                        if len(stitched) > len(md) * 0.7:
                            md = stitched
                            applied["loop_bands"] += 1

            if exp in ("exp1_scan_prep", "exp1_all") and (
                bucket == "olm_old_scans"
                or (bucket == "pb2_hard_tables_text" and name.startswith("text/"))
                or is_rd_scan
            ):
                clean_im, _ = prepare_scan(_fit_long_side(rp.image, 2400))
                c, _ = ocria.build_page_call(clean_im, cfg.page_model, cfg.page_runtime, cfg.prompt_sha)
                r = cache.get(c.key())
                if r is not None:
                    pr = ocria.read_page_result(r)
                    if pr.markdown.strip() and not looped(pr.markdown):
                        md = trim_loop_tail(pr.markdown)
                        applied["scan_prep"] += 1

            if exp == "exp1_scan_band" and bucket == "olm_old_scans":
                clean_im, _ = prepare_scan(_fit_long_side(rp.image, 2400))
                parts = []
                for band_im, _ in slice_horizontal_bands(clean_im, n_bands=2, overlap_frac=0.08):
                    c, _ = ocria.build_page_call(band_im, cfg.page_model, cfg.page_runtime, cfg.prompt_sha)
                    r = cache.get(c.key())
                    if r is not None:
                        parts.append(trim_loop_tail(ocria.read_page_result(r).markdown))
                if len(parts) == 2:
                    md = stitch_bands(parts)
                    applied["scan_band"] += 1

            if exp in ("exp1_col", "exp1_all") and bucket == "olm_mc_tbl_math" and name.startswith("multi_column/"):
                ccrops = column_crops(dets, w, h)
                parts = []
                for cb in ccrops:
                    crop_im = render_crop(rp.image, cb.box, target_long_side=1800, max_upscale=1.8, border=20)
                    c, _ = ocria.build_page_call(crop_im, cfg.page_model, cfg.page_runtime, cfg.prompt_sha)
                    r = cache.get(c.key())
                    if r is not None:
                        parts.append(trim_loop_tail(ocria.read_page_result(r).markdown))
                if ccrops and len(parts) == len(ccrops):
                    md = "\n\n".join(p for p in parts if p.strip())
                    applied["columns"] += 1

            if exp in ("exp1_table", "exp1_all") and bucket in (
                "pb2_hard_tables_text",
                "olm_mc_tbl_math",
                "realdoc_hard",
            ):
                tcrops = table_crops(dets, w, h, min_area=0.05, max_area=0.92, min_score=0.50)
                tcrops = sorted(tcrops, key=lambda c: -((c.box[2] - c.box[0]) * (c.box[3] - c.box[1])))[:2]
                pg_ir = parse_page(md, 0, "crop")
                page_tables = [(reg.id, reg.chosen().raw) for reg in pg_ir.regions if reg.kind == "table"]
                replaced = False
                for cb in tcrops:
                    crop_im = render_crop(rp.image, cb.box, target_long_side=1800, max_upscale=2.0, border=24)
                    c, _ = ocria.build_page_call(crop_im, cfg.page_model, cfg.page_runtime, cfg.prompt_sha)
                    r = cache.get(c.key())
                    if r is None:
                        continue
                    pr = ocria.read_page_result(r)
                    if looped(pr.markdown):
                        continue
                    crop_ir = parse_page(pr.markdown, 0, "crop")
                    crop_tables = [reg.chosen().raw for reg in crop_ir.regions if reg.kind == "table"]
                    if not crop_tables:
                        continue
                    best_crop_tbl = max(crop_tables, key=len)
                    hit_id = match_cropped_table(best_crop_tbl, page_tables)
                    if hit_id is None and len(page_tables) == 1 and len(tcrops) == 1:
                        hit_id = page_tables[0][0]
                    if hit_id is not None:
                        for reg in pg_ir.regions:
                            if reg.id == hit_id:
                                md = md.replace(reg.chosen().raw, best_crop_tbl, 1)
                                replaced = True
                                break
                if replaced:
                    applied["table_crops"] += 1

            if exp in ("exp1_chart", "exp2_qwen3vl_charts") and (
                bucket == "pb2_hard_charts" or (bucket == "realdoc_hard" and name in rd_charts)
            ):
                ch_list = _select_chart_crops(data, rp, dets, max_crops=2)
                pg_ir = parse_page(md, 0, "chart")
                page_tables = [(reg.id, reg.chosen().raw) for reg in pg_ir.regions if reg.kind == "table"]
                new_blocks: list[str] = []
                for box, hint_str in ch_list:
                    crop_im = render_crop(rp.image, box, target_long_side=1600, max_upscale=2.0, border=16)
                    if exp == "exp1_chart":
                        c, _ = ocria.build_page_call(crop_im, cfg.page_model, cfg.page_runtime, cfg.prompt_sha)
                        r = cache.get(c.key())
                        if r is None:
                            continue
                        pr = ocria.read_page_result(r)
                        crop_ir = parse_page(pr.markdown, 0, "crop")
                        ctbls = [reg.chosen().raw for reg in crop_ir.regions if reg.kind == "table"]
                        if ctbls:
                            best_t = max(ctbls, key=len)
                            hit = match_cropped_table(best_t, page_tables)
                            if hit is None and len(page_tables) == 1 and len(ch_list) == 1:
                                hit = page_tables[0][0]
                            if hit is not None:
                                for reg in pg_ir.regions:
                                    if reg.id == hit:
                                        md = md.replace(reg.chosen().raw, best_t, 1)
                                        applied["chart_ocria_replace"] += 1
                                        break
                    else:
                        c, _ = vlm_specialists.build_chart_call(crop_im, hint_str)
                        r = cache.get(c.key())
                        if r is None:
                            continue
                        cr = vlm_specialists.read_chart_result(
                            r, hint=hint_str or None, include_axes=(bucket == "realdoc_hard")
                        )
                        if cr.table_html:
                            hit = match_cropped_table(cr.table_html, page_tables)
                            if hit is None and len(page_tables) == 1 and len(ch_list) == 1:
                                hit = page_tables[0][0]
                            if hit is not None:
                                for reg in pg_ir.regions:
                                    if reg.id == hit:
                                        repl = cr.markdown if cr.title else cr.table_html
                                        md = md.replace(reg.chosen().raw, repl, 1)
                                        applied["chart_qwen_replace"] += 1
                                        break
                            else:
                                new_blocks.append(cr.markdown)
                                applied["chart_qwen_append"] += 1
                        elif bucket == "realdoc_hard" and cr.markdown:
                            new_blocks.append(cr.markdown)
                            applied["chart_qwen_meta"] += 1
                if new_blocks:
                    md = md.rstrip() + "\n\n" + "\n\n".join(new_blocks)

            if exp == "exp2b_chart_verify" and (
                bucket == "pb2_hard_charts" or (bucket == "realdoc_hard" and name in rd_charts)
            ):
                pg_ir = parse_page(md, 0, "draft")
                draft_regions = [reg for reg in pg_ir.regions if reg.kind == "table"]
                draft_tbls = "\n\n".join(reg.chosen().raw for reg in draft_regions)
                is_pdf = data.lstrip()[:5].startswith(b"%PDF-")
                full_hint = ""
                if is_pdf and rp.text_words:
                    try:
                        h_obj = extract_chart_hint(
                            words_in_points(rp), load_shapes(data, 0), bbox=None, page_size=(rp.width_pt, rp.height_pt)
                        )
                        full_hint = format_chart_hint(h_obj, max_chars=5500)
                    except Exception:
                        full_hint = ""
                c, _ = vlm_specialists.build_chart_verify_call(rp.image, draft_tbls, full_hint)
                r = cache.get(c.key())
                if r is not None:
                    new_tbls = vlm_specialists.read_chart_verify_tables(r, hint=full_hint or None)
                    draft_rows = sum(t.lower().count("<tr") for t in [reg.chosen().raw for reg in draft_regions])
                    new_rows = sum(t.lower().count("<tr") for t in new_tbls)
                    draft_cells = sum(
                        t.lower().count("<td") + t.lower().count("<th")
                        for t in [reg.chosen().raw for reg in draft_regions]
                    )
                    new_cells = sum(t.lower().count("<td") + t.lower().count("<th") for t in new_tbls)
                    if new_tbls and (
                        not draft_regions or (new_rows >= 0.80 * draft_rows and new_cells >= 0.75 * draft_cells)
                    ):
                        if len(new_tbls) == len(draft_regions) and draft_regions:
                            for reg, nt in zip(draft_regions, new_tbls, strict=True):
                                md = md.replace(reg.chosen().raw, nt, 1)
                            applied["chart_verify_1to1"] += 1
                        elif draft_regions:
                            # Replace first draft table with all verified tables and remove other old draft tables
                            md = md.replace(draft_regions[0].chosen().raw, "\n\n".join(new_tbls), 1)
                            for reg in draft_regions[1:]:
                                md = md.replace(reg.chosen().raw, "", 1)
                            applied["chart_verify_multi"] += 1
                        else:
                            md = md.rstrip() + "\n\n" + "\n\n".join(new_tbls)
                            applied["chart_verify_append"] += 1
                    else:
                        applied["chart_verify_guarded_skip"] += 1

            if exp == "exp4_chandra2_scans" and (bucket == "olm_old_scans" or is_rd_scan or is_pb2_cth):
                c, _ = vlm_specialists.build_chandra_call(rp.image)
                r = cache.get(c.key())
                if r is not None:
                    drop_f = bucket == "olm_old_scans"
                    sr = vlm_specialists.read_chandra_result(r, drop_furniture=drop_f)
                    if sr.finish_reason != "length" and sr.markdown.strip() and not looped(sr.markdown):
                        md = sr.markdown
                        applied["chandra2_scan"] += 1
                    else:
                        applied["chandra2_loop_fallback"] += 1

            if exp == "exp4_chandra2_all":
                c, _ = vlm_specialists.build_chandra_call(rp.image)
                r = cache.get(c.key())
                if r is not None:
                    drop_f = bucket in ("olm_old_scans", "olm_mc_tbl_math")
                    sr = vlm_specialists.read_chandra_result(r, drop_furniture=drop_f)
                    if sr.finish_reason != "length" and sr.markdown.strip() and not looped(sr.markdown):
                        md = sr.markdown
                        applied["chandra2_all"] += 1
                    else:
                        applied["chandra2_loop_fallback"] += 1

            if exp == "exp3_olmocr2_scans" and (bucket == "olm_old_scans" or is_rd_scan or is_pb2_cth):
                c, _ = vlm_specialists.build_olmocr_call(rp.image)
                r = cache.get(c.key())
                if r is not None:
                    sr = vlm_specialists.read_olmocr_result(r)
                    if sr.markdown.strip():
                        md = sr.markdown
                        applied["olmocr2_scan"] += 1

            if bucket in ("olm_old_scans", "olm_mc_tbl_math"):
                olm_pages[name] = [md]
            elif bucket == "realdoc_hard":
                rd_md[name] = md
            else:
                pb_md[name] = md

    print(f"Applied upgrades for {exp}: {dict(applied)}")
    _score_custom_md(exp, olm_pages, rd_md, pb_md)
    cmd_compare("clean_base", exp)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalkit.pilot_experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    p_imp = sub.add_parser("import")
    p_imp.add_argument("--exp", required=True)
    p_imp.add_argument("--results", type=Path, required=True)
    p_ev = sub.add_parser("eval")
    p_ev.add_argument(
        "--exp",
        required=True,
        choices=[
            "exp1_loop",
            "exp1_table",
            "exp1_scan_prep",
            "exp1_scan_band",
            "exp1_col",
            "exp1_chart",
            "exp1_all",
            "exp2_qwen3vl_charts",
            "exp2b_chart_verify",
            "exp3_olmocr2_scans",
            "exp4_chandra2_scans",
            "exp4_chandra2_all",
        ],
    )
    args = parser.parse_args(argv)
    if args.cmd == "plan":
        return cmd_plan()
    if args.cmd == "import":
        from docparse.models.client import result_from_output
        from docparse.models.manifest import read_manifest

        mdir = MANIFESTS_DIR / args.exp
        cache = Cache(paths.cache_dir())
        rows: dict[str, dict[str, Any]] = {}
        for line in args.results.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and "key" in row and "output" in row and row.get("ok", True):
                rows[row["key"]] = row
        added = 0
        for call, meta in read_manifest(mdir):
            row = rows.get(call.key())
            if row:
                added += cache.put(result_from_output(call, row["output"]), call, meta)
        print(f"Imported {added} new results from {args.results} into cache")
        return 0
    return cmd_eval(args.exp)


if __name__ == "__main__":
    sys.exit(main())
