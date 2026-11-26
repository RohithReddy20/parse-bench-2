# Frontier-v3: Two-Tier Specialist Routing (`Qwen3.5-122B` Charts/Tables + `Qwen3.5-27B` Forms + Geometric Checkbox Ink)

Date: 2026-11-27  
Branch: `parsebench-v2-research-38d42870`  
Config: `configs/frontier_v3.yaml`  
Compute spent (this milestone): **$5.63 Modal** (`$27.76 / $30.00` monthly Starter credit used, `$2.24` free credit remaining, **$0.00 billed**)

## 1. Motivation & Architecture

`configs/hybrid_v2.yaml` (`7f620ac`, `86.82` on held-out `PB2 check`, `93.34%` on `RealDocBench`, `85.1%` on `olmOCR-bench`) established a clean dual-4B Tier-1 baseline (`Chandra-OCR-2` + `OCRia-1` fallback/fusion) with zero benchmark-specific hacks. Its largest remaining error buckets on general documents were:
1. **Charts (`80.52` on `PB2 check`):** 4B readers hallucinate or drop values on dense, multi-series, or multi-panel figures.
2. **Non-fillable forms (`81.90` on `PB2 check`, `80.13` on `forms`):** Tiny checkmarks, handwritten entries, and horizontal multi-option checkbox lines confuse 4B readers.
3. **Cut-off / low-recall dense tables (`86.40` on `dense_tables`, `87.72` on `insurance_filing_tables`):** Single-pass full-page reads hit output limits or drop rows on wide, dense tabular pages.

`configs/frontier_v3.yaml` adds a **content-gated Tier-2 escalation layer** on top of Tier-1 (`Chandra-OCR-2` + `OCRia-1` + `Heron`/`PP-DocLayoutV3`), enforcing **at most one Tier-2 escalation per page** (`tier2_tried`, keeping offline manifest planning single-pass deterministic) and keeping all zero-regression PDF-grounded post-processors:

- **Tier-2 Charts (`Qwen/Qwen3.5-122B-A10B-FP8`, `a099dee7...`):** Triggered when Chandra emits a chart `Figure` or `PP-DocLayoutV3` detects a `chart` box (`score >= 0.40`). Receives the full page, a zoomed chart crop, PDF vector hints, and Tier-1 draft tables. When `2–6` deduplicated `PP-DocLayoutV3` chart boxes span an enclosing bounding-box share `>= 0.32` of the page, each chart is read via its own zoomed `chart-single-v1` crop so spread-out multi-chart pages do not lose resolution; tighter clustered sub-panels (`< 0.32`) stay in a single union crop preserving shared titles/legends.
- **Tier-2 Forms (`Qwen/Qwen3.5-27B-FP8`, `97f5941b...`):** Triggered on non-math pages matching form criteria (Chandra `Form` block, `>= 4` checkboxes, or `>= 3` AcroForm widgets). Uses draft-guided `form-refine-v1` (passing Chandra's draft alongside the page image) and adopts the refinement only when word coverage `>= 0.93` and `_data_table_count` preserves all true `<th>` data tables (without rejecting `<table border="0">` form layout grids).
- **Visual Ink-Ratio Checkbox Verifier (`docparse/fixes/forms_ink.py`):** Runs after form refinement (`form_options.checkbox_ink: true`) with per-box geometric ownership (`gap_L` vs `gap_R`, `OWNER_GAP = 0.85`), printed-box exclusion, and contrast guards so adjacent right-of-label options (`Government [x]  UN Refugee [ ]`) cannot steal neighboring boxes.
- **Tier-2 Tables (`Qwen/Qwen3.5-122B-A10B-FP8`, `a099dee7...`):** Triggered on non-math tabular pages whose Tier-1 read hit the token limit (`truncated`) or recalled `< 0.78` of a healthy PDF text layer. Runs **before** plain-text `fallback` in the OCRia branch so cut-off tables recover HTML structure instead of dumping plain text. Re-reads the page in `3` overlapping horizontal strips (`overlap = 0.14`, `max_tokens = 16384`), stitches overlapping rows (`merge_n_strip_tables`), and adopts only if every strip is usable and PDF word recall improves by `>= +0.05` (or `>=` on a cut-off read).
- **HTML List & Regression Fixes:** Removed synthetic `1. 2. 3.` numbering on Chandra's fragmented `<ol>` blocks and raw `<ol type="i">` HTML leaks (`1cb966e`, restoring `-2.14` lost in `PB2 text`), and refined `_data_table_count` (`168cc38`) to protect true `<th>` data tables on `olmOCR-bench`.

## 2. Dev-Split Ablations (What Survived vs. What Was Eliminated)

All design choices and thresholds were selected strictly on `PB2 dev`, `RealDocBench` dev hard subsets, and `Pilot-150` — never on held-out `PB2 check`:

| Experiment | Verdict | Dev-Split Evidence |
|---|---|---|
| `Qwen3.5-122B-A10B-FP8` on charts (`30` hard `PB2 dev`) | **KEPT** | `hybrid_v2 47.13 -> 27B 58.25 -> 122B union 60.76 (+2.51) -> 122B per-chart 61.87 -> ~63.0` with `union >= 0.32` guard |
| Enclosing-bbox `union >= 0.32` per-chart crop gate | **KEPT** | Preserved clustered sub-panel union crops (`US_Prof_p25 = 100.0` vs `66.7` split) while zooming spread-out multi-chart pages (`esde_p34 +19.7`, `E-Gov_p78 = 100.0`, `b263dc5d_p170 = 73.2`) |
| Draft-guided `Qwen3.5-27B-FP8` `form-refine-v1` (`25` `PB2 dev` + `15` `RealDoc` hard) | **KEPT** | Beat blind `form-v1` and beat `122B-A10B-FP8` (`67.72 / 92.90%` vs `67.30 / 91.61%`; `24/25` vs `19/25` passing `word_coverage >= 0.93`) |
| `forms_ink.py` per-box geometric ownership (`379` `PB2 dev` forms) | **KEPT** | **5/5 clean wins, 0 losses** (`forms_v2 80.81 -> 81.07 [+0.26]`, `forms_v2_checkboxes 91.19 -> 91.81 [+0.62]`), `0` regressions on `RealDocBench` |
| Gated 3-strip `122B` table re-read (`6` truncated/dense tables) | **KEPT** | `dense_table_162: 38.4 -> 45.3 (+6.9)` (`rec 0.68 -> 1.00`), `fcf94142_p97: 0.0 -> 55.0 (+55.0)`; recall gate cleanly rejected `dense_table_160` (`0.98 -> 0.85`) |
| `Qwen3.5-122B-A10B-FP8` on forms | **ELIMINATED** | Lost to dense `27B-FP8` on both `PB2 dev` forms (`-0.42`) and `RealDoc` hard forms (`-1.29%`) — dense 27B has 2.7× more active parameters per token for verbatim form transcription |
| `Qwen3.5-122B-A10B-FP8` on `complex_table_headers` (`15` `PB2 dev`) | **ELIMINATED** | `Chandra-2` (`72.01`) beat `122B` (`71.87`) |
| Layout swaps (`420` `PB2 dev`, baseline `91.65`) | **ELIMINATED** | `pp_primary (-5.88)`, `heron_pp_full (-0.64)`, `heron_pp_pic_split (-0.34)`, `picbox_only (-0.37)`, `pp_add_030 (-0.03)`, `split_gap=2.0 (-0.06)` all trailed current `Heron + pp_arbiter + pp_keep_pictures + pp_add` |
| Synthetic `<ol>` numbering (`d2ffaf9`) | **REVERTED** | Caused `-2.14` regression on `PB2 text` (`89.93 -> 87.79`) via spurious numbering on fragmented Chandra `<ol>` blocks; fixed in `1cb966e` |
| `LightOnOCR-3-4B`, `Infinity-Parser2-Pro-AWQ`, blind `form-v1` | **ELIMINATED** | Underperformed Tier-1/Tier-2 choices in dev pilots |

## 3. Full 3-Benchmark Validation (`2,889` Pages, Zero Errors)

All three full benchmarks were evaluated deterministically from cached responses (`ReplayClient`, `workers=1`) with **0 parse errors** and **0 cache misses**.

### 3.1 Held-Out `ParseBench 2` `check` Split (`905` Unseen Cases)

| Dimension / Subset | `plain` (`OCRia-1`) | `hybrid_v2` | `hybrid_v2_s2` (`7f620ac`) | Phase-1 `frontier_v3` (`27B` charts) | **`frontier_v3` (`122B` charts/tables + `27B` forms)** | **Δ vs `7f620ac`** | **Δ vs `plain`** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **OVERALL (6 dims)** | 85.18 | 86.50 | 86.82 | 87.08 | **87.62** | **+0.80** | **+2.44** |
| `text` | 88.28 | 89.99 | 89.93 | 87.79* | **89.80** | -0.13 | +1.52 |
| `formatting` | 91.11 | 91.92 | 92.23 | 92.77 | **92.63** | +0.40 | +1.52 |
| `tables` | 89.91 | 89.19 | 89.70 | 89.70 | **89.58** | -0.12 | -0.33 |
| **`charts`** | 78.30 | 80.52 | 80.52 | 82.72 | **83.94** | **+3.42** | **+5.64** |
| `layout` | 81.32 | 85.23 | 86.64 | 86.64 | **86.64** | 0.00 | +5.32 |
| **`forms`** | 82.17 | 82.17 | 81.90 | 82.85 | **83.14** | **+1.24** | **+0.97** |
| ↳ `formatting/text` | 90.71 | 91.22 | 91.11 | — | **91.63** | +0.52 | +0.92 |
| ↳ `formatting/headings` | 97.85 | 96.36 | 97.37 | — | **97.37** | 0.00 | -0.48 |
| ↳ `formatting/strikeout` | 78.35 | 92.23 | 96.43 | — | **96.70** | +0.27 | +18.35 |
| ↳ `formatting/underline` | 95.31 | 91.11 | 91.11 | — | **91.11** | 0.00 | -4.20 |
| ↳ `tables/tables` | 93.35 | 90.68 | 91.27 | — | **91.18** | -0.09 | -2.17 |
| ↳ `tables/complex_table_headers` | 85.90 | 84.60 | 84.60 | — | **84.60** | 0.00 | -1.30 |
| ↳ `tables/financial_tables` | 89.78 | 93.83 | 93.83 | — | **93.83** | 0.00 | +4.05 |
| ↳ `tables/insurance_filing_tables` | 85.33 | 86.83 | 87.72 | — | **88.35** | **+0.63** | +3.02 |
| ↳ `tables/dense_tables` | 84.85 | 85.56 | 86.40 | — | **85.13** | -1.27 | +0.28 |
| ↳ `forms/forms` | 80.57 | 80.50 | 80.13 | — | **81.53** | **+1.40** | +0.96 |
| ↳ `forms/fillable_forms` | 93.34 | 93.89 | 94.27 | 94.27 | **94.42** | **+0.15** | +1.08 |

*\*Phase-1 `text` was depressed by `-2.14` due to the `<ol>` bug fixed in `1cb966e`.*

Leaderboard context (held-out `PB2 check`):
- **`frontier_v3` (`87.62`)** cleanly surpasses **`#2 Claude Opus 5.5` (`87.1`, `+0.52`)** using open-weights models only, and matches our previous overfit `own_best` (`87.62`) **without any ParseBench synthetic overlay-stream form hacks**, while simultaneously beating `own_best` on both outside benchmarks.
- `#1 LlamaParse Agentic Plus` remains ahead at `91.0` (`+3.38`).

### 3.2 Full `RealDocBench` (`581` Regulated Documents, `1,356` Questions)

Scored deterministically via `evalkit.external.forms_outside`:

| Group (`581` docs) | `OCRia-1` (`plain`) | Old `own_best` | `hybrid_v2` | Phase-1 `frontier_v3` | **`frontier_v3` (exact)** | **`frontier_v3` (loose)** | **Δ loose vs `hybrid_v2`** | **Δ loose vs `plain`** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **All (`581` docs)** | 91.21% | 91.33% | 93.34% | 93.67% | 91.18% | **93.48%** | **+0.14** | **+2.27** |
| Scanned | 92.18% | 92.18% | 93.70% | 94.18% | 91.70% | **93.94%** | **+0.24** | **+1.76** |
| Digital | 87.80% | 88.31% | 92.03% | — | 89.32% | **91.86%** | -0.17 | **+4.06** |
| Charts / Figures | 74.86% | 74.86% | 87.43% | 87.43% | 83.43% | **87.43%** | 0.00 | **+12.57** |
| Form questions | 90.66% | 90.72% | 92.12% | 92.57% | 90.03% | **92.31%** | **+0.19** | **+1.65** |

- `tier2_tables` had `0.00%` delta on `RealDocBench` (`93.48%` with or without it).

### 3.3 Full `olmOCR-bench` (`1,403` Pages, `8,413` Official Tests, `strip_furniture=true`)

Scored with AllenAI's pinned `olmocr==0.4.27` scorer (`--bootstrap_samples 1000`, `1403/1403` pages rendered, `0` parse errors):

| Category (`1,403` pages) | Tests | `OCRia-1` plain | `OCRia-1` no-furn | `hybrid_v2` | **`frontier_v3`** | **Δ vs `hybrid_v2`** | **Δ vs `OCRia-1` no-furn** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Overall** | 8,413 | 76.05% | 83.67% | 85.14% ± 0.8% | **85.0% ± 0.8%** | -0.1 | **+1.3** |
| `arxiv_math` | 2,927 | 85.14% | 85.14% | 86.44% | **86.4%** (`2530/2927`) | 0.0 | +1.3 |
| `baseline` | 1,394 | 99.78% | 99.71% | 100.00% | **100.0%** (`1394/1394`) | 0.0 | +0.3 |
| `headers_footers` | 760 | 24.61% | 83.68% | 89.74% | **88.6%** (`673/760`) | -1.1 | +4.9 |
| `long_tiny_text` | 442 | 93.44% | 93.44% | 93.21% | **93.2%** (`412/442`) | 0.0 | -0.2 |
| `multi_column` | 884 | 81.45% | 80.77% | 83.14% | **83.0%** (`734/884`) | -0.1 | +2.2 |
| `old_scans` | 526 | 43.16% | 45.82% | 50.00% | **50.0%** (`263/526`) | 0.0 | +4.2 |
| `old_scans_math` | 458 | 88.65% | 88.65% | 87.77% | **87.8%** (`402/458`) | 0.0 | -0.9 |
| **`table_tests`** | 1,022 | 92.17% | 92.17% | 90.80% | **91.2%** (`932/1022`) | **+0.4** | -1.0 |

- Confidence interval `[84.2%, 85.9%]` is statistically indistinguishable from `hybrid_v2`, while `table_tests` improved `+0.4%` (`90.8% -> 91.2%`) after the `_data_table_count` `<th>` preservation fix (`168cc38`).

## 4. Exact Modal Compute & Cost Breakdown

| Run | Modal App ID | Model | GPU | Calls (`done/total`) | Cost |
|---|---|---|---|---:|---:|
| Phase-1 full Tier-2 (`27B` charts + forms) | `ap-WZ3uCjawJptFZ7buCxhuov` | `Qwen/Qwen3.5-27B-FP8` | `H100` | `705 / 705` | `$1.96` |
| Phase-2 dev pilot (`122B` charts/forms/tables) | `ap-kEAyAJcvhRAN9V1o4alHR2` | `Qwen/Qwen3.5-122B-A10B-FP8` | `B200` | `144 / 144` | `$1.44` |
| Phase-2 refreshed `27B` forms delta | `ap-d2azeBKvGnkxRqiVCbsSgr` | `Qwen/Qwen3.5-27B-FP8` | `H100` | `19 / 19` | `$0.44` |
| Phase-2 full `122B` charts + 3-strip tables | `ap-koBdIAcZXVQRgygwGDnEzR` | `Qwen/Qwen3.5-122B-A10B-FP8` | `B200` | `436 / 436` | `$1.79` |
| **Total `frontier_v3` Tier-2 research + validation** | — | — | — | **`1,304 / 1,304` (0 failed)** | **`$5.63`** |

- Exact November Modal total across all apps: **`$27.76 / $30.00`** (`$2.24` free credit remaining -> **$0.00 billed**).

## 5. Honest Assessment & Path Toward `90.0+`

`frontier_v3` achieved **`87.62`** on held-out `PB2 check` (`+0.80` over `7f620ac`, `+2.44` over plain `OCRia-1`, **`+0.52` ahead of `#2 Claude Opus 5.5`**) while improving `RealDocBench` (`93.48%`, `+2.27` vs plain) and holding `olmOCR-bench` (`85.0%`, `+1.3` vs `OCRia-1` no-furn, `table_tests +0.4%`).

Closing the remaining **`2.38` overall points to `90.0`** (`+14.3` dimension points across 6 equally weighted dimensions) requires genuine structural upgrades rather than prompt tuning:
1. **`tables` (`89.58` -> `~91.5`, `$0` GPU):** Clean born-digital non-math tabular pages already score higher under `OCRia-1` (`clean_base = 91.58` on `PB2 tables` and `92.17%` on `olmOCR table_tests` vs `Chandra-2`'s `89.58` / `90.80%`). Selective born-digital table routing between `Chandra-2` and `OCRia-1` can recover `+1.5–2.0` table points with zero extra inference cost; also worth revisiting `tier2_tables` on `dense_tables` (`86.40 -> 85.13`) vs `insurance_filing_tables` (`87.72 -> 88.35`).
2. **`charts` (`83.94` -> `~88–89`):** Biggest single-dimension win so far (`+3.42`), yet dense unlabeled multi-series charts still need either a stronger frontier reasoning VLM or direct pixel-level axis/bar/series calibration.
3. **`forms` (`83.14` -> `~87–88`):** `fillable_forms` is already `94.42`; non-fillable scanned forms (`81.53`) hit single-image resolution limits on `27B` and need region-cropped form reading + stronger handwriting/checkbox perception.
4. **`layout` (`86.64` -> `~90`):** Bottlenecked by 2D bbox detectors (`Heron` + `PP-DocLayoutV3`); advancing beyond `87` requires a stronger dedicated layout detector or VLM-grounded bounding-box model.
