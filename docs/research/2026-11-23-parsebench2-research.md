# ParseBench 2 — research report (2026-11-23)

Goal: an open PDF parser that scores as high as possible on ParseBench 2 (PB2), building on
[docparse](https://github.com/RohithReddy20/docparse) (80.60 on PB v1), plus a Next.js app to inspect
results on our own files. Budget: $30 Modal credits, 200 Colab compute units, free Kaggle GPUs.

Sources: the PB2 paper ([arXiv 2611.17400](https://arxiv.org/abs/2611.17400)), the harness at
`run-llama/ParseBench` HEAD `6463ac1` (scorers read directly), the HF dataset card, the published outputs
of every leaderboard system ([ParseBench-2-results](https://huggingface.co/datasets/llamaindex/ParseBench-2-results)),
vendor posts, and model cards. Code references are to `src/parse_bench/evaluation/` in the harness.

Updated 2026-11-24 with corrections from a Gemini 4 Neon review, each checked against the code (§13).

---

## 1. TL;DR

1. **PB2 = 4,317 single-page cases from 2,645 documents, 13 subsets, six equal-weight dimensions**
   (text, formatting, tables, charts, layout, forms). One critical error caps a page at 90.
   **Not making errors matters more than recovering more content.**
2. **#1 is LlamaParse Agentic Plus at 90.93.** The best open model is KDL-nano-v2 at 80.13 (AGPL);
   the best permissively licensed one is Infinity-Parser2 Pro at 78.50. **Routing between existing
   open models tops out around 80–81**, so a high score has to come from building, not routing.
3. **Charts decide 85 vs 90.** They are 6.2 of the 12.4-point gap between Agentic Plus and the best
   permissive model (Infinity-Parser2 Pro), and 3.2 of the 10.8-point gap to KDL. Most systems describe a chart instead of extracting its data table, and a caption-only chart
   scores 0.
4. **About 91% of pages have a usable text layer.** Every sampled chart page was born-digital and
   mostly vector-drawn. Form values are flattened text in the text layer. This favours the docparse
   approach: deterministic extraction from the PDF, with models only where needed.
5. **Strikethrough and underline are nearly free for a text-layer system** (worth about +1.6 overall).
   Every open model sits at the "no decoration" floor (strike 13–17, underline 18–36).
6. **Several output-shape mistakes silently score 0:**
   - layout items without both `bbox` and `layout_segments`
   - a non-canonical layout label
   - a chart written only as a caption
   - a blank form whose labels are left out
   - text placed in HTML comments
7. **Budget is enough if iteration runs on cached model outputs** (CPU re-normalisation, as in v1).
   One full run of a ~4B VLM costs about $1.5–2.5 on a Modal H100. The $30 Modal credit renews
   monthly. 200 Colab units ≈ 37 A100-40GB hours, which is the cheapest compute available and best
   spent on fine-tuning.
8. **Integrity constraints:**
   - There is no hidden test split, so hold out by source document and disclose what was tuned on.
   - Filenames leak labels (`blank__…`, `filled__…`), so routing must use page content only.
   - The dataset and CLI change weekly, so pin both versions.

---

## 2. How PB2 scores

### Dimensions and weights

Overall = unweighted mean of the six dimensions. Within a dimension, each subset's page scores are
averaged, then the subsets are combined with fixed weights (`next_score.py:37-59`).

| Dimension | Subsets (weight) | Cases | Metric | Reads |
|---|---|---:|---|---|
| Text | `text` | 508 | `text_next` | page markdown + header/footer markdown |
| Formatting | `text` 3/4, `headings` 1/8, `strikeout` 1/16, `underline` 1/16 | 508 + 353 | `formatting_v2` | page markdown + header/footer markdown |
| Tables | `tables` 1/2; `complex_table_headers`, `financial_tables`, `insurance_filing_tables`, `dense_tables` 1/8 each | 1,753 | `table_next` | page markdown only |
| Charts | `charts` (1,025 charts, 36,958 values) | 568 | `chart_next` | page markdown only |
| Layout | `layout` | 535 | `layout_grounding_pass_rate` | `layout_pages[].items` only |
| Forms | `forms` 7/8 (450 typed fills + 50 blank), `fillable_forms` 1/8 (100 real AcroForm pages) | 600 | `forms_v2` | page markdown only |

### Severity cap

- **Formula:** page score = min(proportional score, cap), where
  `cap = 1 − 0.10·f(critical) − 0.05·f(major) − min(0.15, 0.01·minor)`.
- **Repeated errors:** the first error at each severity costs the full weight; later ones are scaled
  by `min(1, K/N)`. K is 400 tokens for text, 80 cells for tables, 10 cells for charts, and 10 items
  for formatting and forms.
  - **Exception:** on text, formatting and forms, minors are never scaled. Each costs 0.01, up to the
    0.15 cap (`metric_policy.py:194-196`).
- **Reading order (text only):** reading-order majors are capped separately at 0.30.
- **Layout has no cap.**
- **What counts as critical:**
  - a wrong digit, sign, unit or date
  - a flipped negation
  - an omission of 12 or more tokens
  - an invented sentence, table row or form value (an invented chart series is only major)
  - swapped table headers
  - a flipped checkbox
  - a strikethrough lost or added
  - an exponent lost (10<sup>9</sup> read as 109)
- **What is free:** equivalent formats (Markdown vs HTML tables, `**` vs `<b>`, transposed chart
  tables, units stated in headers).

### Pitfalls that zero or heavily cap a page

- **Layout:**
  - an item missing either `bbox` or `layout_segments` produces 0 boxes, with no error
  - a non-canonical label (`figure`, `paragraph`) raises
  - `page_number` ≠ 1 produces 0 boxes
  - a `raw_output` that looks like LlamaParse pages hijacks the adapter
  - a page with no layout scores 0
  - if no layout adapter is registered, scoring falls back to the LlamaParse adapter, which needs
    `llama-cloud` installed
  - **leaderboard CI re-scores with the harness's own code** (`leaderboard/README.md:59-60`), so an
    adapter registered in our package is never loaded there: the output must meet the default
    adapter's contract natively
- **Charts:**
  - a chart written only as a caption, `[Figure: …]` or a description scores 0
  - a unit that contradicts the chart (e.g. `(%)` on a USD chart) makes the whole column critical
  - placeholder legend names (`Series 1`, colours) are critical when the chart has several series and
    major for a single series
- **Forms:**
  - if under half of the labels are found, the score is capped at 2 × coverage
  - an invented value in a blank field (`0`, an example value, a name under an unsigned `Signature:`)
    is critical
- **Tables:**
  - a table with no header row is critical
  - every missing row is critical, and a whole missing table charges every row
  - "Table could not be extracted" scores 0
- **Text:**
  - running headers, footers and page numbers count as content; dropping them cost 5–18 points per
    sample page
  - a figure described in unmarked prose is an "invented passage" (critical); alt text or
    `[Image: …]` is free
  - a looped or repeated page scores about 0.2
- **Tables and forms:** two renderings of the same content with different values count as a
  "contradicting copy" (critical; major for table text).
- **Charts:** a second table under a placed chart's title is scored too. Each cell it gets wrong where the
  placed table is right is a value error (critical, or major for "value off") and an uncredited output cell
  (`chart_next.py:3755`, `:3932`). One such copy 4% off took a sample page from 0.80 to 0.40. An exact
  duplicate cost nothing, and neither did an untitled copy, but don't rely on that.
- **Everywhere:** HTML comments are invisible to every scorer.

---

## 3. Leaderboard and ceilings (live `leaderboard-v2.csv`, 59 rows)

| System | Overall | Text | Fmt | Tables | Charts | Layout | Forms | ¢/page |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LlamaParse Agentic Plus | **90.93** | 90.22 | 93.60 | 94.72 | **92.58** | 88.23 | 86.22 | 2.81 |
| Claude Opus 5.5 | 87.11 | **92.62** | 89.38 | 94.20 | 74.12 | 89.17 | 83.16 | 6.49 |
| LlamaParse Agentic | 86.96 | 86.96 | 92.75 | 89.90 | 77.39 | 89.17 | 85.62 | 1.25 |
| Reducto r-1 | 86.23 | 82.42 | 92.45 | 90.97 | 78.45 | 89.48 | 83.58 | 1.00 |
| LlamaParse Cost Effective | 85.52 | 85.47 | 91.94 | 87.22 | 73.07 | 89.49 | 85.92 | 0.38 |
| Gemini 4 argon (low) | 85.32 | 89.60 | 89.22 | 92.85 | 68.20 | 89.75 | 82.31 | 2.68 |
| Muse Ember 1.0 (low, closed, Meta) | 75.66 | 90.14 | 85.96 | 89.08 | 60.49 | 45.85 | 82.41 | 0.16 |
| KDL-Frontier-Parser-nano-v2 (open, **AGPL**) | 80.13 | 80.21 | 84.58 | 80.54 | 73.55 | 88.28 | 73.60 | — |
| Infinity-Parser2 Pro (open, Apache) | 78.50 | 85.55 | 78.70 | 86.27 | 55.64 | 87.04 | 77.79 | — |
| LightOnOCR-3 4B (open, Apache) | 78.39 | 84.30 | 81.44 | 78.25 | 62.90 | 84.25 | 79.19 | — |
| Chandra 2 (open, non-compete) | 77.54 | 77.18 | 88.80 | 85.60 | 62.48 | 72.44 | 78.73 | — |
| LightOnOCR-3 0.8B (open, Apache; docparse v1 base) | 77.20 | 81.24 | 80.87 | 78.58 | 61.22 | 82.90 | 78.41 | — |
| MinerU2.5-Pro (open, Apache weights) | 76.86 | 76.92 | 79.04 | 80.58 | 63.96 | 86.21 | 74.46 | — |

**Ceilings:**
- **Best score per dimension on the whole board:** text 92.62 / fmt 93.60 / tables 95.23 /
  charts 92.58 / layout 90.31 / forms 86.67, which averages to **91.8**.
- **Perfect per-page routing among permissive open models** reaches only **about 81**. Charts are the
  one dimension with a large gap (best permissive about 65 vs 92.6).
- **Where Agentic Plus loses points:** about 3.9 points to severity caps, against 5.8–9.1 for open
  models. The rest of the open models' gap is content they never output: no decorations, no chart
  tables, missed rows.
- **The published scores were computed at an older harness commit** (`results-v2.yaml`:
  `scored_with: d7efdbc`). Re-score baseline systems on our pinned version before comparing.

---

## 4. What the data looks like

- **Text layer:** from a sample of 30 PDFs per subset (390 total), about 91% have a native text layer,
  about 4% are scans with an OCR layer, and about 5% are image-only.
  - About one third of the `text` core is scanned. v1 listed 129 scanned, 47 multilingual and
    23 handwritten documents out of 553.
  - **22 of 30 sampled `complex_table_headers` pages are image-only.** That subset needs a strong
    table VLM.
- **Charts:** all 30 sampled chart pages are born-digital, and 25 of them have no large raster image,
  so the charts are mostly vector-drawn. In v1, 89% of chart pages were vector and only 42% of
  reference values were printed as labels.
- **Forms:** values are flattened pdf-lib fills; on the 3 sampled filled pages with values, 40 of 40
  appeared in the text layer. `fillable_forms` are real AcroForm pages, and the one sample checked was
  100% blank.
- **Dataset size:** about 1.5 GB.

---

## 5. How to win each dimension (from the scorer code)

### Text (`text_next`)

- **Transcribe everything:**
  - running headers, footers and page numbers
  - struck text, margin notes
  - signatures as `[signature: Name]`
- **Keep numbers and signs exact.** Key tokens (numbers, IDs, units, negators) weigh 1.5× in the
  recovery score.
- **Image descriptions** go only in alt text, `[Image: …]` or `<figure>`.
- **Never loop or duplicate.** Loops cost word precision plus "repeated text" or "invented passage"
  errors. (The `repetition_overshoot` rule is not part of the PB2 score.)

### Formatting (`formatting_v2`)

- **Syntax doesn't matter:** `**`/`<b>`, `~~`/`<del>`, `<u>`, `<sup>`/`<sub>`, `==`/`<mark>` all count.
  Exception: inside HTML `<td>`, Markdown `**` is read literally, so use tags there.
- **Strikethrough lost or added is critical.** Dropped strike or underline markup makes the page score
  0.00 on those subsets, and one false strike costs 10 points. So mark strikes only when confident.
- **Headings:** only relative order counts, so a uniform level shift is free.
  - A missed heading costs about 5 points.
  - An extra heading costs about 2–5.
  - Flattening all headings to one level is one major (5 points).
- **Redline pages** (any strikethrough in the reference): a lost or added underline becomes major
  instead of minor (`formatting_v2.py:1126`).

### Tables (`table_next`)

- **How it scores:** each cell becomes a triple (row-label path, column-header path, value).
- **Value credit:**

  | Case | Credit |
  |---|---|
  | Exact | 1 |
  | Case only differs | 0.95 |
  | Number reformatted | 0.9 |
  | Unit lost | 0.5 (major) |
  | Any digit or sign wrong | 0 (critical) |

- **Free:** table-syntax choice only. Each column or row out of order is a major error
  (`table_next.py:2818-2822`). A transposed table is major only if every cell keeps its labels and
  value; otherwise it is critical.
- **Format:**
  - Use HTML with explicit `colspan`/`rowspan`. Markdown can't express a blank header cell correctly.
  - Always include a header row.
  - Put the title in a spanning first row, a `<caption>`, or the line directly above the table.
- **Never drop or invent rows.**
- **Copy cell text exactly**, including `$`, `%` and accounting parentheses.

### Charts (`chart_next`)

- **One table per chart and per panel.** Copy legend names exactly as headers and put the title on the
  line directly above the table.
- **Shape is free:** series as rows or columns, long format (`Panel | Category | Series | Value`) and
  multi-level headers all score the same. LlamaParse Agentic Plus emits long format with `<th>` row
  labels and a `<caption>`.
- **Tolerance:**
  - 1% when data labels are printed (fewer decimals that round correctly count as exact);
  - otherwise the larger of 1% and 2% of the value range;
  - credit decays linearly to zero at 3× the tolerance.
- **Vector reconstruction from drawings meets these tolerances.**
- **Stacked charts** need segment values, not cumulative totals. **Pies** need the printed shares.
- **Never add `Total` or `Average` rows.**
- **Omissions** cost at most 0.10 per chart on the cap, but recall still drops.

### Layout (`layout_grounding_pass_rate`)

- **The label is not scored.** An element passes when:
  - IoA(GT, pred) ≥ 0.5 and IoA(pred, GT) ≥ 0.2;
  - token F1 ≥ 0.8 against the box text (words of other GT elements are removed first);
  - for tables, 90% of word centres fall inside the box.
- **Credit** ramps linearly from IoU 0.5 to 0.9. Elements are weighted by √area.
- **Extra boxes are penalised:**
  - duplicates
  - page-sized or container boxes
  - text boxes inside pictures
- **Box tightness:** the scorer already forgives text boxes that are looser than the reference, within
  annotated margins, but not tables, pictures, captions or rotated items
  (`layout_grounding_score.py:146-193`). Our "+10% of word height" came from two sample pages only.
  Never pad tables or pictures; re-measure text padding on the dev split.
- **One box per picture, heading, table, page header and page footer.** In the simulations, dropping
  pictures cost 33–49%.
- **Granularity:** line-level or paragraph-level boxes both pass for text.

### Forms (`forms_v2`)

- **Print every label, even blank ones**, preferably as `Label: [___]` (Neon: a bare `Label:` may let
  the next line be read as its value; unverified).
- **Write one field per line** as plain `Label: value`. Bold or bulleted `key: value` lines that match
  no reference field are charged as "extra field" (minor, 0.01 each, unscaled), so never bold page
  furniture such as `OMB No.: 1545-0112` (`forms_v2.py:5185-5224`).
- **Keep form tables as HTML grids.** "Form table not kept as a grid" is minor, and flattening can
  separate values from their captions, which is major.
- **Keep printed line numbers and IDs** beside captions (`1a`, `7`); dropping or changing one is minor.
- **Checkboxes:** `[x] Option` / `[ ] Option`, with the glyph immediately before its label.
  `✗ ☒ ● ■` all read as **checked**, so never use them as bullets.
- **Signatures:** `[signature: Name]` when signed; nothing after the colon when unsigned. A
  `[signature]` placeholder on an unsigned line is critical ("signature invented").
- **Never fill a blank field.**
- **Severity:**
  - critical: wrong or swapped values, invented values, flipped boxes
  - major: missing field, value lost
  - minor: blank field omitted
- **The cap binds** on 75–84% of core form pages for every system.
- **Where the points are:** typed `forms` (7/8 of the weight), where open models score 71–77 against
  about 85 for the best systems. On `fillable_forms` open models already score 87–92 against about 95,
  so AcroForm work is worth about 0.1 overall (`docs/forms-v203-all-parser-summary.json`).
- **For `fillable_forms`:** read the AcroForm widgets (`/V`, `/AS`, signature fields) for the field
  inventory and states. The scorer never reads the PDF itself, only our markdown.

---

## 6. How the top systems work (vendor-reported)

- **LlamaParse:**
  - Every tier runs its own model, **OCRia-1/1.5**: Qwen3.5-4B trained with SFT on about 101k pages,
    then GRPO with TEDS/GriTS/chart-keying rewards ([report](https://arxiv.org/abs/2611.00092)).
    The weights are not public.
  - A Parse Gateway routes pages by text-layer signals.
  - **Agentic Plus is a separate multi-pass pipeline.** Its published outputs show about 42K input
    and 6K output tokens per chart page, with specialized chart parsing on by default. Turning that on
    adds +9.7 to +14.2 chart points on the lower tiers.
- **KDL-nano-v2 (AGPL):** a single 1.2B model doing layout, then per-region crops, then per-category
  prompts, then rule-based post-processing. Probably MinerU2.5-derived.
- **Extend Parse 2:** layout and reading-order models, then per-block fine-tuned VLMs, plus checkbox,
  signature and strike detectors.
- **Reducto r-1:** a single-pass model, with an optional slow chart agent.
- **Pulse Ultra 2:** a single end-to-end model.

---

## 7. Open components

### Page models

| Model | Size | License | PB2 strengths | Notes |
|---|---|---|---|---|
| LightOnOCR-3-4B | ~4B | Apache-2.0 | forms 79.2, text 84.3, grounding 84.3 | One-pass grounding (boxes, labels, text, HTML tables). 3.36 p/s on H100. |
| Infinity-Parser2-Pro | 35B MoE (~3B active) | Apache-2.0 | tables 86.3 (best permissive), grounding 87.0 | Needs an A100-80GB or H100. |
| MinerU2.5-Pro-2605 | 1.2B | Apache weights (code moved off AGPL in v3.1) | grounding 86.2, charts 64.0 | Two-step (layout, then crops). |
| Nemotron 3.5 Parse | 0.63B | OpenMDW (commercial OK) | formatting 82.0 | 18.6 p/s on H200. Re-reading crops added +2.5 on v1. |
| Qwen3.8-27B (thinking) | 27B | Apache-2.0 | charts 65.2 (best permissive) | Too expensive for this budget. |
| PaddleOCR-VL-1.6 | 0.9B | Apache-2.0 | — | Chart-crop fallback in docparse v1. |
| dots.mocr | — | MIT + custom agreement | text 86.3 (best open), tables 83.4 | Charts 0.8, so a text/table component only. |
| TeleOCR | — | Apache-2.0 (per Neon) | layout 87.0 | Overall 65.9. |
| MiniCPM-OCR-1.0 | — | Apache-2.0 (per Neon) | tables 80.6 | |
| Granite-Vision-4.1-4B | 4B | Apache-2.0 (per Neon) | not on the board | `<tables_html>` and `<chart2csv>` modes; provider already in the harness. |
| aeddix-alpine-ocr | 1.2B | Apache-2.0 (per Neon) | not on the board | MinerU2.5-Pro fine-tune for tables and charts; provider in the harness. |
| OvisOCR3 | 0.8B | Apache-2.0 (per Neon) | not on the board | Chart-crop prompt; provider in the harness. |

### Specialists and data

- **Layout detectors:** Docling heron (RT-DETRv2, Apache-2.0, 17 classes including
  Checkbox-Selected/Unselected and Key-Value), egret, and PP-DocLayoutV3 (Apache-2.0).
  - DocLayout-YOLO is AGPL; avoid it.
- **Forms:**
  - PyMuPDF widgets give `field_value`, `on_state()` and `is_signed`, but PyMuPDF is **AGPL-3.0**
    (or a commercial Artifex licence). Permissive alternatives: pypdf (BSD-3) for AcroForms, pypdfium2
    (Apache-2.0/BSD-3) and pdfplumber/pdfminer.six (MIT) for text, styles and drawings.
  - tech4humans conditional-DETR signature detector (Apache-2.0).
- **Chart training data:**
  - usable: ChartNet (2.5M samples, CDLA-P-2.0) and CoSyn-400K (ODC-BY);
  - avoid: ChartGalaxy and Infinity-Doc2-5M (non-commercial).
- **Licenses to avoid for a product:** KDL and PyMuPDF (AGPL), Chandra 2, Surya and OXR (non-compete),
  HunyuanOCR (region exclusions). Check OCRia-nano-1 (Neon says non-compete RAIL-M; unverified) and
  dots.mocr (MIT plus a custom agreement).

### Techniques with evidence

- **Re-reading table and chart crops at higher resolution** (Nemotron: +2.5 on v1).
- **Per-element agreement voting** (TeleOCR) and an OCR-correctness verifier (MiniCPM-OCR-Jev,
  AUROC 93.6). Naive word-level voting hurts.
- **Repetition control** with an n-gram logits processor and a higher-temperature retry.
- **Fine-tuning on ChartNet:** granite-vision-2B went from 53.8 to 70.3 on chart extraction.
- **GRPO with unit-test rewards** (olmOCR 2: +12 on tables). Full-page GRPO needs far more compute
  than we have. A crop-level chart model (1.2–4B, LoRA) with `chart_next` as the reward might fit in
  about 20 A100-hours (Neon's estimate; unproven).

---

## 8. Budget

| Resource | What it buys |
|---|---|
| Modal $30/month (Starter plan, renews) | H100 at $3.95/h, A100-80GB at $2.50/h, L4 at $0.80/h. One full PB2 run (4,317 pages) of a ~4B VLM ≈ $1.5–2.5 on H100, so about 13–22 runs/month. ~1B model ≈ $1.0–1.6. 27B-class model ≈ $6–12. |
| Colab 200 CU (expire after 90 days) | ≈ 37 h A100-40GB, 117 h L4, or 168 h T4. Cheapest A100 time available; use it for fine-tuning and detector or chart-model runs. |
| Kaggle | T4×2, about 30 h/week, 12 h sessions. No BF16, FP8 or FlashAttention-2, so Qwen3.5-based models may not run. Fine for detectors and small models. |

- **Iteration should use cached outputs.** Run each GPU model once over the dataset, cache the raw
  outputs, and iterate on fusion and normalisation on CPU, as docparse v1 did.
- **Cheap-API reference points** (break the open-weight constraint; can't be paid with these credits):

  | Model | Full run (standard) | Batch |
  |---|---|---|
  | GPT-6 luna | $5.5 | $2.7 |
  | Claude Haiku 5.5 | $6.7 | $3.4 |
  | Muse Ember 1.0 | $6.9 | — |
  | Gemini 3 Flash | $38 | $19 |
  | Claude Opus 5.5 | $280 | $140 |

---

## 9. What docparse v1 brings, and what PB2 adds

**Reusable (about 6k lines, already registered as a parse-bench provider):**
- PyMuPDF text layer with styles (bold, italic, underline, strike, highlight, font sizes); PyMuPDF
  is AGPL, see §7
- LightOnOCR-3-0.8B grounding on Modal vLLM
- Docling heron layout fusion with word attribution
- vector chart reconstruction (bars, lines, areas, pies, legends, panels)
- PaddleOCR-VL-1.6 chart-crop fallback
- native-style alignment
- DocLayNet Section/Text reranker
- an oracle/paired-report tool
- the cached-output re-normalisation workflow

**New or changed for PB2:**

| Area | Change |
|---|---|
| Forms | A whole new dimension: label inventory, values, checkbox states, signatures, blank discipline, AcroForm widgets. |
| Layout output contract | `bbox` + `layout_segments`, canonical labels, `md` text, no stray HTML. Emit the default adapter's contract natively: leaderboard CI never loads our own adapter. |
| Charts | Must output the **full** reference table (every series × category), not v1's spot-checked points. Plus legend exactness, unit discipline and panel titles. |
| Tables | Scorer is now triple-based with severity. Need a stronger table model (Infinity-Parser2-Pro, or LightOnOCR-3-4B plus text-layer cell snapping), especially for image-only complex headers and dense tables. |
| Text | Page furniture now counts as content. About one third of the `text` core is scanned or handwritten, so it needs a good OCR VLM (the 4B over the 0.8B). |
| Strike/underline subsets | Text-layer decoration is a large advantage here. |

---

## 10. Integrity and eligibility

- **No hidden split.** All cases and answer keys are public (Apache-2.0), and no rule forbids tuning on
  them. The KDL submission stated "no output ensemble or group-wise routing".
  - **Our rule:** hold out by source document, route on page content only (never filenames or
    subset folders), never read `.test.json` during inference, and disclose what was tuned on.
- **Listing:**
  1. Publish outputs to `ParseBench-2-results` (`pack.py`, `validate.py`, HF PR).
  2. Open a PR on the leaderboard CSV/YAML.
  3. CI re-scores the row and fails if any score differs by more than 0.5.
- **Category is uncertain for a hybrid.** The "Open Weight" category already contains multi-stage
  pipelines (KDL, Nemotron, MinerU, PaddleOCR-VL). Open an issue to confirm before submitting.
- **The target moves:** 11 CLI releases and 8 dataset revisions since Nov 11. A scorer fix alone moved
  one vendor's layout score from 73.3 to 82.4. Pin the CLI and dataset revision for every reported
  number.

---

## 11. Implications (decisions still open)

**Realistic per-dimension targets for an open-weights text-layer hybrid** (estimates, not
measurements):

| Dimension | Target | Main lever |
|---|---|---|
| Text | 86–90 | Text layer for native pages, 4B VLM for scans, page furniture kept |
| Formatting | 88–93 | Native styles, strike/underline from drawings |
| Tables | 85–90 | Strong table VLM + text-layer value snapping + HTML spans |
| Charts | 80–88 | Vector reconstruction + VLM for raster/pies + strict table conventions |
| Layout | 86–90 | heron + text attribution + exact output contract |
| Forms | 82–87 | Text-layer values + widgets + checkbox detection + blank discipline |
| **Overall** | **≈ 84.5–89.7** | |

- **Where that lands:** first among open systems by a wide margin, and in range of Claude Opus 5.5
  (87.1) and LlamaParse Agentic (87.0).
- **Beating Agentic Plus (90.93) would need** charts of about 90 and tables of about 93.
- **Second opinion (Gemini 4 Neon):** 86.7–88.9 realistic (centre 87.8), about 90 at a stretch.
  Per dimension: text 89–90.5, formatting 90–92, tables 90–92, charts 79–83.5, layout 88.8–90,
  forms 83.5–85.5. It ranks the work as formatting and text first (+2.5 overall), then charts (+3.0–3.7,
  largest but slowest), forms (+1.0), tables (+0.9) and layout (+0.5). Sending the ~600 hardest pages
  to a paid model would cost about $10–14 per full run for +1.7–2.6 (roughly 90–91). Its advice: ship
  open-weights first, and add paid escalation later as an optional tier.

**Open decisions:**
- open weights only vs. selective paid-API escalation
- whether to fine-tune (charts and forms on synthetic data with Colab A100 hours)
- leaderboard submission vs. personal result

---

## 12. Results UI notes

**Reference screen:** LlamaParse's Parse page.
- PDF viewer on the left with page navigation and zoom.
- Right-hand tabs: Build (config), Results (Markdown / text / JSON / images, per page), Code, History.
- Upload/paste at the top, and a Run button with estimated cost.

**Existing code to borrow:** `apps/visual_grounding_viewer` in the harness (Apache-2.0; React + pdf.js +
react-markdown/rehype-raw) already does grounding overlays and markdown↔box linking.

**What our app adds:**
- uploading our own files
- runs and comparison across our pipeline versions
- per-case score breakdown with severity events
- side-by-side view against any leaderboard system's published output for the same page
- per page, which bound set the score: proportional score, severity cap or forms coverage cap
- a ranked list of error types by overall points lost, with subset weights applied (one chart page is
  worth about 2.9 table pages overall)
- per-dimension diff views: table and chart cells against the reference, a forms field checklist, and
  reference vs predicted layout boxes with the reason each failed
- holdout pages hide their answer keys and error details by default

The last four came from Neon's review. Keep the Python, PDF and Modal work in a FastAPI service rather
than Next.js route handlers, and index published leaderboard outputs locally (SQLite or DuckDB) so
switching systems is instant.

---

## 13. Second opinion: Gemini 4 Neon (2026-11-24)

Asked through the Antigravity CLI (`agy`, model `gemini-4-neon-xhigh`) with this report (GitHub link
removed) and the harness code. Full review: `2026-11-24-gemini-neon-review.md`; its line references
point to the report before these edits.

**Checked in code and applied above:**
- the layout output must meet the default adapter contract, because CI never loads ours
- column or row out of order is major; transposition is critical unless no cell changes
- bold or bulleted lines that are not real form fields cost "extra field" minors
- minors are not scaled on text, formatting and forms
- an invented chart series and a single-series placeholder legend are major, not critical
- formatting reads page headers and footers
- redline pages promote underline errors to major
- flattened headings cost 5, not 10
- "contradicting copy" is a table and forms event; charts charge a wrong second copy through their own
  copy check instead (added 2026-11-24 after the Fable design pass; re-run and confirmed)
- the chart-gap arithmetic in the TL;DR
- PyMuPDF is AGPL
- fillable forms have little headroom

**Disputed:** box padding. The scorer's outward tolerance makes padding text boxes roughly neutral and
padding tables or pictures harmful. Our +10% came from two pages, so settle it on the dev split.

**New ideas worth keeping:**
- **Charts:**
  - extract geometry from the vector drawing, then give those values to a VLM as hints so it binds
    colours to legend names and resolves stacked and dual-axis charts;
  - on raster charts, take 3–5 sampled reads and keep the per-cell median;
  - run a CPU check before output: no placeholder names, no Total rows, consistent units, and every
    printed label present.
- **Tables:**
  - use Infinity-Parser2-Pro, and re-read hard tables from 300–400 DPI crops;
  - snap numbers to the PDF text layer, and flag text-layer lines missing from the table;
  - on scans, run a second model and flag disagreements;
  - always emit HTML with rectangular spans and KDL-style multi-row `<th>` marking.
- **Text-layer health check:** look for broken font maps, invisible OCR layers and symbol fonts before
  trusting the PDF text.
- **Compute:** about 30–35 Colab A100-hours for chart fine-tuning (SFT, then GRPO); Kaggle for
  detectors; Modal for cached full passes.
