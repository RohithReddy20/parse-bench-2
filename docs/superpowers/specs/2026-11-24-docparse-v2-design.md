# docparse v2: ParseBench 2 pipeline and results viewer

Date: 2026-11-24. Status: revised the same day around OCRia-1 (revision 2). The owner set the plan in §2;
this document records it.

Companion documents in this repo. This design overrides them where they differ.

- `docs/research/2026-11-23-parsebench2-research.md`: the benchmark, its scoring rules and output traps,
  model survey, budget. Called "the report" below. It predates OCRia-1's release.
- `docs/research/2026-11-24-fable-architecture-plan.md`: the detailed architecture (IR schema, module
  tree, model-call protocol, per-dimension checks, viewer API). Called "the plan" below. Its sections are
  part of this design except where §2 or §11 here says otherwise.
- `docs/research/2026-11-24-gemini-neon-review.md`: a second opinion on the report.

---

## 1. What this is and why

The goal is **at least second place on the ParseBench 2 leaderboard**: an overall score above Claude Opus
5.5's 87.13, aiming for 89. The pipeline uses open-weights models only and **trains nothing**. A local web
app, modelled on LlamaParse's Parse screen, runs the same pipeline on one's own PDFs and shows exactly what
it produced.

ParseBench 2 (LlamaIndex, released 2026-11-12) scores 4,317 single-page cases on six equally weighted
dimensions: text, formatting, tables, charts, layout and forms. Today's board (`leaderboard-v2.csv` at
harness commit `6e4479a`):

| System | Overall | Text | Formatting | Tables | Charts | Layout | Forms |
|---|---:|---:|---:|---:|---:|---:|---:|
| LlamaParse Agentic Plus (#1) | 90.95 | 90.26 | 93.59 | 94.72 | 92.65 | 88.23 | 86.24 |
| Claude Opus 5.5 (#2, the score to beat) | 87.13 | 92.62 | 89.37 | 94.20 | 74.24 | 89.17 | 83.18 |
| **OCRia-1** (#7, best open model, our base) | 85.24 | 87.87 | 91.58 | 86.28 | 77.66 | 85.96 | 82.10 |
| KDL-Frontier-Parser-nano-v2 (next open system) | 80.10 | 80.23 | 84.58 | 80.54 | 73.34 | 88.28 | 73.61 |

**OCRia-1** (`huggingface.co/llamaindex/OCRia-1`) is LlamaIndex's open 4B document model, a Qwen3.5-4B
fine-tune. It reads one page image per call and writes markdown with HTML tables, charts as data tables,
inline styles, check boxes, `[___]` blanks and page-furniture tags. It is the best open model on five of
the six dimensions; only its layout is behind (KDL nano v2 has 88.28). We need 1.9 points over it to pass
Claude Opus 5.5 and 3.8 to reach 89.

Prior work is the owner's docparse v1 (github.com/RohithReddy20/docparse at `9451b48`). It combined the
PyMuPDF text layer, LightOnOCR-3-0.8B, Docling heron layout and vector chart extraction, and scored 80.60 on
ParseBench v1. v2 reuses its text-layer, style and chart-geometry code.

## 2. Decisions already made

**By the owner:**

1. No model training of any kind (no fine-tuning, SFT, GRPO or LoRA).
2. Accuracy first; efficiency only after.
3. Any open-weights model may be used if it is the best at its job. Licences are not a criterion.
4. Leaderboard submission rules are not a criterion. The goal is the score.
5. Development uses the benchmark's own data: all 4,317 pages and their answer keys.
6. Budget: $30 a month of Modal credit (renews monthly), 200 Google Colab compute units, free Kaggle (T4×2).
   Every model call goes through the cache.
7. A Next.js viewer like LlamaParse's Parse screen: the PDF on the left; Results, Code and History tabs on
   the right; Upload; Run with an estimated cost.
8. **The OCRia-1 plan (approach E in §4):**
   - OCRia-1 reads every page. No reader is built from scratch. Its output is parsed into the IR, then
     corrected with the PDF itself and a few specialist models.
   - M0 scores OCRia-1's published outputs with the harness, with no GPU, and builds its loss table. That
     table orders all later work.
   - Each fix in §6 is kept only if it raises the score.
   - Models: OCRia-1, Docling heron, KDL nano v2 (layout) and the tech4humans signature detector. Every
     model OCRia-1 already beats is dropped, including Infinity-Parser2 Pro and Qwen3.8-27B. Second reads
     on scans use OCRia-1 at a different resolution or crop.
   - Kept from the first design: the IR, the model-call cache and manifest runner, the checks, the writers
     and output contract, the evaluation loop, the viewer and the guard tests.
9. Working rules: before any GPU run, the number of new model calls and the estimated cost are shown and
   approved. Only scores actually run with the harness's own scorer are reported.

**Calls I made (open to correction):**

- The code lives in this repo. The package keeps the name `docparse`. v1 code is copied in module by
  module as each fix needs it (text layer, styles, chart geometry), not imported wholesale.
- PyMuPDF stays as the PDF library, behind a small interface.
- The viewer runs locally for one user, with no accounts. GPU endpoints scale to zero, and cold starts of
  1–3 minutes on uploads are acceptable.
- Published OCRia-1 outputs are used for the M0 baseline and loss table, and as stand-in readings while a
  fix is prototyped. From M1 on, every reported score comes from our own model runs.
- No deadline was given, so the plan runs to the end. If a date appears, cut M9 (the full viewer) first.
- ParseBench 2 is the only scored measure. Multi-page uploads are parsed and shown page by page, with no
  stitching across pages (§3 non-goals). A few real-world PDFs go through the viewer at each milestone gate
  as a smoke test, with no score gate.
- New harness and dataset releases are adopted at milestone gates, never mid-milestone (§8.1).

## 3. Goals, non-goals, anti-goals

**Goals**

1. ParseBench 2 overall **above 87.13**, aiming for 89. It is measured with the benchmark's own scorer at a
   pinned harness and dataset revision. Layout is scored through the harness's default adapter, so the
   number means the same as the leaderboard's numbers.
2. The exact same pipeline runs on any PDF uploaded in the viewer.
3. After accuracy: the same score (within 0.5) at 40% or less of the GPU seconds per page.

**Non-goals**

- Training or fine-tuning.
- Building page readers from scratch, or serving any model OCRia-1 already beats.
- Licence vetting, and leaderboard submission and its category rules.
- Paid APIs in the pipeline. A disabled `external` model slot may exist; nothing depends on it.
- A hosted or multi-user viewer.
- Stitching across pages: removing repeated headers and footers, joining a table that continues onto the
  next page, or ranking heading levels across a whole document. ParseBench 2 cases are single pages and
  can't measure it. A multi-page upload's markdown is its pages in order.

**Anti-goals** (failures even if the number goes up)

- Anything at parse time keyed to benchmark file names, folders, case ids or answer keys. Uploads have
  arbitrary names, so the same bytes must give the same output under any name. (The harness's own `ocria_1`
  pipeline computes boxes only for `layout/` cases. Ours computes them for every page.)
- Rules that only fit the pages they were written on. The check slice (§8.2) exists to catch these.
- Exploiting scorer quirks that don't make the output better for a reader. Quirks also vanish with harness
  updates.
- A score that cannot be reproduced from saved model outputs.
- Paying twice for the same model call.

## 4. Approaches considered

| | Approach | Expected overall | Trade-off | Verdict |
|---|---|---:|---|---|
| A | One strong open model end to end, plus clean-up | 85.2 (OCRia-1, measured by its publishers) | Simplest. Inherits the model's errors: wrong digits, form values, chart values, loose boxes. | The base of E |
| B | Send each region type to the best open model for it | about 85.6 | OCRia-1 is already the best open model on five dimensions; only layout gains. | Only its layout part is kept |
| C | PDF first: build text, table, chart and form readers from the PDF, with models for structure (the first design) | 86–89 | Most engineering. Rebuilds what OCRia-1 already does well. | Superseded by E |
| D | Fine-tune a VLM | — | — | Ruled out by the owner |
| E | **OCRia-1 first, PDF-corrected:** OCRia-1 supplies structure and a first reading of everything; the PDF replaces characters, digits, values, styles and boxes wherever it holds the exact answer; a few specialists cover layout and signatures | 87.5–89.5 | Starts at 85.24. Each fix is small, separately measured, and dropped if it doesn't pay. | **Chosen** |

Why E works: the costliest scoring events (a wrong digit, an invented form value, a wrong chart value, a
lost strikeout) have an exact source of truth inside born-digital PDFs, and OCRia-1 never sees the PDF: it
reads only the image. Its structure (which rows, which columns, which caption owns which value, which
series) is the hard part and is already good. So the work is alignment and substitution, not reading.

## 5. Architecture

The full design of the kept parts is in the plan, §1. Summary of the revised flow:

```
PDF bytes → S0 ingest → S1 PDF evidence ──────────────────────────────┐
                │                                                      ▼
                └→ S2 OCRia-1 page read → S3 parse answer into IR → S4 correctors (one per fix, each a flag)
                                                                       ↓ candidates
                 S7 lint ← S6 writers ← S5 select + check → repair → more model calls (≤ 2 rounds)
```

- **S0 Ingest:** hash the bytes, split pages, render them. Nothing downstream sees the file name.
- **S1 PDF evidence (CPU):** words and characters with font, size, flags, colour and content-stream order;
  drawings; images; annotations; AcroForm widgets; the typed-in "fill layer" on flattened forms; health
  scores per line (can the text layer be trusted here?).
- **S2 OCRia-1 page read:** one cached model call per page, with the model's own prompt and the settings of
  the harness's `ocria_1` pipeline.
- **S3 Parse into the IR:** OCRia-1's answer becomes regions in reading order (headings, paragraphs, lists,
  tables, chart tables, form fields, check boxes, pictures, page furniture). This is the *base candidate*
  of every region. Written back out unchanged, it must score the same as OCRia-1 alone (§5.4).
- **S4 Correctors:** one per fix in §6. A corrector reads a region's base candidate and the PDF evidence
  and adds a corrected candidate with provenance. Each is behind a config flag, so each is ablated alone.
  Layout boxes come from the layout corrector (§6), not from the answer.
- **S5 Select and check:** a fixed rule picks one candidate per region. Validators report findings and never
  edit content. A failure triggers the repair ladder: deterministic fix, then an OCRia-1 re-read at a
  different resolution or on a zoomed crop, then the least costly safe fallback (usually the base
  candidate).
- **S6 Writers:** one markdown writer and one layout writer, both reading the same IR.
- **S7 Lint:** hard output guards for every format trap that silently scores zero.

Principles: OCRia-1's structure is the default; the PDF wins on characters, values, styles and boxes
wherever it is healthy; each fact is rendered exactly once; every model call is content-addressed and
cached; batch runs and viewer runs share one code path; the parser knows nothing about the benchmark.

### 5.1 Code layout and v1

Use the plan's §1.4 module tree, with `readers/` replaced by `base/` and `fixes/`:

```
docparse/
  core/        ir.py, config.py, hashing.py, geometry.py
  pdf/         backend.py, pymupdf_backend.py, textlayer.py, drawings.py, fill_layer.py, widgets.py,
               health.py, render.py
  models/      client.py, calls.py, cache.py, manifest.py, adapters/ (ocria.py, heron.py, kdl.py, signature.py)
  base/        ocria_parse.py (answer → IR), postprocess.py
  fixes/       layout.py, forms.py, tables.py, charts/ (vector.py, textlabels.py, bind.py), text.py,
               formatting.py
  checks/      text.py, table.py, chart.py, form.py, page.py, repair.py
  write/       markdown.py, layout_items.py, lint.py
  pipeline.py  the only public entry point: parse(pdf_bytes, config, client)
  bench/       parse-bench provider and pipeline registration (the only importer of parse_bench.inference)
  api/         FastAPI service for the viewer
evalkit/       pins, data fetch, splits, scoring driver, ledger, loss table, paired report
serve/         Modal apps, the manifest runner, Colab and Kaggle notebooks that call the same runner
apps/viewer/   Next.js app
```

From v1 at `9451b48`, copy in when the matching fix starts:

- **Text layer and styles** (`textlayer.py`, `styles.py`, `textalign.py`): for the text, formatting and
  table fixes.
- **Chart geometry** (`charts/vector.py`, `charts/textlabels.py`): for the chart fix.
- **Paired report and oracle:** into `evalkit/`.
- **Not taken:** `hybrid.py`, `fusion.py`, `charts/integrate.py` (it appends extra chart tables), the
  LightOnOCR provider and all path-keyed caches.

Two import rules are enforced by a test: the parser never imports `parse_bench` outside `bench/`, and never
imports `evalkit`.

### 5.2 Model calls and the cache

As in the plan's §1.5–1.6:

- **Call key:** each `ModelCall` is keyed by a hash of the model revision, runtime, prompt, parameters
  (including sample index and seed) and input hashes.
- **Clients:** three, sharing one interface. `CacheOnlyClient` writes misses to a manifest for batch GPU
  sessions; `EndpointClient` is for the viewer; `ReplayClient` fails on any miss and is used for reported
  scores.
- **Runner:** one resumable manifest runner works on Modal, Colab and Kaggle.
- **Storage:** results are stored compressed and indexed in SQLite with tokens and GPU seconds. That index
  is the source for cost estimates and for the efficiency phase.
- **Plan before spend:** `docparse cache plan` prints the number of new calls per model and the estimated
  cost. No GPU session starts before the owner has seen that and agreed.
- **Rounds:** a parse runs at most three call rounds (round 0, plus two repair rounds).

### 5.3 Models

| Model | Job | Where it runs |
|---|---|---|
| **OCRia-1** (4B, BF16 weights 10.35 GB, needs vLLM 0.30 or later) | Reads every page. Re-reads dense tables, chart panels and scan regions from zoomed crops or at another resolution. | Colab A100 or Modal. Kaggle's T4s have no BF16. |
| **Docling heron** (RT-DETRv2 detector) | Layout region boxes, check-box classes | Kaggle T4 |
| **KDL-Frontier-Parser-nano-v2** (layout 88.28) | Layout region boxes | Kaggle T4 if it runs in FP16, else Colab |
| **tech4humans signature detector** | Signatures on scanned forms | Kaggle T4 |

- **Dropped:** every model OCRia-1 already beats, including Infinity-Parser2 Pro, Qwen3.8-27B,
  LightOnOCR-3, dots.mocr, Chandra 2, surya-ocr-2 and MinerU2.5-Pro. There are no bake-offs.
- **OCRia-1 settings** (from the harness's `ocria_1` pipeline, which produced the 85.24):
  - page rendered at 300 DPI with pypdfium2 (CropBox and rotation applied);
  - the model's `prompt.txt` as the system message, the image as the only user content, thinking off;
  - greedy decoding, up to 24,576 output tokens in a 32,768-token context;
  - `ocria_postprocess` turns the raw answer into scored markdown.
  M1 reproduces these exactly before anything is changed.
- **Its code:** `ocria_grounding/` and `examples/` in the model repository may be run. Everything else
  downloaded is data.
- **Unverified:** OCRia-1 throughput on stock vLLM on an A100 (the card gives about 320 tokens per second for
  one request); the effective image size the model sees (the card says the benchmark numbers used about 4.1
  megapixels per page); KDL nano v2's size, runtime and output format. M1 checks them on a small pilot
  before the full run is costed.

### 5.4 Writers and the output contract

As in the plan's §1.8, with the report's format rules.

**Markdown:**

- Running headers, footers and page numbers stay in the body, and also go in the page header and footer
  fields with identical strings.
- Tables are HTML with `<th>` header rows, explicit `colspan`/`rowspan` and the title on the line above.
  Cell styles use HTML tags.
- Each chart or panel is exactly one HTML table, with legend names verbatim as headers, plain numbers, and
  no hedges, ranges or derived totals.
- Pictures get nothing or alt text, never prose.
- Form fields, check boxes and blanks keep OCRia-1's spelling (`[x]`, `[ ]`, `[___]`) until an A/B on the
  development part shows another spelling scores higher.
- Never an HTML comment. Never a second rendering of the same content.

**Round-trip rule.** With every corrector off, the pipeline's markdown for a page must score the same as
OCRia-1's own post-processed markdown. This is tested in M0 against the published outputs: the answer is
parsed into the IR and written back, and the re-scored dimensions must match the baseline within 0.1.
Anything the IR can't represent is carried as an opaque block and written back verbatim.

**Layout:**

- The output satisfies the harness's default adapter: `layout_pages[]` in PDF points, every item with both
  `bbox` and `layout_segments`, canonical labels read from the pinned harness, and item text in `md`.
- `raw_output` carries nothing another adapter could match on.
- Boxes are tight to the ink, with no padding (never on pictures, tables, captions or rotated items).
- A contract test proves the default adapter is selected.

Lint (S7) blocks HTML comments, unbalanced tags, header-less tables, chart regions without a table,
placeholder series names, invented `Total`/`Average` rows, duplicate captions, items missing a box, and
looped text.

## 6. Per-dimension design

OCRia-1 supplies the structure in every dimension. Each row below is one corrector. "Now" is OCRia-1's
board score; the targets are estimates until measured. Checks and repair actions are in the plan, §2.1–2.6,
applied to the corrected candidates.

| Dimension | Now | Target | Fix |
|---|---:|---:|---|
| Forms | 82.10 | ~87.5 | Keep OCRia-1's structure: which caption owns which value, the order, the grids. Replace each value with the PDF's typed-in text (the fill layer: late content-stream text in fonts the base page doesn't use, sitting on field geometry) or the AcroForm widget value. Take check-box states and blank fields from the PDF: a field with no fill is blank, a tick is a fill-layer glyph or a widget state. A model value with no support on a healthy page is removed. Scans keep OCRia-1's reading, plus heron's check-box classes and the signature detector. |
| Tables | 86.28 | ~90 | Keep its rows and columns. Align each cell's text to the PDF words inside the table box and replace the cell text with the words assigned to that cell. PDF words left unassigned expose dropped rows. Dense or flagged tables are re-read by OCRia-1 from zoomed crops and the readings merged by the same word assignment. Image-only tables keep the model's text. |
| Charts | 77.66 | 81+ | Keep its series names and categories where they match text printed on the chart; otherwise take the chart's own strings. Replace values with printed data labels from the text layer, or with numbers measured from the chart's vector drawing against calibrated axis ticks (v1's chart code). Measured series are bound to OCRia-1's series by legend colour, legend order and agreement with its values. Exactly one table per chart or panel. Raster charts keep OCRia-1's values. |
| Layout | 85.96 | ~88.8 | Region boxes from Docling heron and/or KDL nano v2, each tightened to the PDF words (or grid, or image) it contains. Item text is the words of OCRia-1's answer that fall in the box. Start from OCRia-1's own `ocria_grounding` package, which already combines heron regions with text-layer word boxes and falls back to the model's attention on pages without a text layer. The gains to test: KDL boxes, tightening, splitting multi-paragraph boxes, removing duplicates and containers, and keeping low-scoring pictures. |
| Text | 87.87 | ~89.5 | Align its text to the PDF text layer line by line. Wherever the text layer is healthy, correct characters and digits from the PDF and restore omitted lines. Its reading order and block structure stay. Scans get a second OCRia-1 read at another resolution; disagreements on numbers, ids, units and negators are re-read from a crop. |
| Formatting | 91.58 | ~93 | Add or remove strikeout and underline using the PDF's thin drawn lines and markup annotations (inside the word span, at the right height, not a table or field line). Add or remove bold and italic using the PDF's fonts and flags. Its headings stay. |
| **Overall** | **85.24** | **~88.3** | The six targets average 88.3. Reaching 89 needs charts near 85, or several dimensions above target. Above 87.13 needs about 40% of the planned gains. |

Each fix is kept only if it passes §8.3 on the development part. The order of work comes from the M0 loss
table, not from this table.

## 7. Data

- **Benchmark data:** `llamaindex/ParseBench-2` at the `data_revision` pinned in the harness's
  `leaderboard/results-v2.yaml` (`2e954075…` at harness `6e4479a`).
- **Published outputs:** `llamaindex/ParseBench-2-results`, folder `results/ocria_1/` at the
  `results_revision` in the same file (`497e243f…`): one JSONL per subset, 13 files, about 62 MB. Each
  record holds the post-processed markdown and, for layout cases, the layout items. `raw_output` is empty,
  so the model's raw answers (with page-furniture tags) come only from our own run in M1.
- **Reference rows:** LlamaParse Agentic Plus and Claude Opus 5.5 are scored from the same repo when a
  comparison is needed.
- **OCRia-1:** `huggingface.co/llamaindex/OCRia-1`, pinned by revision (`dae83aa5…` today).
- **Downloaded files are untrusted data:** each download goes in its own directory outside the repo, is read
  with `python -I`, and is never executed. The exceptions are the harness (a pinned dependency) and
  OCRia-1's own code from the official repository.

## 8. Evaluation loop

### 8.1 Scoring

- **Harness:** `run-llama/ParseBench` at main commit `6e4479a`, pinned as a git dependency. The latest
  tagged release, v2.2.1 (`6463ac1`), predates OCRia-1 and its `ocria_1` pipeline.
- **Command:** the harness's own scoring command, the same one its `scripts/leaderboard/rescore.py` uses:
  `parse-bench evaluation run --next … --next_only` over all 13 subsets.
- **Speed:** model outputs are cached, so a full re-score is CPU-only, about 20 minutes. Per-page outputs
  are memoised, so a change that touches only charts re-runs only chart pages.
- **Loss table:** points lost per error type per dimension, ranked, for OCRia-1 first and for every later
  run. It is computed as counterfactuals from the saved event counts through `metric_policy`, weighted by
  page weight (one chart page is worth about 2.9 table pages overall). This table is the work queue.
- **Ledger:** one row per run, with harness commit, dataset revision, code sha, config hash, cache snapshot
  hash, per-dimension and per-subset scores, GPU seconds per page, zero-score pages and lint fallbacks.
- **Version policy:** the harness and dataset change often (see the harness `CHANGELOG.md`). The pin is
  fixed within a milestone and moved to the newest main commit at each milestone gate, as its own change:
  1. Re-score the last accepted run on the new pin from cache, and record the delta.
  2. Re-score OCRia-1's published outputs and the reference rows on the same pin.
  3. Run model passes only for added or changed pages.
  4. Assign new cases to folds by the split rule (§8.2).
  5. Re-base the targets if a scorer change moved the references.

  The final number (M8) is on the newest pin, next to the references re-scored on that same pin.

### 8.2 Development part and check slice

- **The split:** 80% of source documents are the development part (look at pages, write rules, tune
  thresholds). 20% are a check slice: scored on every run, pages not inspected. Group by source document
  (forms by template) and stratify by subset. A case's fold is a seeded hash of its group key, so cases
  added in later dataset releases get a fold without moving existing ones. Assignments are recorded in
  `evalkit/splits_data/v1.json`.
- **Final score:** all pages of the release in use at M8 (4,317 today).
- **Overfitting signal:** when a milestone's check-slice gain is under half its development gain, the
  milestone's new thresholds are reviewed and the simplest kept.

### 8.3 Keeping a change

A fix is kept when, on the development part, against the same run with its flag off:

- overall rises by at least 0.05, or it is a declared refactor moving overall by less than 0.02;
- no dimension drops more than 0.3 and no subset more than 0.5, unless the trade is written down;
- zero-score pages and lint fallbacks don't increase;
- the contract, rename-invariance and isolation tests pass.

A claimed gain also needs at least 90% probability from a paired bootstrap over documents. A fix that fails
stays in the code behind its flag, off, with the measurement recorded in the ledger.

### 8.4 Guard tests

- **Rename invariance:** 50 PDFs copied under random names give byte-identical outputs.
- **Isolation:** a page parsed alone equals the same page inside a full run.
- **No answer keys:** an audit hook fails the run if any answer-key file is opened during parsing.
- **Default adapter:** the contract test in §5.4.
- **Round trip:** the rule in §5.4.

## 9. Viewer

- **Stack:** Next.js (App Router, TypeScript) in `apps/viewer`, plus a FastAPI service in `docparse/api`
  that calls `pipeline.parse` with an `EndpointClient`. Local, one user, SQLite, files stored by hash. API
  as in the plan's §4.3.
- **Parse screen:**
  - **Top bar:** Upload, document name, preset (Standard or Max accuracy), and Run showing the estimated
    seconds and dollars plus endpoint state.
  - **Left:** PDF viewer with page navigation, zoom and overlays (regions, lines, words, table cells, form
    fields, fill layer, and reference boxes on benchmark pages).
  - **Right tabs:**
    - **Results**, with sub-views:
      - Rendered: markdown linked to boxes on hover and click.
      - Markdown.
      - JSON: the page IR.
      - Layout: the items as sent to the scorer.
      - Evidence: for the selected region, OCRia-1's base candidate and every corrected candidate, with
        provenance, checks and repair history.
      - Score (benchmark pages only): score, which bound set it, events, and diffs per dimension.
      - Compare: against another run of ours, OCRia-1 alone, or any leaderboard system's published output.
    - **Code:** the CLI and Python call that reproduces this run with its exact config.
    - **History:** past runs of this document with config, cost, duration and score.
- **Other screens:** Runs (all history), and Benchmark (scores per dimension and subset, the loss table,
  worst pages, and a diff between two runs).
- **Cost estimate:** the CPU stages run first and plan round 0 exactly, including cache hits. Repair rounds
  are estimated from historical trigger rates.
- **Reuse:** port the overlay, markdown-linking, reference-box, hover and text-diff code from ParseBench's
  `apps/visual_grounding_viewer`, as listed in the plan's §4.5.

## 10. Work order and exit criteria

All targets are on the development part. M3 to M7 are listed in the default order; the M0 loss table sets
the real order. A missed target sends the loss table back to decide whether to continue or move on. Effort
is for one engineer, and GPU spend is an estimate that is replaced by a measured plan before each run.

| # | Milestone | Exit criteria | Effort | GPU |
|---|---|---|---|---|
| M0 | Foundations and baseline: harness and dataset pins, data fetch, split, scoring driver, ledger; **score OCRia-1's published outputs** (no GPU); **OCRia-1 loss table**; package skeleton and IR; OCRia-1 answer parser and writers with the round-trip rule; model-call cache and manifest runner; guard tests | OCRia-1's 85.24 reproduced within 0.1 overall and 0.5 per dimension; loss table written; round trip within 0.1 per dimension; guard tests green | 5 d | $0 |
| M1 | Our own OCRia-1 run and the layout swap: pilot, costed plan, full page pass through the cache; `ocria_grounding` reproduced; then heron and/or KDL boxes tightened to PDF words | Our run within 0.5 of the published score on every dimension; layout ≥ 87.5 with the swap, or the swap is recorded as not paying | 6 d | 10–25 CU or $5–10 Modal (to be measured on the pilot) |
| M2 | Viewer MVP: Parse screen with Results sub-views (Rendered, Markdown, JSON, Layout, Evidence, Score), benchmark case list, loss table | A benchmark page and an uploaded PDF both run end to end from the UI | 4 d | < $1 |
| M3 | Forms: fill layer, widgets, value replacement, check boxes, blanks, signatures | Forms ≥ 86; no value on a fill-layer page that the PDF doesn't support | 5 d | $0–1 |
| M4 | Tables: word-to-cell replacement; dropped-row detection; zoomed re-reads of dense tables | Tables ≥ 89; no born-digital table digit absent from the PDF | 6 d | 3–6 CU |
| M5 | Charts: name matching, printed labels, vector measurement, one-table rule | Charts ≥ 81 | 8 d | 2–4 CU |
| M6 | Text and formatting: text-layer alignment and correction; strikeout, underline, bold and italic from the PDF | Text ≥ 89; formatting ≥ 92.5 | 5 d | 2–4 CU |
| M7 | Second reads on scans, repair tuning, layout polish | Layout ≥ 88.5; repair calls on ≤ 15% of regions | 3 d | 3–6 CU |
| M8 | Freeze: final run on all pages; check-slice review | Each check-slice dimension within 1.0 of development; **overall > 87.13** | 3 d | $0 |
| M9 | Full viewer: Code and History tabs, Compare, Runs, Benchmark dashboard, presets | All screens in §9 | 5 d | < $1 |
| M10 | Efficiency: ablate each fix, route re-reads by page content, try quantised OCRia-1 | Overall within 0.5 of M8 at ≤ 40% of its GPU seconds per page; cost-per-page table | 5 d | $5–10 |

Expected path if every fix lands on target: 85.2 (M0) → 85.7 (layout) → 86.6 (forms) → 87.2 (tables) →
87.8 (charts) → 88.3 (text and formatting).

## 11. Where this overrides the plan

- **Plan §0, §1.2, §2 (readers):** superseded by approach E. There are no from-scratch readers and no
  native-first text rendering: OCRia-1's answer is the base candidate and the PDF corrects it.
- **Plan §1.4 (v1 table):** only the text layer, styles, chart geometry and the paired report are taken
  (§5.1).
- **Plan §1.5 and §2 (model rosters, bake-offs):** replaced by §5.3. Four models, no bake-offs.
- **Plan §1.7 (repair ladder):** the "second model" and "arbiter" rungs become one rung: OCRia-1 at a
  different resolution or crop.
- **Plan §1.9 (integrity guards):** keep the bytes-only entry point, rename invariance, isolation and the
  answer-key audit hook. Drop the submission disclosure notes.
- **Plan §2.4 (charts):** the VLM schema read and the K=3 median path are dropped. OCRia-1's chart table is
  the schema and the raster fallback.
- **Plan §3.1:** the pins are those in §7 and §8.1. The "holdout" becomes the check slice of §8.2. The final
  number is on all pages.
- **Plan §5 (phases):** replaced by §10.
- **Plan §6.2:** answered by §2.
- **Plan §4.4:** the right-hand tabs are Results, Code and History as the owner asked, with the plan's views
  under Results.

## 12. Left to the builder

- All thresholds and alignment settings, and which of heron, KDL or both supplies layout boxes.
- Per-line versus per-region layout segments, and whether text inside pictures belongs to the picture item.
  Both are measured on the development part.
- Blank form fields as `[___]`, bare `Caption:` or another spelling, by A/B in M3.
- OCRia-1 render settings for second reads (resolution, crop padding).
- Package and environment tooling (v1 uses uv and hatchling), Next.js version, styling and component
  library.
- SQLite schemas beyond the plan's §4.2, and how jobs stream progress (SSE or polling).

## 13. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Our OCRia-1 run doesn't reproduce the published score (render size, vLLM version, post-processing) | Every later delta is measured from the wrong base | M1 copies the harness pipeline's settings exactly, pilots on 50 pages against the published outputs, and gates on 0.5 per dimension |
| The gains are smaller than estimated because OCRia-1's losses aren't the ones the PDF can fix | Overall stalls near 86.5 | The M0 loss table measures this before any fix is built; work goes where the points are |
| A corrector damages pages OCRia-1 had right (misaligned words, wrong cell, false strike) | A fix nets zero | Each fix sits behind a flag and is kept only if it passes §8.3; the base candidate is always the fallback |
| Charts stay near 78 because vector measurement covers few charts | Overall about 87.7; 89 out of reach | Measure decompiler coverage on the development part first; printed labels alone are the cheap first step |
| Layout gain needs KDL nano v2, whose runtime is unchecked | Layout stays near 86–87 | heron tightening first (no new model); KDL checked on a pilot |
| OCRia-1 GPU cost on stock BF16 vLLM is higher than estimated | Fewer re-reads | Pilot first; cache everything; crops only for re-reads |
| Harness or dataset releases mid-project | Scores move without code changes | Pin within a milestone; adopt releases at gates with the references re-scored (§8.1) |
| Rules fitted to how ParseBench 2 was built (e.g. synthetic form fills) pass the check slice | Weaker results on real PDFs | Build detectors on general signals (appended overlays, field geometry), never benchmark fingerprints; real-PDF smoke test at each gate |
