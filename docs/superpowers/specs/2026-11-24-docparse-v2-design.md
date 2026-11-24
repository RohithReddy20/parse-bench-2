# docparse v2: ParseBench 2 pipeline and results viewer

Date: 2026-11-24. Status: draft for the owner's approval.

Companion documents in this repo. This design overrides them where they differ.

- `docs/research/2026-11-23-parsebench2-research.md`: the benchmark, its scoring rules and output traps,
  model survey, budget. Called "the report" below.
- `docs/research/2026-11-24-fable-architecture-plan.md`: the detailed architecture (IR schema, module
  tree, model-call protocol, per-dimension checks, viewer API, phase table). Called "the plan" below.
  Its sections are part of this design except where §2 or §11 here says otherwise.
- `docs/research/2026-11-24-gemini-neon-review.md`: a second opinion on the report.

---

## 1. What this is and why

The goal is the highest ParseBench 2 score a pipeline of open-weights models can reach **without training
any model**, plus a local web app, modelled on LlamaParse's Parse screen, for running that pipeline on
one's own PDFs and seeing exactly what it produced. In the owner's words, we are "running to find a proper
solution": a real parser that is strong on any PDF, not a set of benchmark tricks.

ParseBench 2 (LlamaIndex, released 2026-11-12) scores 4,317 single-page cases on six equally weighted
dimensions: text, formatting, tables, charts, layout and forms. Today's board (`leaderboard-v2.csv` at
harness commit `6463ac1`):

| System | Overall | Text | Formatting | Tables | Charts | Layout | Forms |
|---|---:|---:|---:|---:|---:|---:|---:|
| LlamaParse Agentic Plus (#1) | 90.93 | 90.22 | 93.60 | 94.72 | 92.58 | 88.23 | 86.22 |
| Claude Opus 5.5 | 87.11 | 92.62 | 89.38 | 94.20 | 74.12 | 89.17 | 83.16 |
| KDL-Frontier-Parser-nano-v2 (best open system) | 80.13 | 80.21 | 84.58 | 80.54 | 73.55 | 88.28 | 73.60 |
| Best open model per dimension | (83.73) | 86.29 dots.mocr | 88.80 Chandra 2 | 86.27 Infinity-Parser2 Pro | 73.55 KDL nano v2 | 88.28 KDL nano v2 | 79.19 LightOnOCR-3 4B |

Prior work is the owner's docparse v1 (github.com/RohithReddy20/docparse at `9451b48`). It combined the
PyMuPDF text layer, LightOnOCR-3-0.8B, Docling heron layout and vector chart extraction, and scored 80.60 on
ParseBench v1.

## 2. Decisions already made

**By the owner:**

1. No model training of any kind (no fine-tuning, SFT, GRPO or LoRA). v1's small bundled section-text
   classifier, already trained, may stay as it is.
2. Accuracy first; efficiency only after.
3. Any open-weights model may be used if it is the best at its job. Licences are not a criterion.
4. Leaderboard submission rules are not a criterion. The goal is the score.
5. Development uses the benchmark's own data: all 4,317 pages and their answer keys.
6. Budget: $30 a month of Modal credit (renews monthly), 200 Google Colab compute units, free Kaggle (T4×2).
7. A Next.js viewer like LlamaParse's Parse screen: the PDF on the left; Results, Code and History tabs on
   the right; Upload; Run with an estimated cost.
8. The approach in §4 (approach C) and the work order in §10, approved with "go ahead". That includes
   reading typed form values from the PDF's own text and keeping a 20% check slice (§8.2).

**Calls I made (open to correction):**

- The code lives in this repo. The package keeps the name `docparse`, and v1 is imported as the starting
  point (§5.1).
- PyMuPDF stays as the PDF library, behind a small interface.
- The viewer runs locally for one user, with no accounts. GPU endpoints scale to zero, and cold starts of
  1–3 minutes on uploads are acceptable.
- Published outputs of other systems are used for baselines, loss maps and as stand-in model readings while
  prototyping. Every reported score comes from our own model runs, because the viewer needs those anyway.
- No deadline was given, so the plan runs to the end. If a date appears, cut M9 (the full viewer) and the
  measured raster-chart path first.
- ParseBench 2 is the only scored measure, since the owner said the goal is the score. Multi-page uploads
  are parsed page by page and shown page by page, with no stitching across pages (§3 non-goals). A few
  real-world PDFs go through the viewer at each milestone gate as a smoke test, with no score gate.
- New harness and dataset releases are adopted at milestone gates, never mid-milestone. The final number is
  reported on the newest release at M8, with the reference systems re-scored on that same version (§8.1).

## 3. Goals, non-goals, anti-goals

**Goals**

1. ParseBench 2 overall in the 86–89 range, centre about 88. It is measured with the benchmark's own scorer
   at a pinned harness and dataset revision. Layout is scored through the harness's default adapter, so the
   number means the same as the leaderboard's numbers.
2. The exact same pipeline runs on any PDF uploaded in the viewer.
3. After accuracy: the same score (within 0.5) at 40% or less of the GPU seconds per page.

**Non-goals**

- Training or fine-tuning.
- Licence vetting, and leaderboard submission and its category rules.
- Paid APIs in the pipeline. A disabled `external` model slot may exist; nothing depends on it.
- A hosted or multi-user viewer.
- Stitching across pages: removing repeated headers and footers, joining a table that continues onto the
  next page, or ranking heading levels across a whole document. ParseBench 2 cases are single pages and
  can't measure it. A multi-page upload's markdown is its pages in order.

**Anti-goals** (failures even if the number goes up)

- Anything at parse time keyed to benchmark file names, folders, case ids or answer keys. Uploads have
  arbitrary names, so the same bytes must give the same output under any name.
- Rules that only fit the pages they were written on. The check slice (§8.2) exists to catch these.
- Exploiting scorer quirks that don't make the output better for a reader, for example the fact that an
  untitled contradicting chart copy is currently not charged. Quirks also vanish with harness updates.
- A score that cannot be reproduced from saved model outputs.
- Paying twice for the same model call.

## 4. Approaches considered

| | Approach | Expected overall | Trade-off | Verdict |
|---|---|---:|---|---|
| A | One strong open model end to end (KDL nano v2 or Infinity-Parser2 Pro) plus clean-up | 80–82 | Simplest. Inherits the model's errors: charts at most 74, forms at most 79, wrong digits anywhere. | Rejected |
| B | Send each region type to the best open model for it | about 83.7 | Moderate effort. Still trusts model characters, digits and values. | Kept inside C as its model layer |
| C | **PDF first:** the PDF supplies characters, digits, styles, chart geometry and typed form values; models supply structure, names, and readings of scanned pixels; checks and a repair ladder catch the errors the scorer punishes hardest | 86–89 | Most engineering. The largest gains land exactly where models are weakest (charts, forms, strikeout and underline, digits). | **Chosen** |
| D | Fine-tune a VLM for charts and forms | — | — | Ruled out by the owner |

Why C works: the costliest scoring events (a wrong digit, a dropped row, an invented value, a renamed chart
series, a lost strikeout) have an exact source of truth inside born-digital PDFs. The scanned minority falls
back to approach B's best-model readings, cross-checked by a second model.

## 5. Architecture

The full design is in the plan, §1. Summary:

```
PDF bytes → S0 ingest → S1 PDF evidence → S2 trust map → S3 regions → S4 readers
                                                                      ↓ candidates
                 S7 lint ← S6 writers ← S5 select + check → repair → more model calls (≤ 2 rounds)
```

- **S0 Ingest:** hash the bytes, split pages, render them. Nothing downstream sees the file name.
- **S1 PDF evidence (CPU):** characters with font, size, flags, colour and content-stream order; drawings;
  images; annotations; AcroForm widgets; the typed-in "fill layer" on flattened forms; health scores.
- **S2 Trust map:** per text line and per region, decide whether the PDF text can be trusted there
  (native, partly native, OCR layer, or pixels only). The page class is only a summary.
- **S3 Regions:** boxes proposed by a layout detector and a grounding VLM, snapped to the ink they contain,
  plus a reading order.
- **S4 Readers:** one per region kind (text, table, chart, form, picture). Each reader returns candidates
  with provenance.
- **S5 Select and check:** a fixed rule picks one candidate per region. Validators report findings and never
  edit content. A failure triggers the repair ladder: deterministic fix, sharper crop, second model, the
  biggest model shown the conflicting readings, then the least costly safe fallback.
- **S6 Writers:** one markdown writer and one layout writer, both reading the same intermediate
  representation (IR).
- **S7 Lint:** hard output guards for every format trap that silently scores zero.

Principles: the PDF wins wherever it is healthy; each fact is rendered exactly once; every model call is
content-addressed and cached; batch runs and viewer runs share one code path; the parser knows nothing about
the benchmark.

### 5.1 Code layout and v1

Use the plan's §1.4 module tree: `docparse/{core,pdf,models,layout,readers,checks,write,bench,api}`,
`evalkit/` for scoring and the ledger, `serve/` for Modal, Colab and Kaggle runners, and `apps/viewer/` for
Next.js. Import v1 at `9451b48` and apply the plan's keep, rewrite and drop table:

- **Keep:** text layer, styles, chart geometry (`charts/vector.py`, `charts/textlabels.py`), the paired
  report and the oracle.
- **Rewrite:** the core that combines readings (`hybrid.py`, `fusion.py`) so native text is the default.
- **Drop:** `charts/integrate.py`, which appends extra chart tables, and path-keyed caches.

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
- **Rounds:** a parse runs at most three call rounds (round 0, plus two repair rounds).

### 5.3 Models (bake-off candidates)

With licences no longer a criterion, the candidates per job are below. The number in brackets is that
model's board score on the relevant dimension.

| Job | Candidates |
|---|---|
| Layout proposals | Docling heron (detector, also check-box classes); KDL nano v2 (88.28); Infinity-Parser2 Pro (87.04); MinerU2.5-Pro (86.21); LightOnOCR-3 4B (84.25) |
| OCR on scans and untrusted lines | dots.mocr (86.29); Infinity-Parser2 Pro (85.55); surya-ocr-2 (84.91); LightOnOCR-3 4B (84.30) |
| Styles on scans | Chandra 2 (formatting 88.80); surya-ocr-2 (86.32) |
| Table structure | Infinity-Parser2 Pro (86.27); Chandra 2 (85.60); dots.mocr |
| Chart naming and raster charts | KDL nano v2 (73.55); Qwen3.8-27B thinking (65.15); MinerU2.5-Pro (63.96) |
| Form structure | LightOnOCR-3 4B (79.19); Chandra 2 (78.73); surya-ocr-2 (78.11) |
| Signatures on scans | tech4humans signature detector |

- **Bake-off rule:** each job is decided on a fixed stratified sample of about 300 pages from the
  development part. Pick by dimension score, then by GPU seconds.
- **Unverified properties:** sizes, memory, runtime support and output formats for KDL nano v2, Chandra 2,
  dots.mocr and surya-ocr-2 have not been checked. M0 and M1 check them.
- **Where models run:**
  - Kaggle runs detectors and small models in FP16 (it has no BF16).
  - Colab is the main inference pool for models up to about 8B.
  - Modal runs models that need more than 40 GB (Infinity-Parser2 Pro, a 35B MoE; the 27B) and the
    viewer's scale-to-zero endpoints.

### 5.4 Writers and the output contract

As in the plan's §1.8, with the report's format rules.

**Markdown:**

- Running headers, footers and page numbers stay in the body, and also go in the page header and footer
  fields with identical strings.
- Tables are HTML with `<th>` header rows, explicit `colspan`/`rowspan` and the title on the line above.
  Cell styles use HTML tags.
- Each chart or panel is exactly one HTML table, with the title in `<caption>`, legend names verbatim as
  headers, plain numbers, and no hedges, ranges or derived totals.
- Pictures get nothing or alt text, never prose.
- Form fields are plain `Caption: value` lines, with no bold or bullets. A blank is `Caption:`. Check boxes
  are `[x] Option` / `[ ] Option`. A signed line is `Caption: [signed]`. Form tables are HTML grids, and
  printed line ids are kept.
- Never an HTML comment. Never a second rendering of the same content.

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

Details, checks and repair actions are in the plan, §2.1–2.6.

| Dimension | Best open now | Target | How |
|---|---:|---:|---|
| Charts | 73.6 | ~81 | Find each chart and panel. Decompile born-digital charts from their drawing (bars, lines, areas, pie wedges, markers against calibrated axis ticks) and take printed data labels from the text layer. A VLM reads the schema (title, series names in legend order, categories, units). Bind measured series to names by legend colour, legend order and agreement with the VLM's rough values. Resolve each value: printed label, then geometry, then (raster only) the per-cell median of 3 VLM reads. One table per chart, always. Hard checks cover placeholder legends, invented totals, missing titles and label/geometry disagreement. A 2-day CPU feasibility spike in M1 measures how many charts decompile cleanly. |
| Forms | 79.2 | ~87 | Detect the appended "fill layer" (late content-stream text in fonts the base page doesn't use, sitting on field geometry). Its strings are the values: exact, blanks stay blank, ticks are its glyphs. Field geometry (boxes, underlines, comb cells, check-box squares, grids) and AcroForm widgets decide ownership. A model page read supplies captions and structure, with its values snapped to fill-layer strings. Scans and pages without a fill layer use the model path, heron check-box classes and the signature detector. |
| Formatting | 88.8 | ~91.5 | Bold and italic from fonts and flags. Strikeout and underline from thin drawn lines and markup annotations, with tightened geometry tests (inside the word span, at the right height, not a table or field line). Superscript and subscript from flags and geometry. Heading levels from font-size ranks. On scans: a line detector plus a VLM yes/no on the crop. |
| Tables | 86.3 | ~90 | A model gives the grid. Every cell's text is re-rendered from the PDF words assigned to it, which removes wrong digits and exposes dropped rows. A PDF-grid candidate from ruling lines and word gaps cross-checks it. Dense or flagged tables are re-read on a 300–400 DPI crop. Image-only tables are read by two models and compared cell by cell. HTML is normalised (rectangular spans, header rows, original column order). |
| Text | 86.3 | ~89.5 | Trusted lines are rendered from PDF characters (spacing, ligatures, hyphenation, superscripts). Headers, footers and page numbers are kept. Every trusted line lands in exactly one region. Reading order comes from the grounding VLM, checked against an XY-cut of native lines. Scans get two OCR models; where they disagree on critical tokens (numbers, ids, units, negators), a high-resolution re-read decides by majority. |
| Layout | 88.3 | ~88.8 | Detector and VLM boxes are proposals. Final boxes are the union of the words or grid or image they contain. Multi-paragraph boxes are split at paragraph gaps, and duplicates and page-sized containers are removed. Item text is the native words inside the box. A third detector is added only if an ablation pays. |
| **Overall** | 80.1 best system; 83.7 best per dimension | **~88** | Realistic range 86–89; some dimension usually underperforms. |

## 7. Data

- **Benchmark data:** `llamaindex/ParseBench-2` at the `data_revision` pinned in `results-v2.yaml`.
- **Published outputs:** `llamaindex/ParseBench-2-results`. All top open systems are verified rows with
  outputs for every page: KDL nano v2, Infinity-Parser2 Pro, LightOnOCR-3 4B, Chandra 2, MinerU2.5-Pro,
  Qwen3.8-27B, surya-ocr-2 and others. LlamaParse Agentic Plus is published too, as the upper reference.
- **Downloaded files are untrusted data:** read them as data and never execute them.

## 8. Evaluation loop

### 8.1 Scoring

- **Command:** the harness's own scoring command, the same one CI's `rescore.py` uses, at harness
  `6463ac1` (release 2.2.1).
- **Speed:** model outputs are cached, so a full re-score is CPU-only, about 20 minutes. Per-page outputs
  are memoised, so a change that touches only charts re-runs only chart pages.
- **Loss table:** points lost per error class per dimension, ranked. It is computed as counterfactuals from
  the saved event counts through `metric_policy`, weighted by page weight (one chart page is worth about 2.9
  table pages overall). This table is the work queue.
- **Ledger:** one row per run, with harness commit, dataset revision, code sha, config hash, cache snapshot
  hash, per-dimension and per-subset scores, GPU seconds per page, zero-score pages and lint fallbacks.
- **Version policy:** the harness and dataset change often (see the harness `CHANGELOG.md`). The pin is
  fixed within a milestone and moved to the newest release at each milestone gate, as its own change:
  1. Re-score the last accepted run on the new pin from cache, and record the delta.
  2. Re-score the reference systems' published outputs on the same pin.
  3. Run model passes only for added or changed pages.
  4. Assign new cases to folds by the split rule (§8.2).
  5. Re-base the next milestone's targets if a scorer change moved the reference systems.

  The final number (M8) is on the newest release, next to the reference systems re-scored on that same
  release. "All pages" means all pages in that release.

### 8.2 Development part and check slice

- **The split:** 80% of source documents are the development part (look at pages, write rules, tune
  thresholds). 20% are a check slice: scored on every run, pages not inspected. Group by source document
  (forms by template) and stratify by subset. A case's fold is a seeded hash of its group key, so cases
  added in later dataset releases get a fold without moving existing ones. Assignments are recorded in
  `evalkit/splits/v1.json`.
- **Final score:** all pages of the release in use at M8 (4,317 today).
- **Overfitting signal:** when a milestone's check-slice gain is under half its development gain, the
  milestone's new thresholds are reviewed and the simplest kept.

### 8.3 Keeping a change

A change is kept when, on the development part:

- overall rises by at least 0.05, or it is a declared refactor moving overall by less than 0.02;
- no dimension drops more than 0.3 and no subset more than 0.5, unless the trade is written down;
- zero-score pages and lint fallbacks don't increase;
- the contract, rename-invariance and isolation tests pass.

A claimed gain also needs at least 90% probability from a paired bootstrap over documents.

### 8.4 Guard tests

- **Rename invariance:** 50 PDFs copied under random names give byte-identical outputs.
- **Isolation:** a page parsed alone equals the same page inside a full run.
- **No answer keys:** an audit hook fails the run if any answer-key file is opened during parsing.
- **Default adapter:** the contract test in §5.4.

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
      - Evidence: every candidate for the selected region, with provenance, checks and repair history.
      - Score (benchmark pages only): score, which bound set it, events, and diffs per dimension.
      - Compare: against another run of ours or any leaderboard system's published output.
    - **Code:** the CLI and Python call that reproduces this run with its exact config.
    - **History:** past runs of this document with config, cost, duration and score.
- **Other screens:** Runs (all history), and Benchmark (scores per dimension and subset, the loss table,
  worst pages, and a diff between two runs).
- **Cost estimate:** the CPU stages run first and plan round 0 exactly, including cache hits. Repair rounds
  are estimated from historical trigger rates.
- **Reuse:** port the overlay, markdown-linking, reference-box, hover and text-diff code from ParseBench's
  `apps/visual_grounding_viewer` (Apache-2.0), as listed in the plan's §4.5.

## 10. Work order and exit criteria

All targets are on the development part. A missed target sends the loss table back to decide whether to
continue or move on. Effort is for one engineer, and GPU spend is an estimate.

| # | Milestone | Exit criteria | Effort | GPU |
|---|---|---|---|---|
| M0 | Foundations: harness and dataset pins, split, ledger, loss table; **score the published outputs of the top open systems and LlamaParse Agentic Plus** (no GPU); package skeleton and IR; PDF evidence including the fill layer; model-call cache and manifest runner; guard tests | Published scores of two reference systems reproduced within 0.5; loss maps per system written; guard tests green; manifest round trip on Kaggle | 5–6 d | $0 |
| M1 | First full score: round-0 model passes; native-first text; v1 table and chart paths under the one-table rule; new writers and lint; **chart feasibility spike** | Overall ≥ 81; layout ≥ 86; lint fallbacks < 1% of pages; spike report | 6 d | $6–9 Modal, 6–10 CU |
| M2 | Viewer MVP: Parse screen with Results sub-views (Rendered, Markdown, JSON, Layout, Evidence, Score), benchmark case list, loss table | A benchmark page and an uploaded PDF both run end to end from the UI | 4 d | < $1 |
| M3 | Text and formatting | Text ≥ 89; formatting ≥ 91; strikeout and underline subsets ≥ 88 | 6 d | $1–2, 4–6 CU |
| M4 | Forms | Forms ≥ 86; no invented values on pages with a fill layer | 6 d | $0–1 |
| M5 | Tables | Tables ≥ 89.5; complex headers ≥ 80; no born-digital table digit absent from the PDF | 7 d | $5–8, 4–6 CU |
| M6 | Charts | Charts ≥ 80; structural events (missing title, renamed series, lost legend) on < 5% of charts | 12 d | $8–12, 8–12 CU |
| M7 | Layout polish and repair tuning | Layout ≥ 88.5; repair calls on ≤ 15% of regions | 3 d | $1–2 |
| M8 | Freeze: final run on all pages; check-slice review | Each check-slice dimension within 1.0 of development; overall ≥ 86 | 3 d | $0 |
| M9 | Full viewer: Code and History tabs, Compare, Runs, Benchmark dashboard, presets | All screens in §9 | 5 d | < $1 |
| M10 | Efficiency: ablate each component, route big models by page content, try smaller models | Overall within 0.5 of M8 at ≤ 40% of its GPU seconds per page; cost-per-page table | 6 d | $5–10 |

Totals (estimate): about 63 working days; $30–45 of Modal over two to three monthly cycles; 30–40 of the 200
Colab units on the main line, with the rest for bake-offs and reruns.

## 11. Where this overrides the plan

- **Plan §1.9 (integrity guards):** keep the bytes-only entry point, rename invariance, isolation and the
  answer-key audit hook. They protect the viewer use and score reproducibility. Drop the submission
  disclosure notes.
- **Plan §2 and §1.5:** the model rosters are replaced by §5.3 here (licences no longer filter).
- **Plan §3.1:** the "holdout" becomes the check slice of §8.2. The final number is on all pages.
- **Plan §6.2:** answered as follows:
  - fill layer: yes;
  - submission: not a goal;
  - licences: no filter;
  - target: the highest score with open weights;
  - budget: as given;
  - tuning on public data: yes;
  - published outputs as prototyping stand-ins: yes;
  - viewer: local;
  - deadline: none given.
- **Plan §4.4:** the right-hand tabs are Results, Code and History as the owner asked, with the plan's views
  under Results.

## 12. Left to the builder

- All thresholds, prompts and sampling settings, and the bake-off winners.
- Per-line versus per-region layout segments, and whether text inside pictures belongs to the picture item.
  Both are measured on the development part.
- Blank form fields as bare `Caption:` versus a placeholder, by A/B in M4.
- How v1 is imported (plain copy recording the source commit, or a history-preserving merge).
- Package and environment tooling (v1 uses uv and hatchling), Next.js version, styling and component
  library.
- SQLite schemas beyond the plan's §4.2, and how jobs stream progress (SSE or polling).

## 13. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| The chart decompiler handles far fewer PB2 charts than hoped | Charts stuck near the low 70s; overall about 86 | M1 spike measures it before M6 is scheduled; strengthen the model path with KDL nano v2 |
| Form ownership on dense, multi-column forms | Forms nearer 84 than 87 | Geometry plus model owner agreement; arbiter on disagreement; forms work starts with the loss table |
| Serving the 35B MoE and 27B within the Modal budget | Fewer table and chart re-reads | Crops only; cache everything; fall back to 4B–8B models on Colab |
| Harness or dataset releases mid-project | Scores move without code changes; targets go stale | Pin within a milestone; adopt releases at gates with the reference systems re-scored (§8.1) |
| Rules fitted to how ParseBench 2 was built (e.g. synthetic form fills) pass the check slice | Weaker results on real PDFs | Build detectors on general signals (appended overlays, field geometry), never benchmark fingerprints; real-PDF smoke test at each gate |
| Unchecked properties of newly added models | Bake-off delays | Check size, runtime and output format in M0 before planning passes |
