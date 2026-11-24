# docparse v2 for ParseBench 2: architecture and phased build plan

> Copied from the Fable design pass on 2026-11-24. Its two largest claims (A1 chart copy penalty, A3 fill-layer
> values) were re-run and reproduced the same numbers. A3 locates fields with the reference boxes, so it is an
> upper bound for value extraction, not an end-to-end forms score.

Date: 2026-11-24. Author: Fable (design pass on the research report, the Neon review, docparse v1 and the
harness at `6463ac1`).

**How to read the evidence tags.** Every factual claim carries one of three tags.

- **[V]** I checked it in code or data during this pass. Appendix A lists each check and its numbers.
- **[R]** It comes from the research report (whose §13 lists what was code-verified) and I did not re-check it.
- **[U]** Unverified: an estimate, a vendor claim, or a model property nobody here has measured.

All score targets in this document are **[U]** estimates. No GPU job was run for this plan.

---

## 0. The design in one page

**What we are building.** A parser that treats the PDF itself as the primary evidence and uses open models
only for what the PDF cannot say: structure, semantics, and pixels on scans. It keeps docparse v1's assets
(text layer, style alignment, vector chart reconstruction, cached-output workflow) and replaces v1's core
(`hybrid.py`, `fusion.py`, `charts/integrate.py`), which is shaped around PB1 metrics that rewarded recall.

**The six decisions that matter most.**

1. **PDF first for characters, digits, styles and geometry.** On pages with a healthy text layer, no model
   output ever overrides a PDF character. v1 did the opposite: it kept the VLM's characters and transferred
   styles onto them [V]. PB2 makes one wrong digit a critical error that caps the page at 90 [R], so the
   exact source has to win.
2. **One intermediate representation with candidates.** Each region keeps every reading it received, with
   provenance, and a deterministic selector picks one. Check-and-repair is "validators ask for another
   candidate", not ad-hoc patching.
3. **Every model call is content-addressed and cached; the pipeline is a pure function of
   (PDF bytes, config, cache).** Batch runs and the viewer's online runs use the same code through one
   `ModelClient`. In batch mode a cache miss becomes a line in a request manifest, which is how multi-pass
   repair stays compatible with "run each GPU model once".
4. **Exactly one rendering of each fact.** v1 appends extra candidate chart tables because PB1 only measured
   recall [V]. On PB2 a second, disagreeing version of a chart took a real page from 0.80 to 0.40 [V].
5. **Forms are read from the fill layer.** In the sampled filled forms, the typed values are a trailing block
   of text objects in fonts used nowhere else on the page [V]. That gives exact values, exact blank
   discipline and exact tick marks without a model. The remaining forms problem is structure
   (which caption owns which value), and that is where the model and geometry work goes [V].
6. **Charts are measured, not read.** Values come from printed labels, then from vector geometry, and from a
   VLM only when neither exists. The VLM's main job on charts is naming: titles, series, categories, units.

**Realistic outcome** [U]: 86–89 overall, centre about 87.5, first among open systems by a wide margin and
level with Claude Opus 5.5 (87.11) and LlamaParse Agentic (86.96) [V, leaderboard CSV]. Reaching 90.93 needs
charts near 90 with every other dimension at the top of its range; I do not plan on it without the optional
paid-escalation switch.

**Where I diverge from the draft** is collected in §6.1. The short version: triage per region rather than per
page, one detector plus one grounding VLM rather than three-way box fusion, table content from PDF words
rather than "numbers corrected", one chart table rather than candidates, fill-layer forms, repair as manifest
rounds, two writers over one IR, and an early chart feasibility spike.

---

## 1. Architecture

### 1.1 Principles

| # | Principle | Why (scorer or budget reason) |
|---|---|---|
| P1 | The PDF is the authority wherever it is healthy; models decide structure and semantics. | Wrong digit, sign or unit is critical on text, tables, charts and forms [R]. |
| P2 | One IR; regions hold candidates with provenance; selection is deterministic. | Makes repair, ablation and the viewer's "why did it say that" view the same mechanism. |
| P3 | Every expensive step is a cached, content-addressed call. | $30/month Modal; v1's cached-output workflow is the only way to iterate [R]. |
| P4 | Batch and online are one code path. | The viewer must run uploads through the same pipeline. |
| P5 | Emit only what evidence supports, once. | Invented content and contradicting copies are critical [R][V]. Abstaining costs less than guessing. |
| P6 | The parser knows nothing about the benchmark. | Integrity: no filenames, no subset folders, no `.test.json`. |
| P7 | Do not rely on scorer quirks. | The leaderboard re-scores every row daily with the newest harness [V], so quirks disappear. |

### 1.2 Components and data flow

```
PDF bytes ──► S0 ingest ──► S1 PDF evidence ──► S2 trust map ──► S3 region graph ──► S4 readers
              sha256,        text layer,          per-region        proposals from     text / table /
              page split,    drawings, images,    health scores,    detector + VLM,    chart / form /
              renders        fill layer,          page class        snapped to ink,    picture
                             widgets                                reading order
                                   │                                                      │
                                   ▼                                                      ▼
                            ModelClient ◄──────────────── model calls ───────────── candidates
                            (cache │ manifest │ endpoint)                                 │
                                                                                          ▼
              S7 lint ◄── S6 writers ◄── S5 select + validate ──► repair plan ──► new calls (round n+1)
              output      markdown,       per-region checks,
              guards      layout items    page checks
```

- **S0 Ingest.** Hash the bytes, split to single pages, render each page deterministically at the DPIs the
  models need. Nothing downstream sees the original path or name.
- **S1 PDF evidence (CPU, PyMuPDF).** Characters with font, size, flags, colour, content-stream sequence
  number; drawings; images; annotations; AcroForm widgets; the fill layer (§2.6); health scores.
- **S2 Trust map.** For each text line and each region: can the text layer be trusted here? See §2.1.
- **S3 Region graph.** Layout proposals from a detector and a grounding VLM become regions, snapped to the
  ink they contain, with a reading order.
- **S4 Readers.** One reader per region kind. Each produces one or more candidates.
- **S5 Select and validate.** A deterministic selector picks a candidate per region; validators run; failures
  produce a repair plan, which is a list of new model calls.
- **S6 Writers.** A markdown writer and a layout writer, both reading the same IR.
- **S7 Lint.** Output guards for every format trap that silently scores 0.

The orchestrator runs S3–S5 in **rounds**. Round 0 contains the unconditional calls. Round 1 contains calls
that depend on round 0 results (crops, second reads). Round 2 is the last allowed repair round. In batch mode
each round is one manifest per model; in online mode the rounds run back to back.

### 1.3 Intermediate representation

Coordinates are PDF points, origin top-left, in the page frame after applying `/Rotate`. Rendered-image
coordinates are converted at the model-adapter boundary and never stored in regions.

```python
# docparse/core/ir.py  (pydantic models; JSON-serialisable; this is what the viewer's JSON tab shows)

class Provenance(BaseModel):
    source: Literal["pdf_text", "pdf_fill", "pdf_widget", "pdf_drawing", "model", "rule"]
    ref: str            # evidence id, or model call key, or rule id
    detail: str = ""    # e.g. "printed data label", "axis fit r2=0.9998"

class Confidence(BaseModel):
    value: float                 # 0..1, calibrated only in the loose sense of "ranked"
    basis: list[str]             # which checks or agreements support it

class Span(BaseModel):           # inline text run
    text: str
    bbox: Box | None
    style: set[Literal["b", "i", "u", "s", "sup", "sub", "mark", "code"]]
    prov: Provenance
    critical: bool = False       # number, id, unit, negator: used by validators

class TextContent(BaseModel):
    lines: list[list[Span]]
    role: Literal["paragraph", "heading", "list_item", "caption", "footnote", "furniture", "code"]
    heading_level: int | None = None
    list_depth: int | None = None

class Cell(BaseModel):
    row: int; col: int; rowspan: int = 1; colspan: int = 1
    header: bool = False
    spans: list[Span]            # cell text with styles
    word_ids: list[str] = []     # PDF words assigned to this cell (empty on scans)

class TableContent(BaseModel):
    cells: list[Cell]; n_rows: int; n_cols: int
    title: str | None = None

class ChartPoint(BaseModel):
    category: str; series: str; value: str        # value is the string that will be printed
    numeric: float | None
    prov: Provenance                              # "label" | "geometry" | "model"
    tolerance_margin: float | None = None         # estimated error / allowed error, when known

class ChartContent(BaseModel):
    title: str | None; panel_title: str | None
    chart_type: str | None; unit: str | None
    categories: list[str]; series: list[str]
    points: list[ChartPoint]
    diagnostics: dict[str, float]                 # axis residual, legend margin, unassigned marks, ...

class FormField(BaseModel):
    caption: list[Span]; printed_id: str | None
    kind: Literal["text", "checkbox", "signature"]
    value: list[Span] | None                      # None means blank
    checked: bool | None
    grid: tuple[int, int] | None                  # row, col when the field sits in a form table
    prov: Provenance

class FormContent(BaseModel):
    fields: list[FormField]; tables: list[TableContent]

class Candidate(BaseModel):
    id: str
    reader: str                                   # "text.native", "table.vlm.infinity", "chart.vector", ...
    content: TextContent | TableContent | ChartContent | FormContent | None
    prov: list[Provenance]
    checks: list[CheckResult] = []
    confidence: Confidence

class Region(BaseModel):
    id: str
    kind: str                    # canonical layout class, plus "chart" as a picture subtype
    bbox: Box                    # snapped to ink
    segments: list[Box] = []     # line boxes when the region is multi-line text
    order: int                   # reading order
    trust: Literal["native", "native_partial", "ocr_layer", "pixels_only"]
    proposals: list[Provenance]  # which detector / VLM boxes produced it
    candidates: list[Candidate]
    selected: str | None         # candidate id
    repairs: list[RepairRecord] = []

class Page(BaseModel):
    index: int; width: float; height: float; rotation: int
    page_class: Literal["born_digital", "scanned", "mixed"]
    regions: list[Region]
    findings: list[Finding]      # validator output, page and region level
    cost: CostRecord             # model calls, gpu seconds, tokens

class Document(BaseModel):
    sha256: str; n_pages: int; pages: list[Page]
    config_hash: str; code_version: str
```

Evidence (the output of S1 and the raw model outputs) is stored separately and is immutable. Regions refer to
it by id. That split is what lets a normalisation change re-run on CPU without touching the cache.

### 1.4 Module boundaries, and what happens to v1

```
docparse/
  core/        ir.py, config.py, hashing.py, geometry.py
  pdf/         backend.py (interface), pymupdf_backend.py, textlayer.py, drawings.py, images.py,
               fill_layer.py, widgets.py, health.py, render.py
  models/      client.py, calls.py, cache.py, manifest.py, prompts/ (one versioned file per prompt),
               adapters/ (one file per model: build the call, parse the answer)
  layout/      propose.py, graph.py, snap.py, order.py, section_text.py
  readers/     text.py, picture.py, table/ (pdf_grid.py, vlm.py, assign.py), chart/ (vector.py,
               textlabels.py, schema.py, bind.py, raster.py), form/ (fields.py, boxes.py, structure.py)
  checks/      text.py, table.py, chart.py, form.py, page.py, repair.py
  write/       markdown.py, layout_items.py, lint.py
  pipeline.py  orchestrator (rounds), the only public entry point: parse(pdf_bytes, config, client)
  bench/       parse-bench provider and pipeline registration (the only importer of parse_bench.inference)
  api/         FastAPI service for the viewer
evalkit/       separate package: splits, scoring driver, ledger, loss tables, paired report
serve/         Modal apps, the batch runner, Colab and Kaggle notebooks that call the same runner
apps/viewer/   Next.js app
```

Two import rules are enforced by a test: `docparse.*` (outside `bench/`) never imports `parse_bench`, and
nothing in `docparse/` imports `evalkit`.

| v1 file | Decision | Reason |
|---|---|---|
| `textlayer.py` | **Keep, extend.** Add font name, colour, sequence number per span (v1 already uses `get_texttrace()` sequence numbers for visibility [V]), geometric superscript/subscript, fill-layer flag. | It is the core asset. |
| `styles.py`, `textalign.py` | **Keep, demote.** Used only when a model's characters are the chosen source (scans with partial native text). | Native-first rendering needs no transfer. |
| `hybrid.py` | **Rewrite** as `readers/text.py`. Reuse `render_line`, `merge_rows`, `_paragraphs`, the heading and list heuristics, orphan recovery. Invert `reconcile_page`: native text is the default, the VLM is the cross-check. | v1 keeps VLM characters and falls back to native only above a similarity threshold [V]. |
| `fusion.py` | **Rewrite** as `layout/graph.py`. Reuse `_relabel` and the heading tests. Drop `load_detections`, which finds cached detections by parent folder name and file stem [V]. | Path-keyed caches break the integrity rule and the viewer. |
| `section_text.py` + bundled forest | **Keep behind a flag; re-validate on PB2.** | It was calibrated on DocLayNet and validated on PB1 only [V, README]. |
| `charts/vector.py`, `charts/textlabels.py` | **Keep as the chart decompiler.** Refactor outputs to `ChartContent` with per-point provenance and fit diagnostics. | 1,800 lines of working geometry. |
| `charts/integrate.py` | **Drop.** It appends extra candidate tables "because they can only add recall" [V]. | Harmful on PB2 [V]. Replaced by `chart/bind.py`, which produces one table. |
| `charts/vlm_fallback.py`, `serve/modal_chart_crops.py` | **Fold into** a model adapter and the generic batch runner. | Same job, new cache keys. |
| `tables.py` | **Park.** Its own docstring says each rule was net negative on PB1 [V]. Re-test the ideas against `table_next`, which scores header structure differently. | |
| `providers/lightonocr3.py` | **Split** into `models/adapters/lightonocr.py` (prompt, parse) and `write/layout_items.py` (`build_layout_page`, the label map). | |
| `providers/hybrid.py` | **Replace** with a thin `bench/provider.py` that reads bytes and calls `pipeline.parse`. | |
| `serve/modal_vllm.py` | **Keep** as the online endpoint. Add a batch function that consumes a manifest. | |
| `serve/modal_*_dump.py` | **Replace** with the generic manifest runner. They write `<group>/<name>.json` [V]. | Content-addressed keys instead. |
| `oracle.py`, `scripts/paired_report.py` | **Move to `evalkit/`** and extend to the six PB2 dimensions. | The paired report is the acceptance tool. |

**PyMuPDF stays**, behind `pdf/backend.py`. I agree with the current direction: it is AGPL, the leaderboard
lists AGPL entries, and the features we need (sequence numbers, drawings, widgets) are already used by v1 [V].
The interface keeps a later port contained. I confirmed with pypdfium2 that the fill-layer signal is visible
through a second library too [V], so the port is feasible if ever wanted.

### 1.5 Model-serving layer

**Abstraction.**

```python
class ModelCall(BaseModel):
    model: ModelRef          # hf repo, revision sha, dtype/quantisation
    runtime: str             # "vllm-0.31", "transformers-5.x"
    prompt_id: str           # name of a file in models/prompts/
    prompt_sha: str          # sha256 of the fully rendered request minus the images
    params: dict             # temperature, top_p, max_tokens, seed, sample_index
    inputs: list[str]        # sha256 of each input blob (PNG bytes or text)
    def key(self) -> str: ...   # sha256 of the canonical JSON of all fields above

class ModelClient(Protocol):
    def get(self, call: ModelCall, blobs: dict[str, bytes]) -> ModelResult | Pending: ...
```

Three implementations: `CacheOnlyClient` (returns `Pending` and appends to a manifest on a miss),
`EndpointClient` (cache, then an OpenAI-compatible HTTP endpoint), and `ReplayClient` (cache only, raises on
a miss; used for the frozen leaderboard run and for CI).

**Batch protocol.** A manifest is a JSONL of calls plus a directory of input blobs named by hash. One runner
script (`serve/run_manifest.py`) starts the model locally in the GPU session, executes the manifest, and
writes a results JSONL keyed by call key. It is resumable: keys already present are skipped. That matters for
Kaggle's 12-hour sessions and Colab disconnects. The same script runs on Modal (function with a volume),
Colab (Drive folder) and Kaggle (dataset in, output files out).

**Which model runs where.**

| Model | Job | Where (batch) | Why there | Online (viewer) |
|---|---|---|---|---|
| Docling heron (RT-DETRv2, Apache-2.0 [R]) | Layout boxes, checkbox classes | Kaggle T4 | Small detector, free hours | Modal T4, or locally on the Mac [U] |
| PP-DocLayoutV3 (Apache-2.0 [R]) | Layout alternate (ablation only) | Kaggle T4 | Same | Off by default |
| tech4humans signature detector (Apache-2.0 [R]) | Signatures on scanned forms | Kaggle T4 | Same | Modal T4 |
| LightOnOCR-3-4B (Apache-2.0 [R]) | Page grounding, reading order, form structure | Colab A100-40GB | About 4–7 CU per full pass [U, from the report's throughput scaling] | Modal L4 |
| Infinity-Parser2-Pro (35B MoE, Apache-2.0 [R]) | Tables, scans, second page read | Modal A100-80GB or H100 | Too large for 40 GB unquantised [R] | Modal A100-80GB, "max accuracy" preset only |
| dots.mocr (MIT + custom agreement [R]) | Scan OCR candidate | Colab A100-40GB | Size not confirmed [U] | Off unless the licence is accepted |
| Granite-Vision-4.1-4B, MinerU2.5-Pro-2605, OvisOCR3 (Apache-2.0 per report/Neon [R][U]) | Table and chart bake-off candidates | Colab A100-40GB | ≤ 4B | Off unless they win a bake-off |
| Qwen3.8-27B thinking (Apache-2.0 [R]) | Chart schema, hard-case arbiter | Modal H100, crops only | Full-page run costs $6–12 [R]; crops cost far less [U] | Modal H100, "max accuracy" preset only |

Kaggle has no BF16 [R], so anything above the detectors is tested there only if it runs in FP16 [U].

**Colab changes role.** The report reserved the 200 Colab units for fine-tuning. With training ruled out,
those roughly 37 A100-40GB hours [R] become the main inference pool for every model up to 8B, and Modal is
kept for the two models that need more than 40 GB and for the viewer's endpoints.

**Online serving for uploads.** Modal endpoints scale to zero (v1's `modal_vllm.py` already does this with a
120-second idle window [V]). The viewer has two presets that differ only in config: *Standard*
(heron + LightOnOCR-3-4B on an L4) and *Max accuracy* (adds Infinity-Parser2-Pro and the 27B chart model).
Cold starts of one to three minutes are expected [U]; the Run button shows endpoint state. Born-digital
uploads with no tables, charts or forms need no GPU beyond layout, and the cost estimate says so before
the run.

**Paid escalation stays possible.** Any OpenAI-compatible endpoint is a `ModelRef`. The repair ladder has a
top rung named `external` that is disabled in the default config. Every candidate records its provenance, so
the open-weights build can assert in a test that no selected candidate came from an external model.

### 1.6 The output cache

- **Key.** `ModelCall.key()` as defined above. The image inputs are keyed by the hash of the rendered PNG
  bytes, so a change in rendering invalidates naturally. `(pdf_sha256, page, dpi, crop box)` is stored as
  metadata for traceability and never as part of the key.
- **Sampled reads.** `sample_index` and a fixed seed are in the key, so K reads are K distinct cache entries
  and are reproducible from cache.
- **Invalidation.** Nothing is ever deleted or overwritten. A new prompt, model revision or parameter makes a
  new key, which misses, which shows up in the next manifest. `docparse cache plan --config X` prints the
  number of misses per model and the estimated GPU cost **before** anything is spent.
- **First write wins.** vLLM results can differ slightly with batch composition even at temperature 0 [U].
  The stored result is the truth. The leaderboard run uses `ReplayClient` against a snapshot whose hash is
  recorded in the score ledger.
- **Storage.** `cache/calls/<aa>/<key>.json.zst` plus a SQLite index
  `(key, model, prompt_id, pdf_sha256, page, tokens_in, tokens_out, gpu_seconds, gpu_type, created_at)`.
  Cost per page in the efficiency phase is a query on this table.
- **Deterministic stages** are not cached for correctness. Per-page outputs are memoised by
  `(pdf_sha256, page, config_hash, code_hash)` only to make full re-scores fast.

### 1.7 Check and repair

A validator returns findings; it never edits content.

```python
class Finding(BaseModel):
    region: str | None; check: str
    level: Literal["hard", "soft"]     # hard: the candidate must not be emitted as is
    evidence: str                      # human-readable, shown in the viewer
    actions: list[RepairAction]        # ordered suggestions
```

**The ladder.** Every repair action is one of five kinds, tried in this order:

1. **Deterministic fix.** Snap a number to the PDF word it aligns with; drop a derived `Total` row; drop an
   unsupported strike mark.
2. **Re-read at higher resolution.** Same model, tight crop, 300–400 DPI.
3. **Second model.** A different model on the same crop.
4. **Arbiter with evidence.** The largest available model, given the crop and the conflicting candidates or
   the measured geometry as hints.
5. **Abstain in the least costly way.** For example: emit the best remaining chart table rather than none
   (caption-only scored 0.000 on the test page [V]); emit text without the uncertain style.

**Selection is evidence-based.** After repair, the selector ranks candidates by, in order: number of hard
findings, agreement with PDF evidence, agreement between independent readers, then a fixed reader priority.
There is no learned selector and no reference to scores.

**Budget.** At most two repair rounds. Each page has a call budget in config. The share of regions that
trigger rungs 2–4 is tracked per validator; a validator that fires on more than about 15% of regions is
treated as a bug in the validator or the reader, not as a reason to spend GPU [U, threshold to tune].

**Why this differs from "re-read with a bigger model".** A bigger model is not always available inside the
budget, and a re-read that is selected because it merely differs is noise. Each rung has to produce evidence
the selector can use.

### 1.8 Writers and the layout contract

Two writers read the selected candidates. They are independent because the layout scorer takes each item's
own text and never requires it to be a substring of the page markdown [V, `parse_pred_blocks`].

**Markdown writer.**

| Region | Output |
|---|---|
| Heading | ATX heading. Only relative level order is scored [R]. |
| Paragraph, list | Plain markdown. Inline styles as `**`, `*`, `<u>`, `~~`, `<sup>`, `<sub>`, `<mark>`. |
| Running header, footer, page number | In the body at the top or bottom in reading order, **and** in `page_header_markdown` / `page_footer_markdown` with byte-identical strings. `output_hypothesis` adds only the header and footer text the markdown does not already contain [V, docstring]; a unit test pins that behaviour. |
| Table | HTML `<table>`, `<th>` header rows, explicit `colspan`/`rowspan`, title on the line directly above. Styles inside cells as HTML tags, because `**` is literal inside `<td>` [R]. |
| Chart | One HTML table per chart or panel. `<caption>` with the chart title, panel title on the line above when it differs. Legend names verbatim as headers, `<th>` row labels. Plain numbers: no hedges, no ranges, no `Total` rows [R][V]. Placed where the figure sits in reading order. |
| Picture | Nothing, or `![]()`-style alt text only. No prose description [R]. |
| Form field | `Caption: value` on one line, plain text. Blank fields `Caption:`. Check boxes `[x] Option` / `[ ] Option` with the glyph before the label. Signed lines `Caption: [signed]`; unsigned lines `Caption:`. Form tables as HTML grids. Printed line ids kept beside captions. |
| Anything | Never an HTML comment; never a second rendering of the same content. |

On forms, the scorer's own reference renderer writes `- **Caption**: value` and a bare `Caption:` for blanks,
and that text scores 1.000 [V]. Bold bullets are safe there only because every line is a real field: a bold
or bulleted `key: value` line that is not a reference field costs an unscaled minor [R]. We therefore write
plain lines. Whether a bare `Caption:` can absorb a following non-caption line (Neon's concern) is still
unverified; it is an A/B on the dev split in milestone M3.

**Layout writer.** Leaderboard CI re-scores published outputs with the base branch's harness and never loads
our code [V]. The output must satisfy the default adapter natively:

- `layout_pages[]` with `page_number` starting at 1, `width` and `height` in PDF points.
- Every item has **both** `bbox` and `layout_segments` [R]; the default path passes
  `require_layout_aware_segments=True` and skips items without segments [V].
- Each segment's `label` is a canonical label accepted by `_canonical_label` [V]. The exact strings are read
  from the pinned harness by a test, not typed by hand.
- Item text goes in `md` (preferred over `value` by the reader [V]). Table items carry their HTML in `md`;
  the reader reduces it to cell text [V].
- A multi-line text region may carry one segment per line with `start_index`/`end_index` into its `md` [V];
  we start with one segment per region and test per-line segments on the dev split.
- `raw_output` holds only our version, config hash and cost numbers. It must not contain a `pages` list whose
  items look like LlamaParse items, nor keys other adapters match on (`parse_quality_score`, `_config.mode`,
  page-level `elements`) [V].
- Boxes are tight to the ink: the union of member words for text, the visual extent for tables and
  pictures, no padding [R].

**Contract test.** `tests/contract/test_default_adapter.py` builds an `InferenceResult` from our writer,
calls `create_layout_adapter_for_result` in an environment where docparse is **not** registered, and asserts
that the adapter is the default one and that the number of attribution blocks equals the number of items
with text plus pictures. It runs on every commit against the pinned harness.

**Lint (S7).** Hard failures stop a page from being written and substitute a safe fallback:
HTML comments; unbalanced tags; a table without a header row; a chart region with no table; a series header
matching the scorer's placeholder patterns; a `Total`/`Average` row not present as a category; two tables
with the same caption; checkbox glyphs used as bullets; a layout item missing either box; an unknown label;
a `page_number` other than the real one; a looped passage (n-gram repetition).

### 1.9 Integrity guards

| Guard | Mechanism |
|---|---|
| No routing on names or folders | `pipeline.parse(pdf_bytes, config, client)` receives bytes only. `bench/provider.py` reads the file and drops `source_file_path` and `example_id` (which is `subset/stem` in published records [V]) before the call. |
| Proof of the above | **Rename-invariance test**: copy 50 PDFs to random names in a flat temp folder, run with `ReplayClient`, assert byte-identical outputs. Runs in CI. |
| No `.test.json` at inference | `sys.addaudithook` in the provider raises on any `open` of a path ending `.test.json` or under the test-case root while inference runs. The parser package has no dependency on `parse_bench.test_cases`. |
| No dataset-level leakage | **Isolation test**: a page parsed alone equals the same page parsed inside a full run. No statistic is computed across documents at inference. |
| Hold out by document | `evalkit/splits/v1.json` lists example ids per fold, built once, committed, never regenerated (§3.1). |
| Disclosure | The submission notes state: what was tuned on the dev split, that the holdout was used only at milestone gates, that the PDF's own content (text layer, drawings, fill layer) is used, and the model list with revisions. |
| No fabricated provenance | Published outputs are produced by `ReplayClient` from a cache snapshot whose hash is in the ledger. |

---

## 2. Per-dimension design

Leaderboard numbers below are from `leaderboard-v2.csv` at `6463ac1` [V]. Those rows were scored at an older
harness commit (`d7efdbc`) [V], so M0 re-scores the reference systems on our pin before any comparison.

**Candidate models to benchmark per job.** Each bake-off runs on a fixed 300-page stratified dev sample and
is decided by dev score per GPU-second.

| Job | Candidates | Evidence for inclusion |
|---|---|---|
| Layout detector | Docling heron; PP-DocLayoutV3; Docling egret | heron took v1 layout from 74.2 to 80.4 [V, v1 notes]; its classes are the GT's 17 [V, v1 code]. |
| Page grounding and reading order | LightOnOCR-3-4B; Infinity-Parser2-Pro (JSON layout); MinerU2.5-Pro-2605 | Layout 84.25 / 87.04 / 86.21 [V]. |
| Scan OCR | dots.mocr; Infinity-Parser2-Pro; LightOnOCR-3-4B | Text 86.29 / 85.55 / 84.30 [V]. |
| Table structure on crops | Infinity-Parser2-Pro; dots.mocr; Granite-Vision-4.1-4B (`<tables_html>`) | Tables 86.27 / 83.40 [V]; Granite is not on the board [U]. |
| Chart naming and fallback values | Qwen3.8-27B thinking; Granite-Vision-4.1-4B (`<chart2csv>`); MinerU2.5-Pro-2605 | Charts 65.15 / not listed / 63.96 [V]. |
| Form structure | LightOnOCR-3-4B; Infinity-Parser2-Pro; Qwen3.8-27B | Forms 79.19 / 77.79 / 73.63 [V]. |
| Signatures, check boxes on scans | tech4humans conditional-DETR; heron checkbox classes | [R] |

### 2.1 Text (`text_next`, 508 pages)

**Deterministic.**

- **Health per line, not per page.** A line's native text is trusted when: its font has a usable Unicode
  map (no private-use or replacement characters), the text is visible (render mode, opacity, not covered by
  a later image; v1 already implements this test [V]), and it is not a glyphless OCR font. Lines that fail
  are `ocr_layer` or `pixels_only`. A page is `mixed` when both kinds occur.
- **Native rendering.** Trusted lines are rendered from PDF characters: words, spacing, ligature folding,
  hyphenation at line ends, superscripts from flags and from geometry (smaller size and raised baseline).
- **Completeness.** Every trusted line must end up in exactly one region (v1's orphan recovery, kept).
  Running headers, footers and page numbers are content [R].
- **Loop and duplication guard** on any model text.

**Model.**

- **Reading order** from the grounding VLM's block order, checked against a deterministic column analysis
  (XY-cut on native lines). v1 found the VLM's order scored better than the detector's [V, code comment].
  Disagreement is a soft finding; the arbiter is the second page model's order.
- **Scans and untrusted lines.** Two OCR reads from different models. Lines are aligned; where both agree
  the text is accepted. Disagreements on critical tokens (numbers, ids, units, negators) trigger a
  high-resolution crop re-read, and the majority of three wins. Non-critical disagreements take the primary
  model's reading. Naive word-level voting is not used; the report records that it hurts [R].

**Checks.** Unassigned trusted lines; duplicated passages; critical-token mismatch between native text and
the VLM on trusted lines (reported, native wins); order disagreement; figure prose outside alt text; passage
of 12 or more tokens present in one OCR read and absent from the other (the omission threshold that is
critical [R]).

**Expected** [U]: 88–91, centre 89.5. Reasoning: Infinity-Parser2 Pro scores 85.55 with roughly 6.8 text
points lost to severity events in the Metric Lab snapshot [R, `kindcosts.json`]; native-first removes the
wrong-number and omission events on the born-digital two thirds of the subset [R]. The scanned and
handwritten third stays near the best open OCR level. The board's best is 92.62 (Claude Opus 5.5) [V].

### 2.2 Formatting (`formatting_v2`: text pages 3/4, headings 1/8, strikeout 1/16, underline 1/16)

**Deterministic.**

- **Bold, italic, highlight** from font flags, font names and fill rectangles (v1, kept). Minors are not
  scaled on formatting and cap at 15 points per page [R], so every spurious or missing bold costs a point:
  whole-word boundaries and joining of adjacent bold runs (v1's style logic) stay.
- **Strike and underline** from thin drawings and from markup annotations (v1 `_apply_drawings`,
  `_apply_annots` [V]). Tightened rules: the stroke must lie within the span's horizontal extent (a rule
  that continues past the text is a table border or a field line), sit in the mid-height band for a strike
  or just under the baseline for an underline, and not coincide with a detected table grid line or a form
  field box. Sampled strikeout and underline PDFs are mostly born-digital with dozens of thin horizontal
  paths per page [V]; the report counted 27 of 30 and 29 of 30 born-digital [R].
- **Confidence asymmetry.** A lost strike and an added strike are both critical [R], and dropping all
  strike markup scores those pages 0 [R]. So the rule is not "only when confident": on a page where any
  strike evidence exists, each candidate strike is kept unless a specific counter-test fails.
- **Superscript and subscript** glued to a digit are critical when lost or added [V, docstring]. Detect
  from flags and geometry, and verify against the VLM's `<sup>` when it disagrees.
- **Headings.** Detector class and text-layer emphasis must agree (v1 rule). Levels from font-size ranks
  on the page; a uniform shift is free [R]. Open models already score 94–97 on the headings subset [R], so
  the goal here is to not regress the VLM's heading set.
- **Redline pages.** When any strike is present, underline errors become major [R]. Underline detection
  uses the same tightened rules.

**Model.** On scanned strike and underline pages (about 3 of 30 and 1 of 30 samples are image-only [R]):
word boxes from the grounding model, a classical line detector over each text line, and a yes/no check by a
VLM on the crop. A mark is emitted when the line detector and the VLM agree.

**Checks.** Every style span maps to whole words; no style on whitespace; heading count versus the VLM's;
no heading longer than one line of body-sized text unless the detector says so; strike candidates
overlapping table or form geometry.

**Expected** [U]: 90–93, centre 91.5. Reasoning: open models sit at 86–90 on text pages and 94–97 on
headings but 13–17 on strikeout and 18–36 on underline [R]. Lifting those two subsets to about 90 is worth
roughly 9 formatting points by the slot weights; exact bold from the PDF moves the text-page slot toward
LlamaParse Agentic Plus's 93.1 [R].

### 2.3 Tables (`table_next`: tables 1/2; complex headers, financial, insurance filings, dense 1/8 each)

**Core idea: structure from a model, content from the PDF.** On born-digital tables every cell's text is
re-rendered from the PDF words assigned to it. The model's characters are used only to make the assignment.

**Deterministic.**

- **Word-to-cell assignment.** Align each VLM cell's text to the PDF words inside the table box
  (monotone alignment per row band, then per column band). Each PDF word is assigned to exactly one cell.
- **PDF grid candidate.** An independent structure from ruling lines and from column gaps in word
  positions. It is a candidate in its own right for ruled tables and the cross-check for the VLM's grid.
- **HTML normalisation.** Rectangular span matrix, `<th>` for header rows, title line, original column
  order (a reordered column is major [R]).

**Model.** Infinity-Parser2-Pro page pass, then a crop re-read at 300–400 DPI for tables that are dense,
small-font or flagged. Image-only tables (22 of 30 sampled `complex_table_headers` pages [R]) get a second
model on the same crop.

**Checks.**

| Check | Catches | Repair |
|---|---|---|
| Every PDF word in the table box is assigned once | Missing rows (each is critical [R]), dropped cells | Re-read crop; fall back to PDF grid |
| No cell contains words from two row bands or two column bands | Merged rows or columns, shifted cells | PDF grid candidate |
| Row count equals text-row clusters; column count equals gap clusters | Split or merged structure | Second model |
| Span matrix is rectangular | Span overflow that shifts values under wrong headers (critical [R]) | Deterministic repair, else re-read |
| Header row exists; title captured | "Header rows missing" critical; "title missing" major [V, policy] | Promote from PDF emphasis; take the line above |
| On scans: two models agree on row count, column count and each numeric cell | Digit errors | Third read at higher DPI; majority per cell |
| No two tables share 90% of cell keys | Duplicate or contradicting copy [R] | Drop the lower-ranked one |

**Expected** [U]: 88–91.5, centre 90. Reasoning: Infinity-Parser2 Pro's per-subset snapshot is 88.9 / 74.1 /
86.7 / 89.7 / 83.3 [R], and its largest table losses are wrong numbers in a column and missing rows [R],
exactly what word assignment removes on born-digital pages (nearly all sampled pages of four of the five
subsets [R]). A plausible outcome is 92 / 80 / 91 / 93 / 90, which gives 90.3 by the slot weights. Complex headers
stay the weak slot because most of it is image-only.

### 2.4 Charts (`chart_next`, 568 pages, 1,025 charts)

This is the dimension that sets the final range. The constraint is no training.

**What the leaderboard says about reading charts with a model** [V]. Full-page reads by frontier models
reach 74.1 (Claude Opus 5.5), 78.9 (GPT-6.2 sol) and 41.4 (Gemini 4 neon). The best permissive open model
reaches 65.2. The systems above 85 are chart-specific pipelines (LlamaParse Agentic Plus 92.6, Reducto 89.1,
Pulse 86.8, Extend 85.1–85.4). LlamaParse Agentic Plus spends a median of 42,200 input and 5,800 output
tokens and 30.7 seconds per chart page [V]. Conclusion: more or larger open-model reads will not close the
gap. Measurement has to.

**What the scorer rewards** [V unless marked].

- The score is point F1 capped by a severity floor. Precision counts.
- Structural events are charged in full and are not scaled: a missing chart, a missing series, a lost
  legend with several series (critical); a missing or wrong title, a renamed series, an invented series,
  a derived total (major). On the sample page, LlamaParse Agentic Plus had every value right and still
  scored 0.80, because four charts each had one "series renamed" event [V].
- Tolerance is 1% when data labels are printed, otherwise the larger of 1% and 2% of the value range;
  credit decays to zero at three times the tolerance [R].
- A second version of a placed chart under its title is scored too. Appending one with values 4% off took
  the page from 0.80 to 0.40 [V]. An exact duplicate cost nothing [V], and an untitled contradicting copy
  also cost nothing [V], but that is a quirk we do not rely on (P7).
- With the tables removed and only captions left, the page scored 0.000 [V].

**Pipeline.**

1. **C0 Find charts and panels.** Union of detector picture boxes, the VLM's chart blocks and the vector
   plot cores (v1 `_plot_cores`). One chart region per axis frame; small multiples are separate charts with
   their panel titles.
2. **C1 Decompile (CPU, born-digital).** From drawings and text: marks (bars, polylines, area polygons, pie
   wedges, markers), axes (tick text matched to tick positions; linear or log fit with residual), legend
   (swatch to label), printed data labels, category labels, title, panel title, axis titles, unit strings.
   Output is a `ChartContent` plus diagnostics: axis fit residual as a share of the value range, number of
   ticks, legend assignment margin (best cost versus runner-up), unassigned marks, unassigned numeric
   labels. All 30 sampled chart pages are born-digital and 25 have no large raster image [R].
3. **C2 Schema read (model, crop at 300 DPI).** The VLM is asked for the chart's schema as JSON: type,
   title, panel title, series names in legend order with their colours or markers, category labels in
   order, axis units and scale, stacked or grouped, labels printed or not, which series uses which axis.
   A second request asks for the full table; that is the fallback and the binding aid, not the answer.
4. **C3 Bind (CPU).** Match decompiled series to names: legend swatch colour first, then legend order, then
   end-of-line labels, then value agreement with the VLM's table. Value agreement is the important new
   signal: even imprecise VLM values identify which measured series is which, by Hungarian assignment on
   median relative difference. Categories bind the same way. The names written are the PDF's text strings
   (exact characters), chosen by the binding.
5. **C4 Resolve each value** by precedence:
   1. the printed data label from the text layer (exact, 1% tolerance applies);
   2. the geometric value from the calibrated axis (bars: height from baseline; stacked bars: segment
      height; lines and areas: sampled at category positions; pies: sweep angle share);
   3. the per-cell median of K=3 VLM reads (raster charts only).
   Geometric values are rounded to a precision derived from the tick step (v1 `_round_value`).
6. **C5 Validate.**

   | Check | Level | What it catches |
   |---|---|---|
   | Exactly one table per chart region; no two tables with the same caption | hard | Contradicting copies [V] |
   | No series header matches placeholder or colour-name patterns | hard | "Legend lost", critical with several series [R] |
   | Every legend text on the chart is a series header; every header is a text string on the chart | hard | Renamed and invented series |
   | No row or column named Total/Sum/Average unless it is a plotted category | hard | "Derived total" |
   | Where a printed label and a geometric value both exist, they agree within 3% of the range | hard | Wrong axis fit, wrong binding, stacked read as cumulative |
   | Axis fit residual below 0.5% of the range; at least three ticks | soft | Log axes, broken axes, dual axes |
   | Every printed numeric label inside the plot appears once in the table | soft | Missed bars or series |
   | Stacked segments sum to the printed total when one is printed | soft | Segment order errors |
   | Pie shares sum to 100 ± 1 | soft | Dropped slice |
   | Title string is present on a line of the page or in the caption | hard | "Title missing", 5 points per chart |
   | Unit in a header appears verbatim on the chart | hard | A contradicting unit makes the column critical [R] |

7. **C6 Repair.** Per-panel crop at higher DPI; alternative model for the schema; the arbiter model given
   the crop plus the decompiled table as a hint ("these are measured values per colour; name the series").
   Last rung: emit the best single VLM table. Never two tables.
8. **C7 Write** as in §1.8.

**Raster charts** (at most about 17% of chart pages by the sample [R]). First version: K=3 reads, per-cell
median, same validators. Second version, only if the dev split shows it pays: measurement by classical image
processing guided by the schema read (colour quantisation, connected components for bars, per-colour column
scans for lines, calibrated by OCR'd tick labels). No training is involved. This is the last chart item.

**Why this can work without training.** Each failure class that separates 65 from 90 has a deterministic
source of truth on born-digital pages: values (labels or geometry), names (legend text), structure (one
table per axis frame), titles (text above the frame). The VLM is used where VLMs are reliable, which is
describing what is there, and is cross-checked by measured values.

**What can go wrong.** Charts drawn as clipped images or with gradients; legends without swatches;
categories on a continuous time axis; combination charts on two axes; text converted to outlines (then the
page behaves like a raster chart). The diagnostics in C1 exist to route these to the model path instead of
emitting a confident wrong table.

**Chart feasibility spike (in M1, CPU only, 2 days).** Run the refactored v1 decompiler on the dev chart
pages and report, per chart: decompiled or not, hard checks passed or not, and `chart_next` for pages where
all charts passed versus the rest. This turns the estimate below into a measurement before the main chart
work is scheduled.

**Expected** [U]: 76–86, centre 81. Reasoning by bucket, all shares assumed: about 60% of pages fully
decompiled and bound, scoring about 95; about 25% partly right, about 70; about 15% on the model path,
about 45. That gives 81. v1's decompiler was only ever validated against PB1's sparse rules (76.8 there
[V, README]), so the 60% share is the number the spike has to confirm. The report's 80–88 assumed
fine-tuning was available for the model path; without it I lower the top end.

### 2.5 Layout (`layout_grounding_pass_rate`, 535 pages)

**Deterministic.**

- **Regions from the bottom up.** Detector and VLM boxes are proposals. A text region's final box is the
  union of the trusted words it contains; a table's is the extent of its grid and cell words; a picture's
  is the extent of its image or drawing cluster. Credit ramps linearly from IoU 0.5 to 0.9 [R], and the GT
  boxes are tight to the ink [R], so snapping is worth more than a better detector.
- **One region per element.** Split a detector box that covers several paragraphs when the text layer shows
  a paragraph gap; keep headings separate from body text; one box per picture, table, header and footer
  [R]. Remove duplicates, page-sized containers, and text boxes inside pictures [R].
- **Item text** is the native words whose centres fall inside the box (v1 `_native_box_text`), which gives
  the token F1 ≥ 0.8 the scorer requires [R]. Tables must cover 90% of their word centres [R].
- **Class** from the detector, with v1's rules for heading versus text. The label is not scored [R], so
  class errors cost only through grouping rules.

**Model.** heron as the detector. LightOnOCR-3-4B grounding (or Infinity-Parser2-Pro's JSON layout) for
elements the detector misses and for scans, where there are no native words to snap to. PP-DocLayoutV3 is
added only if an ablation on the dev split shows a gain. I do not plan a three-way weighted box fusion: the
board's best layout score is 90.31 and five open systems sit between 86.2 and 88.3 [V], so the headroom is in
snapping, splitting and de-duplication, and each extra detector adds duplicates to remove.

**Checks.** Every trusted word belongs to at most one text-like item; no item box contains another item of
the same kind; picture regions have one box; no box larger than 80% of the page unless it is a table or
picture; item count sanity versus the markdown block count.

**Open measurement on the dev split.** Whether text inside pictures should be attributed to the picture
item (v1 does this [V]) or emitted as separate boxes (penalised per the report [R]); whether per-line
segments beat one box per paragraph.

**Expected** [U]: 88–90, centre 88.8. Reasoning: v1's heron fusion with native word attribution is the same
recipe as the open systems at 86.2–88.3; snapping and splitting are the increments; the board saturates near 90.

### 2.6 Forms (`forms_v2`: forms 7/8 with 450 typed fills and 50 blanks; fillable forms 1/8)

**Finding: the fill layer** [V]. On 30 sampled form PDFs:

- All 4 blank pages have no trailing block of text in page-exclusive fonts.
- 23 of 26 filled pages end with a block of text objects whose fonts occur nowhere earlier on the page
  (for example `TrebuchetMS-439`, `CourierPrime-Regular-7900`), usually the last objects in the content
  stream. Tick marks are text glyphs (`X`, `✓`) in the same block.
- On the 4 filled pages with ground truth, reading that block at the reference positions reproduced 209 of
  211 scored items exactly: 46 of 48 text values, 114 of 114 blanks, 49 of 49 check boxes. Both misses are
  signatures, which are not text.
- 3 of 26 filled pages showed no such block and need the fallback path.
- A Word-generated fillable form produced a false positive for the naive "trailing exclusive fonts" test,
  so detection needs more than one signal.

**Second finding: structure is the hard part** [V]. A plain dump of the page's text scored 0.75–0.96 on the
4 blank forms but 0.27–0.74 on the 4 filled ones, and a naive attempt to attach values to the caption on
their left made one page worse (0.74 to 0.23, sixteen critical events from misplaced check marks). The
scorer's own rendering of the reference scores 1.000. Values are easy; ownership is not.

**Deterministic.**

- **Fill-layer detection** as a general "appended overlay" detector, not a font-name pattern: text objects
  late in the content stream, in fonts not used by the base page, positioned inside or on field geometry
  (boxes, underlines, comb cells), forming a minority of the page's text. Appended small paths or images
  near a signature caption count as ink. The same detector serves user uploads that were filled and
  flattened by other tools [U].
- **Field geometry** from the base page: rectangles and underlines as field boxes, small squares as check
  boxes, comb boxes as one field, ruled grids as form tables. On AcroForm pages the widgets give field
  rectangles, types, values and states directly [R].
- **Ownership.** Each fill item belongs to the field box that contains it. A field box's caption is the
  base text inside the box, else the nearest base text to the left on the same row, else directly above,
  with form-table cells inheriting row and column captions. Comb characters are concatenated. A tick mark
  belongs to the check-box square that contains it, and the option label is the base text adjacent to that
  square in reading direction.
- **Blank discipline.** A field with no fill item is blank. Nothing is ever written into it.
- **Captions** are base-layer text, exact.

**Model.** A page read for form structure (which lines are captions, which regions are grids, reading
order). Its values are then **snapped**: each model value is replaced by the fill-layer string it aligns
with; a model value with no fill-layer support is removed when the page's text layer is healthy; a
fill-layer string the model omitted is inserted at its geometric owner. On pages without a fill layer and
without widgets (scans, the 3-of-26 case), the model path stands alone, with heron's checkbox classes and
the signature detector.

**Checks.**

| Check | Catches |
|---|---|
| Every fill-layer item appears exactly once in the output | Value lost (major); contradicting copy (critical) |
| No value in the output lacks fill-layer or widget support (healthy pages) | Invented value (critical) |
| Geometric owner and model owner agree for each value | Value of another field (critical); disagreement goes to the arbiter |
| Each check-box square has exactly one state; single-choice groups have at most one tick | Box state flipped (critical) |
| Signature captions: ink present ↔ `[signed]`, else empty | Signature invented (critical), lost (major) |
| No bold or bulleted `key: value` lines | Extra field (minor, unscaled) |
| Form tables emitted as grids | "Not kept as a grid" (minor) and detached values (major) |
| At least the model's caption inventory is present | Coverage cap at 2 × coverage [R] |

**Expected** [U]: 85–90, centre 87.5. Reasoning: LightOnOCR-3 4B scores 79.19 [V]. In the Metric Lab
snapshot its largest forms losses are wrong value, invented value, value lost and box state flipped, about
5.6 forms points together, plus about 1.2 for garbled captions [R, `kindcosts.json`]; the fill layer and
base-layer captions address exactly those. Adding them naively gives about 86; ownership errors remain.
The board's best is 86.67 [V], so this would be a leading forms score if it holds. Fillable forms already
score 90–92 for open models [R] and move little.

**A point for the user** (also in §6.2): the benchmark's fills are synthetic [V, case tags], so this method
gains more here than on scanned real-world forms. It uses page content only and is the same principle as
using the text layer, and I recommend using it and disclosing it.

### 2.7 Expected scores together

| Dimension | Best permissive open [V] | LlamaParse Agentic Plus [V] | Low [U] | Centre [U] | High [U] |
|---|---:|---:|---:|---:|---:|
| Text | 86.29 | 90.22 | 88 | 89.5 | 91 |
| Formatting | 82.04 | 93.60 | 90 | 91.5 | 93 |
| Tables | 86.27 | 94.72 | 88 | 90 | 91.5 |
| Charts | 65.15 | 92.58 | 76 | 81 | 86 |
| Layout | 87.04 | 88.23 | 88 | 88.8 | 90 |
| Forms | 79.19 | 86.22 | 85 | 87.5 | 90 |
| **Overall** | **78.50** (one system) | **90.93** | **85.8** | **88.1** | **90.3** |

The dimension ranges are not independent best cases that will all land together. I quote **86–89 with a
centre near 87.5** as the realistic range, slightly below the arithmetic centre, because some dimension
usually underperforms its estimate.

---

## 3. Evaluation harness (`evalkit/`)

### 3.1 Pins and split

- **Harness pin.** `parse-bench` at commit `6463ac1` (release 2.2.1 [V]) as a git dependency in
  `evalkit`'s lock file.
- **Dataset pin.** `llamaindex/ParseBench-2` at the `data_revision` in `leaderboard/results-v2.yaml`,
  currently `2d9a2344…` [V]. That is the revision CI scores against.
- **Every ledger row** records harness commit, dataset revision, docparse git sha, config hash and cache
  snapshot hash.
- **Bumps.** The harness and dataset change often (the changelog shows three releases in five days [V]).
  A bump is its own change: re-score the last accepted run on the new pin from cached outputs, record the
  delta, then continue.
- **Split.** 80% dev, 20% holdout, grouped by source document, stratified by subset, fixed seed, written
  once to `evalkit/splits/v1.json`.
  - A group is the source document across all subsets. Several sampled files are pages of one report
    (for example four pages of `ac8b3538-en` in the chart sample [V]).
  - Forms are grouped by form template, not by page, so the holdout contains unseen forms.
  - Group keys are derived from test-case metadata and file stems. This is evaluation-side code; the parser
    never sees it.
- **Holdout discipline.** Scored only at milestone gates. Per-page details of holdout cases are not
  inspected. The viewer hides them by default.

### 3.2 Full scoring from cached outputs

```
docparse bench run   --config configs/x.yaml --split dev     # CPU; ReplayClient; writes results per subset
docparse bench score --run <id>                              # parse-bench evaluation run --next=<subset>=<dir>,... --next_only
docparse bench ingest --run <id>                             # _next_score.json + per-case metadata → ledger.sqlite
```

- The scoring command is the one CI's `rescore.py` uses [V], so local numbers are the numbers CI will
  compute. CI fails a submission that differs by more than 0.5 [V].
- A cache miss during `bench run` is an error that writes a manifest; it never silently skips a page.
- Per-page outputs are memoised (§1.6), so a change that touches only the chart writer re-runs chart pages
  only. CI needs about 20 minutes to re-score one row [V]; a local full dev re-score should be in the same
  range [U].
- **M0 also scores the published outputs of reference systems** (LlamaParse Agentic Plus, Infinity-Parser2
  Pro, LightOnOCR-3 4B, KDL) on the pin and the split. That costs no GPU and gives per-page baselines, a
  per-page oracle, and the dev/holdout gap for systems that were certainly not tuned on our split.

### 3.3 Error classes ranked by overall points lost

For each case, the ledger stores the score, the proportional score, the cap, coverage, and the event counts
by class that the harness already writes as `policy_inputs` [V].

- **Page weight in overall points:** `slot_weight / pages_in_subset / 6`. One chart page weighs about 2.9
  table pages [R].
- **Loss attributed to an event class:** the counterfactual gain if that class were absent on that page.
  `metric_policy.score_metric` recomputes a page's cap from the saved counts under a policy, and a policy
  can set a class to `ignore` [V]. So the counterfactual is a pure function, with no re-parse.
- **Loss not explained by the cap** (the proportional score binds) is reported as a separate row per
  metric: "content recall/precision".
- **Layout** has no cap: losses are grouped by failure reason per element (text F1 below threshold, table
  word coverage, IoU ramp, extra box), weighted by the scorer's √area weights.
- The result is one table, **points lost per error class per dimension**, sorted, with the worst pages
  behind each row. It is the work queue and it is the viewer's dashboard.

### 3.4 Regression tracking and acceptance

- **Ledger.** One row per run: scores per dimension and per subset on dev, on holdout when evaluated, GPU
  seconds per page, counts of lint fallbacks and zero-score pages.
- **Paired report** (extended from v1's `scripts/paired_report.py`, which already rejects mismatched
  evaluator versions and coverage [V]): per-page wins and losses between two runs, per dimension.
- **A change is accepted when all of these hold on the dev split:**
  1. Overall improves by at least 0.05, or the change is declared a refactor and moves overall by less
     than 0.02.
  2. No dimension falls by more than 0.3 and no subset by more than 0.5, unless the trade is written down
     and overall still improves.
  3. For a claimed gain, a paired bootstrap over documents gives at least 90% probability that the
     difference is positive.
  4. Zero-score pages and lint fallbacks do not increase.
  5. Contract, rename-invariance and isolation tests pass.
  6. The change in GPU seconds per page is recorded.
- **Overfitting review** at each milestone gate: if the holdout gain is less than half the dev gain for
  that milestone, the thresholds added in the milestone are reviewed and the simplest ones kept.
- **Tunables** live in one config file. Each has a comment with the run that set it. Thresholds are tuned
  on dev only.
- **Quirk register.** A short list of scorer behaviours we observed and deliberately do not exploit
  (first entry: untitled contradicting chart copies are not charged [V]).

---

## 4. Viewer

### 4.1 Shape

- **`apps/viewer`**: Next.js (App Router, TypeScript). It renders and calls the API. It does not parse
  PDFs, call Modal, or read the dataset.
- **`docparse/api`**: FastAPI. Uploads, runs, results, benchmark data, cost estimates. It calls
  `pipeline.parse` with an `EndpointClient`, so uploads run through the same code as benchmark pages.
- **Local first.** Single user, localhost, files on disk, SQLite. Hosting is an open question (§6.2).

### 4.2 Data model (SQLite)

| Table | Key columns | Notes |
|---|---|---|
| `documents` | `sha256`, name, n_pages, created_at | Uploads and benchmark PDFs alike; files stored by hash. |
| `runs` | id, document_sha, config_name, config_hash, code_version, status, started_at, cost_usd, gpu_seconds | One row per press of Run. History and comparison read this. |
| `run_pages` | run_id, page, ir_path, markdown_path, layout_path, status | The IR JSON is the source for every tab. |
| `run_calls` | run_id, call_key | Joins to the cache index for cost and for "which model said this". |
| `bench_cases` | example_id, subset, document_sha, fold | Benchmark pages only. |
| `bench_scores` | run_id, example_id, metric, score, proportional, cap, coverage, events_json | From the ledger. |
| `board_outputs` | system, example_id, markdown_path | Index of published leaderboard outputs, downloaded once. |
| `jobs` | id, run_id, state, progress, message | Polled or streamed to the UI. |

### 4.3 API

```
POST /documents                       upload; returns sha256 and page count
POST /documents/{sha}/estimate        CPU pre-pass (S0–S2) → planned calls, cache hits, est. seconds and USD
POST /documents/{sha}/runs            start a run with a preset; returns run id
GET  /runs/{id}                       status, cost, per-page status          (SSE at /runs/{id}/events)
GET  /runs/{id}/pages/{n}             IR, markdown, layout items, findings
GET  /documents/{sha}/pages/{n}.png   page render for thumbnails
GET  /documents/{sha}/file            the PDF for pdf.js
GET  /bench/cases?subset=&fold=       case list with scores for a run
GET  /bench/cases/{id}/score?run=     score breakdown, binding bound, events   (holdout: hidden unless ?reveal=1)
GET  /bench/cases/{id}/board/{system} that system's published output for the page
GET  /bench/losses?run=               error-class table (§3.3)
GET  /bench/diff?a=&b=                paired report between two runs
```

**Estimated cost on the Run button.** The estimate endpoint runs the CPU stages, which take well under a
second per page [U], and plans round 0 exactly: which pages need a page model, how many table and chart
crops, how many calls are already cached. Repair rounds are estimated from the historical rate per
validator. The button shows seconds and dollars, plus the cold-start state of each endpoint.

### 4.4 Screens

1. **Parse** (the main screen, modelled on LlamaParse's).
   - Top bar: Upload, document name, preset selector, **Run** with estimated cost, run selector (history).
   - Left: PDF viewer with page navigation, zoom, and overlay toggles (regions, lines, words, table cells,
     form fields, fill layer, reference boxes on benchmark pages).
   - Right tabs, per page:
     - **Rendered**: markdown rendered with tables, linked to boxes on hover and click.
     - **Markdown**: raw text.
     - **JSON**: the page IR.
     - **Layout**: the item list as emitted to the scorer.
     - **Evidence**: for the selected region, every candidate, its provenance, its check results, and
       the repair history. This is the tab LlamaParse does not have and the one that makes debugging fast.
     - **Score** (benchmark pages): score, proportional score, cap, which bound sets the result, the
       severity events, and per-dimension diffs (table and chart cells against the reference, a form field
       checklist, reference versus predicted boxes with the reason each failed).
     - **Compare**: side by side against another run of ours or any leaderboard system's published output.
2. **Runs**: history across documents with cost, duration and config.
3. **Benchmark**: scores per dimension and subset for a run; the error-class table; click-through to the
   worst pages; diff between two runs with per-page wins and losses.

### 4.5 Reuse from `apps/visual_grounding_viewer` (Apache-2.0 [V])

| Their file | Use |
|---|---|
| `frontend/src/lib/grounding.ts` | Overlay layer model and `boxesForPage`. Port as is. |
| `frontend/src/lib/markdownGrounding.ts` | Splitting markdown into blocks and linking blocks to items. Port as is. |
| `frontend/src/lib/gtOverlay.ts` | Reference-box overlay and match partitioning for benchmark pages. Port. |
| `frontend/src/lib/itemGranularPreview.ts` | Line, word and cell hover inside an item. Port. |
| `frontend/src/lib/textDiff.ts` | Text diff for the Compare tab. Port. |
| `frontend/src/components/ViewerPane.tsx` | pdf.js canvas with overlay. Port as a client component loaded with `ssr: false`. |
| `frontend/src/components/MarkdownPane.tsx`, `ItemMarkdownPane.tsx` | react-markdown with rehype-raw and rehype-sanitize. Port; extend the sanitize schema for `rowspan`, `colspan`, `caption`, `u`, `del`, `sup`, `sub`, `mark`. |
| `frontend/src/types/api.ts` | Adopt its grounding page and item types as our wire format so the library code ports unchanged. |
| `backend/gt_rules.py`, `backend/loader.py` | Reference for turning test-case rules into overlay boxes. Reimplement the small part we need in `docparse/api`. |

Not reused: its folder browser, deep-link logic and indexer (it browses result folders; we have a
database), and its metric panels, which Neon reports cover PB1 metrics only [U].

---

## 5. Phased plan

Effort is for one strong engineer. GPU spend is my estimate from the report's throughput figures [U].
Scores are on the dev split, on the pinned harness. Targets are exit criteria; if a milestone misses, the
loss table decides whether to continue or move on.

| # | Milestone | Work | Exit criteria (dev) | Effort | GPU | Main risks |
|---|---|---|---|---|---|---|
| M0 | Foundations | Package layout; IR; PDF evidence incl. fill layer and health; `ModelClient`, cache, manifest runner; `evalkit` with pins, split, ledger; re-score 3–4 published systems; contract, rename-invariance and isolation tests | Published scores of two reference systems reproduced within 0.5; all three tests green; manifest round-trip works on Kaggle with heron | 5 d | $0 (Kaggle) | Harness or dataset moves during setup: pin first |
| M1 | First full score | Round-0 passes: heron (Kaggle), LightOnOCR-3-4B (Colab), Infinity-Parser2-Pro (Modal). Native-first text reader, v1 table and chart paths behind the one-table rule, PB2 writers, lint. **Chart spike** (CPU) | Overall ≥ 81; layout ≥ 86; lint fallbacks < 1% of pages; chart spike report written | 6 d | $6–9 Modal, 6–10 CU | Infinity-Parser2-Pro serving cost or memory; if so, start with LightOnOCR-3-4B alone |
| M2 | Viewer MVP | Parse screen with Rendered / Markdown / JSON / Layout / Evidence / Score, benchmark case list, loss table | A benchmark page and an uploaded PDF both run end to end from the UI | 4 d | < $1 | Scope creep: no Compare or history polish yet |
| M3 | Text and formatting | Health gate per line; order arbitration; strike, underline, sup/sub rules; scan OCR bake-off and two-read arbitration; heading rules | Text ≥ 89; formatting ≥ 91; strikeout and underline subsets ≥ 88 | 6 d | $1–2, 4–6 CU | Scanned third of the text subset; false strikes |
| M4 | Forms | Field geometry; ownership; snapping; check boxes; signatures; AcroForm widgets; blank-field A/B | Forms ≥ 86 (core ≥ 85, fillable ≥ 93); zero invented values on pages with a fill layer | 6 d | $0–1 (Kaggle) | Ownership on dense multi-column forms; the pages without a fill layer |
| M5 | Tables | Word-to-cell assignment; PDF grid candidate; crop re-reads; second model on image-only tables; HTML normalisation | Tables ≥ 89.5; complex headers ≥ 80; no born-digital table with a digit that is not in the PDF | 7 d | $5–8, 4–6 CU | Image-only complex headers; span repair regressions |
| M6 | Charts | Decompiler refactor and diagnostics; schema read; binding; validators; repair; raster median path | Charts ≥ 80; structural events (missing title, renamed series, lost legend) on < 5% of charts | 12 d | $8–12, 8–12 CU | The share of charts the decompiler handles; 27B serving cost |
| M7 | Layout polish and repair tuning | Snapping, splitting, de-duplication; picture text decision; repair trigger rates | Layout ≥ 88.5; repair calls on ≤ 15% of regions | 3 d | $1–2 | Small gains near the ceiling: time-box it |
| M8 | Freeze and holdout | Holdout run; overfitting review; pack and validate outputs; dry-run of CI scoring from the packed files | Each holdout dimension within 1.0 of dev; packed-output score equals ledger score within 0.1; **overall ≥ 86** | 3 d | $0 | Holdout gap in charts or forms |
| M9 | Viewer complete | Compare against leaderboard systems, run history, diff views, benchmark dashboard, presets | All screens of §4.4 | 5 d | < $1 | — |
| M10 | Efficiency | Ablate each component; content-based routing of big models; smaller-model swaps; cost per page report | Overall within 0.5 of the M8 score at ≤ 40% of its GPU seconds per page; cost-per-page table by page class | 6 d | $5–10 | Routing must use page content only |

**Totals** [U]: about 63 working days; about $30–45 of Modal credit across two to three monthly cycles;
about 30–40 of the 200 Colab units in the main line, leaving the rest for bake-offs and reruns.

**Order, and why.** Text and formatting, then forms, come before tables and charts because they are mostly
CPU work with the largest certain gains. Charts come last among the dimensions because they are the largest
and least certain block, but the spike in M1 measures their feasibility before the schedule depends on it.
The efficiency phase is last, as required, and is cheap to do then because every call's GPU seconds are
already in the cache index.

**Expected trajectory** [U].

| After | Text | Fmt | Tables | Charts | Layout | Forms | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 87 | 87 | 86 | 65 | 87 | 79 | 81.8 |
| M3 | 89.5 | 91.5 | 86 | 65 | 87 | 79 | 83.0 |
| M4 | 89.5 | 91.5 | 86 | 65 | 87 | 87 | 84.3 |
| M5 | 89.5 | 91.5 | 90 | 65 | 87.5 | 87 | 85.1 |
| M6 | 89.5 | 91.5 | 90 | 81 | 87.5 | 87.5 | 87.8 |
| M7–M8 | 89.5 | 91.5 | 90 | 81 | 88.8 | 87.5 | 88.1 |

**Final range.** 86–89 overall, centre about 87.5 (§2.7). Below 86 would mean the chart decompiler covers
well under half of the charts. Above 89.5 needs charts at 86 or more together with forms near 90.

---

## 6. Disagreements and open questions

### 6.1 Where I disagree with the draft, the research report, or the Neon review

| # | Their position | Mine | Basis |
|---|---|---|---|
| 1 | Draft: triage each page as born-digital, scanned or mixed. | Trust is decided per line and per region; the page class is only a summary. Forms add a third kind of evidence, the fill layer. | Mixed pages; visible text over images (v1 already tests visibility per text trace [V]). |
| 2 | Draft: fuse boxes from heron, PP-DocLayoutV3 and Infinity-Parser2-Pro. | One detector plus one grounding VLM, regions built bottom-up and snapped to ink. A third source only if a dev ablation pays for it. | Board layout ceiling 90.31; five open systems at 86.2–88.3 [V]; credit depends on IoU 0.5–0.9 [R]. |
| 3 | Draft: tables from Infinity-Parser2-Pro crops "with numbers corrected from the PDF text". | All cell content from PDF words by word-to-cell assignment; the model supplies structure only; a PDF-grid candidate cross-checks it. | "Corrected" leaves the model's text as the default. Assignment also finds missing rows, each of which is critical [R]. |
| 4 | Draft: raster charts get 3 reads and a median. v1: extra candidate tables. | The median path is the fallback for a minority of charts. The main path is label, then geometry, with the VLM for names. **One table per chart, always.** | Disagreeing copy: 0.80 → 0.40 [V]. Frontier full-page chart reads top out below 79 [V]. |
| 5 | Draft: form values from the PDF text, heron check-box states. | Separate the fill layer from the base page; tick marks are fill-layer glyphs; heron is the fallback. Most of the effort goes to caption ownership. | 209 of 211 items exact from the fill layer; naive ownership scored 0.23–0.74 [V]. |
| 6 | Draft: a failed check triggers a re-read with a bigger model. | A five-rung ladder ending in abstention, run as manifest rounds, with evidence-based selection and a cap on trigger rates. | Budget, and a re-read that merely differs is not evidence. |
| 7 | Draft: one writer for markdown "plus layout". | Two writers over one IR; layout item text is independent of the page markdown. | `parse_pred_blocks` reads each item's own `md` [V]. |
| 8 | Draft order: the viewer after the first full score; charts late. | Same, plus a 2-day CPU chart spike inside M1 and a viewer MVP immediately after M1 as the analysis tool. | Charts decide the range; measure early. |
| 9 | v1: keep the VLM's characters and transfer native styles onto them. | Native characters by default on healthy lines. | Wrong digits are critical in PB2 [R]. |
| 10 | Report: spend Colab on fine-tuning. | Colab is the inference pool for models up to 8B. | Training is ruled out. |
| 11 | Report: charts 80–88, forms 82–87. Neon: charts 79–83.5 with training, forms 83.5–85.5. | Charts 76–86 without training; forms 85–90 with the fill layer. | §2.4, §2.6. |
| 12 | Neon: blank fields must be written `Label: [___]`. | Either form scores 1.000 on the scorer's own reference rendering [V]; the spill risk with a following non-caption line is untested. Decide by A/B in M4. | Appendix A. |
| 13 | Report §5: mark strikes "only when confident". | On pages with strike evidence, keep each candidate unless a counter-test fails. | Lost and added strikes are both critical, and losing all scores the page 0 [R]. |
| 14 | "Build on docparse rather than restart." | Agree, with the qualification that the core is rewritten and the assets are kept (§1.4). | v1's core encodes PB1's recall-only incentives [V]. |
| 15 | Keep PyMuPDF; open weights only; paid escalation as a later switch. | Agree on all three. The switch is the `external` rung, off by default. | §1.5. |

### 6.2 Questions only the user can answer

1. **Fill layer on forms.** The benchmark's filled forms are synthetic, so reading the fill layer helps
   more here than on real scanned forms. It is page content, not a filename. Are you comfortable using it
   and disclosing it? I recommend yes.
2. **Is a leaderboard submission the goal, and in which category?** A hybrid of deterministic PDF parsing
   and open models may or may not be accepted under "Open Weight" [R]. If you intend to submit, the
   question should go to the maintainers during M0, not at the end.
3. **Licence bar for models.** Apache-2.0 and MIT only? dots.mocr is MIT plus a custom agreement [R], and it
   is the best open text model on the board. Several candidates' licences come from the Neon review and are
   unchecked [U].
4. **Target.** "Clearly first among open systems" (about 86–89, robust) or "beat 90.93"? The second likely
   needs the paid-escalation switch on a few hundred pages, which Neon estimated at $10–14 per full run [U].
5. **Budget horizon.** The plan uses two to three monthly Modal cycles and about a fifth of the Colab
   units. Is that timeline acceptable, and may Colab be used for inference?
6. **Tuning on the public set.** There is no hidden split [R]. Are you content with "thresholds tuned on an
   80% dev split, holdout number reported alongside" as the disclosed policy?
7. **Published outputs of other systems.** They are used for comparison and analysis. May published outputs
   of open models also be used as stand-in model outputs for prototyping before our own GPU passes? They
   would never feed a reported number.
8. **Viewer deployment.** Local only, or hosted with accounts? Are one-to-three-minute cold starts on
   uploads acceptable, or should one endpoint stay warm (which costs credit)?
9. **Deadline.** About 63 working days as planned. If there is a date, charts beyond the first version and
   M9 are the items to cut.

---

## Appendix A. What I verified, and how

All checks were CPU-only, used existing tools, and read downloaded samples as data with `python3 -I`.

| # | Check | Result |
|---|---|---|
| A1 | `chart_next` on one sample page (`chart1.test.json`, 6 charts) with the published LlamaParse Agentic Plus output and variants | As published: 0.800 (precision 1.0, recall 1.0, four major "series renamed"). Plus an exact duplicate table: 0.800. Plus a second version 4% off under the title: 0.400. Same at 10% off: 0.400. Untitled second version 10% off: 0.800. Second version with `Series n` names: 0.600. Tables removed: 0.000. One table per chart from the reference: 1.000; one value 30% wrong: 0.900. |
| A2 | Fill-layer detection on 30 sampled form PDFs and the first 12 sampled fillable-form PDFs (pypdfium2 page objects, fonts, stream order) | 4 of 4 blank: none. 23 of 26 filled: trailing block in page-exclusive fonts, 21 of them ending the stream; 3 of 26 none. Fillable forms: 1 false positive in 12 (Word-generated page). |
| A3 | Fill-layer text against reference `form_field` rules on 4 filled pages, located by the reference boxes | Text 46 of 48, blank 114 of 114, check boxes 49 of 49. Both misses are signatures. |
| A4 | `forms_v2` on model-free outputs for 8 pages | Text dump: 0.927, 0.750, 0.950, 0.958 on blanks; 0.586, 0.481, 0.344, 0.430 on filled. Geometric rows: 0.927, 0.850, 0.930, 0.920; 0.740, 0.591, 0.267, 0.567. Naive caption attachment: one page fell to 0.227, one rose to 0.667. Scorer's reference rendering: 1.000 on all 8. |
| A5 | Blank-field spelling on the scorer's reference rendering (2 pages) | Bare `Caption:`, `[___]` and `___` all 1.000. |
| A6 | Strikeout and underline samples (pypdfium2) | Listed pages have 3,000–5,000 characters and 38–88 thin horizontal paths; image-only pages have none. |
| A7 | Published LlamaParse Agentic Plus chart records (568) | `example_id` is `charts/<stem>`; `source_file_path` empty; `raw_output` holds cost fields only; median 42,200 input and 5,800 output tokens, $0.056 and 30.7 s per page; 476 pages use `<caption>`; 355 pages have one table, 147 two. |
| A8 | Harness code read | `next_score.py` weights; `metric_policy.py`; `paper_policy.json`; `next_metrics.py`; default adapter selection and `_resolve_llamaparse_pages`; `parse_pred_blocks`; `chart_next` placement, copies and F1; `forms_v2`, `formatting_v2`, `table_next` docstrings; `leaderboard/README.md`; `results-v2.yaml` pins; `CHANGELOG.md`; `leaderboard-v2.csv`. |
| A9 | docparse v1 code read | `providers/hybrid.py`, `providers/lightonocr3.py`, `hybrid.py`, `fusion.py`, `textlayer.py`, `styles.py`, `charts/__init__.py`, `charts/integrate.py`, outline and entry points of `charts/vector.py` and `charts/textlabels.py`, `tables.py` header, `serve/modal_vllm.py`, heads of the dump scripts, README, chart and GPT research notes. |

**Not verified:** every model's behaviour, speed, memory and licence; every score target; the share of
charts the decompiler handles on PB2; raster chart share beyond the report's 30-page sample; cold-start
times; whether the fill-layer signal holds across all 450 filled pages (26 sampled).

## Appendix B. Scratch files created during this pass

The only deliverable is this file. To run the checks in Appendix A, I also wrote eight small throwaway
scripts in the session scratchpad (temporary, not kept in the repo):
(`chart_copy_check.py`, `chart_events.py`, `deco_presence.py`, `form_fill_layer.py`,
`form_overlay_stats.py`, `form_overlay_vs_gt.py`, `forms_baseline.py`, `forms_render_ref.py`). They touch
neither repo and can be deleted, or kept as the seed of the M0 tests for the fill layer and the one-table
rule.
