# Layout: a second detector, KDL-Frontier-Parser-nano-v2 (M1, Task 16 step 4)

Date: 2026-11-25. Everything ran on this Mac (Apple M4, 16 GB, MPS) at no cost; no cloud GPU. Numbers are
the layout dimension (`layout_grounding_pass_rate`) from the harness scorer, on the 535 layout pages; "dev"
is the development part (420 pages), "check" the check slice (115). Every variant keeps the refinements of
`configs/standin_layout.yaml` on (tightening, PDF text per item, items for uncovered PDF words).

## 1. The model

`KDLAI/KDL-Frontier-Parser-nano-v2`, pinned at revision `5b5e0edda3c019563a2f6205ec49095288fafeb8` (the weights
file is byte-identical to the revision the authors validated, `766505d5`, sha256 `596389309a85…`).

- **What it is.** A 1.2B-parameter Qwen2-VL (a 0.5B Qwen2 language model, 24 layers, width 896, under a
  32-layer vision encoder), bf16, one 2.58 GB safetensors file. License **AGPL-3.0**.
- **What it runs.** Nothing of its own: the repository holds only weights, configs, the tokenizer, a Jinja
  chat template and the README. It loads with the stock `transformers` class `Qwen2VLForConditionalGeneration`
  (transformers 5.24.1, torch 2.15.0, the existing `detect` dependency group). No `trust_remote_code`.
- **What it outputs.** It is a pipeline model, not a one-shot page parser: one call per page with the prompt
  `"\nLayout Detection:\n"` returns the page's regions in reading order, each as
  `<|box_start|>x0 y0 x1 y1<|box_end|><|ref_start|>label<|ref_end|><|rotate_up|>` on a 0–1000 grid, with 26
  raw labels (text, title, table, image, chart, image_block, list, header, footer, page_number, captions,
  footnotes, equation, …). Further prompts read each cropped region (text, OTSL tables, formulas, figures);
  those are not used here. There are no confidence scores.
- **How it is called.** Exactly as the harness's own `kdl_frontier_nano` provider does: the page image resized
  to 1036 × 1036 and sent as JPEG quality 95 (1,394 prompt tokens), greedy decoding, at most 6,000 new tokens,
  `no_repeat_ngram_size` 100. Region reading follows the provider's `normalize_native_layout_items` (a `list`
  or `image_block` holding child regions is dropped for its children; formula parts inside an
  `equation_block` are dropped for the block; labels map to Docling classes as the provider maps them).

The ParseBench 2 leaderboard row for this model has layout **88.28** (its own end-to-end pipeline; we
re-scored its published outputs: dev 88.64, check 87.00), against 85.96 for OCRia-1's published layout and
88.99 for our refined layout. Page by page it beats our refined layout on 221 of the 524 pages where the
two differ (ledger run `kd-published-kdl`), so the two are complementary.

## 2. Running it locally

It runs on the Mac. The adapter is `src/docparse/models/adapters/kdl.py` (pinned `ModelRef`, one call per page
keyed on the page image's pixel hash, `read_detections(result) -> list[Detection]`, the raw answer cached so
reading rules can change without a model call); the executor is `kdl_executor` in `serve/local_detector.py`
(`--model kdl`). The manifest came from `evalkit run --plan` with a KDL option on the layout subset (524 page
images for 535 cases), and the answers went into the cache with `serve.run_manifest.import_results`: 524 new
entries, nothing else in the cache touched.

| | |
|---|---|
| Time per page, machine lightly loaded (first pages) | 26–35 s (about 22–25 ms per generated token, MPS, bf16) |
| Time per page per worker while four other agents ran evaluations (load average 40–100) | median 55 s, mean 66 s, 90th percentile 122 s |
| Generated tokens per page | mean 830, median 744, max 4,312 (none reached the 6,000 limit; no empty answers) |
| The 524 page images | 4 h 50 min wall with two worker processes (one from each end of the manifest), 9.6 worker-hours, $0 |
| Four pages batched in one `generate` | 25 s per page against 30–35 s one at a time under the same load (not used) |

The step-2 threshold (30 s per page) is met when the machine is quiet and missed under the other agents'
load; it was run locally anyway because it is free and finished the same day. For all 4,247 pages it would
be about 30–40 hours here when the machine is quiet. A cloud run was not needed and was not started. A rough,
unmeasured estimate for the record: the decoder is 0.5B parameters, so a batched vLLM server should do the
layout stage at around 1–2 s per page on an L4 ($0.80/h: about $1–2 for all 4,247 pages) or under 1 s on an
A100-40GB ($2.10/h: about $1.50). That would need the owner's approval.

**Same answers as the published run?** Not exactly: greedy decoding on MPS in bf16 and on a CUDA vLLM server
branch apart on some pages. On 185 pages compared, 92.1% of the published regions have a local region of the
same class at IoU ≥ 0.9, and 162 of the 185 pages have the same number of regions. Most differences are one
grid unit; a few pages diverge completely (on one page the published run has one table region where we have
17 text regions).

## 3. Experiments

Variants were first tried in a scratch copy of the data root, with KDL's published elements standing in for
the model (nothing from that was written to the real cache or ledger), then run for real on the local KDL
answers once all 524 were cached. The real numbers matched the stand-in ones within about 0.1 for every
variant. Base: `kd-base`, our own run of `configs/standin_layout.yaml` (88.99 on all pages, 89.98 dev, 85.35
check, same as `lay-orph-ink`). Gain and probability are from the paired bootstrap over source documents on the
development part, against the previous kept configuration.

**Round one, each against the base**

| Variant (run) | Dev | Check | All | Dev gain | P(gain > 0) | Kept |
|---|---:|---:|---:|---:|---:|---|
| (a) KDL regions instead of heron, filled with OCRia-1's grounding code (`kd-replace`) | 88.71 | 84.99 | 87.91 | -1.27 | 0.02 | no |
| (a) KDL regions as items, each holding its PDF words (`kd-words`) | 88.64 | 85.99 | 88.07 | -1.34 | 0.04 | no |
| (b) heron items + KDL regions that overlap no item (`kd-add`) | 90.16 | 85.74 | 89.21 | +0.18 | 0.91 | passes; kept in round two |
| (c) split a Table/Picture holding ≥ 2 KDL text regions with PDF words that no other item covers, unless KDL sees a table/picture there (`kd-split`) | 90.04 | 85.87 | 89.14 | +0.06 | 0.58 | no |
| (c) keep such a Table and add the KDL text regions inside it (`kd-children`) | 89.68 | 84.65 | 88.60 | -0.30 | 0.05 | no |
| (d) split a Picture holding ≥ 2 KDL pictures (`kd-pictures`) | 89.95 | 85.14 | 88.92 | -0.03 | 0.35 | no |
| (d) a Picture takes the box of the KDL picture it matches (`kd-picbox`) | 90.03 | 85.31 | 89.02 | +0.05 | 0.66 | no |
| (d) arbiter: of two nested items, drop the one no KDL region matches when KDL matches the other (`kd-arb`) | 90.74 | 85.70 | 89.66 | +0.76 | 0.99 | passes; the next row is better |
| (d) arbiter, pictures never dropped (`kd-arb-kp`) | 90.80 | 86.13 | 89.80 | +0.82 | 1.00 | **yes** |
| (d) arbiter, pictures and tables never dropped (`kd-arb-kpt`) | 90.26 | 86.80 | 89.52 | +0.28 | 0.97 | no (less on dev) |

**Round two, against `kd-arb-kp`**

| Variant (run) | Dev | Check | All | Dev gain | P(gain > 0) | Kept |
|---|---:|---:|---:|---:|---:|---|
| + KDL regions that overlap no item (`kd-arb-kp-add`) | 91.02 | 87.30 | 90.22 | +0.22 | 0.95 | **yes** |
| + split Table/Picture (`kd-arb-kp-split`) | 90.72 | 87.10 | 89.94 | -0.08 | 0.35 | no |
| + KDL picture boxes (`kd-arb-kp-picbox`) | 90.78 | 85.96 | 89.75 | -0.02 | 0.43 | no |

**Round three, against `kd-arb-kp-add`**

| Variant (run) | Dev | Check | All | Dev gain | P(gain > 0) | Kept |
|---|---:|---:|---:|---:|---:|---|
| + split Table/Picture (`kd-r3-split`) | 90.93 | 87.77 | 90.25 | -0.09 | 0.34 | no |
| + tables never dropped by the arbiter (`kd-r3-kpt`) | 90.44 | 87.15 | 89.74 | -0.58 | 0.02 | no |

What the experiments showed:

- **KDL is not a better region source than heron.** With the same grounding and the same refinements, on the
  507 layout pages with a text layer, heron's regions score 89.05 (`kd-heron-ground`) and KDL's 88.10
  (`kd-replace`); the published items we start from score 89.23. KDL's own end-to-end layout (88.49 on those
  pages) is good because it reads its own regions, not because its boxes suit OCRia-1's text better.
- **The granularity rules (c) do not pay.** Where heron has a Table and KDL sees text regions, the reference
  has a table about as often as it has text (on the stand-in run: 47 containers with text in the reference, 12
  with a table or picture). Replacing gains the text elements (+0.44 page shares) and loses the true tables and
  pictures (−0.34 and −0.24). Keeping the table and adding the text regions is worse: when the reference is
  text, the kept table becomes an extra box holding words locked to them, a large penalty. KDL regions that
  another item already covers had to be left out (without that the split lost 1.0, mostly to duplicate boxes);
  also requiring PDF words inside the text regions changed little.
- **What KDL is good for is breaking ties.** The largest remaining loss was "extra box: holds words locked to
  another element" (1.66 points): two of our items overlap and the scorer matches one of them. Round two of
  the layout swap found that no fixed rule picks the right one. A second detector does: when exactly one of two
  nested items has a KDL region at IoU ≥ 0.5, the other goes. That loss falls from 1.66 to 0.64. Pictures must
  never be dropped (dropping them loses true pictures that KDL draws differently); tables may be (keeping them
  helps the check slice but costs 0.5 on dev).
- **Adding KDL regions that overlap no item** adds +0.2 on top. Uncovered PDF words already become items
  through `orphans`, so the gain comes mostly from tables and pictures heron missed (on the stand-in run, tables
  that no box had matched were its largest element-level gain).

## 4. What is kept, and what is left

**Kept:** `kdl_arbiter` with `kdl_keep_pictures`, and `kdl_add` (`configs/standin_layout_kdl.yaml`, config hash
`013a8c2840d2757e`, run `kd-arb-kp-add`). Layout is now **90.22 on all pages** (before 88.99, +1.23; published
OCRia-1 85.95, +4.26), **91.02 on the development part** (+1.04, P 1.00) and **87.30 on the check slice**
(+1.95, P 1.00). That is +0.21 on the overall score. The check slice gains more than the development part.
The options are off by default and only change layout items; the markdown is untouched. KDL answers are cached
for the 535 layout pages only, which are the only pages layout is scored on; the config would need KDL calls
for any other page.

Dimension points lost, all pages (11.01 before, 9.78 after):

| Outcome | Before (`kd-base`) | After (`kd-arb-kp-add`) | Change |
|---|---:|---:|---:|
| extra box: holds words locked to another element | 1.66 | 0.64 | -1.01 |
| element not located (no box matches): text | 1.55 | 1.46 | -0.09 |
| box looser or tighter than the reference (IoU ramp): text | 1.15 | 1.08 | -0.07 |
| box found, text does not match (token F1 < 0.8): text | 1.20 | 1.04 | -0.16 |
| element not located (no box matches): picture | 0.86 | 0.96 | +0.09 |
| box looser or tighter than the reference (IoU ramp): picture | 0.68 | 0.68 | -0.00 |
| box looser or tighter than the reference (IoU ramp): heading | 0.59 | 0.58 | -0.01 |
| element not located (no box matches): heading | 0.62 | 0.58 | -0.05 |
| element not located (no box matches): table | 0.41 | 0.54 | +0.14 |
| box looser or tighter than the reference (IoU ramp): header/footer | 0.35 | 0.38 | +0.02 |
| box found, text does not match (token F1 < 0.8): heading | 0.37 | 0.35 | -0.02 |
| extra box: blank area | 0.30 | 0.31 | +0.01 |

What is left is spread thin: text elements no box matches (1.46) or whose box holds the wrong text (1.04),
box tightness (text 1.08, pictures 0.68, headings 0.58), and pictures and tables not located (0.96, 0.54; the
arbiter drops some true tables, which is where the +0.14 comes from). The worst pages after the change are
large true tables that KDL reads as text regions (the arbiter then drops the table).

Notes for the owner:

- The model's license is **AGPL-3.0**.
- The KDL options need the page image only, so they apply to our own OCRia-1 run unchanged; the arbiter only
  acts on pages with a text layer.
- Ledger runs from this work all start with `kd-`; `kd-published-kdl` is KDL's published output re-scored.
