# Layout: reproducing OCRia-1's boxes, and the first swap experiments (M1, Tasks 15–16)

Date: 2026-11-25. Everything here ran on the Mac with no cloud GPU, on OCRia-1's published readings
(stand-ins). Layout alone re-scores in about 22 seconds, so each experiment is one run plus one score.
Numbers are the layout dimension from the harness scorer; "dev" is the development part (420 layout pages),
"check" the check slice (115).

## 1. Rebuilding the layout ourselves, without the attention pass

OCRia-1's published layout comes from its `ocria_grounding` package: Docling heron regions, every word of the
answer placed by its PDF text-layer box, and the model's attention for words that have no PDF box. We ran
heron locally on all 4,247 page images (about 0.3 s per page on Apple MPS, $0) and ran the package's own
`page_words` and `place` with no attention maps (`configs/standin_ground.yaml`, run `m1-layout-standin-ground`).

| | Dev | Check | All |
|---|---:|---:|---:|
| Published layout | 87.17 | 81.52 | 85.95 |
| Rebuilt here: heron + PDF words, no attention, on the published (post-processed) markdown | 83.15 | 76.30 | 81.68 |

- Our heron boxes are the published boxes: on the pages checked, every box we produce equals a published box.
- On the 506 layout pages with a text layer the rebuild scores 85.44 against 86.05 published.
- The whole gap is the 29 layout pages without a text layer: 16.06 against 84.44, which is 3.7 points of the
  dimension. Those pages need the attention pass (or another source of word positions).
- So the attention pass is needed for pages without a usable text layer only: 29 layout pages, 249 pages in
  the whole benchmark. It needs our own run's raw answers, so it comes after the full OCRia-1 run.

## 2. Swap experiments on the published layout items

Each row changes the published items on pages with a text layer and nothing else. Gain and probability are
from a paired bootstrap over source documents on the development part, against the unchanged items.

| Variant | Dev | Check | All | Dev gain | P(gain > 0) | Kept |
|---|---:|---:|---:|---:|---:|---|
| Published layout items, unchanged | 87.17 | 81.52 | 85.95 | | | |
| tighten, no pad | 85.44 | 79.90 | 84.25 | -1.73 | 0.00 | no |
| tighten, pad 0.08 | 87.45 | 81.85 | 86.25 | +0.28 | 0.98 | no (0.15 is better) |
| tighten, pad 0.12 | 87.88 | 82.33 | 86.69 | +0.71 | 1.00 | no (0.15 is better) |
| tighten, pad 0.15 | 88.09 | 82.57 | 86.90 | +0.92 | 1.00 | **yes** |
| tighten, pad 0.20 | 87.92 | 82.50 | 86.76 | +0.75 | 1.00 | no (0.15 is better) |
| tighten, pad 0.30 | 85.74 | 80.60 | 84.64 | -1.43 | 0.00 | no |
| native text, words split at glyph gaps | 70.11 | 70.09 | 70.11 | -17.06 | 0.00 | no (broken words) |
| native text, words split at spaces, no health test | 87.45 | 82.45 | 86.38 | +0.29 | 0.83 | no (P = 0.83) |
| tighten 0.15 + native text, no health test | 88.42 | 83.59 | 87.38 | +1.25 | 1.00 | no (P = 0.86 over tighten) |
| tighten 0.15 + native text where 80% of tokens agree with the reading | 89.07 | 83.61 | 87.89 | +1.90 | 1.00 | **yes** |
| same, 90% agreement | 89.07 | 83.70 | 87.91 | +1.90 | 1.00 | no (same as 80%) |
| tighten 0.15 + split at gaps over 1.0 line heights | 88.04 | 82.56 | 86.86 | +0.87 | 1.00 | no (-0.05) |
| tighten 0.15 + split at gaps over 1.5 line heights | 88.07 | 82.57 | 86.89 | +0.90 | 1.00 | no (-0.02) |

**Kept:** `tighten` with `tighten_pad: 0.15`, and `native_text` with its health test
(`configs/standin_layout.yaml`). Together: layout 85.95 → 87.89 on all pages (+1.94), 87.17 → 89.07 on the
development part and 81.52 → 83.61 on the check slice. That is +0.32 on the overall score.

What the experiments showed:

- **Tight-to-ink boxes lose.** The reference boxes are a little larger than the glyphs. A box that is exactly
  the union of its words scores 1.7 below heron's own box; the same box grown by 15% of the word height on
  each side scores 0.9 above it. The optimum is narrow (0.08 and 0.30 are both worse), so this value is
  re-checked on our own run.
- **Item text from the PDF helps only with a health test.** Taking the PDF words in a box as the item's text
  fixes boxes whose model text belonged to a neighbour, but on pages with a broken text layer it replaces good
  text with bad. Requiring 80% of the item's PDF tokens to occur in OCRia-1's reading of the page makes the
  gain +0.98 over tightening alone, at probability 1.00 on both folds.
- **OCRia-1's gap rule for words is wrong for reading text.** It joins glyphs closer than 0.35 line heights,
  which splits words in fonts with loose glyph boxes ("R egul atory"). Its grounding code needs that rule; for
  reading we split at the spaces PDFium reports (`RenderedPage.text_words`). With the gap rule, native text
  scored 70.11.
- **Splitting text items at paragraph gaps does not pay** (-0.02 to -0.05), so it stays off.

## 3. Second round: words no item covers, and nested boxes

After round one the largest remaining loss was reference elements that no predicted box located (5.4 of the
11 points still lost). A diagnostic over the 535 layout pages split them by cause, in page shares:

| Cause | Elements | Share of pages lost |
|---|---:|---:|
| No predicted box touches the element at all | 596 | 8.2 |
| Text or heading covered by a much larger Table box | 304 | 5.5 |
| Text covered by, or half covered by, a Picture box | 152 | 3.3 |
| Picture box covers under half of the reference picture | 22 | 1.7 |
| Everything else | | 10.0 |

Of the elements with no box at all, nearly every one has PDF words inside it, and about half of them are text
that OCRia-1 did not write into its markdown at all. So the fix needs no model: PDF words that lie in no
layout item are grouped into lines and blocks, and each block becomes a new Text item (`orphans`).

Gains are against the round-one configuration (87.89 on all pages), same paired bootstrap.

| Variant | Dev | Check | All | Dev gain | P(gain > 0) | Kept |
|---|---:|---:|---:|---:|---:|---|
| Round one (tighten 0.15 + health-tested native text) | 89.06 | 83.61 | 87.89 | | | |
| + items for uncovered PDF words, block gap 1.0 line heights | 89.80 | 85.27 | 88.83 | +0.74 | 1.00 | **yes** |
| same, block gap 0.7 | 89.80 | 85.28 | 88.83 | +0.74 | 1.00 | no (same) |
| same, block gap 1.5 | 89.76 | 85.13 | 88.77 | +0.70 | 1.00 | no |
| same, blocks need 3 letters or digits | 89.78 | 85.28 | 88.81 | +0.71 | 1.00 | no |
| + skip blocks where the rendered page shows nothing (invisible text) | 89.98 | 85.35 | 88.99 | +0.92 | 1.00 | **yes** |
| + skip blocks whose words OCRia-1 did not read (50% support) | 89.69 | 84.76 | 88.63 | -0.11 vs plain | 0.22 | no |
| + skip blocks whose box holds words of another item | 89.79 | 85.29 | 88.82 | -0.01 vs plain | 0.42 | no |
| + drop text items inside a Key-Value Region | 90.00 | 85.62 | 89.06 | +0.02 vs kept | 0.76 | no |
| + drop text items inside a Picture | 89.69 | 85.43 | 88.77 | -0.29 vs kept | 0.00 | no |
| + drop a Table that mostly is a Picture | 90.01 | 85.43 | 89.02 | +0.02 vs kept | 0.62 | no |

**Kept:** `orphans` with `orphan_ink`. Layout is now **88.99 on all pages** (published 85.95, +3.04), 89.98
on the development part and 85.35 on the check slice. That is +0.51 on the overall score, on stand-in
readings and pages with a text layer.

- The check slice gains more than the development part (+1.74 against +0.92), so this is not fitted to the
  development pages.
- Words OCRia-1 did not read are worth keeping: requiring support from the reading loses 0.2.
- The nested-box rules (plan Task 16, step 3) do not pay. Which of two overlapping boxes the reference wants
  differs from page to page; no fixed rule wins.

## 4. What is left in layout after the kept refinements

Dimension points lost, all pages (11.0 in total).

| Outcome | Round one | Now | Change |
|---|---:|---:|---:|
| extra box: holds words locked to another element | 1.56 | 1.66 | +0.09 |
| element not located (no box matches): text | 2.75 | 1.55 | -1.20 |
| box found, text does not match (token F1 < 0.8): text | 0.99 | 1.20 | +0.20 |
| box looser or tighter than the reference (IoU ramp): text | 1.14 | 1.15 | +0.00 |
| element not located (no box matches): picture | 0.87 | 0.86 | -0.00 |
| box looser or tighter than the reference (IoU ramp): picture | 0.68 | 0.68 | -0.00 |
| element not located (no box matches): heading | 0.94 | 0.62 | -0.31 |
| box looser or tighter than the reference (IoU ramp): heading | 0.57 | 0.59 | +0.02 |
| element not located (no box matches): table | 0.41 | 0.41 | -0.00 |
| box found, text does not match (token F1 < 0.8): heading | 0.34 | 0.37 | +0.03 |
| box looser or tighter than the reference (IoU ramp): header/footer | 0.31 | 0.35 | +0.04 |
| extra box: blank area | 0.24 | 0.30 | +0.06 |

What remains is mostly disagreement about granularity: a Table or Picture box where the reference has
separate text elements (about 1.6 points), and overlapping boxes where one is counted as extra (1.7). Those
are the detector's region choices, which is what a second detector (KDL nano v2) is for.

## Still open in M1

- The same refinements on our own run's layout (needs the full OCRia-1 run and the attention pass on pages
  without a text layer).
- KDL nano v2 boxes (plan Task 16, step 4).
