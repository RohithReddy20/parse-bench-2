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

## 3. What is left in layout after the kept refinements

Dimension points lost, all pages.

| Outcome | Published | With refinements | Change |
|---|---:|---:|---:|
| element not located (no box matches): text | 2.83 | 2.75 | -0.08 |
| extra box: holds words locked to another element | 1.94 | 1.56 | -0.38 |
| box looser or tighter than the reference (IoU ramp): text | 1.28 | 1.14 | -0.14 |
| box found, text does not match (token F1 < 0.8): text | 1.63 | 0.99 | -0.64 |
| element not located (no box matches): heading | 0.92 | 0.94 | +0.02 |
| element not located (no box matches): picture | 0.86 | 0.87 | +0.01 |
| box looser or tighter than the reference (IoU ramp): picture | 0.68 | 0.68 | -0.00 |
| box looser or tighter than the reference (IoU ramp): heading | 0.74 | 0.57 | -0.17 |
| element not located (no box matches): table | 0.41 | 0.41 | +0.00 |
| element not located (no box matches): header/footer | 0.35 | 0.34 | -0.01 |
| box found, text does not match (token F1 < 0.8): heading | 0.76 | 0.34 | -0.42 |
| box looser or tighter than the reference (IoU ramp): header/footer | 0.36 | 0.31 | -0.05 |
| box found, text does not match (token F1 < 0.8): table | 0.26 | 0.26 | +0.00 |
| extra box: blank area | 0.24 | 0.24 | +0.00 |
| table box misses its words (< 90% of word centres): table | 0.22 | 0.22 | +0.00 |
| box found, text does not match (token F1 < 0.8): header/footer | 0.24 | 0.14 | -0.10 |

The remaining loss is mostly elements with no matching box and boxes with the wrong text. Those are
segmentation errors in the detector's regions, which is what a second detector (KDL nano v2) is for.

## Still open in M1

- The same refinements on our own run's layout (needs the full OCRia-1 run and the attention pass on pages
  without a text layer).
- KDL nano v2 boxes (plan Task 16, step 4) and the picture-threshold and de-duplication variants (step 3).
