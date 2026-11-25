# Our own run with every kept correction: 87.67

Date: 2026-11-25. Ledger run `own-best4`, config `configs/own_best.yaml`, pipeline name `docparse_best`. All 13
subsets, 4,317 pages, scored by the harness. The page readings are our own OCRia-1 run (vLLM 0.34.0 on Modal);
nothing here uses OCRia-1's published outputs.

| Dimension | Published OCRia-1 | Our run + corrections | Change | Dev | Check | Target |
|---|---:|---:|---:|---:|---:|---:|
| **Overall** | 85.24 | **87.67** | +2.42 | 87.71 | 87.52 | 89 |
| Text | 87.87 | 88.43 | +0.55 | 88.38 | 88.60 | 89.5 |
| Formatting | 91.58 | 93.04 | +1.46 | 93.19 | 92.55 | 93 |
| Tables | 86.28 | 88.97 | +2.68 | 88.32 | 91.58 | 90 |
| Charts | 77.66 | 80.83 | +3.18 | 80.88 | 80.61 | 81 |
| Layout | 85.96 | 90.24 | +4.28 | 91.14 | 86.95 | 88.8 |
| Forms | 82.10 | 84.49 | +2.39 | 84.39 | 84.81 | 87.5 |

- **Against the board:** second place is Claude Opus 5.5 at 87.13 and first is LlamaParse Agentic Plus at
  90.95. 87.67 is 0.54 above second place. That is a real but thin margin: one OCRia-1 run differs from
  another by up to 0.4 on a single dimension (charts), which is under 0.1 on the overall score.
- **The check slice** (pages no rule was tuned on) scores 87.52, close to the development part (87.71), so
  the result is not fitted to the development pages.
- **History of this number:** 86.93 with the first round of corrections, 87.14 with KDL nano v2 added as a
  second layout detector (run `own-best`), 87.41 with the second round on forms, charts and tables (run
  `own-best2`), 87.54 with pages printed sideways or upside down read upright (run `own-best3`), 87.67 with
  the third round on forms (printed hints moved out of blank fields' slots).
- **What is on:** upright reading of turned pages; looped-read and empty-read fallbacks; table grid repairs; form comb values, check-box
  states and row repairs (line numbers and box codes joined to captions); chart values measured from the
  vector drawing and bound by legend colour; strikeout, bold, list-marker and respelling repairs; layout
  rebuilt from heron regions, PDF words and OCRia-1's attention, tightened to PDF words, with items for
  uncovered words and KDL nano v2 voting between nested boxes.

## How the run was put together

- Layout is built only for the 535 pages of the layout subset, the one place it is scored. It needs one
  attention pass per page (524 calls, about $1.20 on an A10G) and one KDL call per page (run on the Mac, free).
  The other subsets were run with `--set layout=none`; their markdown does not depend on layout. Layout for
  every page of the benchmark would cost about $9 more in attention passes and 30 to 40 hours of KDL on the Mac.
- Our own layout, before any refinement, reproduces OCRia-1's published layout: 85.94 against 85.96 (run
  `m1-own-layout-all`). With attention only on the 29 pages without a text layer it was 84.53, because picture
  references are placed by attention on every page.
- GPU cost of the whole own run: OCRia-1 page reads $8.94, attention passes $1.20.

## Plan milestones

M1's exit conditions are met: our run is within 0.5 of the published score on every dimension without
corrections (`2026-11-25-ocria1-own-run.md`, layout 85.94 against 85.96), and layout is above 87.5 with the
swap (90.24).

## What is left to reach 89

| Dimension | Now | Target | Gap in overall points |
|---|---:|---:|---:|
| Forms | 84.49 | 87.5 | 0.50 |
| Tables | 88.97 | 90 | 0.17 |
| Text | 88.43 | 89.5 | 0.18 |
| Charts | 80.83 | 81 | 0.03 |

Formatting and layout are at or above their targets. Also open: the 58 page reads that hit the token limit
lose about 0.5 overall points between them; second reads for those pages are being evaluated.

## Second reads of truncated pages: measured, not kept

58 of the 4,247 page reads hit the 24,576-token limit. Each was read once more in two ways (116 calls, 38
minutes on an A100-40GB, about $1.35), and the second read used when it ended normally
(`reread_truncated`, runs `own-rr-img` and `own-rr-rp` against `own-best2`, five non-layout dimensions).

| Variant | Second reads that ended normally | Text | Formatting | Tables | Charts | Forms |
|---|---:|---:|---:|---:|---:|---:|
| Image at 75% size | 23 of 58 | -0.03 | +0.03 | +0.04 | +0.00 | +0.10 |
| Repetition penalty 1.05 | 20 of 58 | -0.03 | +0.00 | +0.04 | +0.16 | +0.02 |

Each variant is worth about +0.02 to +0.03 overall, and both lose 0.13 to 0.15 on text on the check slice, so
neither passes the keep rule as it stands. The upper bound had looked like 0.5 points, but most second reads
that finish are no better than the first read cut short, and the looped-read fallback already covers the
worst pages. The option stays in the code, off. A sharper rule (use the second read only when it holds more
of the PDF's words than the first) is the obvious next try if this is revisited.

## Upright reading of turned pages: kept

45 pages of the benchmark are printed sideways or upside down (41 in `tables`, 2 in `financial_tables`, 1 in
`dense_tables`, 1 in `text`). The text layer shows which way the text runs; the pipeline turns such a page
upright before OCRia-1 reads it and before any correction (`upright_read`, `src/docparse/pdf/upright.py`).
61 upright page reads were made on Modal (about 10 minutes, $0.35) for the first, looser detector.

| Variant | Tables dev | Tables check | Tables all | Text | Formatting | Charts |
|---|---:|---:|---:|---:|---:|---:|
| `own-best2` | 87.39 | 91.49 | 88.22 | 88.41 | 93.06 | 80.83 |
| Upright, first detector (61 pages) | 88.35 | 91.85 | 89.04 | 88.09 | 92.85 | 80.62 |
| Upright, guarded detector (45 pages) | 88.32 | 91.58 | 88.97 | 88.43 | 93.04 | 80.83 |

The first detector also turned Farsi, Arabic and Hebrew pages (right-to-left text is stored "backwards" on an
upright page) and charts with a few rotated labels, which cost text 0.32 and charts 0.21. The guarded
detector skips right-to-left scripts, needs nearly every line to agree after the turn, and for a quarter turn
needs nearly all longer words to stand on end. It keeps almost all of the tables gain (+0.75, P 1.00 on the
development part, +0.09 on the check slice) and leaves the other dimensions where they were.
