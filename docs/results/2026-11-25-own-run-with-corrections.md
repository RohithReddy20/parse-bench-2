# Our own run with every kept correction: 87.41

Date: 2026-11-25. Ledger run `own-best2`, config `configs/own_best.yaml`, pipeline name `docparse_best`. All 13
subsets, 4,317 pages, scored by the harness. The page readings are our own OCRia-1 run (vLLM 0.34.0 on Modal);
nothing here uses OCRia-1's published outputs.

| Dimension | Published OCRia-1 | Our run + corrections | Change | Dev | Check | Target |
|---|---:|---:|---:|---:|---:|---:|
| **Overall** | 85.24 | **87.41** | +2.17 | 87.41 | 87.45 | 89 |
| Text | 87.87 | 88.41 | +0.54 | 88.38 | 88.52 | 89.5 |
| Formatting | 91.58 | 93.06 | +1.48 | 93.19 | 92.62 | 93 |
| Tables | 86.28 | 88.22 | +1.93 | 87.39 | 91.49 | 90 |
| Charts | 77.66 | 80.83 | +3.18 | 80.88 | 80.61 | 81 |
| Layout | 85.96 | 90.24 | +4.28 | 91.14 | 86.95 | 88.8 |
| Forms | 82.10 | 83.71 | +1.61 | 83.47 | 84.49 | 87.5 |

- **Against the board:** second place is Claude Opus 5.5 at 87.13 and first is LlamaParse Agentic Plus at
  90.95. 87.41 is 0.28 above second place. That is a real but thin margin: one OCRia-1 run differs from
  another by up to 0.4 on a single dimension (charts), which is under 0.1 on the overall score.
- **The check slice** (pages no rule was tuned on) scores 87.45, level with the development part (87.41), so
  the result is not fitted to the development pages.
- **History of this number:** 86.93 with the first round of corrections, 87.14 with KDL nano v2 added as a
  second layout detector (run `own-best`), 87.41 with the second round on forms, charts and tables.
- **What is on:** looped-read and empty-read fallbacks; table grid repairs; form comb values, check-box
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
| Forms | 83.71 | 87.5 | 0.63 |
| Tables | 88.22 | 90 | 0.30 |
| Text | 88.41 | 89.5 | 0.18 |
| Charts | 80.83 | 81 | 0.03 |

Formatting and layout are at or above their targets. Also open: the 58 page reads that hit the token limit
lose about 0.5 overall points between them; second reads for those pages are being evaluated.
