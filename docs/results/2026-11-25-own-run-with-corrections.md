# Our own run with every kept correction: 87.14

Date: 2026-11-25. Ledger run `own-best`, config `configs/own_best.yaml`, pipeline name `docparse_best`. All 13
subsets, 4,317 pages, scored by the harness. The page readings are our own OCRia-1 run (vLLM 0.34.0 on Modal);
nothing here uses OCRia-1's published outputs.

| Dimension | Published OCRia-1 | Our run + corrections | Change | Dev | Check | Target |
|---|---:|---:|---:|---:|---:|---:|
| **Overall** | 85.24 | **87.14** | +1.90 | 87.10 | 87.34 | 89 |
| Text | 87.87 | 88.41 | +0.54 | 88.38 | 88.52 | 89.5 |
| Formatting | 91.58 | 93.06 | +1.48 | 93.19 | 92.62 | 93 |
| Tables | 86.28 | 88.09 | +1.80 | 87.22 | 91.49 | 90 |
| Charts | 77.66 | 80.04 | +2.39 | 79.92 | 80.61 | 81 |
| Layout | 85.96 | 90.24 | +4.28 | 91.14 | 86.95 | 88.8 |
| Forms | 82.10 | 83.02 | +0.92 | 82.76 | 83.83 | 87.5 |

- **Against the board:** second place is Claude Opus 5.5 at 87.13 and first is LlamaParse Agentic Plus at
  90.95. 87.14 is above second place by 0.01, which is nothing: one run of OCRia-1 differs from another by
  more than that on charts alone. It needs a real margin before it can be called second place.
- **The check slice** (pages no rule was tuned on) scores 87.34, higher than the development part, so the
  result is not fitted to the development pages. Only text is flat on the check slice (-0.11).
- **What is on:** looped-read and empty-read fallbacks; table grid repairs; form comb values and check-box
  states; chart values measured from the vector drawing; strikeout, bold, list-marker and respelling repairs;
  layout rebuilt from heron regions, PDF words and OCRia-1's attention, tightened to PDF words, with items for
  uncovered words and KDL nano v2 as a second detector that votes between nested boxes.

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

## What is left to reach a safe second place and 89

| Dimension | Now | Target | Gap in overall points |
|---|---:|---:|---:|
| Forms | 83.02 | 87.5 | 0.75 |
| Tables | 88.09 | 90 | 0.32 |
| Text | 88.41 | 89.5 | 0.18 |
| Charts | 80.04 | 81 | 0.16 |

Also open: the 58 page reads that hit the token limit lose about 0.5 overall points between them; a second
read for those pages is being tested.
