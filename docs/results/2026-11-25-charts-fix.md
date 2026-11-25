# Charts: values measured from the PDF's vector drawing (M5)

Date: 2026-11-25. Everything here ran on the Mac with no GPU and no model calls, on OCRia-1's published
readings (stand-ins). Numbers are the charts dimension (`chart_next`, 568 pages) from the harness scorer; "dev" is
the development part (467 pages), "check" the check slice (101 pages, never inspected). "Gain" and "P" are the
paired bootstrap over source documents (`evalkit paired`) against the previous kept run.

**Result:** charts 77.66 → **80.77** on all pages (+3.11), dev 77.48 → 80.72 (+3.24), check 78.50 → 80.99
(+2.49). That is +0.52 on the overall score. The target was 81: this falls 0.23 short on all pages (0.28 on dev).
No variant reached 81 on all pages; the two closest (legend binding, 80.93–80.96; restored minus signs, 80.84)
failed the keep rule, on P and on the check slice.

Config: `configs/standin_charts.yaml` (`chart_options: {measure, labels, panels}`).

## 1. Where the loss was, and what the PDF can reach

OCRia-1 loses 22.3 points on charts. The loss table splits it into "points missing" (8.3), "points extra or wrong"
(7.3), "value off" (4.8) and "wrong value" (4.6), but the first two are mostly values too: the scorer's point F1
weights each matched point by its value score, so a point outside tolerance costs recall and precision at once.

Before building anything, the dev pages were split by what the PDF holds (analysis script, ground truth read
only there):

| Dev pages (467) | Pages | Points lost (dim) | Mean score |
|---|---:|---:|---:|
| Reference says "no data labels", bars/lines are vector paths | 233 | 15.63 | 68.7 |
| Reference says "no data labels", few paths (raster, or text as outlines) | 90 | 4.36 | 77.4 |
| Data labels shown and ≥ 80% of the reference values printed in the text layer | 112 | 1.43 | 94.0 |
| Data labels shown, values not in the text layer | 28 | 1.00 | 83 |

Only 6,619 of the 30,663 reference values (22%) are printed in the text layer, and the pages where they are
already score 94: OCRia-1 reads printed labels well. **Printed labels cover little; the loss is on unlabeled
charts drawn as vector paths** (line charts alone 6.5 points, then combos, stacked bars and bars). For those the
scorer's tolerance is 2% of the chart's value range (up to 6% is "value off"), and OCRia-1 reads values off the
image a few percent out. So the effort went to measurement.

## 2. What the fix does (`src/docparse/fixes/charts*.py`, `src/docparse/pdf/graphics.py`)

1. **Vector shapes** (`pdf/graphics.py`): every painted path of the page through pypdfium2 (no PyMuPDF), in display
   points, with Form XObject matrices composed, fill/stroke colours and Bezier end points.
2. **Value axes** (`charts_geometry.find_axes`): numeric tick labels aligned in a column (or row), evenly spaced,
   strictly monotone, linear or log-linear. A run that the next aligned label continues in value but not in
   spacing is a designed, non-linear scale and is dropped. Labels are tied to their gridlines when every label
   sits at the same offset from one (labels printed above their gridline are common: up to 11 units of error
   without this). An unlabelled zero line one step past the labels counts as the base.
3. **Marks**: filled axis-aligned rectangles (any subpath), stroked polylines (pieces chained), small closed
   shapes and stubs as markers.
4. **Zones**: each axis governs the region on either side of its labels up to the next label column. With
   `panels`, the region left of an axis that has its own plot on its right goes to the axis of that plot.
5. **Measured series**: bars grouped into one slot per table row (overlaps stack; side-by-side groups split at
   the widest gaps, which must stand out) and read as their length from zero (or their end on an axis above
   zero), one series per colour; lines by vertex (one per row) or through the numeric category axis (years);
   markers like vertices. With `labels`, bars that do not fall into one slot per row are placed by the table's
   category text printed under (or beside) the plot.
6. **Binding**: a table series (column, or row of the transposed table) binds to a measured series that agrees
   with OCRia-1's own reading: ≥ 60% of shared cells within 10% of the range (the axis range, or the table's own
   value range when smaller), at least 3 shared cells covering 70% of the series, the two correlated where the
   series moves, never a flat reading for a moving series. Best agreement first; each side binds once, and two
   readings of one drawn series (a line and its markers) count as one.
7. **Writing**: bound cells get the measured number with the cell's own sign character, currency and separators,
   to a tenth of the axis step (or the cell's own decimals if finer). A cell whose number is printed on the page
   outside the tick labels (a data label) is never changed. Every other byte of the table, and every table that
   does not bind, is written back unchanged (a new candidate in the IR; the base stays). An exception inside the
   fix keeps the reading and records a page finding.

## 3. Variants

| Run | Variant | Dev | Check | All | Dev gain | P | Kept |
|---|---|---:|---:|---:|---:|---:|---|
| ca-base | no options | 77.48 | 78.50 | 77.66 | | | |
| ca-m1 | `measure`: axes, marks, agreement binding, printed labels kept | 80.44 | 80.99 | 80.54 | +2.96 | 1.00 | **yes** |
| ca-m2 | (run aborted by a crash on a check page; not scored) | — | — | — | | | — |
| ca-m2b | + zero line one step past labels, bent-scale rule v1, '07 year labels, crash guard | 80.62 | 80.96 | 80.68 | +0.18 | 0.98 | no (check −0.03) |
| ca-m2-noreach | m2b without the zero-line reach | 80.46 | 80.96 | 80.55 | +0.02 | 0.76 | no |
| ca-m2-noyears | m2b without the year expansion (identical to m2b) | 80.62 | 80.96 | 80.68 | +0.18 | 0.98 | no |
| ca-m2-nobent | m2b without the bent-scale rule | 80.47 | 80.99 | 80.57 | +0.03 | 0.60 | no |
| ca-m3 | m2b with the bent rule v2 (neighbour within reach, jitter-relative) | 80.60 | 80.99 | 80.67 | +0.16 | 0.95 | **yes** |
| ca-m4-legend | m3 + legend binding | 80.77 | 81.42 | 80.89 | +0.17 | 0.86 | no (P) |
| ca-m3-signs | m3 + lost minus signs restored, inverted axes refused | 80.59 | 80.96 | 80.66 | −0.01 | 0.00 | no |
| ca-m3-reverse | m3 + bottom-to-top reading of horizontal charts | 80.61 | 80.99 | 80.68 | +0.01 | 0.63 | no |
| ca-m3-agree07 | m3 with agreement 0.7 instead of 0.6 | 80.53 | 80.95 | 80.61 | −0.07 | 0.19 | no |
| ca-m4-labels | m3 + bars placed by printed category labels | 80.69 | 80.99 | 80.75 | +0.10 | 0.96 | **yes** |
| ca-m5-legend | m4 + legend binding | 80.87 | 81.42 | 80.96 | +0.17 | 0.86 | no (P) |
| ca-m5-panels | m4 + small multiples keep their own axis | 80.72 | 80.99 | 80.77 | +0.03 | 0.99 | **yes** |
| ca-m6-legend | m5 + legend binding tied to the table's panel | 80.91 | 80.99 | 80.93 | +0.19 | 0.86 | no (P) |
| ca-m6-signs | m5 + lost minus signs v2 (several zeros per column, monotone pieces), inverted axes refused | 80.83 | 80.90 | 80.84 | +0.11 | 0.95 | no (check −0.09) |

Screened on dev only with the harness's `chart_next` called directly (a faster loop than `evalkit`; not part of
the keep decision and not reported as dimension scores), and dropped:

| Screen (vs the configuration then current) | Dev change |
|---|---:|
| keep OCRia-1's value when within 1% of the range of the measurement (`replace_min 0.01`) | −0.10 |
| agreement 0.5 | −0.09 |
| agreement tolerance 0.07 / 0.15 of the range | +0.01 / +0.01 |
| category labels also used to sample lines | −0.08 (label positions are not data positions on dense lines) |
| zones cut to the gridlines' extent (`sides`) | −0.05 |
| printed number beside a bar replaces the measurement (`snap`) | −0.07 (picks a neighbour's label) |

**Kept:** `measure`, `labels`, `panels`. Charts 80.77 on all pages (+3.11), 80.72 dev (+3.24), 80.99 check
(+2.49). The check slice gains about as much as dev (2.49 against 3.24), so the rules are not fitted to the dev
pages.

What the variants showed:

- **Measurement works where it binds.** In the fast loop the kept configuration changes 176 dev pages: 104 up,
  16 down, the rest equal (values rewritten to the same rounding). Lines with one vertex per category and long
  annual series sampled at their year ticks recover nearly exactly (a combo chart of 55 years goes from 0.53 to
  0.94).
- **Lost minus signs are real but the fix does not pass.** An OECD report's tick labels read "20 10 0 10 20 30"
  in the text layer (the minus glyph has no text mapping). Restoring them gains on dev (+0.11, P 0.95, nine
  documents) but loses 0.09 on the check slice, so it stays off (`signs`, `upright`).
- **Data labels must win.** On labeled charts the scorer allows 1% relative error, and a measured value rounded to
  a tenth of the tick step loses to the printed label. Before the "printed numbers are never changed" rule, ten
  labeled pages fell from 1.0 to 0.6–0.9.
- **Agreement with OCRia-1 is the binding evidence, and it is also the main risk.** Most regressions were bindings
  to a wrong but similar series: a flat stub series, the cumulative top of a stacked bar, the same bars measured
  against a neighbouring panel's axis. Each became a rule (flat readings refused, the table's own range as the
  tolerance base, panels).
- **Legend binding pays (+0.16 to +0.22 on all pages) but cannot pass P ≥ 0.9.** Its dev gains sit in two source
  documents; a bootstrap resample of the dev documents leaves both out often enough (roughly one time in eight,
  by a back-of-envelope count) that P stays at 0.86 however clean the rule is. It is in the code
  (`legend: true`), off.

## 4. What is left (dimension points lost, all pages)

| Error type | Severity | OCRia-1 alone | Kept |
|---|---|---:|---:|
| content not recovered | — | 15.61 | 13.39 |
| … points missing (recall) | — | 8.29 | 7.25 |
| … points extra or wrong (precision) | — | 7.32 | 6.14 |
| severity cap | — | 6.73 | 5.84 |
| value off | major | 4.82 | 3.83 |
| wrong value | critical | 4.63 | 3.96 |
| invented series or category | major | 0.33 | 0.35 |
| point missing | major | 0.23 | 0.23 |
| category missing | major | 0.19 | 0.18 |
| series missing | critical | 0.10 | 0.14 |

On dev, the unlabeled vector pages still lose 13.5 points (from 15.6); unlabeled pages with few paths 3.3 (from
4.4); labeled pages 2.3 (unchanged). The dev pages the first kept configuration (ca-m1) left untouched, classified
by why (12.6 points then; the later rules took part of the last row):

| Why untouched | Pages | Dim pts |
|---|---:|---:|
| measured series exist but none agrees with OCRia-1's reading | 40 | 3.2 |
| no value axis in the text layer (tick labels outlined or in an image) | 55 | 3.8 |
| no vector shapes (raster chart) | 14 | 1.1 |
| bars or lines that do not map to the table's rows (counts differ, no category axis) | ~40 | 2.8 |

The largest single document left is an OECD report (37 dev pages, 1.8 points, 2 touched) whose tick labels lost
their minus glyphs in the text layer ("20 10 0 10 20 30" down the column) and whose category labels are rotated.

The other option names in the code (`sides`, `snap`, `reverse`, `anchors`, `bar_mode`, `replace_min`, `agree`,
`agree_tol`, `coverage`, `zero_reach`, `years`, `bent`) exist for the ablations above; the defaults are the kept
behaviour. Tests: `tests/fixes/test_charts.py` (synthetic PDFs: bar chart with gridlines, legend, two panels,
lost minus signs, non-linear scale).

## 5. Next

- **Rotated category labels** (vertical text is split into one-letter words by the text layer): reading them as
  columns would place categories for many OECD-style bar charts.
- **Legend binding** under a document-level keep rule, or with more documents covered, is worth about +0.2.
- **Lines whose categories are month or quarter labels** (no numeric category axis): parse dates to positions.
- On our own OCRia-1 run (not stand-ins) the binding thresholds should be re-checked, since they lean on the
  reading's agreement.
