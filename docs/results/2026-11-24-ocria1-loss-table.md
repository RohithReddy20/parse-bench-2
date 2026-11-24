# OCRia-1 loss table (M0, Task 6)

Date: 2026-11-24. Where OCRia-1 loses its points on ParseBench 2, per error type per dimension. This table
orders the fixes. It is computed from the baseline run with no GPU: `uv run python -m evalkit losses --run-id published-ocria_1-baseline`.

Pins: harness `run-llama/ParseBench@6e4479a`, dataset `llamaindex/ParseBench-2@2e954075`,
published outputs `llamaindex/ParseBench-2-results@497e243f` (`results/ocria_1/`).
Ledger run `published-ocria_1-baseline`, code `7e972ef`.

## How to read it

- All numbers are points of a dimension (0–100). Overall points are a sixth of that.
- A page's score is the smaller of two numbers: its **content** score (how much of the reference was
  recovered) and its **severity cap** (what its error events allow). `Content` rows are what the first costs;
  the cap rows are what the events cost on top. The two add up exactly to the points lost.
- An **event** row is the gain if that error type never happened, recomputed page by page with the harness's
  own `metric_policy.score_metric`. Event rows overlap (the first error of a severity costs the most), so they
  are a ranking, not a sum.
- Text and formatting save their counts per severity class only. Their rows share each class's gain among the
  error kinds on the page, by count.
- `content:` rows share a page's content loss among the metric's own sub-scores, in proportion to their
  shortfall. They say which kind of content is missing.
- Layout has no cap. Its rows split each page's lost score among the scorer's own outcomes per element.
- Checked on the real run: the saved counts rebuild all 4,290 capped page scores to 1e-6, and the rows add up
  to the lost score in every dimension (`tests/evalkit/test_losses.py::test_baseline_run_invariants`).

## Where the points are

| Dimension | Score | Lost | Content not recovered | Severity cap | Failed pages | Layout outcomes | Overall pts lost |
|---|---:|---:|---:|---:|---:|---:|---:|
| charts | 77.66 | 22.34 | 15.61 | 6.73 | — | — | 3.72 |
| forms | 82.10 | 17.90 | 11.38 | 6.53 | — | — | 2.98 |
| layout | 85.96 | 14.04 | — | — | — | 14.04 | 2.34 |
| tables | 86.28 | 13.72 | 7.48 | 6.24 | — | — | 2.29 |
| text | 87.87 | 12.13 | 6.18 | 5.95 | — | — | 2.02 |
| formatting | 91.58 | 8.42 | 7.20 | 1.22 | — | — | 1.40 |

Getting from 85.24 to above 87.13 needs 1.89 overall points, which is 11.3 dimension points in total.

## Can the PDF reach these losses?

A page counts as having a text layer when its PDF text has at least 50 characters. That is presence, not
health: an OCR layer on a scan also counts.

| Dimension | Subset | Pages with text layer | Mean score | Dim pts lost | Pages without | Mean score | Dim pts lost |
|---|---|---:|---:|---:|---:|---:|---:|
| charts | charts | 567 | 77.6 | 22.32 | 1 | 89.2 | 0.02 |
| formatting | headings | 100 | 97.1 | 0.36 | 0 | — | 0.00 |
| formatting | strikeout | 125 | 77.4 | 1.38 | 3 | 93.9 | 0.01 |
| formatting | text | 384 | 91.7 | 4.97 | 96 | 90.0 | 1.50 |
| formatting | underline | 122 | 96.9 | 0.19 | 3 | 97.9 | 0.00 |
| forms | fillable_forms | 100 | 92.6 | 0.93 | 0 | — | 0.00 |
| forms | forms | 499 | 80.7 | 16.88 | 1 | 46.7 | 0.09 |
| layout | layout | 506 | 86.0 | 13.20 | 29 | 84.4 | 0.84 |
| tables | complex_table_headers | 48 | 80.3 | 0.79 | 101 | 69.8 | 2.56 |
| tables | dense_tables | 373 | 84.1 | 1.99 | 0 | — | 0.00 |
| tables | financial_tables | 319 | 90.0 | 1.25 | 0 | — | 0.00 |
| tables | insurance_filing_tables | 412 | 84.0 | 2.00 | 0 | — | 0.00 |
| tables | tables | 494 | 89.6 | 5.12 | 6 | 99.0 | 0.01 |
| text | text | 403 | 89.3 | 8.47 | 105 | 82.3 | 3.66 |

| Dimension | Lost on pages with a text layer | Lost on pages without | Share the PDF could reach |
|---|---:|---:|---:|
| charts | 22.32 | 0.02 | 100% |
| formatting | 6.90 | 1.51 | 82% |
| forms | 17.81 | 0.09 | 99% |
| layout | 13.20 | 0.84 | 94% |
| tables | 11.15 | 2.56 | 81% |
| text | 8.47 | 3.66 | 70% |
no text layer: 249 of 4317

4,068 of the 4,317 pages have a text layer. Charts, forms and layout lose almost everything on such pages.
The exception is `complex_table_headers`: 101 of its 149 pages are images, and they hold 2.56 of the 13.72
table points lost. The text subset loses 3.66 of 12.13 on its 105 pages without a text layer.

## Work queue (all subsets of a dimension merged)

| # | Dimension | Error type | Severity | Pages | Dimension pts | Overall pts |
|---:|---|---|---|---:|---:|---:|
| 1 | charts | content: points missing (recall) | — | 355 | 8.30 | 1.383 |
| 2 | charts | content: points extra or wrong (precision) | — | 355 | 7.31 | 1.219 |
| 3 | tables | content: cell content | — | 639 | 7.01 | 1.169 |
| 4 | charts | value off | major | 281 | 4.82 | 0.804 |
| 5 | charts | wrong value | critical | 210 | 4.63 | 0.771 |
| 6 | forms | content: filled values | — | 315 | 4.33 | 0.721 |
| 7 | text | content: words missing (recall) | — | 257 | 3.92 | 0.653 |
| 8 | layout | element not located (no box matches): text | — | 163 | 2.83 | 0.472 |
| 9 | forms | content: blank fields | — | 310 | 2.55 | 0.426 |
| 10 | formatting | content: inline styles (bold, italic, ...) | — | 393 | 2.37 | 0.395 |
| 11 | text | content: words extra or wrong (precision) | — | 199 | 2.13 | 0.355 |
| 12 | forms | content: under half of the items found (coverage cap) | — | 20 | 2.10 | 0.350 |
| 13 | tables | systematic column error: wrong numbers (3+ in one column) | critical | 198 | 1.97 | 0.329 |
| 14 | forms | content: check boxes | — | 201 | 1.94 | 0.324 |
| 15 | layout | extra box: holds words locked to another element | — | 191 | 1.93 | 0.322 |
| 16 | layout | box found, text does not match (token F1 < 0.8): text | — | 171 | 1.63 | 0.272 |
| 17 | forms | wrong value | critical | 116 | 1.55 | 0.258 |
| 18 | formatting | content: headings | — | 229 | 1.54 | 0.257 |
| 19 | text | omitted | major | 116 | 1.50 | 0.250 |
| 20 | text | wrong words | major | 68 | 1.36 | 0.227 |
| 21 | layout | box looser or tighter than the reference (IoU ramp): text | — | 266 | 1.29 | 0.215 |
| 22 | forms | invented value | critical | 89 | 1.24 | 0.207 |
| 23 | formatting | content: under half of the items found (coverage cap) | — | 15 | 1.19 | 0.198 |
| 24 | text | wrong number or identifier | critical | 40 | 1.13 | 0.189 |
| 25 | forms | value without its caption | major | 120 | 1.10 | 0.183 |
| 26 | tables | row missing | critical | 147 | 1.07 | 0.179 |
| 27 | forms | empty field omitted | minor | 178 | 1.06 | 0.176 |
| 28 | layout | element not located (no box matches): heading | — | 104 | 0.92 | 0.153 |
| 29 | layout | element not located (no box matches): picture | — | 65 | 0.86 | 0.143 |
| 30 | forms | value lost | major | 132 | 0.85 | 0.141 |
| 31 | formatting | content: strikeout | — | 73 | 0.83 | 0.138 |
| 32 | tables | wrong or misplaced number in a row | critical | 166 | 0.82 | 0.136 |
| 33 | layout | box found, text does not match (token F1 < 0.8): heading | — | 120 | 0.76 | 0.127 |
| 34 | layout | box looser or tighter than the reference (IoU ramp): heading | — | 205 | 0.74 | 0.124 |
| 35 | forms | caption garbled | minor | 126 | 0.68 | 0.114 |
| 36 | layout | box looser or tighter than the reference (IoU ramp): picture | — | 153 | 0.67 | 0.112 |
| 37 | formatting | content: underline | — | 97 | 0.66 | 0.110 |
| 38 | formatting | content: lists | — | 81 | 0.60 | 0.100 |
| 39 | text | name changed | critical | 25 | 0.60 | 0.099 |
| 40 | tables | repeated row | major | 159 | 0.57 | 0.095 |
| 41 | forms | box state flipped | critical | 39 | 0.53 | 0.089 |
| 42 | text | spelling | minor | 84 | 0.50 | 0.084 |
| 43 | tables | content: row order | — | 86 | 0.47 | 0.078 |
| 44 | text | blocks out of order | major | 49 | 0.44 | 0.073 |
| 45 | forms | content: signatures | — | 38 | 0.42 | 0.070 |
| 46 | tables | extra column | major | 144 | 0.41 | 0.069 |
| 47 | layout | element not located (no box matches): table | — | 14 | 0.41 | 0.068 |
| 48 | forms | unchecked state lost | minor | 84 | 0.40 | 0.067 |
| 49 | tables | systematic column error: wrong text or units (3+ in one column) | major | 116 | 0.40 | 0.066 |
| 50 | layout | box looser or tighter than the reference (IoU ramp): header/footer | — | 250 | 0.36 | 0.061 |
| 51 | text | omitted passage | critical | 25 | 0.36 | 0.060 |
| 52 | forms | contradicting copy of a field | critical | 25 | 0.36 | 0.060 |
| 53 | tables | label reworded (different wording) | major | 227 | 0.35 | 0.059 |
| 54 | layout | element not located (no box matches): header/footer | — | 92 | 0.35 | 0.059 |
| 55 | forms | unchecked option omitted | minor | 77 | 0.35 | 0.059 |
| 56 | text | inserted words | major | 41 | 0.35 | 0.058 |
| 57 | forms | checked state lost | major | 55 | 0.34 | 0.056 |
| 58 | charts | invented series or category | major | 30 | 0.33 | 0.055 |
| 59 | tables | invented row | critical | 44 | 0.32 | 0.053 |
| 60 | tables | header rows differ | major | 177 | 0.31 | 0.052 |
| 61 | formatting | bold added | minor | 101 | 0.30 | 0.050 |
| 62 | tables | wrong text, lost unit or partly different value in a row | major | 130 | 0.29 | 0.049 |
| 63 | forms | extra text in value | minor | 107 | 0.27 | 0.045 |
| 64 | layout | box found, text does not match (token F1 < 0.8): table | — | 9 | 0.26 | 0.043 |
| 65 | forms | printed text in a blank field | minor | 79 | 0.25 | 0.042 |
| 66 | layout | box found, text does not match (token F1 < 0.8): header/footer | — | 69 | 0.24 | 0.040 |
| 67 | layout | extra box: blank area | — | 54 | 0.24 | 0.039 |
| 68 | charts | point missing | major | 36 | 0.23 | 0.038 |
| 69 | layout | table box misses its words (< 90% of word centres): table | — | 3 | 0.22 | 0.037 |
| 70 | forms | value of another field | critical | 19 | 0.20 | 0.033 |

## Per dimension, by subset

### Text

| Subset | Error type | Severity | Pages | Dimension pts |
|---|---|---|---:|---:|
| text | content: words missing (recall) | — | 257 | 3.92 |
| text | content: words extra or wrong (precision) | — | 199 | 2.13 |
| text | omitted | major | 116 | 1.50 |
| text | wrong words | major | 68 | 1.36 |
| text | wrong number or identifier | critical | 40 | 1.13 |
| text | name changed | critical | 25 | 0.60 |
| text | spelling | minor | 84 | 0.50 |
| text | blocks out of order | major | 49 | 0.44 |
| text | omitted passage | critical | 25 | 0.36 |
| text | inserted words | major | 41 | 0.35 |
| text | accent only | minor | 38 | 0.17 |
| text | sentences cut | major | 23 | 0.16 |
| text | negator in changed text | critical | 5 | 0.14 |
| text | content: unattributed | — | 15 | 0.13 |

### Formatting

| Subset | Error type | Severity | Pages | Dimension pts |
|---|---|---|---:|---:|
| text | content: inline styles (bold, italic, ...) | — | 319 | 1.75 |
| text | content: headings | — | 188 | 1.18 |
| text | content: under half of the items found (coverage cap) | — | 12 | 1.15 |
| text | content: lists | — | 81 | 0.60 |
| strikeout | content: strikeout | — | 59 | 0.59 |
| strikeout | content: inline styles (bold, italic, ...) | — | 50 | 0.59 |
| text | content: underline | — | 69 | 0.51 |
| headings | content: headings | — | 41 | 0.36 |
| text | bold added | minor | 101 | 0.30 |
| text | content: strikeout | — | 14 | 0.24 |
| underline | content: underline | — | 28 | 0.15 |
| text | strikethrough added | critical | 7 | 0.10 |
| text | list marker lost | minor | 32 | 0.09 |
| text | heading lost | major | 13 | 0.09 |

### Tables

| Subset | Error type | Severity | Pages | Dimension pts |
|---|---|---|---:|---:|
| tables | content: cell content | — | 138 | 2.64 |
| complex_table_headers | content: cell content | — | 97 | 1.61 |
| dense_tables | content: cell content | — | 162 | 1.24 |
| insurance_filing_tables | content: cell content | — | 160 | 0.90 |
| complex_table_headers | systematic column error: wrong numbers (3+ in one column) | critical | 34 | 0.64 |
| financial_tables | content: cell content | — | 82 | 0.63 |
| tables | systematic column error: wrong numbers (3+ in one column) | critical | 31 | 0.52 |
| tables | row missing | critical | 31 | 0.47 |
| insurance_filing_tables | systematic column error: wrong numbers (3+ in one column) | critical | 54 | 0.45 |
| tables | wrong or misplaced number in a row | critical | 35 | 0.29 |
| complex_table_headers | wrong or misplaced number in a row | critical | 34 | 0.27 |
| complex_table_headers | row missing | critical | 19 | 0.23 |
| complex_table_headers | repeated row | major | 36 | 0.22 |
| dense_tables | systematic column error: wrong numbers (3+ in one column) | critical | 61 | 0.21 |

### Charts

| Subset | Error type | Severity | Pages | Dimension pts |
|---|---|---|---:|---:|
| charts | content: points missing (recall) | — | 355 | 8.30 |
| charts | content: points extra or wrong (precision) | — | 355 | 7.31 |
| charts | value off | major | 281 | 4.82 |
| charts | wrong value | critical | 210 | 4.63 |
| charts | invented series or category | major | 30 | 0.33 |
| charts | point missing | major | 36 | 0.23 |
| charts | category missing | major | 24 | 0.19 |
| charts | series renamed | major | 5 | 0.11 |
| charts | series missing | critical | 9 | 0.10 |
| charts | chart missing | critical | 7 | 0.06 |
| charts | wrong title | major | 6 | 0.06 |
| charts | category label reworded | major | 6 | 0.04 |
| charts | colour groups merged | minor | 15 | 0.03 |
| charts | series name reworded | minor | 5 | 0.02 |

### Layout

| Subset | Error type | Severity | Pages | Dimension pts |
|---|---|---|---:|---:|
| layout | element not located (no box matches): text | — | 163 | 2.83 |
| layout | extra box: holds words locked to another element | — | 191 | 1.93 |
| layout | box found, text does not match (token F1 < 0.8): text | — | 171 | 1.63 |
| layout | box looser or tighter than the reference (IoU ramp): text | — | 266 | 1.29 |
| layout | element not located (no box matches): heading | — | 104 | 0.92 |
| layout | element not located (no box matches): picture | — | 65 | 0.86 |
| layout | box found, text does not match (token F1 < 0.8): heading | — | 120 | 0.76 |
| layout | box looser or tighter than the reference (IoU ramp): heading | — | 205 | 0.74 |
| layout | box looser or tighter than the reference (IoU ramp): picture | — | 153 | 0.67 |
| layout | element not located (no box matches): table | — | 14 | 0.41 |
| layout | box looser or tighter than the reference (IoU ramp): header/footer | — | 250 | 0.36 |
| layout | element not located (no box matches): header/footer | — | 92 | 0.35 |
| layout | box found, text does not match (token F1 < 0.8): table | — | 9 | 0.26 |
| layout | box found, text does not match (token F1 < 0.8): header/footer | — | 69 | 0.24 |

### Forms

| Subset | Error type | Severity | Pages | Dimension pts |
|---|---|---|---:|---:|
| forms | content: filled values | — | 312 | 4.31 |
| forms | content: blank fields | — | 267 | 2.34 |
| forms | content: under half of the items found (coverage cap) | — | 20 | 2.10 |
| forms | content: check boxes | — | 186 | 1.89 |
| forms | wrong value | critical | 116 | 1.55 |
| forms | value without its caption | major | 120 | 1.10 |
| forms | invented value | critical | 74 | 1.08 |
| forms | empty field omitted | minor | 152 | 0.95 |
| forms | value lost | major | 130 | 0.83 |
| forms | caption garbled | minor | 122 | 0.68 |
| forms | box state flipped | critical | 39 | 0.53 |
| forms | content: signatures | — | 34 | 0.40 |
| forms | unchecked state lost | minor | 76 | 0.36 |
| forms | contradicting copy of a field | critical | 24 | 0.35 |

## What this says about each planned fix (design §6)

**Charts (22.3 lost, 3.72 overall).** The largest block, and it is almost all values. Titles (98.8), labels
(99.96) and finding the chart (98.4% of charts) are already right; series renamed, legend lost and missing
title together cost under 0.3. Recall is 84.5 and precision 85.8: points are missing or outside tolerance.
`value off` and `wrong value` cost 4.8 and 4.6 on top. So the planned fix is aimed correctly (replace values
with printed labels or measured geometry, keep OCRia-1's names), and name matching can be skipped at first.
46% of the chart loss sits on 84 pages scoring under 0.5; those are the first pages to look at.

**Forms (17.9 lost, 2.98 overall).** Filled values are the weakest lane (83.0 on 433 pages) and the largest
content row (4.3), followed by blank fields (2.6), check boxes (1.9) and a coverage cap on 20 pages (2.1).
On top: wrong value 1.55, invented value 1.24, value without its caption 1.10, empty field omitted 1.06,
value lost 0.85, box state flipped 0.53. 499 of 500 form pages have a text layer. This matches the plan
(values, blanks and check boxes from the PDF). `value without its caption` and the coverage-cap pages are
structure problems the value swap will not fix.

**Layout (14.0 lost, 2.34 overall).** Elements with no matching box cost 5.4 (text 2.8, headings 0.9,
pictures 0.9, tables 0.4, headers and footers 0.35). Boxes with the wrong text cost 2.9. Extra boxes holding
another element's words cost 1.9. Loose or tight boxes cost 3.2 across classes. So tightening alone is worth
at most about 3; most of the loss is segmentation (missed, merged or split elements), which is what a second
detector and paragraph splitting address.

**Tables (13.7 lost, 2.29 overall).** Cell content is the largest row (7.0) and wrong numbers follow:
systematic column errors 2.0, wrong or misplaced number in a row 0.8, row missing 1.1, repeated row 0.6,
extra column 0.4. Replacing cell text with PDF words targets the first group directly on the 1,646 table
pages with a text layer. `complex_table_headers` (73.2) is mostly images and needs the re-read path instead.
More than half of the table loss is on pages scoring under 0.5, where the structure itself has failed.

**Text (12.1 lost, 2.02 overall).** Missing words 3.9, extra or wrong words 2.1, then omitted 1.5, wrong
words 1.4, wrong number or identifier 1.1, name changed 0.6, spelling 0.5, blocks out of order 0.4. 70% of
the loss is on pages with a text layer, so alignment to the text layer can reach it; 13 pages score 0.

**Formatting (8.4 lost, 1.40 overall).** The smallest block. Inline styles 2.4, headings 1.5, a coverage cap
on 15 pages 1.2, strikeout 0.8. The strikeout subset scores 77.75 with 21 pages at 0, but its slot weight is
1/16, so fixing it fully is worth about 1.4 formatting points (0.23 overall). `bold added` is the most
frequent event (1,105 times) and costs only 0.3.

## Order of work this suggests

By points available and by how directly the PDF holds the answer:

1. Charts: values (up to 3.7 overall available; the planned target of 81+ is +0.56).
2. Forms: values, blanks, check boxes (2.98 available; target +0.90).
3. Tables: cell text and numbers (2.29 available; target +0.62).
4. Layout: located elements and segmentation (2.34 available; target +0.47). M1 does this first because it
   needs no new reading of the page.
5. Text (2.02 available; target +0.27), then formatting (1.40 available; target +0.24).

These are OCRia-1's losses on all pages. The same table for the development part alone is one command:
`python -m evalkit losses --run-id published-ocria_1-baseline --fold dev`.
