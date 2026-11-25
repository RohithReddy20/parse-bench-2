# Tables: placing OCRia-1's cells by the PDF text layer (M4 groundwork)

Date: 2026-11-25. Everything here ran on the Mac with no GPU, on OCRia-1's published readings (stand-ins).
Subsets `tables`, `complex_table_headers`, `financial_tables`, `insurance_filing_tables`, `dense_tables`; metric
`table_next`; slot weights 0.5 for `tables` and 0.125 for each of the other four. "Dev" is the development part,
"check" the check slice. Ledger runs are prefixed `ta-`; the baseline `ta-base` (no options) scores 86.20 on all
pages, 85.22 on dev and 90.05 on check, the same as `m0-roundtrip`.

The fixes live in `src/docparse/fixes/tables.py` (with `tables_align.py` and `tables_html.py`) behind
`table_options`, called from `pipeline.py` right after the page read is parsed into the IR. A corrected table is a
new candidate of its region (reader `fix.tables.pdf`); a table no option changes is written back byte for byte.

## 1. Does the PDF know the digits better than OCRia-1? No.

The plan was "PDF words per cell": keep OCRia-1's structure, replace each cell's text with the text-layer words
it covers. The first measurement says the characters are not where the points are.

- **OCRia-1 almost never misreads a digit on these pages.** Comparing numeric cell values page by page (ground
  truth, OCRia-1's cells, and every run of up to eight consecutive text-layer words on a line): of 418,080
  numeric reference cells, only 46 pairs on 21 pages look like a misread number (an OCRia-1 value found neither
  in the reference nor in the PDF, next to a reference value the PDF does hold, a digit or two apart). Most of
  those are formatting (`$591,588` against `591,588`), not wrong digits.
- **The scorer's "wrong number" cells are misplaced, not misread.** Of the 32,000 reference cells the scorer
  flags as "wrong or misplaced number" on pages with a text layer, 27,500 have their exact value elsewhere in
  OCRia-1's output as a cell of its own, 2,000 inside a merged cell, 1,700 are missing with the value in the
  PDF, and 700 are in neither.
- **Substituting PDF text loses** (local re-score of every table page with the harness's `table_next`, not a
  ledger run): replacing a cell that matches a PDF word run "up to a small edit" with that run costs 0.82 points
  (182 pages down, 25 up); restricting it to same-length, digit-for-digit substitutions of at most two digits
  still costs 0.08 (19 pages down, 13 up). The PDF is wrong about as often as OCRia-1: a font map that maps one
  digit to another (`8,973,865` read `2,973,265` from the text layer, `dense_table_392`), repetitive tables where
  a row is matched one line off (`308` against `303`, `dense_table_281`), numbers wrapped in a narrow column
  (`4,70` / `4` on two lines, `dense_table_496`), and one-digit cells.

So, as the brief allowed, the work moved to structure: where OCRia-1 put its (correct) values.

## 2. How a table is matched to the page

1. Text-layer words (`RenderedPage.text_words`, in points) are grouped into lines.
2. Table rows go to lines by a monotone assignment (row order = line order) that maximises the number of cells
   whose text occurs on the line as consecutive words, compared without case, spaces or dash style. A value
   repeated in every row cannot pull a row onto the wrong line, because the whole row votes and order is kept.
3. On its line, a row's cells are matched to runs of consecutive words, in order, by a small dynamic program;
   only exact matches count. Words used by one table are not offered to the next.
4. **Health test:** a table is changed only when at least half of its text cells were found word for word
   (`health`, default 0.5). A garbled text layer (`financial_tables/accenture-2026-08-14_page1`, a custom font
   encoding) finds nothing and the model's table stays.
5. **Columns** are located by the matched single-column cells of full-width rows (the table's width is the most
   common row width, so one row written with a stray extra cell does not widen it).

## 3. Variants

Gains and probabilities are from `evalkit paired` (bootstrap over source documents) against the previous kept
configuration. Ledger runs are marked; "local" rows were re-scored in a scratch script with the harness's own
`table_next` on the same pages (same numbers as the ledger for unchanged pages), and were used to choose what to
run for real.

**Ledger runs** (each against the configuration kept before it):

| Run | Options | Dev | Check | All | Dev gain | P (dev) | Check gain | Kept |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `ta-base` | none | 85.22 | 90.05 | 86.20 | | | | baseline |
| `ta-regrid` | `regrid` | 86.37 | 91.10 | 87.32 | +1.15 | 1.00 | +1.05 | **yes** |
| `ta-regrid-wide` | + `regrid_wide` | 86.44 | 91.32 | 87.43 | +0.07 | 0.99 | +0.22 | **yes** |
| `ta-complete` | + `complete` | 86.58 | 91.36 | 87.54 | +0.13 | 1.00 | +0.04 | **yes** |
| `ta-merge` | + `merge` | 86.84 | 91.63 | 87.81 | +0.27 | 1.00 | +0.26 | **yes** |
| `ta-extend` | + `extend` | 87.10 | 91.70 | 88.03 | +0.25 | 0.99 | +0.07 | **yes** |
| `ta-extend` against `ta-base` | all five | 87.10 | 91.70 | 88.03 | +1.87 | 1.00 | +1.65 | gain on all pages +1.82 |

**Local re-scores** (deltas against `ta-base` unless said otherwise; pages up / down):

| Variant | Dev | Check | All | Pages up / down | Outcome |
|---|---:|---:|---:|---:|---|
| Cell text from the PDF where it matches "up to a small edit" | -0.77 | -1.01 | -0.82 | 25 / 182 | rejected (section 1) |
| Same, digit-for-digit substitutions only (same length, at most 2 digits) | -0.08 | -0.09 | -0.08 | 13 / 19 | rejected |
| Blank corner cell put in front of every header row one cell short, no PDF | +0.10 | -0.06 | +0.06 | 38 / 43 | rejected: the gap is not always in front |
| Short header rows placed by their PDF words | +0.86 | +0.72 | +0.83 | 58 / 9 | became `regrid` |
| Short body rows placed by their PDF words | +0.13 | +0.27 | +0.16 | 22 / 5 | became `regrid` |
| Both (what `regrid` does) | +1.13 | +0.99 | +1.10 | 77 / 12 | ledger: +1.12 |
| + rows wider than the table, empty cells laid out again | +1.20 | +1.19 | +1.19 | 84 / 16 | became `regrid_wide` |
| + full-width rows with empty cells laid out again | +1.01 | +0.90 | +0.98 | 79 / 32 | rejected (worse than without) |
| Unclosed table: close it, add nothing | +0.01 | +0.00 | +0.01 | 10 / 6 | control |
| Unclosed table completed from the text layer | +0.13 | +0.04 | +0.12 | 15 / 0 | became `complete` |
| Two tables side by side merged: right one has a numeric first column | +0.13 | +0.12 | +0.13 | 16 / 21 | refined |
| ... and shares no header text with the left one | -0.01 | 0.00 | -0.01 | 0 / 1 | rejected: sub-headers repeat in real cases |
| ... and does not repeat the left one's row-label header | +0.15 | +0.12 | +0.14 | 16 / 8 | refined |
| ... and the left one may have extra title rows | +0.26 | +0.27 | +0.26 | 21 / 9 | became `merge` |
| `regrid` + `regrid_wide`, then a group header ending a short header row spans the rest (vs. without) | +0.26 | +0.07 | +0.22 | 49 / 4 | became `extend` |
| Same with any single-column last cell too (vs. `regrid` + `regrid_wide`) | +0.25 | +0.07 | +0.22 | 58 / 10 | rejected (two columns at least is a little better) |
| Text lines between matched rows that no row took, fitting the columns, inserted as rows (vs. `regrid` + `regrid_wide`) | -0.66 | -0.24 | -0.58 | 7 / 39 | rejected: most such lines belong to rows matched elsewhere |
| Empty cells of matched rows filled with the unused words of their line inside their column (vs. `regrid` + `regrid_wide`) | -0.29 | -0.54 | -0.35 | 16 / 95 | rejected: those words are mostly parts of other cells |
| Full-width header rows with empty cells laid out again (vs. `regrid` + `regrid_wide`) | -0.01 | 0.00 | -0.00 | 1 / 2 | rejected (almost never applies) |
| Health share 0.3 instead of 0.5 (vs. `regrid` + `regrid_wide`) | +0.00 | 0.00 | +0.00 | 2 / 0 | 0.5 kept (insensitive) |
| Health share 0.7 instead of 0.5 (vs. `regrid` + `regrid_wide`) | -0.01 | -0.31 | -0.07 | 2 / 4 | 0.5 kept |

## 4. What is kept

`configs/standin_tables.yaml`: `table_options: {regrid, regrid_wide, complete, merge, extend}` (health share
0.5). Tables **86.20 → 88.03 on all pages (+1.82)**, 85.22 → 87.10 on the development part (+1.87, P 1.00) and
90.05 → 91.70 on the check slice (+1.65, P 1.00). That is about +0.30 on the overall score. By subset: `tables`
89.75 → 90.58, `complex_table_headers` 72.57 → 73.91, `financial_tables` 90.03 → 91.56,
`insurance_filing_tables` 83.99 → 89.42, `dense_tables` 84.06 → 87.01.

This falls short of the target of about 90.

What each kept option does, and what it found on these pages:

- **`regrid`** (the bulk of the gain). OCRia-1 often writes a header row one cell short: the blank corner cell
  over an index or label column is left out, so every header slides one column left and every value of the
  table is keyed under its neighbour's header. The scorer then charges a "systematic column error" per column.
  On a born-digital page the header words sit above their columns, so the missing cell's place is known: each
  cell of a short row is moved to the column its words lie in by inserting empty cells in front of it. The same
  goes for a body row that dropped an empty cell. Rows are placed only when the result fits the table's width,
  and a header row no text line was matched to is located by its cells' text just above the table body. The
  blind version of this rule (always put the blank cell in front) is a coin flip (38 pages up, 43 down); with
  the PDF it is 77 up, 12 down. A family of insurance filing pages (`... Year | Written Premium | ...` tables)
  goes from about 0.07 to 1.00 each.
- **`regrid_wide`**: a row one cell wider than the rest because of a stray empty cell has its empty cells laid
  out again around the cells found on the page (two at least).
- **`complete`**: a reading cut off by the token limit leaves its last table unclosed. Its complete rows are
  kept and the text layer's lines below the last matched row are added as rows while every word of a line falls
  in one of the table's columns and the line fills at least half of them; the first line that does not fit (a
  note, a footer) ends the table. 14 pages change, mostly long dense tables; none goes down.
- **`merge`**: OCRia-1 sometimes writes one wide table as two tables, one after the other: the left part with the
  row labels and the right part (a second column group such as "Rolling 4-Quarter") without them, starting at
  the first row that has values. The PDF shows both parts on the same text lines, the right one to the right.
  They are merged back when every body row of the right part lies on a line of a body row of the left part, the
  right part's first column holds numbers, and it does not repeat the left part's row-label header (a repeated
  group such as `Pad no. | Pad Name | X | Y` twice is a separate table in the references, and stays one). A
  family of insurance trend pages goes from about 0.5 to 0.9. The first version, which kept tables apart when
  they shared any header text, merged nothing useful: in the real cases the sub-headers repeat.
- **`extend`**: after `regrid`, a header row still narrower than the table whose last cell spans two or more
  columns and ends the row ("Percent Change 2025 vs. 2024" over four of five value columns) gets that cell spanned
  to the table's last column. This one does not use the PDF and also runs on scans, where it helps
  `complex_table_headers`. Its check-slice gain (+0.07) is under half its development gain (+0.25), the design's
  overfitting signal; it has a single threshold (two columns at least), and one column at least was slightly
  worse, so the simplest form is kept.

Pages without a text layer, and tables whose text layer fails the health test, are left as they were (apart from
`extend`).

## 5. What is left

`python -m evalkit losses --run-id ta-extend`: 11.97 dimension points lost (13.80 at `ta-base`). Pages with a
text layer lose 9.55 (11.23 before), pages without one 2.42 (2.56). Pages at 0 cost 2.73, pages under 0.5 cost
3.70, pages between 0.5 and 0.9 cost 4.60.

| Error type | Severity | `ta-base` | `ta-extend` |
|---|---|---:|---:|
| content: cell content | — | 7.04 | 5.80 |
| systematic column error: wrong numbers (3+ in one column) | critical | 2.02 | 1.67 |
| row missing | critical | 1.07 | 0.99 |
| wrong or misplaced number in a row | critical | 0.82 | 0.81 |
| repeated row | major | 0.57 | 0.55 |
| systematic column error: wrong text or units | major | 0.40 | 0.37 |
| extra column | major | 0.41 | 0.35 |
| label reworded (different wording) | major | 0.35 | 0.35 |
| invented row | critical | 0.32 | 0.32 |
| content: row order | — | 0.47 | 0.31 |
| header rows differ | major | 0.31 | 0.27 |

What the remaining losses are, from the pages looked at:

- **Structure the PDF cannot settle.** Most "wrong number" cells are still values OCRia-1 read correctly but
  keyed differently from the reference: multi-level headers flattened or split differently, a currency code
  and amount in one cell or two (`EUR | 20,000,000` against `EUR 20,000,000`), two values in one reference
  cell, a reference that splits side-by-side groups the reading keeps together. Which form the reference wants
  varies from page to page; no geometric rule tried here picks it.
- **Reads that failed outright**: three looped readings (`SERFF_TX_random_pages_1__p649`, `blackrock-funds...p25`:
  the same table repeated over 80 times; `dense_table_018`: a header repeated until the token limit) and a few
  garbled text layers (`financial_tables/accenture-2026-08-14_page1` uses a custom encoding). They need a table
  built from the text layer alone, or a second model read.
- **Dropped rows and columns.** Inserting text lines that no row took, and filling empty cells with unused words,
  both lost points: in these tables most unclaimed words belong to multi-line cells or to rows the assignment
  put on a neighbouring line. A dropped column group (`SERFF_CA_random_pages 1_page46`: seven groups printed, six
  read) is visible in the text layer but was not attempted.
- **Pages without a text layer** (101 of the 149 `complex_table_headers` pages) still lose 2.4 points; only
  `extend` reaches them.

Best next step: build a table from the text layer alone (lines → columns by the gaps that run through all rows),
use it as a second reading for looped or garbled reads, and as a structural vote where OCRia-1's header
structure and the text layer's column count disagree.

# Round two

Date: 2026-11-25. Same subsets, metric and weights. Base: `configs/standin_best.yaml` (every correction kept so far,
in every dimension), ledger run `t2-base`: tables **88.05** on all pages, 87.12 on dev, 91.71 on check (the same as
`v-best2`). Ledger runs are prefixed `t2-`. Target 90.

## 6. How the remaining loss was ranked

From `t2-base` (11.95 points lost) and the per-page metadata:

- **Page weight decides.** A `tables` page is worth 0.10 dimension points, a `complex_table_headers` page 0.084, a
  `financial_tables` page 0.039, an `insurance_filing_tables` page 0.030, a `dense_tables` page 0.034. Content
  (cells never matched) is 6.11 of the 11.95 and the severity cap 5.84; in the cap, "systematic column error: wrong
  numbers" is the largest share (about 1.3), then "row missing" (0.8) and "wrong number in a row" (0.7).
- **By page kind:** pages without a text layer lose 2.4 points (most are `complex_table_headers` crops); pages with
  one lose 9.5, spread over a long tail: 119 `tables` pages with a text layer score under 0.95, and nearly every
  one fails for its own reason (reference taste about joined or split cells, a title row, a units row, empty
  columns).
- **Rotated pages.** 52 pages print their text sideways (21 turned 90°, 18 turned 270°) or upside down (13), and
  OCRia-1 read the rotated image. Between them they lose 1.24 points: 0.024 a page, against 0.005 a page on upright
  pages. Round one's fixes never touched them: the text layer's lines run across the table, so no row matched. Two
  of the looped reads (`SERFF_TX_random_pages_1__p649`, `blackrock-funds...p25`) and the mirrored read of
  `FBLB-134215544__p110` (its columns in reverse order) are upside-down pages.
- **Garbled text layers.** 72 pages have a text layer whose glyphs mostly lack a proper unicode mapping
  (`RenderedPage.map_errors` over 0.1; most `financial_tables` pages from one source, and the Apple DEF 14A pages).
  The word positions are right but the text is not, so no cell matches and the health test keeps OCRia-1's table.

## 7. Variants

Ledger runs (each against the configuration kept before it):

| Run | Options added | Dev | Check | All | Dev gain | P (dev) | Check gain | Kept |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `t2-base` | none (round-one configuration) | 87.12 | 91.71 | 88.05 | | | | baseline |
| `t2-leaf` | `leaf` (`leaf_mode` 3) | 87.26 | 91.71 | 88.16 | +0.13 | 0.98 | +0.00 | **yes** |
| `t2-failed` | `dedupe` + `complete_extent` (one change: failed reads) | 87.30 | 91.71 | 88.19 | +0.04 | 0.95 | +0.00 | **yes** |
| `t2-failed` against `t2-base` | all three | 87.30 | 91.71 | 88.19 | +0.17 | 1.00 | +0.00 | +0.14 on all pages |

Local re-scores. Every page was replayed through `fix_tables` from a capture of its IR just before the table step
(with its text layer and shapes), and scored with the harness's own `table_next` and the page's test flags. The
replay reproduces `t2-base` exactly (no page differs; 87.12 / 91.71 / 88.05) and predicted both ledger runs to the
page. P is the same document bootstrap as `evalkit paired`. Against `t2-base` unless marked "vs leaf":

| Variant | Dev | Check | All | P (dev) | Pages up / down | Outcome |
|---|---:|---:|---:|---:|---:|---|
| `leaf` mode 0 (every body row as wide as the table) | +0.132 | 0.000 | +0.105 | | 3 / 1 | refined |
| `leaf` mode 2 (any body) | +0.132 | 0.000 | +0.105 | | 3 / 1 | refined |
| `leaf` mode 3 (not under a one-column header ending the row above) | +0.135 | 0.000 | +0.108 | 0.98 | 3 / 0 | ledger: kept |
| `dedupe`, any table written 3+ times | -0.613 | -0.480 | -0.585 | | 1 / 22 | rejected: forms print one blank table several times |
| `dedupe`, only on a read `fallback.looped` flags | +0.031 | 0.000 | +0.024 | 0.64 | 1 / 0 | alone under the bar |
| `complete_extent` | +0.009 | 0.000 | +0.007 | | 2 / 0 | alone under the bar |
| `dedupe` + `complete_extent` (vs leaf) | +0.039 | 0.000 | +0.031 | 0.95 | 3 / 0 | ledger: kept |
| `upright` (180° only, first version) | 0 | 0 | 0 | | 0 / 0 | extended to all four turns |
| `upright` (all four turns) | +0.061 | 0.000 | +0.047 | 0.61 | 1 / 0 | not kept (one page, the same page `leaf` fixes) |
| `absorb` | +0.058 | 0.000 | +0.047 | 0.86 | 2 / 0 | not kept (under the bar) |
| `split`, 8 pt gutter | +0.051 | -0.006 | +0.040 | | 4 / 16 | refined: row numbers "(1) 2022" were split off |
| `split`, row numbers excluded | +0.056 | 0.000 | +0.045 | 0.85 | 2 / 0 | not kept (under the bar) |
| `split`, 4 pt gutter | +0.013 | 0.000 | +0.011 | | 2 / 5 | rejected |
| `split`, one cell enough (vs leaf) | +0.062 | 0.000 | +0.050 | | 2 / 0 | the same two pages |
| `unspan` mode 0 (words over one column only) | +0.006 | +0.428 | +0.100 | | 4 / 4 | refined |
| `unspan` mode 2 (word centre inside that column) | +0.000 | +0.455 | +0.100 | 0.49 | 3 / 3 | not kept: the gain is one check document |
| `join`: a label-less row of bracketed values joins the row above (no PDF) | -0.012 | 0.000 | -0.010 | 0.45 | 2 / 5 | rejected: regression tables keep standard errors as rows |
| `prune`: drop columns no text covers (vs leaf) | -0.148 | +0.001 | -0.116 | 0.01 | 10 / 6 | rejected: references keep empty columns |
| `rows`: a free line between matched rows whose numbers occur nowhere in the table becomes a row (vs leaf) | -0.391 | -0.167 | -0.345 | | 1 / 21 | rejected: the line is often another table's row |
| `rows`, only a one-line gap between consecutive rows (vs leaf) | -0.051 | 0.000 | -0.040 | | 1 / 2 | rejected |
| the positive options not kept, together: `absorb`, `split`, `upright`, `unspan` with `dedupe`, `complete_extent` (vs leaf) | +0.154 | +0.455 | +0.223 | 1.00 | 10 / 3 | not run as a bundle (each alone is under the bar) |

`join`, `prune` and `rows` were taken out of the code again; the other options stay behind their keys, off.

## 8. What is kept

`configs/standin_best.yaml` now adds `leaf: true, leaf_mode: 3, dedupe: true, complete_extent: true` to
`table_options`. Tables **88.05 → 88.19 on all pages (+0.14)**, 87.12 → 87.30 on dev (+0.17, P 1.00), 91.71 →
91.71 on check (no check page moved). By subset: `tables` 90.65 → 90.79, `complex_table_headers` 73.91 → 74.23,
`financial_tables` 91.55 → 91.71, `insurance_filing_tables` 89.46 → 89.46, `dense_tables` 86.88 → 86.94. That is
about +0.02 on the overall score. **This falls well short of the target of 90.**

- **`leaf`** (three pages up, none down). Where no page geometry places a table (a scan, or a text layer that did
  not match), a last header row short of its table is moved right so its cells stand over the last columns. Leaf
  headers ("Mean | SD | Mean | SD", "Form | Exhibit | Filing Date") name the value columns, which come after the
  row-label columns, so the cell the reading dropped is the blank corner. It is the blind rule round one rejected
  (38 up, 43 down) narrowed three ways: only the last header row, only where `regrid` could not place the row from
  the page, and not when the row above ends with a one-column header (then the gap may be under it, at the end:
  `dense_table_278`). `complex_header_116` 0 → 0.49, `accenture-2026-08-14_page1` (garbled text layer) 0 → 0.49,
  `Apple_10-k__p59_page1` (a landscape table on a portrait page) 0.53 → 1.00.
- **`dedupe` + `complete_extent`** (three pages up, none down), the two repairs for failed reads (idea 1(a)), judged
  as one change. `dedupe`: on a read that repeats itself (`fallback.looped`), every copy of a table written three
  times or more is dropped after the first, with a last cut-off copy (`blackrock-funds...p25`: 112 copies, 0 →
  0.24). A page that prints one blank table five times (`insurance_filing_tables/87328e997e1c9ca2_p6`) is not a
  looped read and keeps all five. `complete_extent`: when `complete` finishes a table the token limit cut off, only
  the words inside the table's horizontal extent count, so a table printed as column groups side by side is
  completed down its own group instead of stopping at the first line whose other groups do not fit
  (`dense_table_154`: 0.45 → 0.64).

## 9. What was learned

- **OCRia-1's column count is right far more often than the text layer's** (idea 1(b), analysis only). On the 674
  one-table pages with a matched text layer, a column count from the text layer alone (gaps free of words in at
  least 95% of body lines, 4 pt or wider) agrees with the reference on 456, OCRia-1's width on 609. Where the two
  disagree (222 tables), OCRia-1 is right 173 times, the text layer 20 times, neither 29 times. A rule that trusts
  the text layer's count is not worth trying; the 20 are mostly an empty column OCRia-1 added (`dense_table_074`).
- **The upright frame works but finds little to fix.** `upright` turns the words of a rotated page so its lines read
  across (the turn is told by tall word boxes and by the order the PDF writes the words of a line), and round one's
  repairs then match the rows (`Apple_10-k__p59`: 3% of cells found before, 81% after). Only one page changes: on
  most rotated pages the problem is the read, not the grid. The real fix is upstream: render a rotated page upright
  before the model reads it (a new model call, out of scope with stand-ins).
- **Most single-pattern rules reach two or three pages.** `absorb` ("(DKK 1,000)" written as a header cell beside
  the years: `aa2-2026-08-14_page1` 0 → 1.00), `split` (two values on either side of a gutter:
  `alegro-2026-08-13_page1` 0 → 0.89) and `unspan` ("12" spanning MIN/TYP/MAX but printed under MIN:
  `222876fb_page3` +0.70) are clean on the pages they touch but too rare for the bootstrap.
- **Joining and pruning follow the reference's taste.** Bracketed second lines ("40" over "(82)") are one cell in
  some references and their own row in others (regression tables); some references keep empty columns. The
  geometry does not decide either, and both lost.

## 10. What is left

`python -m evalkit losses --run-id t2-failed`: 11.81 points lost. Pages with a text layer lose 9.43 (pages at 0
cost 1.92, under 0.5 3.16, 0.5 to 0.9 3.49), pages without one 2.38.

| Error type | Severity | `t2-base` | `t2-failed` |
|---|---|---:|---:|
| content: cell content | — | 5.80 | 5.67 |
| systematic column error: wrong numbers (3+ in one column) | critical | 1.66 | 1.71 |
| row missing | critical | 0.99 | 0.97 |
| wrong or misplaced number in a row | critical | 0.81 | 0.82 |
| repeated row | major | 0.55 | 0.55 |
| systematic column error: wrong text or units | major | 0.37 | 0.38 |
| extra column | major | 0.35 | 0.36 |
| label reworded (different wording) | major | 0.35 | 0.35 |
| invented row | critical | 0.32 | 0.30 |
| content: row order | — | 0.32 | 0.27 |
| header rows differ | major | 0.27 | 0.27 |

(The event rows are counterfactual and overlap: lifting a page off 0 can raise another label's row.)

Best next ideas, in order:

1. **Read rotated pages upright** (52 pages, 1.24 points, 4.6 times the loss of an upright page). `upright_turn`
   already tells the turn from the text layer. Rendering the page turned before the OCRia-1 call changes the page
   read, so measuring it needs a GPU session. It would likely also fix two of the three looped reads.
2. **Decode garbled text layers.** On the 72 pages with broken font maps the word boxes are right and each glyph code
   stands for one character; OCRia-1's cell text, aligned by position and length, gives the substitution table.
   With it, every repair that needs the text layer would reach those pages.
3. **Finish cut-off reads across column groups.** `dense_table_154`, `dense_table_155` and `dense_table_162` stop in
   the middle of a page printed as side-by-side groups. The groups never read carry the same header as the last
   group, shifted right, so the text layer can supply them. In our own OCRia-1 run 58 of 4,247 answers hit the token
   limit, so this matters more there than on stand-ins.
