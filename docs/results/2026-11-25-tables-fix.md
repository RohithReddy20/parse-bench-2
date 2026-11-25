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
LEDGER_MORE

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
LOCAL_MORE

## 4. What is kept

KEPT_LINE

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
MERGE_EXTEND

## 5. What is left

LEFT_SECTION
