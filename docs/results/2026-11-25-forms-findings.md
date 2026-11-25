# Forms: what the data and the scorer look like (groundwork for the forms fix)

Date: 2026-11-25. No GPU. Numbers come from OCRia-1's published readings (ledger run
`published-ocria_1-baseline`) and from reading the harness scorer `forms_v2.py` at the pinned commit.

## 1. The data

| Subset | Pages | Weight in the forms dimension | OCRia-1 | What the PDFs are |
|---|---:|---:|---:|---|
| `forms` | 500 | 87.5% | 80.7 | 450 filled + 50 blank synthetic government forms from 20+ countries. No AcroForm, no annotations. All have a text layer. |
| `fillable_forms` | 100 | 12.5% | 92.6 | Real AcroForm PDFs with widgets. Nearly all fields are empty (17 filled text values in 3,458 fields). |

- **The plan's premise was partly wrong.** "Read AcroForm values" only applies to `fillable_forms`, which is
  12.5% of the dimension, already scores 92.6 and has almost no values to read. The loss is in `forms`.
- **Every filled value is in the text layer.** In the 450 filled pages, 4,474 of 4,474 reference text values
  (3 characters or longer) occur in the PDF text layer. OCRia-1's markdown has 4,313 of them (96.4%).
- **The fill is a separate overlay.** In all 450 filled pages the typed values and the check marks are the
  last content stream of the page, drawn after a closing `Q`, in fonts used nowhere else on the page
  (for example `CourierPrime-Regular-3680` for values and `ArialUnicodeMS-7198` for `✓`). Check marks are
  text glyphs (`✓` or `X`) with positions. A font-only rule is not enough: 14 of 50 blank pages also end in a
  run of text in its own font; requiring the closing `Q` stream before it leaves no false hit in `forms`.
- Fields in `forms`: 5,102 filled text, 6,718 empty text, 1,220 checked boxes, 3,452 unchecked boxes,
  131 signed and 105 unsigned signature lines.

## 2. How the scorer reads a page (`forms_v2.py`)

- It gets the whole markdown. Tables (pipe, HTML, bare `a | b | c` rows) become grid cells; every other line is
  one cell. Bold, tags and links are stripped, so `K: v`, `**K:** v` and `| K | v |` are equivalent.
- **Caption:** found as a phrase with at least 60% of its token weight in order (short words weigh 0.25).
  A caption split over two table header cells, or printed without its number, is often not found.
- **Value:** read only from the caption's own slot: the rest of its line or cell, else the next cell to the
  right, else the cell below, else the next line plus two continuation lines. Reading stops at the next
  located caption or any box token. Values compare with spaces and punctuation removed.
- **Blank:** nothing, `___`, `N/A`, `none` and similar. `0`, `X`, `No`, `unknown` are not blank.
- **Boxes:** checked is `[x]`, `[✓]`, `[checked]`, `☑`, `☒` and similar; unchecked is `[ ]`, `☐`, `□`. The box
  must be within 4 tokens of its option label, and all options on one line must use the same side.
- **Score** = min(proportional, floor). Proportional = 1 − Σ loss / (fields + 3), with loss 1.0, 0.5 or 0.2
  for a critical, major or minor event. Floor: the first critical event caps the page at 0.90, the first
  major at 0.95, each minor costs 0.01 (0.15 at most). Under 50% of fields found caps the page at 2 × coverage.
- The ideal page is one field per line: `- **12a Wages**: 1,234.50`, `- **Middle name**:`,
  `- [x] I agree`, `- [ ] Send me email`, `- **Signature**: [signed]`.

## 3. Where OCRia-1 loses (500 `forms` pages)

Lost in total: 97 page units, 56 through the proportional part and 41 through the caps.

| Event | Severity | Events | Pages | What it means |
|---|---|---:|---:|---|
| empty field omitted | minor | 2,648 | 226 | The caption of a blank field is not found in the output |
| caption garbled | minor | 1,031 | 191 | Caption found but under 85% match |
| unchecked option omitted / state lost | minor | 1,314 | 121 | Option label or its `[ ]` is missing |
| value without its caption | major | 641 | 129 | Value is on the page, its caption is not found |
| printed text in a blank field | minor | 461 | 128 | A blank field's slot holds printed form text |
| value lost | major | 322 | 139 | Caption found, slot reads blank |
| extra text in value | minor | 283 | 151 | Slot holds the value plus more text |
| wrong value | critical | 254 | 122 | Slot holds something else |
| invented value | critical | 221 | 83 | A blank field's slot holds a value |
| checked state lost / option missing | major | 227 | 60 | Checked box has no readable state |
| field missing | major | 107 | 36 | Neither caption nor value found |
| box state flipped | critical | 99 | 39 | Checked read as unchecked or the reverse |
| contradicting copy / value of another field | critical | 90 | 46 | |

- 38 pages have under 60% of their fields found and lose 19.3 of the 97 page units. Six of them are loops
  (about 69,000 characters of repeated empty cells or blanks).
- 404 pages hold 95% or more of the PDF's words and still average 0.82: most of the loss is structure
  (which value sits in which caption's slot), not misread text.

## 4. First repair measured: text-layer fallback for looped reads

A read that repeats itself (compresses 30 times or better, or hits the token limit) and holds under half of
the PDF text layer's words is replaced by the text layer written as plain lines (`fallback_below: 0.5`,
`src/docparse/fixes/fallback.py`). It applies to any page, not only forms.

| Variant | Forms dev | Forms check | Forms all | Pages changed |
|---|---:|---:|---:|---:|
| Stand-in readings | 82.10 | 82.10 | 82.10 | |
| Fallback whenever under half of the PDF words are read | 82.08 | 83.01 | 82.31 | 9 (3 got worse) |
| Fallback only for looped reads | 82.18 | 83.06 | 82.39 | 6 (largest gains +0.64, +0.37, +0.35, +0.30) |

The ungated version hurts pages where OCRia-1 deliberately skipped fine print, so the loop test stays.

On the whole benchmark (run `all-fbloop` against `m0-roundtrip`, stand-in readings) the gated fallback changes
8 pages: overall 85.23 → 85.32, text 87.87 → 88.06 (one page from 0 to 0.94), formatting 91.59 → 91.66,
forms 82.10 → 82.39; tables, charts and layout unchanged. One page loses 0.01, none loses more. The pages are
too few for the bootstrap to call it (P = 0.6 on the development part), so it is kept as a safety net, not
counted as a tuned gain. It is on in `configs/standin_layout.yaml`.

## 5. Why captions are not found

Unfound captions are behind "empty field omitted", "value without its caption", "field missing" and the
omitted options: about 4,300 events. Two causes, seen by reading pages next to their events:

- **Printed sub-labels the read leaves out.** A phone field prints `COUNTRY CODE`, `AREA CODE` and `NUMBER`
  in small type inside its boxes; OCRia-1 writes `**Office hours**: ( 61 ) ( 03 ) 9608 2035`. The three
  values are right and all three fields score "value without its caption". Date boxes (`Day`, `Month`,
  `YYYY`) and address lines behave the same way. These labels are in the PDF text layer.
- **Captions split over table header cells.** `Sex` in one header row and `M/F/X` in the next never form the
  phrase `Sex (M/F/X)`.

A word-level check of the unfound captions against the markdown overstates how many are "present" (it
matches `NUMBER` inside `Client ID number`), so the split between the two causes still needs the alignment
described below to count properly.

## 6. Overlay detection (built)

`docparse.pdf.overlay.overlay_fonts(pdf_bytes)` returns the fonts of the overlay stream, and every text-layer
word now carries its font, so a word is overlay text when its font is in that set.

| Check | Result |
|---|---|
| Filled `forms` pages with an overlay found | 450 of 450 |
| Blank `forms` pages with an overlay found | 0 of 50 |
| Reference text values present in the overlay words | 4,510 of 4,513 |
| Tick glyphs in the overlay (`✓`, `X`, ...) against checked reference boxes | 1,319 against 1,220 |
| Pages with an overlay found in 3,391 pages of eight other subsets | 3 |

## 7. Repairs worth building next, in order

Each needs the same base: an alignment of the markdown's words with the PDF's words, so that every markdown
word is known to be printed form text, overlay (filled) text, or neither.

1. **Check-box states from the overlay marks.** A mark's position names the option it ticks; set `[x]` there
   and `[ ]` elsewhere. Targets box state flipped (critical, 39 pages) and checked state lost (major, 60).
2. **Exact values from the overlay.** Replace a misread value with the overlay string and insert values the
   read dropped. Targets wrong value, value typo, value lost.
3. **Blank slots.** A slot holding only printed text, on a field with no overlay text, should read blank.
   Targets invented value (critical, 83 pages) and printed text in a blank field.
4. **Printed labels the read left out.** A printed label that no markdown word aligns to, sitting directly
   beside or above a filled value or an empty box, is written in front of that value (`COUNTRY CODE: 61`).
   Targets value without its caption (major, 129 pages) and empty field omitted (226 pages). Probably the
   largest of the four.
5. **Captions in table headers.** Join a two-row header into one caption per column on form pages.

Not a repair: adding PDF text the read lacks to text pages. On the 378 text pages with a text layer the
read misses 2,696 expected words, 1,037 of them in the PDF, but the PDF holds 24,821 tokens the read
lacks and only 1,871 of those are expected. Blanket insertion would add far more wrong text than right.
