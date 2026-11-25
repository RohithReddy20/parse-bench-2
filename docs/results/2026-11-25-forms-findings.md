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

## 5. Repairs worth building next, in order

Each needs the same base: an alignment of the markdown's words with the PDF's words, so that every markdown
word is known to be printed form text, overlay (filled) text, or neither.

1. **Check-box states from the overlay marks.** A mark's position names the option it ticks; set `[x]` there
   and `[ ]` elsewhere. Targets box state flipped (critical, 39 pages) and checked state lost (major, 60).
2. **Exact values from the overlay.** Replace a misread value with the overlay string and insert values the
   read dropped. Targets wrong value, value typo, value lost.
3. **Blank slots.** A slot holding only printed text, on a field with no overlay text, should read blank.
   Targets invented value (critical, 83 pages) and printed text in a blank field.
4. **Captions in table headers.** Fields whose caption is split over header cells (`Sex` / `M/F/X`) are not
   found; this is the bulk of "empty field omitted" and needs a decision on reshaping form tables.
