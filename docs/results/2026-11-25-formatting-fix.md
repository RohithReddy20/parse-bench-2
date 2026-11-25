# Formatting: corrections from the PDF itself (strikes, bold, list markers)

Date: 2026-11-25. Stand-in readings (OCRia-1's published markdown), no GPU, no model calls. Subsets `text`,
`headings`, `strikeout`, `underline`; metric `formatting_v2` (slot weights 0.75 / 0.125 / 0.0625 / 0.0625).
Every number below is the harness scorer's, through `evalkit score` and `evalkit paired` (bootstrap over source
documents). "Dev" is the development part, "check" the check slice. Options live in `format_options`
(`src/docparse/fixes/formatting.py`, validated in `Config.load`, all off by default); the kept set is in
`configs/standin_textfmt.yaml`.

## Result

| | Dev | Check | All |
|---|---:|---:|---:|
| Base (`fm-base`: stand-ins + text-layer fallback for looped reads) | 91.78 | 91.23 | 91.66 |
| Kept (`fm-marker`: cell_strike + strike + unbold + list_marker) | 93.10 | 92.63 | 92.99 |
| Kept + text respell (`fm-respell`, the final configuration) | 93.10 | 92.64 | 93.00 |

Formatting +1.34 on all pages (+1.32 dev, +1.41 check); that is +0.22 on the overall score. The target was 93:
reached on the final configuration (93.00), with no margin.

## Variants, one at a time

Gain and P(gain > 0) are against the previous kept run.

| Run | Variant | Dev | Check | All | Dev gain | P | Kept |
|---|---|---:|---:|---:|---:|---:|---|
| `fm-base` | stand-ins + fallback (base) | 91.78 | 91.23 | 91.66 | | | |
| `fm-cell` | `cell_strike`: `~~x~~` inside an HTML table becomes `<s>x</s>` | 92.83 | 92.38 | 92.73 | +1.05 | 1.00 | **yes** |
| `fm-cell-strike` | + `strike`: struck PDF words get `~~` / `<s>` (geometry only) | 92.91 | 92.46 | 92.81 | +0.08 | 0.95 | superseded |
| `fm-cell-strike-ink` | same, the line must also show in the render between the glyphs | 92.94 | 92.46 | 92.83 | +0.11 | 1.00 | **yes** |
| `fm-unbold` | + `unbold`: `**spans**` the PDF sets in a regular face lose the asterisks | 93.02 | 92.63 | 92.93 | +0.08 | 0.97 | **yes** |
| `fm-marker` | + `list_marker`: `**1.** text` at a line start becomes `1. text` | 93.10 | 92.63 | 92.99 | +0.07 | 0.95 | **yes** |
| `fm-dropu` | + `drop_underline`: `<u>` with no drawn line under its words loses the tags | 93.11 | 92.69 | 93.01 | +0.01 | 0.72 | no |

The ink check is part of the kept `strike` option: against `fm-cell` it gains +0.11 (P 1.00) where the
geometry-only version gained +0.08, and against the geometry-only run it removes the only two false strikes
(+0.03, P 0.87): a box border hidden behind a heading's white panel, and a line inside a background drawing.

## What each kept option does, and why it is right for any PDF

- **`cell_strike`** (the largest gain, +1.05). OCRia-1 writes many redline pages as HTML tables and marks
  deletions inside the cells with `~~`. Markdown is not read inside an HTML block, so a renderer shows literal
  tildes, and the scorer (which reads HTML cells the same way) sees no strike at all: the strikeout subset scored
  77.75 with 21 pages at 0. Writing `<s>` in cells (the design's rule "cell styles use HTML tags", §5.4) takes the
  strikeout subset to about 93.5. No PDF evidence is needed; nothing else changes.
- **`strike`** (`src/docparse/pdf/drawings.py`). Word processors draw a strikethrough as a stroked segment or a
  thin filled rectangle. A character is struck when such a line (at most 0.2 character heights thick) covers half
  its width with its centre 0.10–0.40 character heights above the baseline (height = font ascent − descent, from
  PDFium's loose character box). Calibrated on the ground truth with an analysis script: 5,808 struck words hit,
  117 missed, 20 plain words hit. Two refusals keep it precise: a line whose crossed characters span under 60% of
  its length is a rule (a column rule behind an FDA-label heading, a table border), and the line must show as ink
  in the 300-dpi render in the gaps between the glyphs, clearly more often than the same gaps a little higher up.
  The reading's tokens are aligned to the text layer's in reading order (difflib longest matching blocks); a run of
  struck tokens on one line becomes one `~~...~~` (or `<s>...</s>` in a cell), never crossing markup. Text already
  struck, and blocks holding an unpaired `~~`, are left alone.
- **`unbold`**. "Bold added" was the most frequent formatting event (1,105). A bold span in a paragraph or list loses
  its asterisks when at least half of its words align to the text layer and at least 80% of those are in a regular
  face, on a page whose fonts do carry bold (some word's font is bold by its PostScript name, a weight of 600 or
  more, the ForceBold flag, or fill-and-stroke rendering). On a page with no bold face at all the fonts say
  nothing (Type 3 fonts, generic names), so nothing changes. Headings are left alone (the scorer ignores bold in
  headings).
- **`list_marker`**. OCRia-1 sometimes writes numbered answers as `**1.** c`: markdown reads a bold "1." in a
  paragraph, not a list item. The marker is list syntax either way (the scorer's own comment says so, though its
  pattern misses this spelling), so the emphasis is dropped from the marker.

## Measured with analysis scripts only (local harness scoring), not built

Each script applied a candidate rule to the stand-in markdown and re-scored the page with the harness's
`formatting_v2` (and `text_next`), reproducing the ledger's per-page scores exactly. Numbers are text-subset
formatting points (×100 page mean) unless noted.

| Idea | Dev | Check | Verdict |
|---|---:|---:|---|
| Bold removal, every non-bold PDF word counts as regular | +0.02 | +0.17 | built with the page rule instead |
| Bold removal, page carries bold (the kept rule) | +0.10 | +0.22 | built (`unbold`) |
| Bold removal, regular only when the font names its weight | +0.03 | +0.25 | no (dev) |
| Adding bold where the PDF font is bold and the reading is not | — | — | not tried: the reference leaves 4,518 PDF-bold tokens plain against 117 it bolds |
| Italic removal, same rule as `unbold` | −0.01 | 0.00 | no |
| Underline removal (`drop_underline`, then built and rejected above) | underline subset +0.07, text +0.06 | | no |
| Demoting headings the PDF prints at body size in a regular face | −0.10 (headings subset −1.09) | −0.08 | no: such headings are real headings in the reference |
| Dropping the reading's strikes where no drawn line crosses the words (pages that strike with lines) | strikeout −0.17 | strikeout +0.50 | no: 8 pages up, 8 down |
| Inline code from monospaced fonts (glyph advances constant) | −0.02 | 0.00 | no (see below) |
| `**`/`*` inside table cells as `<b>`/`<i>` | 0 | 0 | no effect on these subsets (left out) |

## What is left in formatting (all pages, dimension points)

`fm-base` against the final run `fm-respell` (`evalkit` loss rows, subsets merged).

| Loss | Base | Final | Change |
|---|---:|---:|---:|
| content: inline styles (bold, italic, ...) | 2.39 | 1.78 | −0.60 |
| content: headings | 1.56 | 1.57 | +0.01 |
| severity cap | 1.24 | 1.13 | −0.11 |
| content: under half of the items found (coverage cap) | 1.03 | 1.03 | 0.00 |
| content: strikeout | 0.85 | 0.28 | −0.57 |
| content: underline | 0.68 | 0.68 | 0.00 |
| content: lists | 0.59 | 0.52 | −0.07 |

- **Headings** (1.57) are now the largest block: false headings ("heading not in the page's heading list", "text
  promoted to heading") and level relations. Font size and weight do not separate them (the demotion test lost).
- **Inline styles** (1.78): what remains is mostly bold and italic the reference marks or leaves out regardless
  of the fonts, and inline code: four pages (`caldera`, `2colpa`, `overlapping`, `paper`) lose about 130 "inline
  code lost" events and sit at the 0.85 minor cap. A monospaced font is easy to detect from the PDF (the advance
  between glyph origins is constant: share at the median 0.95–1.00 for t1xtt, NimbusMonL, CMTT10, Courier; at
  most 0.67 for proportional fonts). A first prototype (backticks around runs of monospaced words, `<code>` in
  cells, on pages with at least 30% proportional text) scored −0.01 on 8 pages: it did not reach the three pages
  that lose the most (their code sits in table cells or in lines the alignment did not pair), and added code
  spans the reference leaves plain elsewhere. It needs that placement worked out, and the forms dimension
  measured with it (form fill values are often set in Courier), before it is worth an evaluation run.
- **Coverage cap** (1.03): pages where the reading lacks the text of most formatted items (scans, failed reads).
- **Underline** (0.68): OCRia-1 already scores 96.9 on the underline subset; the PDF lines under words are
  ambiguous (link underlines, rules under form captions), and removal did not pass.
