# Forms: the second round-two line ("Neon") brought in, reviewed and measured

Date: 2026-11-25. No GPU, no model calls: OCRia-1's published readings (stand-ins), `forms,fillable_forms` (600
pages), harness scorer through `evalkit run/score/paired`. "Dev" is the development part, "check" the check slice; P
is the paired bootstrap probability of a gain over documents on the development part.

Two engineers worked on forms from the same starting point (83.01 on stand-ins). The main line (rounds two and three,
`2026-11-25-forms-fix.md`) reached 84.47 with `ids`, `codes`, `outlines`, `trailing`, `blanks`, `cells`,
`blank_forms`. The other line ("Neon", a separate copy of the repository, one commit `bc3a5ff`) reached 86.45 with
15 new options. This note brings Neon's code in, reviews each of its rules for whether it would be right on a form the
benchmark has never seen, and picks the configuration.

## 1. What was merged

- Neon's modules copied in unchanged: `forms_tables.py`, `forms_lines.py`, `forms_combs.py`, `forms_wave2.py`,
  `forms_wave3.py`, `forms_wave4.py`, `forms_clean.py`, `forms_acro.py`, `forms_blank.py`. Its `forms_grids.py` and
  `forms_slots.py` collide with the main line's modules of those names and are `forms_n_grids.py` and
  `forms_n_slots.py` here; its option `blank_forms` is `n_blank_forms` (`forms_blank.py`). Its tests came along
  (`test_forms_n_grids.py`, `test_forms_n_slots.py`, the rest under their own names); 429 tests pass, ruff clean.
- All 15 options are off by default. In `forms.repair` Neon's steps run in its own order after the first batch of
  edits (`combs`, `boxes`, `bare_ticks`, `outlines`, `trailing`) and before `ids`/`codes` and the slot repairs;
  `acroform`/`acro_clean` run first on any page, `n_blank_forms` before `blank_forms` on pages without an overlay.
  The other order (main line's `ids`/`codes` first) was measured on the chosen set: dev +0.10, check -0.06
  (`ni-general-o2` against `ni-general`), so it was not taken.
- Both lines reproduce exactly: `ni-main` scores 84.47 with every page equal to `f3-final2`, and `ni-neon` (Neon's 15
  on, the main line's round-two/three options off) scores 86.45 with every page equal to Neon's `nn-final`.
- One bug fixed while merging: `acroform_widgets` opened every input with PDFium outside the PDFium lock and crashed the
  whole run on image inputs in the other subsets (`Data format error`). Non-PDF or unreadable bytes now have no widgets
  (test added). Neon never ran its code on the other subsets with this pipeline.
- Nothing in the new code reads ground truth, file names or example ids (inputs are the markdown, the rendered text
  words, the PDF bytes); `tests/guards` and `tests/test_import_rules.py` pass.
- Byte identity, all 15 options on: across the 3,717 pages of the other 11 subsets, 3,717 markdowns are identical
  (`ni-oth-off` against `ni-oth-on2`). One `layout` page has layout boxes up to 1.4 points narrower with identical
  markdown: PDFium measures text a little differently on the first page a worker process renders (checked: the same
  PDF rendered twice in one process gives different word boxes the first time), so this is which worker got the page,
  not a form edit. In the form subsets, with the chosen set on, all 50 blank `forms` pages (no overlay, no widgets)
  are identical; 91 filled pages and 20 `fillable_forms` pages (AcroForm widgets) change. With all 15 on, 6 blank
  pages change: `n_blank_forms` is gated on the markdown alone, not on anything in the PDF.

## 2. Review of the 15 rules (from the code)

(a) corrects something the page shows that the read got wrong, judged from PDF evidence; (b) reshapes correct content
into the form this scorer reads, no PDF evidence needed; (c) keyed to particular wording or one family of forms. An
option that bundles several rules gets the class of its weakest part that changes output; where an (a) part is
bundled with (b) parts it is written "a+b" and judged by the (b) test in section 3.

| Option | Class | What it does | Deciding code |
|---|---|---|---|
| `mini_tables` | b (one hard-coded detail) | a 1- or 2-row table right under a caption line becomes `caption: value` on that line; the overlay only confirms which cells are typed | `edits.append((line_start, e, f"{new_prev}{sep}{val_str}"))`; second-row hints only `_HINT_SINGLE = re.compile(r"^[DMYJAdmyja]{1,2}$")` (English and French date-box letters: nearly general but for this) |
| `table_rows` | b | caption, box-code and value cells merged into one `code caption: value` cell; a 2-row caption-over-value table becomes one row; printed `1 ,00` cents-placeholder cells and template cells repeated 3+ times emptied | `inner=f"{html.escape(prefix + cap_text)}{sep}{html.escape(val_text)}"`, `_CENTS_ONLY`, `_PURE_CODE_CENTS` -> `cell.inner = ""`; the `,00` placeholders are a convention of some countries' forms, but the rule only deletes printed decoration |
| `bare_ticks` | **a** | a bare `X`/`✓` the read wrote as text becomes `[x]` only when an overlay tick glyph sits beside the label on the PDF; `[ ]` added beside sibling labels only where an empty drawn box of the same size is | `if _same_line(w0, tk) and tk.cx <= w0.x0 + 2.0` over `page.ticks`; `if any(_inside((tk.cx, tk.cy), b) for tk in page.ticks): continue` over `drawn`. Defect: its mark regex does not know the `[checked X]` / `[unchecked]` spelling, so on such reads it writes `[checked [x]]` and `[ ] [unchecked]` (seen on the three German `GewA` pages, score-neutral there) |
| `field_lines` | c | on `Caption: value` lines a box code moves into the caption, units and short hints after a blank marker are deleted, bracketed examples deleted | `_UNIT_TOK = ... years?\|months?\|days?\|hours?\|hrs? ...` (English time-unit words) and currency codes; `_EXAMPLE_BEFORE_BLANK` deletes printed text |
| `box_grids` | b | check-box cells of a grid get the column caption (`[x] Fair`); `[ ] Label [x]` becomes `[x] Label`; a newline before inline options after `Caption:` | `box_grid_edits(markdown)`: reads the markdown only, no PDF evidence; the `[ ] Label [x]` rewrite is a guess about which box is real |
| `table_clean` | a+b | 4+-column check-box strips decided from drawn boxes and overlay ticks (a); a printed second line (`<br/>` + other script) deleted from a typed cell; `Caption \| value \| code` and row-id columns merged (b) | `is_ticked = any(_inside((tk.cx, tk.cy), box) for tk in all_ticks)`; `c1.inner = c1.inner[: bm.start()].strip()` deletes printed text |
| `acroform` | a+b | `[ ]`/`[x]` from AcroForm check-box and radio widgets beside the label (a); row-id cells joined to captions, markdown only (b) | `checked = bool(pdfium_c.FPDFAnnot_IsChecked(fh, ann))`, `mark = "[x] " if chk else "[ ] "`; gate `if n_int < 3: return markdown` |
| `field_notes` | c | bracketed examples stripped from cells, `Caption (note) \| value` merged, trailing tax codes moved | `_PAREN_EX`: `z\.\s*B\.`, `e\.?\s*g\.?`, `for\s+example`, `leave\s+blank`, `Beispiel`; `_TRAIL_BOX_CODE`: `[0-9][A-Z]{2}` (French 2042 box codes such as `4BE`) |
| `field_slots` | c | 2-row caption-over-value tables (not signature blocks); `1 = Leave, 2 = Switch` legends; phone blocks; date-format hints deleted | `_PHONE_BLANK_HDR`: literal `Country code ... Area code ... Number`; `_SIG_WORD`: `signature\|sign\|unterschrift\|firma`; `_DATE_HINT`: `YYYY\|MM\|DD\|DAY\|MONTH\|YEAR` |
| `comb_noise` | b | `[1][5][6]` joined when the tokens spell an overlay fill (overlay-checked reshape, like `combs`); `1__1__1__` divider noise -> `[___]` | `edits.append((s0, e1, fill.text))`; `_NOISE_EMPTY_COMB = re.compile(r"(?:1_{1,4}){3,}1?_?")` (how this reader writes empty combs) |
| `table_grids` | c | bold box-number grids -> `code row — column: value`; `(1) (2) (3)` header rows merged; example lines deleted; `d 2 d 7 m 0 m 9` -> `27 09` checked against fills | `_BR_EXAMPLE`: `for\s+example\|e\.?\s*g\.?\|z\.\s*B\.\|Beispiel\s*:\|par\s+exemple`; `_DMY_WATERMARK`: `[dmyjtaDMYJTA]` (date letters of particular languages); `_UNIT_TOK` with `zł\|gr\|元`; keyed to the read's `<strong>3.30</strong>` box numbers |
| `acro_clean` | c | `1) [___]` -> `[___]`; empty Yes/No columns filled `[ ] Yes`; notes after a blank split; SSN mask stripped | `_ACRO_TRAIL_INSTR`: `See\s+(?:separate\s+)?instructions` (US IRS); `_ACRO_SSN_MASK`: `X{2,3}-X{2}-` (US SSN); `c.text.strip().lower() in ("yes", "no")` |
| `field_clean` | c | underlined / piped comb noise joined when it spells overlay fills; equation rows merged; date-unit headers split off | `_DATE_UNITS_ANY`: `Tag\|Day\|Jour`, `Monat\|Month\|Mois`, `Jahr\|Year\|Année` |
| `blank_hints` | c | printed hints deleted from blank slots | `read the notes` (UK HMRC), `See instructions` (US IRS), `ovvero` (Italian), `(999) 999-9999` masks, `,00` / `,—` cells -> `[___]` |
| `n_blank_forms` | c | blank-form tables with bold box numbers or bold codes -> `code caption: [___]`; dense blank pages split, units deleted | `repair_blank_form(markdown)`: markdown only; `zł\|zł,\s*gr` (Polish forms), `Years\|Months\|Days` |

Only `bare_ticks` is pure (a); `table_clean` and `acroform` carry an (a) part. Seven of the 15 are (c). The rules Neon
added after `nn-v9-tableclean` (dev 85.10 -> 86.89, check 84.81 -> 85.04) are, apart from `acroform`, all (c) or
the markdown-only `comb_noise`.

## 3. Measurements

### 3.1 The three starting configurations

| Run | Configuration | Dev | Check | All |
|---|---|---:|---:|---:|
| `ni-main` | `standin_best.yaml` as committed (main line) | 84.36 | 84.81 | 84.47 |
| `ni-neon` | Neon's 15 on, `ids codes outlines trailing blanks cells blank_forms` off | 86.89 | 85.04 | 86.45 |
| `ni-both` | everything on | 87.33 | 85.51 | 86.90 |

`ni-neon` against `ni-main`: dev +2.52 (P 1.00), check +0.23 (P 0.77). `ni-both` against `ni-neon`: dev +0.44
(P 1.00), check +0.47.

Pages where `ni-both` is below the better of `ni-main` and `ni-neon`: 19, together -0.88 page points. The largest:

| Page | ni-main | ni-neon | ni-both | Why |
|---|---:|---:|---:|---|
| `germany--17_GewA3 p1` | 0.890 | 0.708 | 0.708 | Neon's 2-row caption-over-value rule takes the line number as a caption (`<td>4: [___]</td><td>Name: Klimczak</td>`) before `ids` can join it to `Name` |
| `germany--31_GewA2 p1` | 0.863 | 0.712 | 0.712 | same |
| `france--24_2086 p1` | 0.685 | 0.585 | 0.585 | Neon's table merges |
| `germany--05_GewA1 p1` | 0.792 | 0.708 | 0.695 | same as GewA3 |
| `us--13_1040-X p2` | 0.779 | 0.717 | 0.717 | |
| `us--41_I-864 p5` | 0.987 | 0.878 | 0.930 | |
| `rrf1_form p1` (fillable) | 0.701 | 0.890 | 0.851 | |

### 3.2 Neon's rules added to `ni-main` one at a time, in the order Neon kept them

Keep rule of the brief: dev gain with P >= 0.9 against the previous kept configuration, and no check loss. Each row is
measured against the "base" column.

| Run | Added | Class | Base | Dev | Check | All | Dev gain | P | Check gain | Kept |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `ni-i01-minitables` | `mini_tables` | b | ni-main | 84.49 | 84.82 | 84.57 | +0.12 | 1.00 | +0.01 | yes |
| `ni-i02-tablerows` | `table_rows` | b | i01 | 84.91 | 84.96 | 84.92 | +0.42 | 0.98 | +0.15 | yes |
| `ni-i03-bareticks` | `bare_ticks` | a | i02 | 85.18 | 85.03 | 85.14 | +0.27 | 1.00 | +0.06 | yes |
| `ni-i04-fieldlines` | `field_lines` | c | i03 | 85.29 | 85.03 | 85.23 | +0.11 | 1.00 | +0.00 | yes |
| `ni-i05-boxgrids` | `box_grids` | b | i04 | 85.74 | 85.31 | 85.64 | +0.45 | 0.99 | +0.28 | yes |
| `ni-i06-tableclean` | `table_clean` | a+b | i05 | 85.94 | 85.28 | 85.79 | +0.20 | 1.00 | -0.03 | **no** (check loses: `france--26_2044_SPE`, `france--24_2086`, `cerfa_10334` -0.05 to -0.10) |
| `ni-i07-acroform` | `acroform` | a+b | i05 | 85.95 | 85.44 | 85.83 | +0.21 | 1.00 | +0.12 | yes |
| `ni-i08-fieldnotes` | `field_notes` | c | i07 | 86.04 | 85.44 | 85.90 | +0.09 | 0.98 | +0.00 | yes |
| `ni-i09-fieldslots` | `field_slots` | c | i08 | 86.14 | 85.53 | 86.00 | +0.11 | 1.00 | +0.10 | yes |
| `ni-i10-combnoise` | `comb_noise` | b | i09 | 86.23 | 85.53 | 86.07 | +0.09 | 0.95 | +0.00 | yes |
| `ni-i11-tablegrids` | `table_grids` | c | i10 | 86.74 | 85.53 | 86.46 | +0.51 | 0.98 | +0.00 | yes |
| `ni-i12-acroclean` | `acro_clean` | c | i11 | 86.83 | 85.53 | 86.52 | +0.08 | 0.98 | +0.00 | yes |
| `ni-i13-fieldclean` | `field_clean` | c | i12 | 86.99 | 85.53 | 86.65 | +0.16 | 0.98 | +0.00 | yes |
| `ni-i14-blankhints` | `blank_hints` | c | i13 | 87.06 | 85.53 | 86.70 | +0.07 | 0.97 | +0.00 | yes |
| `ni-i15-nblankforms` | `n_blank_forms` | c | i14 | 87.11 | 85.53 | 86.74 | +0.05 | 0.87 | +0.00 | **no** (P < 0.9) |

The best combination by the keep rule is `ni-i14-blankhints` (13 of the 15 on): 86.70, dev 87.06, check 85.53.
From `ni-i09` on, five rules add +0.92 on dev and exactly nothing on the check slice; all five are (c) or
markdown-only. That is what fitting looks like: the rules match the wording of documents in the development part
and find nothing to match in the check slice's documents. (`field_slots`, also (c), did move the check slice, +0.10:
the phone-block and date-hint wording happens to occur there too.)

### 3.3 The general set

`ni-general` = `ni-main` + the (a) rule (`bare_ticks`) + the (a+b) and (b) rules whose check gain in 3.2 is positive and
at least a third of their dev gain: `table_rows` (+0.15 of +0.42), `box_grids` (+0.28 of +0.45), `acroform` (+0.12 of
+0.21). Not in: `mini_tables` (+0.01 of +0.12), `table_clean` (check -0.03), `comb_noise` (+0.00), and every (c) rule.

| Run | Configuration | Dev | Check | All | vs ni-main: dev | P | check |
|---|---|---:|---:|---:|---:|---:|---:|
| `ni-main` | main line | 84.36 | 84.81 | 84.47 | | | |
| `ni-aonly` | + `bare_ticks` | 84.63 | 84.87 | 84.69 | +0.27 | 1.00 | +0.06 |
| **`ni-general`** | + `bare_ticks`, `table_rows`, `box_grids`, `acroform` | **85.71** | **85.43** | **85.65** | +1.35 | 1.00 | +0.62 (P 1.00) |
| `ni-general-o2` | same, main line's `ids`/`codes` before Neon's steps | 85.82 | 85.36 | 85.71 | +1.45 | 1.00 | +0.55 |
| `ni-i14-blankhints` | best keep-rule combination (13 of Neon's 15) | 87.06 | 85.53 | 86.70 | +2.70 | 1.00 | +0.72 |
| `ni-both` | everything | 87.33 | 85.51 | 86.90 | +2.97 | 1.00 | +0.70 |

`ni-general` against `ni-both`: dev -1.62, check -0.08 (P 0.36). Against `ni-neon`: dev -1.18, check +0.39 (P 0.96).
`bare_ticks` alone changes 9 pages, all for the better. `ni-general` changes 60 pages against `ni-main`, 52 up; the
losers are the three German `GewA` pages (-0.18, -0.15, -0.10: the caption-over-value rule of `table_rows` taking the
line number for a caption), `us--13_1040-X p2` (-0.06) and four small ones.

## 4. Recommendation

**Make `ni-general` the project's forms configuration**: the main line plus `bare_ticks`, `table_rows`, `box_grids` and
`acroform`. Forms 85.65 (dev 85.71, check 85.43), against 84.47 for the main line alone. It is now in
`configs/standin_best.yaml` and, with the same form keys, in `configs/own_best.yaml` (not run on own readings).

It gives up 1.05 points on all pages against the best keep-rule combination (86.70) and 1.25 against everything on
(86.90), but only 0.10 and 0.08 on the check slice: what is given up is almost entirely development-part gain from
rules that name particular words (`See instructions`, `read the notes`, `ovvero`, `Country code / Area code / Number`,
`Tag Monat Jahr`, `zł`, French `4BE` codes). On a form the benchmark has never seen those rules either do nothing or,
where the wording happens to match, delete printed text the page really shows. The 10 other options stay in the code,
off.

Two known weaknesses of the chosen set, not fixed here (no new rules in this round): `table_rows`' caption-over-value
rule should not take a bare line number as a caption (the `GewA` pages), and `bare_ticks` should recognise the
`[checked X]` / `[unchecked]` spelling of a box before adding marks.

### How much of the forms gain is real correction

Over plain OCRia-1 (82.10) the chosen configuration gains +3.55 on all pages. By the steps that produced it (each
from its own ledger runs, all-pages column):

| Step | Gain | Judged |
|---|---:|---|
| looped-read fallback to the text layer | +0.29 | a: the read was looped or cut off; the PDF's text replaces it |
| `combs` (values typed one per box written as one string, from the overlay) | +0.33 | a, borderline b: the value is the overlay's string, but the read had its characters |
| `boxes` (check states from overlay ticks and drawn boxes) | +0.28 | a |
| `outlines` (character-box outlines read as empty check boxes dropped) | about +0.10 | a |
| `ids`, `codes`, `trailing` (line numbers and box codes joined to captions, trailing text moved) | about +0.56 | b: built to the scorer's caption chain |
| `blanks`, `cells`, `blank_forms` (printed text out of blank slots, blank markers) | +0.80 | b: built to how the scorer forgives printed text |
| `bare_ticks` | +0.22 | a |
| `acroform` | part of +0.96 | a+b |
| `table_rows`, `box_grids` | rest of +0.96 | b |

My judgement: about **1.3 points (roughly 35 to 40%) of the 3.55 is real correction** of what the read got wrong
(fallback, typed values, check states from ticks and widgets, box outlines), and about **2.2 points (60 to 65%) is
reshaping content the read already had right into the layout this scorer reads** (`code caption: value`, blank
markers, captions moved). Those (b) rules are not tied to particular documents (their gains hold on the check
slice) but they would be worth nothing to a reader that does not score forms this way. Neon's (c) rules, which the
chosen set leaves out, would add another 1.05 points on all pages, 1.35 on dev and 0.10 on the check slice: that part
I judge to be fitting this benchmark, and it is not counted above.

## 5. Runs

All with `configs/standin_best.yaml` and `--set` overrides, `--subsets forms,fillable_forms`, `--workers 6`:
`ni-main`, `ni-neon`, `ni-both`, `ni-i01` to `ni-i15`, `ni-aonly`, `ni-general`, `ni-general-o2`, and `ni-final`
(the committed `standin_best.yaml`). Byte-identity runs (no scoring): `ni-oth-off`, `ni-oth-on2` on the other 11
subsets (`ni-oth-on` died on the PDFium bug of section 1).
