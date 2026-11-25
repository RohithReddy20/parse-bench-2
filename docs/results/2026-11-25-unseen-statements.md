# A first check on documents outside the benchmark: 51 pages of financial statements

Date: 2026-11-25. Nine private PDFs (bank, card, mortgage and pension statements; 51 pages, all digital with a
visible text layer). None is in ParseBench and no rule was written with them in view. There is no answer key,
so each PDF's own text layer is the reference. Only counts are recorded here; the documents and the outputs
stay on the owner's machine.

Both outputs come from the same OCRia-1 page reads (51 calls, about $0.30): "plain" is the model's markdown as
it is, "corrected" is `configs/own_best.yaml` without layout.

| Measure against the PDF text layer | Plain OCRia-1 | Corrected |
|---|---:|---:|
| Printed words found in the output | 98.17% | 98.19% |
| Output words that are printed on the page | 98.61% | 98.63% |
| Printed amounts found (`1,234.56`) | 99.91% | 99.91% |
| Output amounts that are printed on the page | 99.91% | 99.91% |
| Printed lines with two or more amounts kept together in one output row | 359 of 362 | 359 of 362 |

What the corrections changed, on 28 of the 51 pages:

| Edit | Count | Right? |
|---|---:|---|
| Bold marks removed | 52 of 329 bold spans | Yes: those words are set in the regular face in the PDF |
| Word respelled from the text layer | 3 | 1 clearly right (a letter read as a digit); 2 swap between two words that are both on the page |
| Table cell span changed | 1 | Not checked |

## What this does and does not show

- **The corrections are safe here.** No amount changed, no row was broken, and the edits that were made agree
  with the PDF.
- **They add almost nothing here.** OCRia-1 alone already reads these statements nearly perfectly, and the
  statements have none of the things the corrections were built for: charts, multi-level table headers,
  pages printed sideways, looped reads, filled-in forms.
- **So this is not evidence that the benchmark gain carries over.** It shows no harm on easy documents. Whether
  the gain is real on other hard documents needs unseen pages with charts, complex tables and forms, and an
  answer key: ParseBench v1 or OmniDocBench are the obvious candidates.
