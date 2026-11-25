# Do the corrections help outside ParseBench 2? Two public benchmarks

Date: 2026-11-25. The same two outputs are compared on documents the corrections were never tuned on:
"plain" is OCRia-1's markdown as it is (`configs/ocria_own.yaml`), "corrected" is OCRia-1 plus every kept
correction (`configs/own_best.yaml`), both without layout. No rule was changed for these tests and no failure
was studied.

## 1. Real forms: RealDocBench (Extend, 2026)

581 one-page forms (mortgage 222, finance 172, supply chain 113, medical 74), 1,356 questions with typed gold
answers; data `Extend-AI/RealDoc-Bench` at `08dde9d2`. The benchmark's own scorer sends the markdown to a paid
Gemini model before comparing answers, so it was not run. The measure used instead is deterministic and the
same for both outputs: is each gold field value present in the document's markdown (numbers compared by
value, dates by date; "loose" also forgives spacing and punctuation). Check-box states, choices and empty
fields cannot be judged by presence and are not scored, which leaves 1,039 questions and 2,686 fields.
572 OCRia-1 page reads, about 35 minutes on an A100-40GB.

| Measure | Plain OCRia-1 | Corrected | Change |
|---|---:|---:|---:|
| Field values found | 88.46 | 88.57 | +0.11 |
| Field values found, loose | 91.21 | 91.33 | +0.12 |
| Questions with every field found | 79.98 | 80.17 | +0.19 |
| The same on the 174 pages with a text layer (590 fields) | 84.92 | 85.42 | +0.50 |
| The same on the 406 scanned pages (2,096 fields) | 89.46 | 89.46 | +0.00 |

- The corrected markdown differs on 125 of 581 pages. Three fields are found only by the corrected output and
  none only by the plain one; the bootstrap puts the chance of a real gain at 0.63. In plain words: no harm,
  and no measurable benefit.
- **The form corrections tuned on ParseBench do not reach real forms.** They key on typed-in text stored as an
  extra content stream, which is how ParseBench's synthetic forms were filled. That pattern is on 0 of these
  581 pages; 406 pages are scans with no text layer at all.
- What this cannot show: check-box repairs (those answers are not scored here).

## 2. olmOCR-bench (Allen AI)

1,403 one-page PDFs with 7,019 pass/fail tests (8,413 with the scorer's own "page is not empty or looping"
tests); data `allenai/olmOCR-bench` at `54a96a6f`; scorer `olmocr==0.4.27`, run unchanged. 1,220 of the PDFs
have a text layer. 1,402 OCRia-1 page reads, about 65 minutes on an A100-40GB.

| Category | Tests | Plain OCRia-1 | Corrected | Change in tests passed |
|---|---:|---:|---:|---:|
| **Overall** (mean of categories) | | **76.05** | **75.99** | |
| arXiv maths | 2,927 | 85.14 | 84.97 | -5 |
| Headers and footers left out | 760 | 24.61 | 24.21 | -3 |
| Long tiny text | 442 | 93.44 | 93.44 | 0 |
| Multi-column reading order | 884 | 81.45 | 81.56 | +1 |
| Old scans | 526 | 43.16 | 43.16 | 0 |
| Old scans, maths | 458 | 88.65 | 88.65 | 0 |
| Tables | 1,022 | 92.17 | 91.98 | -2 |
| Page not empty or looping | 1,394 | 99.78 | 99.93 | +2 |

- The scorer's own interval is ±0.9 for each output, so the two are the same. The corrected markdown differs
  on 258 of 1,403 pages; 6 tests are failed only by the plain output and 13 only by the corrected one.
- **The corrections do not help here, and are very slightly harmful.** Tables is the category where a gain
  was most expected, and it loses two tests.
- Both outputs score low on headers and footers because OCRia-1 keeps that text by design and this benchmark
  wants it left out; that lowers both equally.
- Not covered by this benchmark: charts, forms, layout boxes and pages printed sideways, which is where the
  largest ParseBench gains are (charts +3.2, layout +4.3). Whether those carry over is still untested.

## What the two tests say together

On 1,984 pages from two independent benchmarks, plus 51 private statement pages, the corrections change the
result by nothing that can be measured. The ParseBench 2 gain in text, tables, formatting and forms should
therefore be read as fit to that benchmark's documents and scorer, not as a better parser in general. On
documents like these the quality comes from OCRia-1 itself.
