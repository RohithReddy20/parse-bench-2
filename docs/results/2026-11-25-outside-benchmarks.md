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

Results are added below when the page reads finish.
