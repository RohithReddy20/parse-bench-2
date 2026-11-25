# Our own OCRia-1 run reproduces the published score (M1, Task 14)

Date: 2026-11-25. Ledger run `m1-own` (config `configs/ocria_own.yaml`, pipeline name `docparse_m1_base`),
scored by the harness against `published-ocria_1-baseline`. No corrections are on. Layout is left out here:
OCRia-1 writes no boxes, so layout is its own step (see the layout results files).

## Scores

| Dimension | Published OCRia-1 | Our run | Difference | Within 0.5 |
|---|---:|---:|---:|---|
| Text | 87.87 | 88.01 | +0.13 | yes |
| Formatting | 91.58 | 91.65 | +0.07 | yes |
| Tables | 86.28 | 86.30 | +0.02 | yes |
| Charts | 77.66 | 77.22 | -0.44 | yes |
| Forms | 82.10 | 82.15 | +0.06 | yes |

The gate of the plan (each of the five within 0.5 of the published score) passes. Charts is the one to watch:
it is 0.44 lower, and the paired bootstrap gives only a 2% chance that the difference is noise in our favour.
The pilot had shown the same kind of run-to-run variation on chart pages.

## The run

- 4,247 page reads (4,317 cases; 70 share a page image), all through the cache: 46 from the pilot, 4,201 new.
- Modal, one A100-40GB, vLLM 0.34.0, BF16, 300 DPI page images, 64 requests in flight.
- Three sessions: 38 minutes (ended by a cancellation from outside the container, cause unknown, 914 answers
  kept), 3.5 hours for the rest, and 9 minutes to retry 7 calls that had timed out after 30 minutes each.
- GPU time 4.26 hours, about **$8.94** at $2.10 an hour. The estimate was $6.74 and the cap $9.45.
- Tokens: 41.6 million in, 9.56 million out (9,790 in and 2,252 out per page on average). No image was
  downscaled.
- **58 answers hit the 24,576-token limit** (`finish_reason: length`); the lint marks 17 pages as looped text.
  Those are the pages the text-layer fallback and a second read are for.

Why it cost a third more than estimated: with 64 requests in flight the GPU's key-value cache sat at 98–99%,
which held generation near 650–800 tokens a second instead of the 1,100 measured in the pilot, and the looped
pages alone produced about 1.4 million tokens. Next time: 40–48 requests in flight, and a lower token limit
with a retry for the few pages that need more.
