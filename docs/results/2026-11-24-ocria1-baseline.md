# OCRia-1 baseline, re-scored (M0, Task 4)

Date: 2026-11-24. OCRia-1's published ParseBench 2 outputs, scored here with the harness's own scorer. No GPU.

Pins: harness `run-llama/ParseBench@6e4479a`, dataset `llamaindex/ParseBench-2@2e954075`,
published outputs `llamaindex/ParseBench-2-results@497e243f` (`results/ocria_1/`).
Ledger run `published-ocria_1-baseline`, code `7e972ef`.

Command: `uv run python -m evalkit score-published --pipeline ocria_1` (explodes the 13 published JSONL files
and runs `parse-bench evaluation run --next ... --next_only`, the command the leaderboard's own re-score uses).

## Result: the board score is reproduced exactly

| | Overall | Text | Formatting | Tables | Charts | Layout | Forms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Board | 85.24 | 87.87 | 91.58 | 86.28 | 77.66 | 85.96 | 82.10 |
| This run, all pages | 85.24 | 87.87 | 91.58 | 86.28 | 77.66 | 85.96 | 82.10 |
| Development part (80%) | 85.23 | 87.66 | 91.68 | 85.32 | 77.47 | 87.18 | 82.10 |
| Check slice (20%) | 85.34 | 88.63 | 91.23 | 90.05 | 78.53 | 81.51 | 82.10 |

Gate (plan Task 4): overall within 0.1 and every dimension within 0.5 of the board. Passed, with every value
equal to two decimals. All 4,317 pages were scored, and no page was left out of an average.

The development part and the check slice are the 80/20 split of `evalkit/splits_data/v1.json` (by source
document, seed `docparse-v2-split-35`). The slice is small, so its tables (90.05) and layout (81.51) differ
from the development part by several points before we have changed anything. Gains will be judged on the
development part and on paired differences, not on the raw gap between the two.

## Per subset

| Dimension | Subset | Metric | Weight | Pages | Score | Pages at 0 | Pages at 1 |
|---|---|---|---:|---:|---:|---:|---:|
| text | text | text_next | 1 | 508 | 87.87 | 13 | 203 |
| formatting | text | formatting_v2 | 0.75 | 508 | 91.38 | 5 | 88 |
| formatting | headings | formatting_v2 | 0.125 | 100 | 97.09 | 0 | 59 |
| formatting | strikeout | formatting_v2 | 0.0625 | 128 | 77.75 | 21 | 66 |
| formatting | underline | formatting_v2 | 0.0625 | 125 | 96.89 | 1 | 97 |
| tables | tables | table_next | 0.5 | 500 | 89.75 | 12 | 294 |
| tables | complex_table_headers | table_next | 0.125 | 149 | 73.20 | 11 | 31 |
| tables | financial_tables | table_next | 0.125 | 319 | 90.03 | 12 | 186 |
| tables | insurance_filing_tables | table_next | 0.125 | 412 | 83.99 | 10 | 173 |
| tables | dense_tables | table_next | 0.125 | 373 | 84.06 | 8 | 142 |
| charts | charts | chart_next | 1 | 568 | 77.66 | 7 | 188 |
| layout | layout | layout_grounding_pass_rate | 1 | 535 | 85.96 | 2 | 52 |
| forms | forms | forms_v2 | 0.875 | 500 | 80.60 | 1 | 68 |
| forms | fillable_forms | forms_v2 | 0.125 | 100 | 92.60 | 0 | 29 |
