# All kept corrections together, on OCRia-1's published readings

Date: 2026-11-25. Run `v-best1` against `m0-roundtrip`, both in the ledger; config `configs/standin_best.yaml`.
All 4,317 pages, CPU only, scored by the harness. The readings are OCRia-1's published outputs (stand-ins),
so this is what the corrections add to that model's text, not yet our own run.

| Dimension | OCRia-1 stand-in | With corrections | Change | Dev | Check | Target |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 85.23 | **86.75** | +1.52 | | | 89 |
| Text | 87.87 | 88.06 | +0.19 | 87.89 | 88.63 | 89.5 |
| Formatting | 91.59 | 91.66 | +0.07 | 91.78 | 91.23 | 93 |
| Tables | 86.20 | 88.03 | +1.82 | 87.10 | 91.70 | 90 |
| Charts | 77.66 | 80.80 | +3.13 | 80.76 | 80.99 | 81 |
| Layout | 85.95 | 88.99 | +3.03 | 89.98 | 85.35 | 88.8 |
| Forms | 82.10 | 83.00 | +0.90 | 82.73 | 83.83 | 87.5 |

- The corrections do not interfere: each dimension scores the same in the combined run as in its own run.
- Tables, charts, layout and forms each gain on the check slice with bootstrap probability 1.00.
- Second place on the board is 87.13. The combined result is 0.38 short of it; forms (4.5 under its
  target) and text and formatting (not yet worked on) are where the rest has to come from.

What each correction is, and what was tried and rejected, is in the per-dimension files:
`2026-11-25-layout-swap.md`, `2026-11-25-tables-fix.md`, `2026-11-25-charts-fix.md`,
`2026-11-25-forms-fix.md` and `2026-11-25-forms-findings.md`.
