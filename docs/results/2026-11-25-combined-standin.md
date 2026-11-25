# All kept corrections together, on OCRia-1's published readings

Date: 2026-11-25. Run `v-best2` against `m0-roundtrip`, both in the ledger; config `configs/standin_best.yaml`.
All 4,317 pages, CPU only, scored by the harness. The readings are OCRia-1's published outputs (stand-ins),
so this is what the corrections add to that model's text, not yet our own run.

| Dimension | OCRia-1 stand-in | With corrections | Change | Dev | Check | Target |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 85.23 | **87.07** | +1.84 | | | 89 |
| Text | 87.87 | 88.42 | +0.55 | 88.35 | 88.66 | 89.5 |
| Formatting | 91.59 | 93.13 | +1.54 | 93.27 | 92.64 | 93 |
| Tables | 86.20 | 88.05 | +1.85 | 87.12 | 91.71 | 90 |
| Charts | 77.66 | 80.80 | +3.13 | 80.76 | 80.99 | 81 |
| Layout | 85.95 | 88.99 | +3.03 | 89.98 | 85.35 | 88.8 |
| Forms | 82.10 | 83.01 | +0.91 | 82.75 | 83.83 | 87.5 |

- The corrections do not interfere: each dimension scores the same in the combined run as in its own run
  (run `v-best1` had everything but formatting and text, at 86.75).
- Every dimension gains on the check slice as well as on the development part.
- Second place on the board is 87.13. The combined result is 0.06 short of it. Forms is 4.5 under its
  target and holds most of what is still missing; tables is 2 under, text 1.1 under.
- The text row includes two safety nets that the bootstrap cannot judge because each touches one page:
  the looped-read fallback and the empty-read fallback.

What each correction is, and what was tried and rejected, is in the per-dimension files:
`2026-11-25-layout-swap.md`, `2026-11-25-tables-fix.md`, `2026-11-25-charts-fix.md`,
`2026-11-25-forms-fix.md`, `2026-11-25-forms-findings.md`, `2026-11-25-formatting-fix.md` and
`2026-11-25-text-fix.md`.
