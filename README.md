# docparse v2

OCRia-1 reads every page; the PDF itself and a few specialist models correct what it gets wrong.
Target: above 87.13 on ParseBench 2.

- Design: `docs/superpowers/specs/2026-11-24-docparse-v2-design.md`
- Plan (M0, M1): `docs/superpowers/plans/2026-11-24-docparse-v2-m0-m1.md`
- Results: `docs/results/`

```bash
uv sync
uv run pytest -q                                  # unit tests (data tests skip without the data)
uv run python -m evalkit fetch gt                 # dataset, pinned, into $DOCPARSE_DATA (default ~/.cache/docparse-v2)
uv run python -m evalkit score-published --pipeline ocria_1
```

Downloaded data is untrusted: it lives outside the repo and is only ever read.
