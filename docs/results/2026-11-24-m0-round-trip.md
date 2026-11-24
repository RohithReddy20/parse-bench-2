# M0 round trip: the pipeline with every corrector off (Task 10)

Date: 2026-11-24. The round-trip rule (design §5.4): OCRia-1's reading, parsed into the IR and written back
by our writers, must score the same as OCRia-1 alone. No GPU.

- Readings: OCRia-1's published outputs, imported into the model-call cache as stand-ins
  (`python -m evalkit standin`): 4,247 page calls for 4,317 cases.
- Run: `python -m evalkit run --config configs/standin.yaml --name docparse_m0 --run-id m0-roundtrip`
  (50 seconds for all pages on 8 cores, `ReplayClient`, answer keys forbidden while parsing), then
  `python -m evalkit score --run-id m0-roundtrip --name docparse_m0 --baseline published-ocria_1-baseline`.
- Pins: harness `6e4479a`, dataset `2e954075`. Config hash `2f0f9270f917b011`,
  cache snapshot `4caba64b655d`, code `a81c7fa`.

## Result

| | OCRia-1 published | docparse_m0 | Delta |
|---|---:|---:|---:|
| overall | 85.24 | 85.23 | -0.01 |
| text | 87.87 | 87.87 | +0.00 |
| formatting | 91.58 | 91.59 | +0.01 |
| tables | 86.28 | 86.20 | -0.08 |
| charts | 77.66 | 77.66 | +0.01 |
| layout | 85.96 | 85.95 | -0.01 |
| forms | 82.10 | 82.10 | +0.00 |

Gate (plan Task 10): every dimension within 0.1 of the baseline, no errors, nothing pending. Passed:
4,317 pages written, 0 errors, 0 pending.

## Why it is not exactly zero

411 of 4797 page scores changed, for two reasons.

1. **Page size in the layout record (394 layout pages, each under 0.01).** The layout items are
   identical on 531 of 535 layout pages. The published records give the page size as the rendered image's
   size (a 792 pt page rendered at 300 DPI is 3,301 px, written as 792.24 pt); ours give the PDF's own size
   (792.0 pt). The boxes are true points in both, so ours is the consistent pairing. The net effect on layout
   is -0.01.
2. **Pages shared by two subsets (12 page scores changed by 0.01 or more).** 70 cases share a page
   image with another case (for example a chart page that is also a layout case), and for 24 of those pairs
   LlamaIndex's two published readings differ. Our cache holds one reading per page image, and the reading
   that carries layout items is kept, so the other case gets a different reading than was published for it.
   One table page (`complex_header_006`) goes from 0.95 to 0 this way, which is the whole -0.08 on tables.

| Case | Metric | Published | docparse_m0 |
|---|---|---:|---:|
| `complex_table_headers/complex_header_006` | table_next | 0.950 | 0.000 |
| `layout/annotated__Starhill-Annual-Report-2024_p37` | layout_grounding_pass_rate | 0.720 | 0.658 |
| `charts/article_thebeatjun2025_p33` | chart_next | 0.781 | 0.731 |
| `text/text_ocr__012-25` | formatting_v2 | 0.800 | 0.850 |
| `charts/SRI-Insights-August 2025_media_embargo_p1` | chart_next | 0.306 | 0.356 |
| `charts/article_thebeatjun2025_p10` | chart_next | 0.288 | 0.324 |
| `charts/sigma-1-2021-en_p20` | chart_next | 0.577 | 0.600 |
| `layout/markdown_block_layout__layout_0001_0100-definitions-8-17-23_p2` | layout_grounding_pass_rate | 0.758 | 0.777 |
| `charts/WEF_Asset_Tokenization_in_Financial_Markets_2025_p22` | chart_next | 0.523 | 0.505 |
| `layout/layout_bad__pos343028_p1` | layout_grounding_pass_rate | 0.472 | 0.458 |
| `charts/(Web_version)_E-Government_Survey_2024_1392024_p81` | chart_next | 0.677 | 0.690 |
| `layout/annotated__pdf_211245533966_p5` | layout_grounding_pass_rate | 0.796 | 0.786 |

Our own OCRia-1 run (M1) reads each page image once, so the second effect is part of our numbers from then on.

## Other checks on this run

- The markdown writer reproduces all 4,317 published pages byte for byte
  (`tests/base/test_ocria_parse.py::test_every_published_page_round_trips`): 36,560 paragraphs, 14,582
  headings, 6,171 tables, 2,171 lists, 10 code blocks, and 40 blocks carried as opaque (38 pages have an
  unclosed table).
- Lint: 17 pages with looped text. No HTML comments, no layout item without boxes.
- Guard tests on real pages: rename invariance on 50 PDFs across all 13 subsets, isolation on 20 pages inside
  a 200-page run, and the answer-key audit hook. All pass.
- The harness reads every `ParseOutput` that carries layout through its `LlamaParseLayoutAdapter` (selected by
  its `matches` rule, and it needs `llama-cloud` installed). That is the adapter OCRia-1's published layout is
  scored through, and `tests/contract/test_default_adapter.py` pins that ours is read by it too.
