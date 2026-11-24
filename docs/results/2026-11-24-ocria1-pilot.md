# OCRia-1 pilot on Modal (M1, Task 13)

Date: 2026-11-24. First run of our own OCRia-1 serving: 46 development-part pages (four per subset, two for
the three smallest), with the settings of the harness's `ocria_1` pipeline.

- Model `llamaindex/OCRia-1@dae83aa5` in BF16, vLLM 0.34.0 (`vllm/vllm-openai:v0.34.0`), one A100-40GB on
  Modal, 32 requests in parallel. Page rendered at 300 DPI, `prompt.txt` as system message, greedy decoding,
  thinking off, up to 24,576 output tokens in a 32,768-token context.
- Manifest `runs/m1-pilot/manifest` (46 calls), ledger run `m1-pilot`, config `configs/ocria_own.yaml`
  (hash `22bdec65abe993c0`).

## Does our run reproduce the published one?

Scored on the same 46 pages with the harness scorer; "published" is OCRia-1's published output for those pages
(ledger run `published-ocria_1-baseline`). Layout is not in this comparison (the pilot config has no layout).

| Subset | Metric | Pages | Ours | Published | Delta |
|---|---|---:|---:|---:|---:|
| text | text_next | 4 | 95.99 | 95.99 | +0.00 |
| text | formatting_v2 | 4 | 94.63 | 94.63 | +0.00 |
| headings | formatting_v2 | 2 | 100.00 | 100.00 | +0.00 |
| strikeout | formatting_v2 | 4 | 98.71 | 98.71 | +0.00 |
| underline | formatting_v2 | 2 | 96.38 | 96.48 | -0.10 |
| tables | table_next | 4 | 85.76 | 85.76 | +0.00 |
| complex_table_headers | table_next | 4 | 81.56 | 81.56 | +0.00 |
| financial_tables | table_next | 4 | 99.66 | 99.66 | +0.00 |
| insurance_filing_tables | table_next | 4 | 95.94 | 95.94 | +0.00 |
| dense_tables | table_next | 4 | 98.18 | 98.18 | +0.00 |
| charts | chart_next | 4 | 89.05 | 87.84 | +1.20 |
| forms | forms_v2 | 4 | 68.22 | 65.90 | +2.32 |
| fillable_forms | forms_v2 | 2 | 100.00 | 100.00 | +0.00 |

- 37 of the 46 page readings are byte-identical to the published ones. Mean word-level similarity is 0.965.
- The nine that differ are small wording differences, except two form pages. One of them
  (`forms/filled__portugal--08_DA_Alteracoes_Atividade--p3__typed`) ran into a loop in our run and stopped at
  the token limit (44,575 characters against 6,660 published); it is the one `finish_reason=length` and the
  one `looped_text` lint finding.
- So the harness setting (the 300 DPI render sent unchanged) is the right one, and vLLM 0.34.0 gives the
  published behaviour. The model card's 4.1-megapixel variant was not needed and not run.
- Greedy decoding on vLLM is not bit-reproducible across runs: about a fifth of pages differ slightly. This is
  the run-to-run noise every later comparison has to beat, which is why fixes are judged on cached readings.

## What it cost

| | |
|---|---:|
| Calls | 46 done, 0 failed |
| Container start to server ready (weights download, load, warm-up) | 265 s |
| Running the 46 calls | 296 s |
| GPU time billed | about 9.4 min, about $0.33 at $0.000583/s |
| Input tokens per page | 9,475 mean (17,340 max) |
| Output tokens per page | 2,707 mean, 1,299 median, 21,747 max |
| Aggregate generation throughput, 15–30 requests running | 1,000–1,230 tokens/s |
| Prompt (image) throughput while few requests generate | 5,000–8,000 tokens/s |
| One request alone | about 110 tokens/s |

A first attempt failed while building the image (`python` is not on the vLLM image's PATH) before any GPU
started, so it cost nothing.

## Plan for the full run

- New calls: 4,201 OCRia-1 page reads. The 4,317 cases need 4,247 (70 share a page image with another case),
  and the 46 pilot calls are cached.
- Tokens: about 40 million in and 9.5–11.5 million out, from the pilot's per-page figures and the length of
  the published readings.
- Time on one A100-40GB: about 3.0–3.4 hours at the throughput above. More parallel requests may help (the
  KV cache was 23% used at 30 requests); the full run uses 64, and its first minutes show the real rate.
- Cost: about $6.30–7.20 at $2.10/h. The run is capped at 4.5 hours ($9.45); it is resumable, so a capped or
  interrupted run loses nothing.
- Alternatives: an H100 ($3.95/h) would be faster but about the same cost; Colab A100 would be about 16–18
  compute units and needs the 3.9 GB manifest uploaded to Drive and the notebook run by hand.
