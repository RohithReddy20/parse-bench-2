"""Unit tests for the Qwen3.5-27B-FP8 tier-2 call builders, strip tiling and stitching, and answer reader."""

from __future__ import annotations

import io
import re

import pytest
from PIL import Image

from docparse.core.hashing import sha256_hex
from docparse.models.adapters.qwen_tier2 import (
    CHART_PROMPT,
    CHART_SYSTEM,
    FORM_PROMPT,
    HARD_HTML_PROMPT,
    QWEN35_27B_FP8,
    build_chart_call,
    build_form_call,
    build_hard_table_call,
    make_horizontal_strips,
    merge_strip_tables,
    read_tier2_result,
)
from docparse.models.adapters.vlm_specialists import RUNTIME
from docparse.models.calls import ModelResult
from docparse.pdf.render import pixel_hash

_CELLS = re.compile(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", re.S)


def _img(color: tuple[int, int, int] = (255, 255, 255), size: tuple[int, int] = (64, 48)) -> Image.Image:
    return Image.new("RGB", size, color)


def _table(rows: list[list[str]], header: list[str] | None = None, thead: bool = True) -> str:
    head = "<tr>" + "".join(f"<th>{c}</th>" for c in header) + "</tr>" if header else ""
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    if header and thead:
        return f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"
    return f"<table>{head}{body}</table>"


def _rows(md: str) -> list[list[str]]:
    return [_CELLS.findall(r) for r in re.findall(r"<tr\b[^>]*>(.*?)</tr>", md, re.S)]


def _items(lo: int, hi: int) -> list[list[str]]:
    return [[f"Item {i}", f"{i * 10}", f"{i * 100}"] for i in range(lo, hi + 1)]


def _result(answer: str, finish: str | None = "stop", reasoning: str = "") -> ModelResult:
    return ModelResult(key="k", output={"answer": answer, "reasoning": reasoning, "finish_reason": finish})


# ── calls ──


def test_model_is_pinned_fp8_revision() -> None:
    assert QWEN35_27B_FP8.repo == "Qwen/Qwen3.5-27B-FP8"
    assert QWEN35_27B_FP8.revision == "97f5941bf617e31c5e237364a8602ce3f03a551a"
    assert QWEN35_27B_FP8.dtype == "fp8"


def test_chart_call_keys_page_crop_and_prompt_by_content() -> None:
    page, crop = _img(), _img((10, 20, 30), (32, 32))
    call, blobs = build_chart_call(page, crop, "bar A = 12.5", "<table><tr><td>A</td></tr></table>")
    page_sha, crop_sha, prompt_sha = call.inputs
    assert (page_sha, crop_sha) == (pixel_hash(page), pixel_hash(crop))
    assert set(blobs) == set(call.inputs) and prompt_sha == sha256_hex(blobs[prompt_sha])
    assert pixel_hash(Image.open(io.BytesIO(blobs[page_sha]))) == page_sha  # PNG blob round-trips the pixels
    prompt = blobs[prompt_sha].decode("utf-8")
    assert "bar A = 12.5" in prompt and "<td>A</td>" in prompt and "{hint}" not in prompt
    assert call.model == QWEN35_27B_FP8 and call.runtime == RUNTIME
    assert call.prompt_sha == sha256_hex(CHART_PROMPT.encode("utf-8"))
    assert call.params == {
        "n_images": 2,
        "system_prompt": CHART_SYSTEM,
        "enable_thinking": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": 10240,
    }
    same, _ = build_chart_call(_img(), _img((10, 20, 30), (32, 32)), "bar A = 12.5", "<table><tr><td>A</td></tr></table>")
    assert same.key() == call.key()
    variants = [
        build_chart_call(page, crop, "bar A = 13", "<table><tr><td>A</td></tr></table>")[0],
        build_chart_call(page, _img((11, 20, 30), (32, 32)), "bar A = 12.5", "<table><tr><td>A</td></tr></table>")[0],
        build_chart_call(page, crop, "bar A = 12.5", "<table><tr><td>A</td></tr></table>", enable_thinking=False)[0],
        build_chart_call(page, None, "bar A = 12.5", "<table><tr><td>A</td></tr></table>")[0],
    ]
    assert len({call.key(), *(v.key() for v in variants)}) == 5


def test_chart_call_without_crop_or_hints() -> None:
    call, blobs = build_chart_call(_img(), max_tokens=4096, temperature=0.0)
    assert len(call.inputs) == 2 and call.params["n_images"] == 1
    assert call.params["max_tokens"] == 4096 and call.params["temperature"] == 0.0
    prompt = blobs[call.inputs[1]].decode("utf-8")
    assert prompt.count("(none)") == 2


def test_form_call_params_and_prompt_blob() -> None:
    call, blobs = build_form_call(_img())
    assert call.inputs[0] == pixel_hash(_img()) and blobs[call.inputs[1]] == FORM_PROMPT.encode("utf-8")
    assert call.params == {
        "n_images": 1,
        "enable_thinking": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": 10240,
    }
    assert build_form_call(_img(), enable_thinking=False)[0].key() != call.key()


def test_hard_table_call_is_html_only_and_non_thinking() -> None:
    call, blobs = build_hard_table_call(_img())
    assert blobs[call.inputs[1]] == HARD_HTML_PROMPT.encode("utf-8")
    assert call.params == {
        "n_images": 1,
        "enable_thinking": False,
        "temperature": 0.1,
        "top_p": 0.95,
        "presence_penalty": 1.0,
        "max_tokens": 16384,
    }
    assert call.key() != build_form_call(_img())[0].key()
    assert "NEVER output LaTeX `\\begin{tabular}`" in HARD_HTML_PROMPT and "pipe tables" in HARD_HTML_PROMPT


@pytest.mark.parametrize("prompt", [CHART_PROMPT, FORM_PROMPT])
def test_thinking_prompts_cap_reasoning_at_400_words(prompt: str) -> None:
    (words,) = re.findall(r"under (\d+) words", prompt)
    assert int(words) <= 400


# ── strips ──


def test_strips_overlap_across_the_middle_in_rgb() -> None:
    page = Image.new("L", (40, 1000))
    page.putdata([y % 256 for y in range(1000) for _ in range(40)])
    top, bot = make_horizontal_strips(page)
    assert top.mode == bot.mode == "RGB"
    assert top.size == (40, 560) and bot.size == (40, 560)
    assert top.crop((0, 440, 40, 560)).tobytes() == bot.crop((0, 0, 40, 120)).tobytes()
    assert top.crop((0, 0, 40, 1)).getpixel((0, 0)) == (0, 0, 0)
    assert bot.crop((0, 559, 40, 560)).getpixel((0, 0)) == (999 % 256,) * 3


def test_strips_without_overlap_and_invalid_overlap() -> None:
    top, bot = make_horizontal_strips(_img(size=(10, 100)), overlap=0.0)
    assert top.size == bot.size == (10, 50)
    with pytest.raises(ValueError):
        make_horizontal_strips(_img(), overlap=1.5)


# ── stitching ──


def test_merge_drops_repeated_thead_and_overlap_rows() -> None:
    header = ["Item", "Qty", "Cost"]
    top = "# Ledger\n\n" + _table(_items(1, 7), header)
    bot = _table(_items(5, 12), header)
    merged = merge_strip_tables(top, bot)
    assert merged.startswith("# Ledger\n\n<table><thead>") and merged.count("<table") == 1
    assert _rows(merged) == [header, *_items(1, 12)]


def test_merge_drops_repeated_leading_th_rows_without_thead() -> None:
    header = ["Name", "Qty", "Cost"]
    merged = merge_strip_tables(_table(_items(1, 4), header, thead=False), _table(_items(4, 6), header, thead=False))
    assert _rows(merged) == [header, *_items(1, 6)]


def test_merge_matches_rows_after_normalising_markup_and_near_identical_text() -> None:
    long_top = ["Total operating expenses before income tax adjustments", "1,234", "5,678"]
    long_bot = ["Total operating expenses before income tax adjustment", "1,234", "5,678"]
    top = _table([*_items(1, 2), ["<b>Item&nbsp;3</b>", "30", "300"], long_top])
    bot = _table([["Item 3", "30", "300"], long_bot, *_items(4, 5)])
    assert _rows(merge_strip_tables(top, bot)) == [
        *_items(1, 2),
        ["<b>Item&nbsp;3</b>", "30", "300"],
        long_top,
        *_items(4, 5),
    ]


def test_merge_keeps_both_tables_when_nothing_overlaps() -> None:
    top, bot = _table([["B", "2"], ["C", "2"]]), _table([["C", "3"], ["D", "4"]])  # "C 2" vs "C 3": distinct rows
    merged = merge_strip_tables(top, bot)
    assert merged == f"{top}\n\n{bot}"
    assert _rows(merged) == [["B", "2"], ["C", "2"], ["C", "3"], ["D", "4"]]


@pytest.mark.parametrize("cut_bot", [False, True])
def test_merge_replaces_rows_cut_off_at_the_strip_edges(cut_bot: bool) -> None:
    top = _table([*_items(1, 6), ["Item 7", "70", ""]])  # the top strip's lower edge slices row 7
    bot = _table(([["", "40", ""]] if cut_bot else []) + _items(5, 9))  # and the bottom strip's upper edge row 4
    assert _rows(merge_strip_tables(top, bot)) == _items(1, 9)


def test_merge_bottom_strip_inside_top_returns_top_table() -> None:
    top = _table(_items(1, 6), ["Item", "Qty", "Cost"])
    assert merge_strip_tables(top, _table(_items(3, 6), ["Item", "Qty", "Cost"])) == top


def test_merge_when_top_strip_shows_only_the_header() -> None:
    header = ["Item", "Qty", "Cost"]
    top = "Schedule A\n\n<table><thead><tr><th>Item</th><th>Qty</th><th>Cost</th></tr></thead></table>"
    bot = _table(_items(1, 3), header)
    assert merge_strip_tables(top, bot) == f"Schedule A\n\n{bot}"


def test_merge_stitches_text_around_the_table_and_keeps_other_tables() -> None:
    header = ["Item", "Qty", "Cost"]
    table_a, table_c = _table([["a", "1"]], ["K", "V"]), _table([["c", "3"]], ["X", "Y"])
    top = f"Intro\n\n{table_a}\n\nBetween text.\n\n{_table(_items(1, 5), header)}\n\nNote A."
    bot = f"{_table(_items(4, 8), header)}\n\nNote A.\n\nAfter text.\n\n{table_c}"
    merged = merge_strip_tables(top, bot)
    assert merged == (
        f"Intro\n\n{table_a}\n\nBetween text.\n\n{_table(_items(1, 8), header)}\n\nNote A.\n\nAfter text.\n\n{table_c}"
    )
    lead_top = "Heading\n\nIntro sentence.\n\n" + _table(_items(1, 4))
    lead_bot = "Intro sentence.\n\n" + _table(_items(3, 6)) + "\n\nFootnote 1."
    assert merge_strip_tables(lead_top, lead_bot) == f"Heading\n\nIntro sentence.\n\n{_table(_items(1, 6))}\n\nFootnote 1."


def test_merge_plain_text_drops_repeated_lines_once() -> None:
    top = "Para one.\n\nPara two line a\nPara two line b\n"
    bot = "Para two line a\nPara two line b\n\nPara three."
    assert merge_strip_tables(top, bot) == "Para one.\n\nPara two line a\nPara two line b\n\nPara three."
    assert merge_strip_tables("Alpha", "Beta") == "Alpha\n\nBeta"
    assert merge_strip_tables("", "Beta") == "Beta"


# ── reading answers ──


def test_read_strips_outer_fence_and_keeps_reasoning() -> None:
    read = read_tier2_result(_result("```html\n<table><tr><td>1</td></tr></table>\n```\n", reasoning=" plan "))
    assert read.markdown == "<table><tr><td>1</td></tr></table>"
    assert read.reasoning == "plan" and read.finish_reason == "stop" and read.usable


def test_read_splits_inline_think_block() -> None:
    read = read_tier2_result(_result("<think>\ncount the bars\n</think>\n\nTitle: Sales"))
    assert read.markdown == "Title: Sales" and read.reasoning == "count the bars" and read.usable


@pytest.mark.parametrize(
    ("answer", "finish"),
    [
        ("<table><tr><td>1</td>", "length"),
        ("Title: Sales", None),
        ("```markdown\n  \n```", "stop"),
        ("", "stop"),
        ("Name: Ann" + "<br>" * 12, "stop"),
        ("Name: Ann" + "<br/> \n" * 15 + "Date: [___]", "stop"),
        ("\n".join(["Total due 5"] * 12), "stop"),
        ("\n\n".join(["<tr><td>Fee</td><td>10</td></tr>"] * 12), "stop"),
    ],
)
def test_read_flags_truncated_empty_or_looping_answers(answer: str, finish: str | None) -> None:
    assert not read_tier2_result(_result(answer, finish)).usable


@pytest.mark.parametrize(
    "answer",
    [
        "Name: Ann" + "<br>" * 11,
        "\n".join(["Total due 5"] * 11),
        "\n".join(["<tr><td></td><td>&nbsp;</td></tr>"] * 20),  # empty grid rows are not a loop
        "\n".join(["[___]"] * 20),
        "\n".join(f"<tr><td>Row {i}</td></tr>" for i in range(50)),
    ],
)
def test_read_accepts_legitimate_repetition(answer: str) -> None:
    assert read_tier2_result(_result(answer)).usable
