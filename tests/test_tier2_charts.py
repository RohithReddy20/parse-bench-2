"""Tier-2 chart reads, form refinement and table strip re-reads (Qwen3.5) wired into the pipeline, against a
scripted model client."""

import io
import re

import pytest
from PIL import Image

from docparse.core.config import Config
from docparse.core.hashing import canonical_json, sha256_hex
from docparse.models.adapters import ocria, ppdoclayout, qwen_tier2, vlm_specialists
from docparse.models.calls import ModelResult, Pending
from docparse.pdf.render import RenderedPage
from docparse.pipeline import _dedup_chart_boxes, _union_share, parse
from docparse.write.markdown import write_document

CHANDRA, OCRIA, PP, TIER2, FORM, SINGLE, STRIP = (
    vlm_specialists.CHANDRA_PROMPT_ID,
    ocria.PROMPT_ID,
    ppdoclayout.PROMPT_ID,
    qwen_tier2.CHART_PROMPT_ID,
    qwen_tier2.FORM_REFINE_PROMPT_ID,
    qwen_tier2.SINGLE_CHART_PROMPT_ID,
    qwen_tier2.HARD_HTML_PROMPT_ID,
)
GPU_SECONDS = {CHANDRA: 1.0, OCRIA: 2.0, PP: 0.0, TIER2: 4.0, FORM: 3.0, SINGLE: 4.0, STRIP: 5.0}
BASE = Config(page_model=ocria.OWN, page_runtime="vllm-0.34.0", prompt_sha="t", reader="chandra_hybrid", layout="none")
V3 = Config.load("configs/frontier_v3.yaml")
FRONTIER = BASE.model_copy(update={"tier2_charts": True, "tier2_chart_options": V3.tier2_chart_options})
FORMS = BASE.model_copy(update={"tier2_forms": True, "tier2_form_options": V3.tier2_form_options})
TABLES = BASE.model_copy(update={"tier2_tables": True, "tier2_table_options": V3.tier2_table_options})
OCRIA_TABLES = TABLES.model_copy(update={"reader": "ocria"})

PROSE_TOP = "Quarterly revenue grew strongly across all regions this year."
PROSE_END = "Margins held steady through the final quarter."
OCRIA_TABLE = "<table><tr><th>Quarter</th><th>Revenue</th></tr><tr><td>Q1</td><td>31.7</td></tr></table>"
QWEN_TABLE = (
    "<table><tr><th>Quarter</th><th>Revenue</th></tr><tr><td>Q1</td><td>12.5</td></tr>"
    "<tr><td>Q2</td><td>14.25</td></tr></table>"
)
SALES_TABLE = (
    "<table><tr><th>Region</th><th>Sales</th></tr><tr><td>North</td><td>40</td></tr>"
    "<tr><td>South</td><td>60</td></tr></table>"
)
CHART_PAGE = (  # Chandra draws the chart as a Figure whose alt text says so
    f'<div data-bbox="50 40 950 120" data-label="Text"><p>{PROSE_TOP}</p></div>'
    '<div data-bbox="100 200 900 600" data-label="Figure"><img alt="Bar chart of revenue by quarter"/>'
    "<table><tr><th>Quarter</th><th>Revenue</th></tr><tr><td>Q1</td><td>77.7</td></tr></table></div>"
    f'<div data-bbox="50 650 950 720" data-label="Text"><p>{PROSE_END}</p></div>'
)
MISSED_CHART_PAGE = (  # Chandra sees no chart; PP-DocLayoutV3 does
    f'<div data-bbox="50 40 950 120" data-label="Text"><p>{PROSE_TOP}</p></div>'
    '<div data-bbox="100 300 900 700" data-label="Image"><img alt="Company logo"/></div>'
    f'<div data-bbox="100 750 900 900" data-label="Table">{SALES_TABLE}</div>'
)
FORM_PAGE = (  # Chandra tags the form (two checkboxes) and the page furniture around it
    '<div data-bbox="50 20 950 60" data-label="Page-Header"><p>ACME Insurance Ltd</p></div>'
    '<div data-bbox="50 80 950 600" data-label="Form"><p>Applicant name: Jane Doe</p>'
    '<p><input type="checkbox"/> Married <input type="checkbox" checked/> Single</p>'
    "<p>Policy number: 4471-0093</p><p>Date of birth: 1984-07-15</p></div>"
    '<div data-bbox="50 940 950 980" data-label="Page-Footer"><p>Form AC-12 page 1 of 2</p></div>'
)
CHECKBOX_PAGE = (  # no Form block, but four checkboxes
    '<div data-bbox="50 80 950 400" data-label="Text"><p>Applicant name: Jane Doe</p>'
    '<p><input type="checkbox"/> Married <input type="checkbox" checked/> Single '
    '<input type="checkbox"/> Divorced <input type="checkbox"/> Widowed</p></div>'
)
FLIP = ("[ ] Married [x] Single", "[x] Married [ ] Single")  # Chandra's misread boxes, and Qwen's correction
UNITS_TABLE = "<table><tr><th>Market</th><th>Units</th></tr><tr><td>East</td><td>7.5</td></tr></table>"
TWO_CHARTS = [  # two large charts (59% of the page), listed bottom first, and a repeat inside the top one
    {"label": "chart", "score": 0.9, "box": [60, 560, 790, 940]},
    {"label": "chart", "score": 0.9, "box": [60, 150, 790, 520]},
    {"label": "chart", "score": 0.8, "box": [70, 160, 780, 510]},
]


def _ledger(lo: int, hi: int) -> str:
    rows = "".join(f"<tr><td>Item {i}</td><td>{i * 10}</td></tr>" for i in range(lo, hi + 1))
    return f"<table><tr><th>Item</th><th>Qty</th></tr>{rows}</table>"


LEDGER_STRIPS = (f"{PROSE_TOP}\n\n{_ledger(1, 6)}", _ledger(5, 11), f"{_ledger(10, 15)}\n\n{PROSE_END}")


def _png(size=(850, 1100), gradient=False) -> bytes:
    buf = io.BytesIO()
    image = Image.linear_gradient("L").resize(size).convert("RGB") if gradient else Image.new("RGB", size, "white")
    image.save(buf, format="PNG")
    return buf.getvalue()


PAGE = _png()
STRIPED = _png(gradient=True)  # every horizontal strip of it differs, and so does its call key


class ScriptedClient:
    """Answers each model by prompt id; prompt ids in `pending` are not in the cache yet. Per-chart and strip reads
    take `singles` and `strips` in call order."""

    def __init__(
        self, chandra, tier2="", pp=(), pending=(), finish="stop", form="", singles=(), strips=(), ocria_finish=None
    ):
        self.answers = {
            CHANDRA: {"answer": chandra, "finish_reason": "stop"},
            OCRIA: {"markdown": f"{PROSE_TOP}\n\n{OCRIA_TABLE}", "finish_reason": ocria_finish},
            PP: {"detections": list(pp)},
            TIER2: {"answer": tier2, "finish_reason": finish},
            FORM: {"answer": form, "finish_reason": "stop"},
        }
        self.scripts = {SINGLE: list(singles), STRIP: list(strips)}
        self.pending, self.calls, self.blobs = set(pending), [], {}

    def get(self, call, blobs, meta=None):
        self.calls.append(call)
        self.blobs[call.key()] = blobs
        if call.prompt_id in self.pending:
            return Pending(key=call.key())
        if call.prompt_id in self.scripts:
            output = {"answer": self.scripts[call.prompt_id][len(self.of(call.prompt_id)) - 1], "finish_reason": "stop"}
        else:
            output = self.answers[call.prompt_id]
        return ModelResult(key=call.key(), output=output, tokens_out=10, gpu_seconds=GPU_SECONDS[call.prompt_id])

    def of(self, prompt_id):
        return [c for c in self.calls if c.prompt_id == prompt_id]


def test_off_by_default_never_asks_for_tier2_and_keeps_the_config_hash():
    client = ScriptedClient(CHART_PAGE, tier2=QWEN_TABLE)
    doc = parse(PAGE, BASE, client).document
    assert not client.of(TIER2) and not client.of(PP)
    assert "31.7" in write_document(doc)  # hybrid_v2's OCRia-1 chart fusion
    hybrid = Config.load("configs/hybrid_v2.yaml")
    dump = {
        k: v for k, v in hybrid.model_dump(mode="json").items() if not k.startswith("tier2_") and k != "select_options"
    }
    assert not hybrid.tier2_charts and hybrid.config_hash() == sha256_hex(canonical_json(dump))[:16]
    frontier = Config.load("configs/frontier_v3.yaml")
    assert frontier.tier2_charts and frontier.config_hash() != hybrid.config_hash()
    off = {
        "tier2_charts": False,
        "tier2_chart_options": {},
        "tier2_forms": False,
        "tier2_form_options": {},
        "tier2_tables": False,
        "tier2_table_options": {},
        "form_options": {
            k: v for k, v in frontier.form_options.items() if k not in {"checkbox_ink", "acro_clean", "route_blanks"}
        },
        "format_options": {k: v for k, v in frontier.format_options.items() if k != "underline_missing"},
        "table_options": {k: v for k, v in frontier.table_options.items() if k != "table_geo"},
        "layout_options": {
            k: v for k, v in frontier.layout_options.items() if k not in {"pp_scan_arbiter", "blank_drop"}
        },
    }
    assert frontier.model_copy(update=off) == hybrid


def test_a_chandra_chart_figure_takes_the_tier2_tables_and_keeps_the_prose():
    client = ScriptedClient(CHART_PAGE, tier2=f"Revenue by quarter\n{QWEN_TABLE}")
    doc = parse(PAGE, FRONTIER, client).document
    md = write_document(doc)
    (call,) = client.of(TIER2)
    blobs = client.blobs[call.key()]
    assert call.params["n_images"] == 2 and call.params["enable_thinking"] is False
    assert call.params["temperature"] == 0.1 and call.params["presence_penalty"] == 1.0
    assert Image.open(io.BytesIO(blobs[call.inputs[1]])).size == (782, 572)  # the Figure box, padded by 6%
    assert OCRIA_TABLE in blobs[call.inputs[-1]].decode()  # OCRia-1's chart table is the draft
    assert md.index(PROSE_TOP) < md.index("12.5") < md.index("14.25") < md.index(PROSE_END)
    assert "77.7" not in md and "31.7" not in md  # Chandra's and OCRia-1's chart tables are replaced
    cost = doc.pages[0].cost
    assert (cost.calls, cost.gpu_seconds) == (3, 7.0)


def test_a_chart_only_pp_doclayout_saw_is_cropped_read_and_appended_without_repeats():
    pp = [
        {"label": "chart", "score": 0.9, "box": [120, 330, 760, 770]},
        {"label": "chart", "score": 0.2, "box": [0, 0, 850, 1100]},  # under pp_chart_score: ignored
        {"label": "table", "score": 0.95, "box": [85, 825, 765, 990]},
    ]
    answer = f"Sales by region\n{SALES_TABLE}\n\nRevenue by quarter\n{QWEN_TABLE}"
    client = ScriptedClient(MISSED_CHART_PAGE, tier2=answer, pp=pp)
    doc = parse(PAGE, FRONTIER, client).document
    md = write_document(doc)
    assert not client.of(OCRIA) and len(client.of(PP)) == 1
    (call,) = client.of(TIER2)
    blobs = client.blobs[call.key()]
    assert call.params["n_images"] == 2
    assert Image.open(io.BytesIO(blobs[call.inputs[1]])).size == (742, 572)  # the chart box, padded by 6%
    assert SALES_TABLE in blobs[call.inputs[-1]].decode()  # Chandra's tables are the draft
    assert md.startswith(PROSE_TOP) and md.endswith(f"Revenue by quarter\n{QWEN_TABLE}")
    assert md.count("North") == 1 and "Sales by region" not in md  # the table Chandra read is not repeated
    assert doc.pages[0].cost.calls == 2


@pytest.mark.parametrize(
    ("answer", "finish"),
    [
        (f"Revenue by quarter\n{QWEN_TABLE}", "length"),
        ("<table><tr><td>Q1" + "<br>" * 20 + "</td></tr></table>", "stop"),
    ],
)
def test_an_unusable_tier2_answer_leaves_the_hybrid_v2_page(answer, finish):
    base = parse(PAGE, BASE, ScriptedClient(CHART_PAGE)).document
    client = ScriptedClient(CHART_PAGE, tier2=answer, finish=finish)
    doc = parse(PAGE, FRONTIER, client).document
    assert len(client.of(TIER2)) == 1
    assert doc.pages == base.pages and "31.7" in write_document(doc)


def test_pending_calls_plan_the_tier2_read_after_its_inputs():
    client = ScriptedClient(CHART_PAGE, pending={OCRIA, PP})
    first = parse(PAGE, FRONTIER, client)
    assert first.document is None and not client.of(TIER2)
    assert first.pending == [client.of(OCRIA)[0].key(), client.of(PP)[0].key()]
    # the layout's PP-DocLayoutV3 call is the chart trigger's call too: asked for once
    shared = FRONTIER.model_copy(update={"layout": "from_page_read", "layout_options": {"pp_arbiter": True}})
    client = ScriptedClient(CHART_PAGE, pending={OCRIA, PP})
    assert len(parse(PAGE, shared, client).pending) == 2 and len(client.of(PP)) == 1
    client = ScriptedClient(CHART_PAGE, tier2=QWEN_TABLE, pending={TIER2})
    second = parse(PAGE, FRONTIER, client)
    (call,) = client.of(TIER2)
    assert second.document is None and second.pending == [call.key()]


@pytest.mark.parametrize("page", [FORM_PAGE, CHECKBOX_PAGE], ids=["form-block", "four-checkboxes"])
def test_a_form_page_sends_its_chandra_draft_and_takes_the_refined_copy(page):
    draft = vlm_specialists.chandra_markdown(page)
    refined = f"{draft.replace(*FLIP)}\n\nSignature: [___]"
    client = ScriptedClient(page, form=refined)
    doc = parse(PAGE, FORMS, client).document
    md = write_document(doc)
    (call,) = client.of(FORM)
    assert not client.of(TIER2) and not client.of(OCRIA)
    assert call.params["n_images"] == 1 and call.params["enable_thinking"] is False and "top_k" not in call.params
    assert (call.params["temperature"], call.params["presence_penalty"], call.params["max_tokens"]) == (0.1, 1.0, 8192)
    assert f"<draft>\n{draft}\n</draft>" in client.blobs[call.key()][call.inputs[-1]].decode()
    assert FLIP[1] in md and FLIP[0] not in md and "Signature: [___]" in md
    assert (doc.pages[0].cost.calls, doc.pages[0].cost.gpu_seconds) == (2, 4.0)
    client = ScriptedClient(page, form=refined, pending={FORM})
    waiting = parse(PAGE, FORMS, client)
    assert waiting.document is None and waiting.pending == [client.of(FORM)[0].key()]


@pytest.mark.parametrize(
    ("drop", "taken"),
    [("Date of birth: 1984-07-15", False), (" Ltd", True)],  # 3 of the draft's 18 words lost (17%), or 1 (6%)
)
def test_a_refined_copy_must_keep_the_drafts_words(drop, taken):
    draft = vlm_specialists.chandra_markdown(FORM_PAGE)
    client = ScriptedClient(FORM_PAGE, form=draft.replace(*FLIP).replace(drop, ""))
    doc = parse(PAGE, FORMS, client).document
    assert len(client.of(FORM)) == 1 and (FLIP[1] in write_document(doc)) is taken
    if not taken:  # Chandra's page, as if Tier-2 forms were off
        assert doc.pages == parse(PAGE, BASE, ScriptedClient(FORM_PAGE)).document.pages


@pytest.mark.parametrize(
    "page",
    [
        FORM_PAGE.replace("<p>Date of birth: 1984-07-15</p>", '<p><math display="block">x^2 + y^2 = r^2</math></p>'),
        CHECKBOX_PAGE + '<div data-bbox="50 500 950 560" data-label="Text"><p>$$E = mc^2$$</p></div>',
        CHECKBOX_PAGE.replace('<input type="checkbox"/> Divorced <input type="checkbox"/> Widowed', ""),
    ],
    ids=["math-tag", "display-math", "two-checkboxes"],
)
def test_math_pages_and_pages_that_are_not_forms_are_never_refined(page):
    client = ScriptedClient(page, form=vlm_specialists.chandra_markdown(page))
    assert parse(PAGE, FORMS, client).document is not None and not client.of(FORM)


def test_strip_furniture_drops_the_page_header_and_footer_the_refined_copy_repeats():
    draft = vlm_specialists.chandra_markdown(FORM_PAGE, drop_furniture=True)
    answer = f"**ACME Insurance Ltd**\n\n{draft.replace(*FLIP)}\n\nForm AC-12 page 1 of 2"
    client = ScriptedClient(FORM_PAGE, form=answer)
    md = write_document(parse(PAGE, FORMS.model_copy(update={"strip_furniture": True}), client).document)
    (call,) = client.of(FORM)
    assert "ACME" not in client.blobs[call.key()][call.inputs[-1]].decode()  # the draft has no furniture either
    assert md.startswith("Applicant name: Jane Doe") and md.rstrip().endswith("Date of birth: 1984-07-15")
    assert FLIP[1] in md and "ACME" not in md and "AC-12" not in md


def test_tier2_forms_off_keeps_the_hybrid_v2_hash_and_unknown_options_fail():
    hybrid = Config.load("configs/hybrid_v2.yaml")
    old = {
        k: v for k, v in hybrid.model_dump(mode="json").items() if not k.startswith("tier2_") and k != "select_options"
    }
    assert not hybrid.tier2_forms and not hybrid.tier2_form_options
    assert hybrid.config_hash() == sha256_hex(canonical_json(old))[:16]
    assert hybrid.model_copy(update={"tier2_forms": True}).config_hash() != hybrid.config_hash()
    frontier = Config.load("configs/frontier_v3.yaml")
    assert frontier.tier2_forms and frontier.tier2_form_options["min_word_coverage"] == 0.93
    with pytest.raises(ValueError, match="unknown tier2 form option"):
        Config.load("configs/hybrid_v2.yaml", ["tier2_form_options.min_coverage=0.9"])


def test_tier2_tables_off_keeps_the_hybrid_v2_hash_and_unknown_options_fail():
    hybrid = Config.load("configs/hybrid_v2.yaml")
    old = {
        k: v for k, v in hybrid.model_dump(mode="json").items() if not k.startswith("tier2_") and k != "select_options"
    }
    assert not hybrid.tier2_tables and hybrid.config_hash() == sha256_hex(canonical_json(old))[:16]
    assert hybrid.model_copy(update={"tier2_tables": True}).config_hash() != hybrid.config_hash()
    assert V3.tier2_tables and V3.tier2_table_options["n_strips"] == 3
    assert (V3.tier2_chart_options["model"], V3.tier2_form_options["model"]) == ("122b", "27b")
    with pytest.raises(ValueError, match="unknown tier2 table option"):
        Config.load("configs/hybrid_v2.yaml", ["tier2_table_options.strips=4"])


# ── per-chart crops ──


def test_chart_boxes_are_deduplicated_in_reading_order_and_their_union_measured():
    inner, left, right, low = (2, 12, 48, 58), (0, 10, 50, 60), (60, 0, 110, 50), (0, 300, 50, 360)
    assert _dedup_chart_boxes([low, right, left, inner]) == [left, right, low]  # one band: left to right
    rp = RenderedPage(image=Image.new("RGB", (100, 100)), dpi=72, width_pt=100, height_pt=100)
    assert _union_share(rp, [(0, 0, 50, 50), (25, 25, 75, 75)]) == pytest.approx(0.5625)
    assert _union_share(rp, [(10, 10, 30, 30), (90, 90, 150, 150)]) == pytest.approx(0.81)


def test_several_large_charts_are_read_one_zoomed_crop_each():
    answers = (f"Revenue by quarter\n{QWEN_TABLE}", f"Units by market\n{UNITS_TABLE}")
    client = ScriptedClient(MISSED_CHART_PAGE, pp=TWO_CHARTS, singles=answers)
    doc = parse(PAGE, FRONTIER, client).document
    calls = client.of(SINGLE)
    assert not client.of(TIER2) and len(calls) == 2
    assert all(c.model == qwen_tier2.QWEN35_122B_A10B_FP8 and c.params["n_images"] == 2 for c in calls)
    sizes = [Image.open(io.BytesIO(client.blobs[c.key()][c.inputs[1]])).size for c in calls]
    assert sizes == [(832, 502), (832, 512)]  # each box padded by 6%, the top chart first; the repeat is not read
    assert SALES_TABLE in client.blobs[calls[0].key()][calls[0].inputs[-1]].decode()  # the page's draft tables
    assert write_document(doc).endswith(f"Revenue by quarter\n{QWEN_TABLE}\n\nUnits by market\n{UNITS_TABLE}")
    assert doc.pages[0].cost.calls == 3
    client = ScriptedClient(MISSED_CHART_PAGE, pp=TWO_CHARTS, pending={SINGLE})
    waiting = parse(PAGE, FRONTIER, client)
    assert waiting.document is None and waiting.pending == [c.key() for c in client.of(SINGLE)]
    assert len(waiting.pending) == 2  # all charts are asked for in one batch


def test_charts_covering_little_of_the_page_keep_one_read_of_their_union():
    pp = [
        {"label": "chart", "score": 0.9, "box": [100, 200, 400, 400]},
        {"label": "chart", "score": 0.9, "box": [450, 200, 750, 400]},
    ]
    client = ScriptedClient(MISSED_CHART_PAGE, tier2=f"Revenue by quarter\n{QWEN_TABLE}", pp=pp)
    md = write_document(parse(PAGE, FRONTIER, client).document)
    (call,) = client.of(TIER2)
    assert not client.of(SINGLE) and call.model == qwen_tier2.QWEN35_122B_A10B_FP8
    assert Image.open(io.BytesIO(client.blobs[call.key()][call.inputs[1]])).size == (752, 332)  # both boxes: 13%
    assert md.endswith(f"Revenue by quarter\n{QWEN_TABLE}")


# ── table strips ──


def test_a_cut_off_table_read_is_re_read_in_three_strips_and_stitched():
    client = ScriptedClient("", strips=LEDGER_STRIPS, ocria_finish="length")
    doc = parse(STRIPED, OCRIA_TABLES, client).document
    md = write_document(doc)
    calls = client.of(STRIP)
    assert len(calls) == 3 and all(c.model == qwen_tier2.QWEN35_122B_A10B_FP8 for c in calls)
    assert all(c.params["max_tokens"] == 16384 and c.params["enable_thinking"] is False for c in calls)
    heights = [Image.open(io.BytesIO(client.blobs[c.key()][c.inputs[0]])).height for c in calls]
    assert heights == [444, 520, 444]  # thirds of the 1100-pixel page, overlapping by 14% of it
    assert _ledger(1, 15) in md and md.startswith(PROSE_TOP) and "31.7" not in md
    assert (doc.pages[0].cost.calls, doc.pages[0].cost.gpu_seconds) == (4, 17.0)
    client = ScriptedClient("", strips=LEDGER_STRIPS, ocria_finish="length", pending={STRIP})
    waiting = parse(STRIPED, OCRIA_TABLES, client)
    assert waiting.document is None and waiting.pending == [c.key() for c in client.of(STRIP)]
    assert len(waiting.pending) == 3  # all strips are asked for in one batch
    for strips, finish in [((LEDGER_STRIPS[0], "", LEDGER_STRIPS[2]), "length"), (LEDGER_STRIPS, None)]:
        client = ScriptedClient("", strips=strips, ocria_finish=finish)  # an unusable strip; a complete scan read
        assert "31.7" in write_document(parse(STRIPED, OCRIA_TABLES, client).document)
        assert len(client.of(STRIP)) == (3 if finish else 0)


def _item_recall(md, words):  # stands in for the PDF word recall of a ledger whose text layer holds Items 1-15
    return sum(bool(re.search(rf"\bItem {i}\b", md)) for i in range(1, 16)) / 15


@pytest.mark.parametrize(
    ("healthy", "page_hi", "last_strip", "taken"),
    [
        (True, 10, _ledger(10, 15), True),  # recall 0.67 -> 1.0
        (True, 10, _ledger(9, 10), False),  # 0.67 -> 0.67: under min_recall_gain
        (True, 12, _ledger(10, 15), None),  # 0.80: over min_recall_trigger, never re-read
        (False, 10, _ledger(10, 15), None),  # no healthy text layer and a complete read: never re-read
    ],
)
def test_a_low_recall_table_page_takes_the_strips_only_when_they_recover_pdf_words(
    monkeypatch, healthy, page_hi, last_strip, taken
):
    from docparse.fixes import verify

    monkeypatch.setattr(verify, "has_healthy_text_layer", lambda rp: healthy)
    monkeypatch.setattr(verify, "pdf_word_recall", _item_recall)
    page = f'<div data-bbox="50 80 950 900" data-label="Table">{_ledger(1, page_hi)}</div>'
    client = ScriptedClient(page, strips=(_ledger(1, 6), _ledger(5, 10), last_strip))
    doc = parse(STRIPED, TABLES, client).document
    assert len(client.of(STRIP)) == (0 if taken is None else 3)
    assert ("Item 15" in write_document(doc)) is bool(taken)
    if taken:
        assert doc.pages[0].cost.calls == 4  # Chandra and the three strips
    else:  # Chandra's page, as if Tier-2 tables were off
        assert doc.pages == parse(STRIPED, BASE, ScriptedClient(page)).document.pages


def test_a_page_already_escalated_to_tier2_charts_is_not_re_escalated_to_tier2_tables(monkeypatch):
    from docparse.fixes import verify

    monkeypatch.setattr(verify, "has_healthy_text_layer", lambda rp: True)
    monkeypatch.setattr(verify, "pdf_word_recall", _item_recall)
    page = (
        '<div data-bbox="50 50 400 250" data-label="Figure"><img alt="Bar chart of monthly totals"/></div>'
        f'<div data-bbox="50 300 950 900" data-label="Table">{_ledger(1, 10)}</div>'
    )
    cfg = FRONTIER.model_copy(update={"tier2_tables": True, "tier2_table_options": V3.tier2_table_options})
    client = ScriptedClient(page, tier2="Decorative diagram without numeric values.", strips=LEDGER_STRIPS)
    doc = parse(STRIPED, cfg, client).document
    assert len(client.of(TIER2)) == 1
    assert len(client.of(STRIP)) == 0
    assert "Item 10" in write_document(doc) and "Item 15" not in write_document(doc)


def _chart_opts(**extra) -> Config:
    return FRONTIER.model_copy(update={"tier2_chart_options": {**V3.tier2_chart_options, **extra}})


def test_draft_format_formats_only_the_chart_models_draft(monkeypatch):
    import docparse.pipeline as pipeline

    real = pipeline._ocria_markdown

    def ocria_md(pdf_bytes, index, rp, read, key, config):  # a cell the PDF draws a line under
        md, tables, measured = real(pdf_bytes, index, rp, read, key, config)
        if config.format_options.get("underline"):
            tables = [t.replace("<td>Q1</td>", "<td><u>Q1</u></td>") for t in tables]
        return md, tables, measured

    monkeypatch.setattr(pipeline, "_ocria_markdown", ocria_md)
    for opts, underlined in (({}, False), ({"draft_format": {"underline": True}}, True)):
        client = ScriptedClient(CHART_PAGE, tier2=f"Revenue by quarter\n{QWEN_TABLE}")
        md = write_document(parse(PAGE, _chart_opts(**opts), client).document)
        (call,) = client.of(TIER2)
        assert ("<u>Q1</u>" in client.blobs[call.key()][call.inputs[-1]].decode()) is underlined
        assert "<u>" not in md and "12.5" in md  # the page itself keeps its own format options


def test_all_crops_off_keeps_the_charts_that_were_read():
    answers = (f"Revenue by quarter\n{QWEN_TABLE}", "Decorative diagram without numeric values.")
    for opts, kept in (({}, False), ({"all_crops": False}, True)):
        client = ScriptedClient(MISSED_CHART_PAGE, pp=TWO_CHARTS, singles=answers)
        md = write_document(parse(PAGE, _chart_opts(**opts), client).document)
        assert len(client.of(SINGLE)) == 2
        assert ("12.5" in md) is kept


def test_table_figures_off_leaves_a_figure_without_chart_words_alone():
    page = (
        f'<div data-bbox="50 40 950 120" data-label="Text"><p>{PROSE_TOP}</p></div>'
        '<div data-bbox="100 200 900 600" data-label="Figure"><img alt="Quarterly figures"/>'
        "<table><tr><th>Quarter</th><th>Revenue</th></tr><tr><td>Q1</td><td>77.7</td></tr></table></div>"
    )
    pp = [{"label": "chart", "score": 0.9, "box": [85, 220, 765, 660]}]
    for opts, replaced in (({}, True), ({"table_figures": False}, False)):
        client = ScriptedClient(page, tier2=f"Revenue by quarter\n{QWEN_TABLE}", pp=pp)
        md = write_document(parse(PAGE, _chart_opts(**opts), client).document)
        assert "12.5" in md and ("77.7" not in md) is replaced
