"""Tier-2 chart reads and form refinement (Qwen3.5-27B-FP8) wired into the Chandra pipeline, against a scripted
model client."""

import io

import pytest
from PIL import Image

from docparse.core.config import Config
from docparse.core.hashing import canonical_json, sha256_hex
from docparse.models.adapters import ocria, ppdoclayout, qwen_tier2, vlm_specialists
from docparse.models.calls import ModelResult, Pending
from docparse.pipeline import parse
from docparse.write.markdown import write_document

CHANDRA, OCRIA, PP, TIER2, FORM = (
    vlm_specialists.CHANDRA_PROMPT_ID,
    ocria.PROMPT_ID,
    ppdoclayout.PROMPT_ID,
    qwen_tier2.CHART_PROMPT_ID,
    qwen_tier2.FORM_REFINE_PROMPT_ID,
)
GPU_SECONDS = {CHANDRA: 1.0, OCRIA: 2.0, PP: 0.0, TIER2: 4.0, FORM: 3.0}
BASE = Config(page_model=ocria.OWN, page_runtime="vllm-0.34.0", prompt_sha="t", reader="chandra_hybrid", layout="none")
FRONTIER = BASE.model_copy(
    update={"tier2_charts": True, "tier2_chart_options": Config.load("configs/frontier_v3.yaml").tier2_chart_options}
)
FORMS = BASE.model_copy(
    update={"tier2_forms": True, "tier2_form_options": Config.load("configs/frontier_v3.yaml").tier2_form_options}
)

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


def _png(size=(850, 1100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


PAGE = _png()


class ScriptedClient:
    """Answers each model by prompt id; prompt ids in `pending` are not in the cache yet."""

    def __init__(self, chandra, tier2="", pp=(), pending=(), finish="stop", form=""):
        self.answers = {
            CHANDRA: {"answer": chandra, "finish_reason": "stop"},
            OCRIA: {"markdown": f"{PROSE_TOP}\n\n{OCRIA_TABLE}"},
            PP: {"detections": list(pp)},
            TIER2: {"answer": tier2, "finish_reason": finish},
            FORM: {"answer": form, "finish_reason": "stop"},
        }
        self.pending, self.calls, self.blobs = set(pending), [], {}

    def get(self, call, blobs, meta=None):
        self.calls.append(call)
        self.blobs[call.key()] = blobs
        if call.prompt_id in self.pending:
            return Pending(key=call.key())
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
    dump = {k: v for k, v in hybrid.model_dump(mode="json").items() if not k.startswith("tier2_")}
    assert not hybrid.tier2_charts and hybrid.config_hash() == sha256_hex(canonical_json(dump))[:16]
    frontier = Config.load("configs/frontier_v3.yaml")
    assert frontier.tier2_charts and frontier.config_hash() != hybrid.config_hash()
    off = {"tier2_charts": False, "tier2_chart_options": {}, "tier2_forms": False, "tier2_form_options": {}}
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
    old = {k: v for k, v in hybrid.model_dump(mode="json").items() if not k.startswith("tier2_")}
    assert not hybrid.tier2_forms and not hybrid.tier2_form_options
    assert hybrid.config_hash() == sha256_hex(canonical_json(old))[:16]
    assert hybrid.model_copy(update={"tier2_forms": True}).config_hash() != hybrid.config_hash()
    frontier = Config.load("configs/frontier_v3.yaml")
    assert frontier.tier2_forms and frontier.tier2_form_options["min_word_coverage"] == 0.93
    with pytest.raises(ValueError, match="unknown tier2 form option"):
        Config.load("configs/hybrid_v2.yaml", ["tier2_form_options.min_coverage=0.9"])
