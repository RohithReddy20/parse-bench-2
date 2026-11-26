"""Tier-2 chart reads (Qwen3.5-27B-FP8) wired into the Chandra pipeline, against a scripted model client."""

import io

import pytest
from PIL import Image

from docparse.core.config import Config
from docparse.core.hashing import canonical_json, sha256_hex
from docparse.models.adapters import ocria, ppdoclayout, qwen_tier2, vlm_specialists
from docparse.models.calls import ModelResult, Pending
from docparse.pipeline import parse
from docparse.write.markdown import write_document

CHANDRA, OCRIA, PP, TIER2 = (
    vlm_specialists.CHANDRA_PROMPT_ID,
    ocria.PROMPT_ID,
    ppdoclayout.PROMPT_ID,
    qwen_tier2.CHART_PROMPT_ID,
)
GPU_SECONDS = {CHANDRA: 1.0, OCRIA: 2.0, PP: 0.0, TIER2: 4.0}
BASE = Config(page_model=ocria.OWN, page_runtime="vllm-0.34.0", prompt_sha="t", reader="chandra_hybrid", layout="none")
FRONTIER = BASE.model_copy(
    update={"tier2_charts": True, "tier2_chart_options": Config.load("configs/frontier_v3.yaml").tier2_chart_options}
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


def _png(size=(850, 1100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


PAGE = _png()


class ScriptedClient:
    """Answers each model by prompt id; prompt ids in `pending` are not in the cache yet."""

    def __init__(self, chandra, tier2="", pp=(), pending=(), finish="stop"):
        self.answers = {
            CHANDRA: {"answer": chandra, "finish_reason": "stop"},
            OCRIA: {"markdown": f"{PROSE_TOP}\n\n{OCRIA_TABLE}"},
            PP: {"detections": list(pp)},
            TIER2: {"answer": tier2, "finish_reason": finish},
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
    assert frontier.model_copy(update={"tier2_charts": False, "tier2_chart_options": {}}) == hybrid


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
