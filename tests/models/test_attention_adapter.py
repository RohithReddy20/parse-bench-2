from docparse.models.adapters.attention import HEADS, attention_call, read_attention
from docparse.models.adapters.ocria import OWN
from docparse.models.calls import ModelResult


def test_the_call_is_identified_by_page_answer_model_and_heads():
    call, blobs = attention_call("pagehash", "# Title\n\nSome text", OWN, "promptsha")
    again, _ = attention_call("pagehash", "# Title\n\nSome text", OWN, "promptsha")
    other, _ = attention_call("pagehash", "# Title\n\nOther text", OWN, "promptsha")
    assert call.key() == again.key() != other.key()
    assert call.inputs[0] == "pagehash" and blobs == {call.inputs[1]: b"# Title\n\nSome text"}
    assert call.params["heads"] == HEADS and call.prompt_id == "ocria-1/attention"


def test_read_attention_rebuilds_dense_maps():
    out = {"grid": [2, 3], "idx": [[4, 5], []], "val": [[0.75, 0.25], []], "has": [True, False]}
    att = read_attention(ModelResult(key="k", output=out))
    maps, has = att.dense()
    assert maps.shape == (2, 2, 3) and has.tolist() == [True, False]
    assert maps[0].tolist() == [[0.0, 0.0, 0.0], [0.0, 0.75, 0.25]] and not maps[1].any()
