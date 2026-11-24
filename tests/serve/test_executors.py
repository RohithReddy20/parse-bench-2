import base64
import io
import json

import httpx
import pytest
from PIL import Image
from serve.executors import ExecutorError, openai_chat_executor

from docparse.models.adapters import ocria
from docparse.models.cache import Cache
from docparse.models.client import EndpointClient
from docparse.pdf.render import render_pages
from tests.pdfgen import simple_pdf

PROMPT = "Transcribe this Page in markdown"


def _call():
    image = render_pages(simple_pdf("Executor page"))[0].image
    call, blobs = ocria.build_page_call(image, ocria.OWN, "vllm-0.31.0", "s" * 64)
    return call, dict(blobs)


def _ok(answer="```markdown\n# Hi\n```", finish="stop"):
    return httpx.Response(200, json={"choices": [{"message": {"content": answer}, "finish_reason": finish}],
                                     "usage": {"prompt_tokens": 8200, "completion_tokens": 12}})  # fmt: skip


def _executor(handler):
    return openai_chat_executor(
        "http://gpu:8000/v1/", "ocria-1", PROMPT, transport=httpx.MockTransport(handler), gpu_type="A100"
    )


def test_request_is_the_harness_request():
    seen = []

    def handler(request):
        seen.append(request)
        return _ok()

    call, blobs = _call()
    out = _executor(handler)(call, blobs)
    (request,) = seen
    assert str(request.url) == "http://gpu:8000/v1/chat/completions"
    body = json.loads(request.content)
    assert body["model"] == "ocria-1" and body["temperature"] == 0.0 and body["max_tokens"] == 24576
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    system, user = body["messages"]
    assert system == {"role": "system", "content": PROMPT}
    assert user["role"] == "user" and len(user["content"]) == 1  # the image is the only user content
    url = user["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == blobs[call.inputs[0]]
    assert out["answer"].startswith("```markdown") and out["finish_reason"] == "stop"
    assert (out["_tokens_in"], out["_tokens_out"], out["_gpu_type"], out["image_scale"]) == (8200, 12, "A100", 1.0)


def test_overflow_shrinks_the_output_budget_then_the_image():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:  # image + 24576 output tokens overflow the context
            return httpx.Response(
                400,
                text="This model's maximum context length is 32768 tokens. However, you passed 9000 input tokens and requested 24576 output tokens",
            )
        if len(bodies) == 2:  # the image alone is too large
            return httpx.Response(400, text="max_tokens must be at least 1, got -500")
        return _ok(finish="length")

    call, blobs = _call()
    out = _executor(handler)(call, blobs)
    assert [b["max_tokens"] for b in bodies] == [24576, 12288, 2048]

    def size(body):
        data = base64.b64decode(body["messages"][1]["content"][0]["image_url"]["url"].split(",", 1)[1])
        return Image.open(io.BytesIO(data)).size

    assert size(bodies[0]) == size(bodies[1])
    assert size(bodies[2])[0] == int(size(bodies[0])[0] * 0.4)  # second retry uses the second downscale step
    assert out["max_tokens"] == 2048 and out["image_scale"] == 0.4 and out["finish_reason"] == "length"


def test_fourth_failure_and_server_errors_raise():
    call, blobs = _call()
    always_over = _executor(lambda r: httpx.Response(400, text="max_tokens must be at least 1, got -1"))
    with pytest.raises(ExecutorError):
        always_over(call, blobs)
    with pytest.raises(ExecutorError, match="HTTP 500"):
        _executor(lambda r: httpx.Response(500, text="CUDA error"))(call, blobs)


def test_endpoint_client_caches_the_answer(tmp_path):
    hits = []

    def handler(request):
        hits.append(1)
        return _ok()

    call, blobs = _call()
    client = EndpointClient(Cache(tmp_path), _executor(handler))
    first, second = client.get(call, blobs), client.get(call, blobs)
    assert len(hits) == 1 and first.output == second.output
    assert first.tokens_in == 8200 and first.gpu_type == "A100" and "_tokens_in" not in first.output
    assert ocria.read_page_result(first).markdown == "# Hi"
