import os
import threading

import pytest

from docparse.models.cache import Cache
from docparse.models.calls import ModelCall, ModelRef, ModelResult

REF = ModelRef(repo="llamaindex/OCRia-1", revision="a" * 40, dtype="bf16")


def call(**over):
    base = dict(model=REF, runtime="vllm-0.31.0", prompt_id="ocria-1/page", prompt_sha="p" * 64,
                params={"temperature": 0.0, "max_tokens": 24576}, inputs=["1" * 64])  # fmt: skip
    base.update(over)
    return ModelCall(**base)


def result(c, text="hello", **kw):
    return ModelResult(key=c.key(), output={"answer": text}, **kw)


@pytest.mark.parametrize(
    "change",
    [
        {"model": ModelRef(repo="llamaindex/OCRia-1", revision="b" * 40, dtype="bf16")},
        {"model": ModelRef(repo="llamaindex/OCRia-1", revision="a" * 40, dtype="fp16")},
        {"runtime": "vllm-0.32.0"},
        {"prompt_id": "ocria-1/crop"},
        {"prompt_sha": "q" * 64},
        {"params": {"temperature": 0.0, "max_tokens": 24576, "sample_index": 1}},
        {"inputs": ["2" * 64]},
    ],
)
def test_key_changes_with_every_field(change):
    assert call(**change).key() != call().key()


def test_key_is_stable_across_dict_order():
    a = call(params={"temperature": 0.0, "max_tokens": 24576})
    b = call(params={"max_tokens": 24576, "temperature": 0.0})
    assert a.key() == b.key() and len(a.key()) == 64


def test_put_get_and_first_write_wins(tmp_path):
    cache, c = Cache(tmp_path), call()
    assert cache.get(c.key()) is None
    assert cache.put(result(c, "first", tokens_in=10, tokens_out=5, gpu_seconds=1.5, gpu_type="A100"), c) is True
    assert cache.put(result(c, "second"), c) is False
    assert cache.get(c.key()).output == {"answer": "first"}
    assert list(cache.keys()) == [c.key()]
    assert cache.usage() == {"calls": 1, "tokens_in": 10, "tokens_out": 5, "gpu_seconds": 1.5}
    assert cache.usage("other/model")["calls"] == 0
    assert Cache(tmp_path).get(c.key()).output == {"answer": "first"}  # survives a reopen


def test_result_key_must_match_call(tmp_path):
    with pytest.raises(ValueError):
        Cache(tmp_path).put(ModelResult(key="0" * 64, output={}), call())


def test_truncated_file_reads_as_absent_and_can_be_rewritten(tmp_path):
    cache, c = Cache(tmp_path), call()
    cache.put(result(c), c)
    path = tmp_path / "calls" / c.key()[:2] / f"{c.key()}.json.zst"
    path.write_bytes(path.read_bytes()[:10])
    assert cache.get(c.key()) is None
    assert cache.corrupt == [c.key()]
    assert cache.put(result(c, "again"), c) is True
    assert cache.get(c.key()).output == {"answer": "again"} and cache.corrupt == []


def test_interrupted_write_leaves_nothing_readable(tmp_path, monkeypatch):
    cache, c = Cache(tmp_path), call()

    def killed(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "link", killed)
    with pytest.raises(KeyboardInterrupt):
        cache.put(result(c), c)
    monkeypatch.undo()
    assert cache.get(c.key()) is None
    assert not list((tmp_path / "calls").rglob("*.tmp"))  # no stray temp file either
    assert cache.put(result(c), c) is True


def test_concurrent_writers_leave_one_valid_entry(tmp_path):
    c = call()
    outcomes = []

    def writer(i):
        outcomes.append(Cache(tmp_path).put(result(c, f"writer {i}"), c))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert outcomes.count(True) == 1
    assert Cache(tmp_path).get(c.key()).output["answer"].startswith("writer ")
    assert Cache(tmp_path).usage()["calls"] == 1


def test_snapshot_hash_tracks_the_key_set(tmp_path):
    cache = Cache(tmp_path)
    empty = cache.snapshot_hash()
    c1, c2 = call(), call(inputs=["3" * 64])
    cache.put(result(c1), c1)
    one = cache.snapshot_hash()
    cache.put(result(c2), c2)
    assert len({empty, one, cache.snapshot_hash()}) == 3


def test_bad_key_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        Cache(tmp_path).get("../../etc/passwd")
