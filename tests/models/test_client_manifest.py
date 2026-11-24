import pytest

from docparse.__main__ import main as docparse_main
from docparse.models.cache import Cache
from docparse.models.calls import CacheMiss, ModelResult, Pending
from docparse.models.client import CacheOnlyClient, EndpointClient, ReplayClient
from docparse.models.manifest import ManifestWriter, Rate, format_plan, load_blobs, plan, read_manifest
from tests.models.test_calls_cache import call, result


def test_cache_only_client_records_each_miss_once(tmp_path):
    cache, manifest = Cache(tmp_path / "cache"), ManifestWriter(tmp_path / "m")
    client, c = CacheOnlyClient(cache, manifest), call()
    blobs = {"1" * 64: b"PNG"}
    assert client.get(c, blobs, {"page": 0}) == Pending(key=c.key())
    assert client.get(c, blobs) == Pending(key=c.key())
    rows = list(read_manifest(tmp_path / "m"))
    assert len(rows) == 1 and rows[0][0] == c and rows[0][1] == {"page": 0}
    assert load_blobs(tmp_path / "m", c) == blobs
    cache.put(result(c), c)
    assert isinstance(client.get(c, blobs), ModelResult)
    assert len(manifest) == 1


def test_manifest_reopen_keeps_dedupe_and_needs_blobs(tmp_path):
    c = call()
    assert ManifestWriter(tmp_path).add(c, {"1" * 64: b"x"}) is True
    again = ManifestWriter(tmp_path)
    assert again.add(c, {}) is False
    other = call(inputs=["4" * 64])
    with pytest.raises(ValueError, match="no blob"):
        again.add(other, {})
    with pytest.raises(ValueError, match="not a blob hash"):
        again.add(call(inputs=["../x"]), {"../x": b"evil"})


def test_replay_client_raises_on_miss(tmp_path):
    cache, c = Cache(tmp_path), call()
    with pytest.raises(CacheMiss) as err:
        ReplayClient(cache).get(c, {})
    assert err.value.key == c.key()
    cache.put(result(c), c)
    assert ReplayClient(cache).get(c, {}).output == {"answer": "hello"}


def test_endpoint_client_calls_executor_once_per_key(tmp_path):
    seen = []

    def executor(c, blobs):
        seen.append(c.key())
        return {"answer": "from gpu", "_tokens_out": 7, "_gpu_seconds": 2.0, "_gpu_type": "L4"}

    client, c = EndpointClient(Cache(tmp_path), executor), call()
    first = client.get(c, {"1" * 64: b"x"})
    second = client.get(c, {"1" * 64: b"x"})
    assert seen == [c.key()]
    assert first.output == {"answer": "from gpu"} == second.output
    assert first.tokens_out == 7 and first.gpu_seconds == 2.0 and first.gpu_type == "L4"


def test_plan_counts_new_against_cached(tmp_path, capsys):
    cache, manifest = Cache(tmp_path / "cache"), ManifestWriter(tmp_path / "m")
    calls = [call(inputs=[f"{i}" * 64]) for i in range(1, 6)]
    for c in calls:
        manifest.add(c, {c.inputs[0]: b"x"})
    for c in calls[:2]:
        cache.put(result(c), c)
    (row,) = plan(tmp_path / "m", cache, {"llamaindex/OCRia-1": Rate(seconds_per_call=12.0, usd_per_hour=3.0)})
    assert (row.new_calls, row.cached_calls) == (3, 2)
    assert row.est_gpu_seconds == 36.0 and row.est_usd == pytest.approx(0.03)
    assert "llamaindex/OCRia-1@aaaaaaaaaaaa" in format_plan([row])
    code = docparse_main(["cache", "plan", "--manifest", str(tmp_path / "m"), "--cache", str(tmp_path / "cache"),
                          "--rate", "llamaindex/OCRia-1:12:3"])  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0 and "total" in out and "0.03" in out
