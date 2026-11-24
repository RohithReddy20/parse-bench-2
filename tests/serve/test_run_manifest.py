import json

from serve.run_manifest import import_results, run_manifest

from docparse.models.cache import Cache
from docparse.models.manifest import ManifestWriter
from tests.models.test_calls_cache import call


def _manifest(tmp_path, n=5):
    writer = ManifestWriter(tmp_path / "m")
    calls = [call(inputs=[f"{i:x}" * 64]) for i in range(1, n + 1)]
    for c in calls:
        writer.add(c, {c.inputs[0]: c.inputs[0][:3].encode()}, {"i": c.inputs[0][0]})
    return tmp_path / "m", calls


def test_runs_every_call_and_imports(tmp_path):
    mdir, calls = _manifest(tmp_path)
    results = tmp_path / "out" / "results.jsonl"
    stats = run_manifest(mdir, results, lambda c, blobs: {"answer": blobs[c.inputs[0]].decode(), "_tokens_out": 3})
    assert (stats.total, stats.done, stats.failed, stats.skipped) == (5, 5, 0, 0)
    cache = Cache(tmp_path / "cache")
    assert import_results(mdir, results, cache) == 5
    assert import_results(mdir, results, cache) == 0
    hit = cache.get(calls[0].key())
    assert hit.output == {"answer": "111"} and hit.tokens_out == 3


def test_resumes_without_rerunning_finished_calls(tmp_path):
    mdir, calls = _manifest(tmp_path)
    results = tmp_path / "results.jsonl"
    ran = []

    def dies_on_third(c, blobs):
        if len(ran) == 2:
            raise KeyboardInterrupt  # the session is killed
        ran.append(c.key())
        return {"answer": "x"}

    try:
        run_manifest(mdir, results, dies_on_third)
    except KeyboardInterrupt:
        pass
    with results.open("a") as fh:
        fh.write('{"key": "torn-li')  # a half-written last line
    second = []
    stats = run_manifest(mdir, results, lambda c, b: second.append(c.key()) or {"answer": "y"})
    assert stats.skipped == 2 and stats.done == 3
    assert set(second) == {c.key() for c in calls} - set(ran)


def test_failed_call_is_recorded_and_retried_next_run(tmp_path):
    mdir, calls = _manifest(tmp_path, n=3)
    results = tmp_path / "results.jsonl"
    bad = calls[1].key()

    def flaky(c, blobs):
        if c.key() == bad:
            raise RuntimeError("CUDA out of memory")
        return {"answer": "ok"}

    stats = run_manifest(mdir, results, flaky)
    assert (stats.done, stats.failed) == (2, 1)
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    assert [r for r in rows if not r["ok"]][0]["error"].startswith("RuntimeError: CUDA")
    cache = Cache(tmp_path / "cache")
    assert import_results(mdir, results, cache) == 2
    stats = run_manifest(mdir, results, lambda c, b: {"answer": "ok now"})
    assert (stats.skipped, stats.done) == (2, 1)
    assert import_results(mdir, results, cache) == 1
    assert cache.get(bad).output == {"answer": "ok now"}


def test_parallel_workers(tmp_path):
    mdir, _calls = _manifest(tmp_path, n=5)
    stats = run_manifest(mdir, tmp_path / "r.jsonl", lambda c, b: {"answer": "x"}, max_workers=4)
    assert stats.done == 5
    assert len((tmp_path / "r.jsonl").read_text().splitlines()) == 5
