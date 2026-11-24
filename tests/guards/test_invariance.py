"""Rename invariance and isolation (design §8.4), on real benchmark pages replayed from the cache.

The parser must give the same output for the same bytes under any file name, and a page parsed alone must
equal the same page parsed inside a larger run. Both need the stand-in readings (python -m evalkit standin).
"""

import json
import random
import shutil

import pytest

from docparse.bench.record import result_record
from docparse.core.config import Config
from docparse.models.cache import Cache
from docparse.models.calls import CacheMiss
from docparse.models.client import ReplayClient
from docparse.pipeline import parse
from evalkit import paths
from evalkit.explode import split_example_id
from evalkit.run import list_examples, run_pipeline
from evalkit.standin import source_file

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def config():
    return Config.load("configs/standin.yaml")


@pytest.fixture(scope="module")
def sample(gt_dir):
    ids = list_examples(gt_dir)
    random.Random(20261124).shuffle(ids)
    by_subset: dict[str, list[str]] = {}
    for example_id in ids:
        by_subset.setdefault(example_id.split("/")[0], []).append(example_id)
    picked = [i for group in by_subset.values() for i in group[:4]]  # every subset is represented
    return sorted(picked)[:50]


def _record(path, config, cache, example_id="x/y"):
    try:
        doc = parse(path.read_bytes(), config, ReplayClient(cache)).document
    except CacheMiss:
        pytest.skip("stand-in readings not imported (python -m evalkit standin)")
    return json.dumps(result_record(example_id, "p", doc), ensure_ascii=False, sort_keys=True)


def test_rename_invariance(gt_dir, config, sample, tmp_path):
    cache = Cache(paths.cache_dir())
    assert len(sample) == 50
    rng = random.Random(7)
    for example_id in sample:
        subset, stem = split_example_id(example_id)
        src = source_file(gt_dir, subset, stem)
        renamed = tmp_path / f"{rng.getrandbits(64):016x}{src.suffix}"  # flat folder, random name
        shutil.copyfile(src, renamed)
        assert _record(renamed, config, cache) == _record(src, config, cache), example_id


def test_isolation(gt_dir, config, sample, tmp_path):
    """A page parsed alone equals the same page inside a 200-page run."""
    cache_root = paths.cache_dir()
    twenty = sample[:20]
    everything = sorted(set(list_examples(gt_dir)[::20][:180]) | set(twenty))
    alone, together = tmp_path / "alone", tmp_path / "together"
    for example_id in twenty:
        one = run_pipeline(config, "p", alone, gt_dir, cache_root, [example_id], workers=1)
        if one.errors and "CacheMiss" in one.errors[0][1]:
            pytest.skip("stand-in readings not imported (python -m evalkit standin)")
        assert one.written == 1, one.errors
    summary = run_pipeline(config, "p", together, gt_dir, cache_root, everything, workers=4)
    assert summary.errors == [] and summary.written == len(everything)
    for example_id in twenty:
        subset, stem = split_example_id(example_id)
        rel = f"{subset}/p/{stem}.result.json"
        assert (alone / rel).read_bytes() == (together / rel).read_bytes(), example_id
