import importlib.util
import re

from evalkit import paths, pins


def test_pins_are_full_shas():
    for v in (pins.HARNESS_COMMIT, pins.DATA_REVISION, pins.RESULTS_REVISION, pins.OCRIA_REVISION):
        assert re.fullmatch(r"[0-9a-f]{40}", v)


def test_harness_has_ocria_pipeline():
    # the ocria_1 provider landed after v2.2.1, so finding it proves the pin is past that release
    assert importlib.util.find_spec("parse_bench.inference.providers.parse.ocria_1")


def test_board_overall_is_mean_of_dimensions():
    dims = [v for k, v in pins.BOARD_OCRIA.items() if k != "overall"]
    assert len(dims) == 6
    assert abs(sum(dims) / 6 - pins.BOARD_OCRIA["overall"]) < 0.01


def test_data_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCPARSE_DATA", str(tmp_path))
    assert paths.data_root() == tmp_path
    assert paths.gt_dir() == tmp_path / "gt" / pins.DATA_REVISION
    assert paths.published_dir().is_relative_to(tmp_path)


def test_data_root_default(monkeypatch):
    monkeypatch.delenv("DOCPARSE_DATA", raising=False)
    assert paths.data_root().name == "docparse-v2"
