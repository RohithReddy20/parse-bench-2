import huggingface_hub
import pytest

from evalkit import fetch, pins


@pytest.fixture
def calls(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCPARSE_DATA", str(tmp_path))
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    seen: list[dict] = []

    def fake(**kwargs):
        seen.append(kwargs)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    return seen


def test_ground_truth_uses_pins(calls, tmp_path):
    dest = fetch.fetch_ground_truth()
    assert dest == tmp_path / "gt" / pins.DATA_REVISION
    assert calls[0]["repo_id"] == pins.DATA_REPO
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["revision"] == pins.DATA_REVISION
    assert calls[0]["local_dir"] == str(dest)
    assert (dest / ".complete").read_text() == pins.DATA_REVISION


def test_second_call_does_not_download(calls):
    fetch.fetch_ground_truth()
    fetch.fetch_ground_truth()
    assert len(calls) == 1


def test_published_only_fetches_one_pipeline(calls):
    dest = fetch.fetch_published("ocria_1")
    assert calls[0]["allow_patterns"] == ["results/ocria_1/*"]
    assert calls[0]["revision"] == pins.RESULTS_REVISION
    assert dest.parts[-2:] == ("results", "ocria_1")


@pytest.mark.parametrize("name", ["../x", "a/b", "", ".hidden", "x y"])
def test_published_rejects_bad_names(calls, name):
    with pytest.raises(ValueError):
        fetch.fetch_published(name)
    assert calls == []


def test_ocria_code_excludes_weights(calls):
    fetch.fetch_ocria_code()
    assert calls[0]["repo_type"] == "model"
    assert not any("safetensors" in p for p in calls[0]["allow_patterns"])
    assert "ocria_grounding/*" in calls[0]["allow_patterns"]


def test_failed_download_leaves_no_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCPARSE_DATA", str(tmp_path))
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    attempts = []

    def boom(**kwargs):
        attempts.append(1)
        raise OSError("connection reset")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    with pytest.raises(OSError):
        fetch.fetch_ground_truth()
    assert len(attempts) == fetch.ATTEMPTS
    assert not (tmp_path / "gt" / pins.DATA_REVISION / ".complete").exists()


def test_retry_then_success(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCPARSE_DATA", str(tmp_path))
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    state = {"n": 0}

    def flaky(**kwargs):
        state["n"] += 1
        if state["n"] < 3:
            raise OSError("reset")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", flaky)
    assert (fetch.fetch_ground_truth() / ".complete").exists()
