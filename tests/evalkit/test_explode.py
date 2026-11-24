import json

import pytest

from evalkit.explode import explode, split_example_id


def _rec(example_id, md="x"):
    return {"request": {"example_id": example_id}, "output": {"markdown": md}}


def _write(folder, subset, records, raw_lines=()):
    folder.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in records] + list(raw_lines)
    (folder / f"{subset}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_counts_layout_and_content(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _write(src, "text", [_rec("text/a", "héllo"), _rec("text/b")], raw_lines=["", "   "])
    _write(src, "charts", [_rec("charts/(Web)_x p1")])
    assert explode(src, out, "ocria_1") == {"text": 2, "charts": 1}
    got = json.loads((out / "text" / "ocria_1" / "a.result.json").read_text(encoding="utf-8"))
    assert got == _rec("text/a", "héllo")
    meta = json.loads((out / "charts" / "ocria_1" / "_metadata.json").read_text())
    assert meta["pipeline"] == {
        "pipeline_name": "ocria_1",
        "provider_name": "ocria_1",
        "product_type": "parse",
        "config": {},
    }
    assert (out / "charts" / "ocria_1" / "(Web)_x p1.result.json").exists()


@pytest.mark.parametrize(
    "bad",
    ["../../x", "/abs/y", "text/a/b", "text/..", "text/", "nosuchsubset/a", "text/.hidden", "text\\a", 7, None],
)
def test_hostile_ids_are_rejected_with_line_number(tmp_path, bad):
    src, out = tmp_path / "src", tmp_path / "out"
    _write(src, "text", [_rec("text/ok"), _rec(bad)])
    with pytest.raises(ValueError, match=r"text\.jsonl:2"):
        explode(src, out, "ocria_1")
    assert not out.exists()  # validation runs before any write
    assert sorted(p.name for p in tmp_path.iterdir()) == ["src"]


def test_missing_request_field(tmp_path):
    src = tmp_path / "src"
    _write(src, "text", [{"output": {}}])
    with pytest.raises(ValueError, match=r"text\.jsonl:1"):
        explode(src, tmp_path / "out", "ocria_1")


def test_malformed_json_line(tmp_path):
    src = tmp_path / "src"
    _write(src, "text", [_rec("text/a")], raw_lines=["{not json"])
    with pytest.raises(ValueError, match=r"text\.jsonl:2"):
        explode(src, tmp_path / "out", "ocria_1")


def test_record_in_wrong_file_and_duplicates(tmp_path):
    src = tmp_path / "src"
    _write(src, "text", [_rec("charts/a")])
    with pytest.raises(ValueError, match="wrong file"):
        explode(src, tmp_path / "out", "ocria_1")
    _write(src, "text", [_rec("text/a"), _rec("text/a")])
    with pytest.raises(ValueError, match="duplicate"):
        explode(src, tmp_path / "out", "ocria_1")


def test_bad_pipeline_name_and_empty_dir(tmp_path):
    (tmp_path / "src").mkdir()
    with pytest.raises(ValueError):
        explode(tmp_path / "src", tmp_path / "out", "../evil")
    with pytest.raises(ValueError, match="no <subset>.jsonl"):
        explode(tmp_path / "src", tmp_path / "out", "ocria_1")


def test_split_example_id():
    assert split_example_id("forms/blank__x--p2__blank") == ("forms", "blank__x--p2__blank")
