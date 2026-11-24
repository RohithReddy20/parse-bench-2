from docparse.models.adapters import heron
from docparse.models.manifest import ManifestWriter, load_blobs, read_manifest
from evalkit.detect import build_detect_manifest
from tests.models.test_calls_cache import call


def test_detect_manifest_reuses_the_page_blobs(tmp_path):
    pages = ManifestWriter(tmp_path / "pages")
    calls = [call(inputs=[f"{i:x}" * 64]) for i in (1, 2, 3)]
    for c in calls:
        pages.add(c, {c.inputs[0]: b"PNG" + c.inputs[0][:1].encode()}, {"page": 0})
    n = build_detect_manifest(tmp_path / "pages", tmp_path / "detect")
    assert n == 3 and build_detect_manifest(tmp_path / "pages", tmp_path / "detect") == 3  # idempotent
    rows = list(read_manifest(tmp_path / "detect"))
    assert [c.model for c, _ in rows] == [heron.HERON] * 3
    assert [c.inputs for c, _ in rows] == [c.inputs for c in calls] and rows[0][1] == {"page": 0}
    assert (tmp_path / "detect" / "blobs").is_symlink()
    assert load_blobs(tmp_path / "detect", rows[1][0]) == {calls[1].inputs[0]: b"PNG2"}
