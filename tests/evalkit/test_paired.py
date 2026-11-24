import pytest

from evalkit.ledger import CaseRow
from evalkit.paired import paired_bootstrap


def _cases(scores):
    return [CaseRow(f"layout/doc{i}_p1", "layout", "layout_grounding_pass_rate", s, True) for i, s in enumerate(scores)]


def test_consistent_gain_is_certain_and_noise_is_not():
    base = _cases([0.5] * 40)
    better = _cases([0.6] * 40)
    out = paired_bootstrap(base, better, "layout", fold="all", samples=300)
    assert out["gain"] == pytest.approx(10.0) and out["p_positive"] == 1.0 and out["pages"] == 40
    noisy = _cases([0.5 + (0.2 if i % 2 else -0.2) for i in range(40)])
    out = paired_bootstrap(base, noisy, "layout", fold="all", samples=300)
    assert out["gain"] == pytest.approx(0.0, abs=1e-9) and 0.2 < out["p_positive"] < 0.8
    assert out["low"] < 0 < out["high"]


def test_no_shared_pages_raises():
    with pytest.raises(ValueError):
        paired_bootstrap(_cases([0.5]), _cases([0.5]), "charts", fold="all")
