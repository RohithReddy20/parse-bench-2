from evalkit.pilot import pilot_ids
from evalkit.splits import fold_of, load_split


def test_pilot_sample_is_fixed_stratified_and_from_the_development_part():
    ids = list(load_split())
    sample = pilot_ids(ids)
    assert sample == pilot_ids(list(reversed(ids))) and len(sample) == 46  # 10 subsets x 4 + 3 x 2
    assert all(fold_of(i) == "dev" for i in sample)
    per_subset = {}
    for i in sample:
        per_subset[i.split("/")[0]] = per_subset.get(i.split("/")[0], 0) + 1
    assert len(per_subset) == 13 and per_subset["headings"] == 2 and per_subset["charts"] == 4
