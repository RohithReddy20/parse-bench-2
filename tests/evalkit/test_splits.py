import pytest

from evalkit import splits
from evalkit.splits import build_split, check_shares, fold, fold_of, group_key, load_split


@pytest.mark.parametrize(
    "subset, stem, key",
    [
        ("forms", "blank__australia--02_WA-DLA1--p2__blank", "australia--02_wa-dla1"),
        ("forms", "filled__australia--02_WA-DLA1--p7__typed", "australia--02_wa-dla1"),
        ("fillable_forms", "f1040--2020__p1", "f1040--2020"),
        ("fillable_forms", "fw9__p2", "fw9"),
        (
            "charts",
            "(Web_version)_E-Government_Survey_2024_1392024_p101",
            "(web_version)_e-government_survey_2024_1392024",
        ),
        (
            "charts",
            "(Web_version)_E-Government_Survey_2024_1392024_p102",
            "(web_version)_e-government_survey_2024_1392024",
        ),
        ("layout", "annotated__annual-report_fy2023_07_en_p5", "annual-report_fy2023_07_en"),
        ("charts", "annual-report_fy2023_07_en_p12", "annual-report_fy2023_07_en"),
        ("layout", "text_core__TOS.2-columns-repeated-block", "tos.2-columns-repeated-block"),
        ("text", "text_simple__registration", "registration"),
        (
            "financial_tables",
            "2025-cisco-full-annual-report-2026-08-21_page48",
            "2025-cisco-full-annual-report-2026-08-21",
        ),
        ("insurance_filing_tables", "6313a48d4cefe4fd_p8", "6313a48d4cefe4fd"),
        ("tables", "JNPR.2011.page_82.pdf_110673_page1", "jnpr.2011"),
        ("tables", "Apple 10-k_page45", "apple 10-k"),
        ("tables", "SERFF_TX_random_pages 1_page402", "serff_tx_random_pages 1_page402"),
        ("headings", "case-0030-4c35d16ab4e0", "case-0030-4c35d16ab4e0"),
        ("dense_tables", "dense_table_418", "dense_table_418"),
    ],
)
def test_group_key(subset, stem, key):
    assert group_key(subset, stem) == key


def test_fold_is_deterministic_and_seeded():
    assert fold("some-document") == fold("some-document")
    assert {fold(f"doc-{i}") for i in range(200)} == {"dev", "check"}
    share = sum(fold(f"doc-{i}") == "check" for i in range(5000)) / 5000
    assert 0.17 < share < 0.23
    assert any(fold(f"doc-{i}", "a") != fold(f"doc-{i}", "b") for i in range(50))


def test_build_split_ignores_input_order_and_keeps_documents_together():
    ids = ["charts/report_p1", "charts/report_p2", "layout/annotated__report_p9", "text/text_simple__other"]
    a, b = build_split(ids), build_split(list(reversed(ids)))
    assert a == b and list(a) == sorted(ids)
    assert a["charts/report_p1"] == a["charts/report_p2"] == a["layout/annotated__report_p9"]


def test_fold_of_falls_back_to_the_hash_rule_for_new_cases():
    assert fold_of("charts/a-brand-new-report_p3") == fold(group_key("charts", "a-brand-new-report_p3"))


def test_committed_split_is_complete_and_stratified():
    split = load_split()
    assert len(split) == 4317
    shares = check_shares(split)
    assert len(shares) == 13 and all(0.15 <= s <= 0.25 for s in shares.values()), shares
    assert split == build_split(list(split))  # the file is exactly what the rule produces
    by_group = {}
    for example_id, f in split.items():
        subset, _, stem = example_id.partition("/")
        by_group.setdefault(group_key(subset, stem), set()).add(f)
    assert all(len(folds) == 1 for folds in by_group.values())
    assert splits.SEED == "docparse-v2-split-35"


@pytest.mark.data
def test_split_covers_exactly_the_dataset(gt_dir):
    from evalkit.run import list_examples

    assert set(list_examples(gt_dir)) == set(load_split())
