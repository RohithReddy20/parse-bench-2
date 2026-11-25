from evalkit.external.forms_outside import (
    Haystack,
    bootstrap_gain,
    close_digit_gaps,
    dates_in,
    field_kind,
    found,
    normalize,
    response_tokens,
    sample_documents,
    score_items,
)


def test_response_tokens_and_kinds():
    tokens = response_tokens("Return exactly: premium=<number>; kind=<Owned|Leased|null>; on=<boolean>; d=<date>")
    assert tokens == {"premium": "number", "kind": "Owned|Leased|null", "on": "boolean", "d": "date"}
    assert field_kind("number", 12800) == "number"
    assert field_kind("Owned|Leased|null", "Leased") == "choice"
    assert field_kind("boolean", True) == "state"
    assert field_kind("text", None) == "absent"
    assert field_kind("date", "2024-05-15") == "date"
    assert field_kind("text", "2024-05-15") == "text"
    assert field_kind(None, ["a", "b"]) == "list"
    assert field_kind("value|null", "x") == "text"  # one real option: a value, not a choice


def test_normalize_and_digit_gaps():
    assert normalize("**Smith** – <br>John  |  “A”") == 'smith-john | "a"'
    assert close_digit_gaps("£ 2 2 8 6 2 . 0 0 and 1 , 234") == "£ 22862.00 and 1,234"


def test_numbers_dates_and_text():
    hay = Haystack.of(
        "| Premium | $12,800.00 |\nDate: January 15, 2024 and 03/04/2025\nTotal 1 2 3 4\nName: **Jo Smith**"
    )
    assert found("number", 12800, hay) == (True, True)
    assert found("number", 1234, hay) == (False, True)  # only after the spacing is collapsed
    assert found("number", 128, hay) == (False, False)
    assert found("date", "2024-01-15", hay) == (True, True)
    assert {"2025-03-04", "2025-04-03"} <= dates_in(normalize("03/04/2025"))
    assert found("text", "Jo Smith", hay) == (True, True)
    assert found("text", "Smith", hay) == (True, True)
    assert found("text", "Smit", hay) == (False, True)  # inside a word: loose only
    assert found("text", "mi", hay) == (False, False)  # short values are never found loosely inside words
    assert found("text", "JoSmith", hay) == (False, True)


def test_sample_documents():
    pages = {f"d{i}": 1 + i % 3 for i in range(100)}
    assert sample_documents({"a": 2, "b": 3}, cap=10) == ["a", "b"]
    picked = sample_documents(pages, cap=50, seed=1)
    assert sum(pages[s] for s in picked) <= 50 and picked == sorted(picked)
    assert picked == sample_documents(pages, cap=50, seed=1)
    assert picked != sample_documents(pages, cap=50, seed=2)


def test_bootstrap_gain():
    gain, p = bootstrap_gain([(1, 2, 2)] * 10)
    assert gain == 50.0 and p == 1.0
    assert bootstrap_gain([(2, 2, 2)] * 5) == (0.0, 0.0)
    assert bootstrap_gain([]) == (0.0, 0.0)


def test_score_items_counts_unscored_fields():
    items = [{"question_id": "q1", "source_file": "d", "domain": "finance", "capabilities": ["checkbox_state"],
              "response_format": "Return exactly: amt=<number>; ok=<boolean>; who=<text>; c=<A|B>",
              "gold_dict": {"amt": 5, "ok": True, "who": "Ann Lee", "c": "A"}}]  # fmt: skip
    rows = score_items(items, {"x": {"d": "Amount 5\nAnn Lee"}, "y": {"d": "Amount 6"}})
    assert (rows["x"][0]["fields"], rows["x"][0]["found"], rows["y"][0]["found"]) == (2, 2, 0)
    assert rows["x"][0]["unscored"] == {"state": 1, "choice": 1} and rows["x"][0]["form"]
