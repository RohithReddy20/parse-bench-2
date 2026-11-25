import random
import time
from difflib import SequenceMatcher

from docparse.fixes.align import key, match, opcodes, tokens


def test_tokens_fold_case_and_accents():
    assert [t[2] for t in tokens("Café, CAFÉ_x 42")] == ["cafe", "cafe", "x", "42"]
    assert key("Ünïcode") == "unicode"


def test_small_pairs_align_exactly_as_difflib():
    rng = random.Random(7)
    words = ["the", "a", "of", "1", "2", "data", "table", "x", "y", "z"]
    for _ in range(50):
        a = [rng.choice(words) for _ in range(rng.randint(0, 60))]
        b = [rng.choice(words) for _ in range(rng.randint(0, 60))]
        assert opcodes(a, b) == SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
        expected = [-1] * len(a)
        for i, j, n in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
            for x in range(n):
                expected[i + x] = j + x
        assert match(a, b) == expected


def test_a_long_table_of_repeated_digits_aligns_quickly():
    rng = random.Random(3)
    a = [str(rng.randint(0, 9)) if i % 5 else f"row{i}" for i in range(12_000)]
    b = list(a)
    b[6001] = "changed"
    start = time.time()
    partner = match(a, b)
    assert time.time() - start < 20
    assert sum(1 for j in partner if j >= 0) >= len(a) - 50
    assert partner[100] == 100 and partner[11_999] == 11_999
