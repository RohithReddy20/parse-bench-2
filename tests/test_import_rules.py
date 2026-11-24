"""The parser knows nothing about the benchmark: static import rules for src/docparse."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "docparse"


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def _files():
    return sorted(SRC.rglob("*.py"))


def test_docparse_never_imports_evalkit():
    bad = [str(p.relative_to(SRC)) for p in _files() if "evalkit" in _imports(p)]
    assert bad == []


def test_docparse_imports_parse_bench_only_in_bench():
    bad = [
        str(p.relative_to(SRC))
        for p in _files()
        if "parse_bench" in _imports(p) and p.relative_to(SRC).parts[0] != "bench"
    ]
    assert bad == []


def test_rule_checker_sees_files():
    assert _files(), "src/docparse has no python files"
