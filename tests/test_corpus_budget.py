"""Controlled nested corpus slices: Table 2 data-size intervention."""
import hashlib

import pytest

from w2v_factory.cli.corpus_budget import make_corpus_budgets


def test_nested_prefix_budgets_exact_and_hashable(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("ONE two three four\nfive six seven\neight nine ten\n", encoding="utf-8")
    root = tmp_path / "slices"
    manifest = make_corpus_budgets([str(source)], [8, 3], root)
    short = (root / "tokens-3.txt").read_text()
    long = (root / "tokens-8.txt").read_text()
    assert short.split() == long.split()[:3]
    assert len(long.split()) == 8
    assert len(short.split()) == 3
    assert manifest["slices"][0]["sha256"] == hashlib.sha256(short.encode()).hexdigest()
    assert manifest["slices"][1]["sha256"] == hashlib.sha256(long.encode()).hexdigest()
    with pytest.raises(FileExistsError):
        make_corpus_budgets([str(source)], [3, 8], root)


def test_budget_errors_are_explicit_and_partial_files_cleaned(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("one two three\n")
    root = tmp_path / "out"
    with pytest.raises(ValueError, match="too few"):
        make_corpus_budgets([str(source)], [5], root)
    assert not (root / "tokens-5.txt").exists()
    assert not (root / "tokens-5.txt.partial").exists()
    with pytest.raises(ValueError, match="distinct"):
        make_corpus_budgets([str(source)], [2, 2], root)
