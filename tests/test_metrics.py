import numpy as np
import pytest

from w2v_factory.trainer.metrics import evaluate_analogy, load_analogy


def test_load_analogy_skips_section_headers(tmp_path):
    f = tmp_path / "q.txt"
    f.write_text(": capital-common-countries\n" "paris france rome italy\n" ": grammar-adjective\n" "x y z w\n")
    assert load_analogy(str(f)) == [("paris", "france", "rome", "italy"), ("x", "y", "z", "w")]


def test_evaluate_analogy_3cosadd():
    vecs = {
        "king": [10.0, 0.0],
        "man": [9.0, 0.0],
        "woman": [0.0, 9.0],
        "queen": [0.0, 10.0],
        "unused": [1.0, 1.0],
    }
    itos = list(vecs)
    stoi = {w: i for i, w in enumerate(itos)}
    emb = np.asarray([vecs[w] for w in itos], dtype=np.float64)
    items = [("king", "man", "woman", "queen"), ("a", "b", "c", "not_in_vocab")]
    acc = evaluate_analogy(emb, stoi, itos, items, topk=1)
    # first item hits; second skipped (d not in vocab)
    assert acc == pytest.approx(1.0)
