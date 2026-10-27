"""Paper-level interventions and evaluation regression tests."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

from w2v_factory.data.dataset import generate_skipgram_pairs
from w2v_factory.experiments.evaluate import evaluate_file
from w2v_factory.experiments.plan import build_plan
from w2v_factory.models import make_model
from w2v_factory.registry import LOSS_REG, MODEL_REG, SCHED_REG
from w2v_factory.trainer.engine import Trainer  # noqa: F401 - registers objectives
from w2v_factory.trainer.lr_schedulers import get_scheduler  # noqa: F401 - registers schedulers


def test_factories_register_actual_training_dispatch():
    assert {"cbow", "skipgram"}.issubset(MODEL_REG._objs)
    assert {"hs", "ns"}.issubset(LOSS_REG._objs)
    assert {"linear", "none"}.issubset(SCHED_REG._objs)
    assert make_model("skipgram", vocab_size=5, dim=4).in_embed.weight.shape == (5, 4)


def test_context_window_sampling_controls_real_pair_distribution():
    tokens = list(range(7))
    fixed = list(generate_skipgram_pairs(tokens, 2, rng=np.random.default_rng(7), window_sampling="fixed"))
    dynamic = list(generate_skipgram_pairs(tokens, 2, rng=np.random.default_rng(7), window_sampling="uniform"))
    assert len(fixed) == 22
    assert len(dynamic) <= len(fixed)
    with pytest.raises(ValueError, match="Unknown"):
        list(generate_skipgram_pairs(tokens, 2, window_sampling="variable"))


def test_cbow_pooling_changes_context_output_and_ignores_padding():
    mean = make_model("cbow", vocab_size=5, dim=3, pooling="mean")
    summed = make_model("cbow", vocab_size=5, dim=3, pooling="sum")
    summed.load_state_dict(mean.state_dict())
    context = torch.tensor([[0, 1, 4]])
    lengths = torch.tensor([2])
    torch.testing.assert_close(summed._mean_context(context, lengths), 2 * mean._mean_context(context, lengths))


def test_plan_pairs_seeds_and_is_deterministic():
    path = Path(__file__).resolve().parents[1] / "configs/paper1301_ablations.yaml"
    a, root = build_plan(path)
    b, _ = build_plan(path)
    assert [r.run_id for r in a] == [r.run_id for r in b]
    seeds_by_variant = defaultdict(set)
    for exp in a:
        seeds_by_variant[exp.variant].add(exp.seed)
        assert str(root) in exp.cfg.RUN.out_dir
    assert all(seeds == {11, 23, 37} for seeds in seeds_by_variant.values())
    with pytest.raises(ValueError, match="exceeds"):
        build_plan(path, max_runs=1)


def test_analogy_coverage_and_semantic_syntactic_categories(tmp_path):
    vectors = tmp_path / "embeddings.txt"
    vectors.write_text("6 2\na 1 0\nb 2 0\nc 0 1\nd 1 1\njunk -1 -1\nother 0 -1\n")
    q = tmp_path / "questions.txt"
    q.write_text(": capital-world\na b c d\na b c out_of_vocab\n: gram1-adjective-to-adverb\na b c d\n")
    report = evaluate_file(vectors, q)
    assert report["overall"]["total"] == 3
    assert report["overall"]["eligible"] == 2
    assert report["overall"]["accuracy"] == pytest.approx(1.0)
    assert report["overall"]["coverage"] == pytest.approx(2 / 3)
    assert report["groups"]["semantic"]["eligible"] == 1
    assert report["groups"]["syntactic"]["eligible"] == 1


def test_no_eligible_questions_have_null_accuracy(tmp_path):
    vectors = tmp_path / "v.txt"
    vectors.write_text("4 2\na 1 0\nb 0 1\nc -1 0\nd 0 -1\n")
    q = tmp_path / "q.txt"
    q.write_text(": gram1-adjective-to-adverb\nx y z zz\n")
    result = evaluate_file(vectors, q)
    assert result["overall"]["accuracy"] is None
    assert result["overall"]["coverage"] == 0
