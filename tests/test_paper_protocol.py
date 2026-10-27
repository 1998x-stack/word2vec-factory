"""Paper-policy and factory regression tests."""
import numpy as np
import pytest

from w2v_factory.data.dataset import (
    generate_cbow_pairs_with_progress,
    generate_skipgram_pairs_with_progress,
)
from w2v_factory.data.sampler import build_unigram_sampler
from w2v_factory.data.subsample import compute_discard_probs
from w2v_factory.factory import make_model, make_optimizer


def test_paper_cbow_fixed_four_words_each_side():
    tokens = list(range(9))
    positions = list(range(1, 10))
    samples = list(
        generate_cbow_pairs_with_progress(
            tokens, positions, 4, rng=np.random.default_rng(19),
            random_window=False,
        )
    )
    assert samples[4] == (([0, 1, 2, 3, 5, 6, 7, 8], 4), 5)


def test_skipgram_fixed_radius_includes_both_directions():
    samples = list(generate_skipgram_pairs_with_progress(
        [0, 1, 2], [1, 2, 3], 2, random_window=False,
    ))
    assert samples == [
        ((0, 1), 1), ((0, 2), 1), ((1, 0), 2),
        ((1, 2), 2), ((2, 0), 3), ((2, 1), 3),
    ]


def test_followup_subsample_formula_retains_more_mass():
    legacy = compute_discard_probs([900, 100], 1000, 0.1, formula="legacy")
    paper = compute_discard_probs([900, 100], 1000, 0.1, formula="mikolov_1310")
    assert paper[0] < legacy[0]
    with pytest.raises(ValueError, match="Unknown subsample formula"):
        compute_discard_probs([1], 1, None, formula="unknown")


def test_negative_sampling_exponent_changes_distribution():
    counts = [100, 1]
    flatter = build_unigram_sampler(counts, power=0.5, rng=np.random.default_rng(2))
    steeper = build_unigram_sampler(counts, power=1, rng=np.random.default_rng(2))
    assert np.mean(steeper.sample(20000) == 0) > np.mean(flatter.sample(20000) == 0)


def test_real_model_optimizer_factories():
    model = make_model("cbow", vocab_size=5, dim=4, share_io=False, out_vocab_size=4)
    optimizer = make_optimizer("sgd", params=model.parameters(), lr=0.02)
    assert model.in_embed.weight.shape == (5, 4)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.02)
    with pytest.raises(KeyError):
        make_model("missing", vocab_size=5, dim=4, share_io=False, out_vocab_size=4)
