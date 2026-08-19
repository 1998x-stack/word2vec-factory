import numpy as np
from w2v_factory.data.sampler import AliasSampler, build_unigram_sampler


def test_alias_matches_distribution():
    probs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    sampler = AliasSampler(probs)
    n = 200_000
    samples = sampler.sample(n)
    hist = np.bincount(samples, minlength=len(probs)) / n
    np.testing.assert_allclose(hist, probs / probs.sum(), atol=0.02)


def test_sample_range_and_length():
    probs = np.array([0.25, 0.25, 0.5])
    sampler = AliasSampler(probs)
    samples = sampler.sample(24)
    assert samples.shape == (24,)
    assert samples.min() >= 0 and samples.max() < 3


def test_unigram_sampler_has_same_support():
    counts = [1, 3, 2, 5, 4]
    sampler = build_unigram_sampler(counts)
    assert sampler.sample(50).min() >= 0
    assert sampler.sample(50).max() < len(counts)