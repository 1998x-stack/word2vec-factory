import numpy as np

from w2v_factory.data.sampler import AliasSampler, build_unigram_sampler


def test_alias_matches_distribution():
    probs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    sampler = AliasSampler(probs, rng=np.random.default_rng(0))
    n = 200_000
    samples = sampler.sample(n)
    hist = np.bincount(samples, minlength=len(probs)) / n
    np.testing.assert_allclose(hist, probs / probs.sum(), atol=0.02)


def test_sample_range_and_length():
    probs = np.array([0.25, 0.25, 0.5])
    sampler = AliasSampler(probs, rng=np.random.default_rng(1))
    samples = sampler.sample(24)
    assert samples.shape == (24,)
    assert samples.min() >= 0 and samples.max() < 3


def test_unigram_sampler_has_same_support():
    counts = [1, 3, 2, 5, 4]
    sampler = build_unigram_sampler(counts, rng=np.random.default_rng(2))
    assert sampler.sample(50).min() >= 0
    assert sampler.sample(50).max() < len(counts)


def test_seeded_alias_sampler_is_replayable():
    probs = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    a = AliasSampler(probs, rng=np.random.default_rng(7)).sample(100)
    b = AliasSampler(probs, rng=np.random.default_rng(7)).sample(100)
    np.testing.assert_array_equal(a, b)
