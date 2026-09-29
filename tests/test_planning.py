import math

from w2v_factory.data.dataset import generate_cbow_pairs, generate_skipgram_pairs
from w2v_factory.data.sampler import build_unigram_sampler
from w2v_factory.trainer.planning import build_epoch_plans, pair_rng
from w2v_factory.utils import make_numpy_rng


def test_skipgram_plan_matches_replayed_window_stream():
    sents = [[0, 1, 2, 3, 4], [4, 3, 2]]
    plans = build_epoch_plans(
        sents=sents,
        arch="skipgram",
        window=3,
        batch_size=4,
        epochs=3,
        seed=17,
    )

    for plan in plans:
        rng = pair_rng(17, plan.epoch)
        actual_examples = sum(
            len(list(generate_skipgram_pairs(sent, 3, rng=rng))) for sent in sents
        )
        assert plan.examples == actual_examples
        assert plan.optimizer_steps == math.ceil(actual_examples / 4)


def test_cbow_plan_matches_generated_target_count():
    sents = [[0, 1, 2, 3], [3, 2]]
    plans = build_epoch_plans(
        sents=sents,
        arch="cbow",
        window=3,
        batch_size=4,
        epochs=2,
        seed=23,
    )

    for plan in plans:
        rng = pair_rng(23, plan.epoch)
        actual_examples = sum(
            len(list(generate_cbow_pairs(sent, 3, rng=rng))) for sent in sents
        )
        assert plan.examples == actual_examples == 6
        assert plan.optimizer_steps == 2


def test_negative_sampling_does_not_perturb_window_stream():
    sent = list(range(12))
    expected = list(generate_skipgram_pairs(sent, 4, rng=pair_rng(5, 0)))

    sampler = build_unigram_sampler(
        [10, 8, 6, 4, 2],
        rng=make_numpy_rng(5, 2),
    )
    sampler.sample(10_000)

    actual = list(generate_skipgram_pairs(sent, 4, rng=pair_rng(5, 0)))
    assert actual == expected
