import pytest

from w2v_factory.data.dataset import generate_skipgram_pairs
from w2v_factory.data.sampler import build_unigram_sampler
from w2v_factory.trainer.planning import build_training_plan, pair_rng
from w2v_factory.utils import make_numpy_rng


def test_training_plan_uses_in_vocab_token_mass():
    plan = build_training_plan([7, 3, 2], epochs=4)
    assert plan.trainable_tokens_per_epoch == 12
    assert plan.epochs == 4
    assert plan.total_progress_tokens == 48


def test_pair_rng_is_replayable_per_epoch():
    sent = list(range(12))
    first = list(generate_skipgram_pairs(sent, 4, rng=pair_rng(5, 2)))
    replay = list(generate_skipgram_pairs(sent, 4, rng=pair_rng(5, 2)))
    other_epoch = list(generate_skipgram_pairs(sent, 4, rng=pair_rng(5, 3)))
    assert first == replay
    assert first != other_epoch


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


@pytest.mark.parametrize(
    "seed,stream_ids",
    [(-1, ()), (True, ()), (1, (-1,)), (1, (False,))],
)
def test_rng_factory_rejects_invalid_seed_components(seed, stream_ids):
    with pytest.raises(ValueError, match="RNG"):
        make_numpy_rng(seed, *stream_ids)
