from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..data.dataset import count_skipgram_pairs
from ..utils import make_numpy_rng

PAIR_RNG_STREAM = 3


@dataclass(frozen=True)
class EpochPlan:
    epoch: int
    examples: int
    optimizer_steps: int


def pair_rng(seed: int, epoch: int) -> np.random.Generator:
    """Return the deterministic context-window RNG for one epoch."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    return make_numpy_rng(seed, PAIR_RNG_STREAM, epoch)


def _count_epoch_examples(
    sents: list[list[int]],
    arch: str,
    window: int,
    seed: int,
    epoch: int,
) -> int:
    if arch == "cbow":
        return sum(len(sent) for sent in sents if len(sent) >= 2)
    if arch != "skipgram":
        raise ValueError(f"Unsupported Word2Vec architecture: {arch}")

    rng = pair_rng(seed, epoch)
    return sum(count_skipgram_pairs(sent, window, rng=rng) for sent in sents)


def build_epoch_plans(
    sents: list[list[int]],
    arch: str,
    window: int,
    batch_size: int,
    epochs: int,
    seed: int,
) -> list[EpochPlan]:
    """Plan exact optimizer-step counts for deterministic per-epoch windows."""
    if window <= 0 or batch_size <= 0 or epochs <= 0:
        raise ValueError("window, batch_size, and epochs must be positive")

    plans: list[EpochPlan] = []
    for epoch in range(epochs):
        examples = _count_epoch_examples(sents, arch, window, seed, epoch)
        if examples <= 0:
            raise ValueError(f"Epoch {epoch + 1} has no training examples")
        plans.append(
            EpochPlan(
                epoch=epoch,
                examples=examples,
                optimizer_steps=math.ceil(examples / batch_size),
            )
        )
    return plans
