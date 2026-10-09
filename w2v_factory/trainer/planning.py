from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils import make_numpy_rng

PAIR_RNG_STREAM = 3


@dataclass(frozen=True)
class TrainingPlan:
    """Static token-progress denominator for a streaming training run."""

    trainable_tokens_per_epoch: int
    epochs: int
    total_progress_tokens: int


def pair_rng(seed: int, epoch: int) -> np.random.Generator:
    """Return the deterministic context-window RNG for one epoch."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    return make_numpy_rng(seed, PAIR_RNG_STREAM, epoch)


def build_training_plan(counts: list[int], epochs: int) -> TrainingPlan:
    """Build a token-based plan without materializing or pre-counting pairs."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    trainable_tokens = sum(counts)
    if trainable_tokens <= 0:
        raise ValueError("Training requires positive in-vocabulary token mass")
    return TrainingPlan(
        trainable_tokens_per_epoch=trainable_tokens,
        epochs=epochs,
        total_progress_tokens=trainable_tokens * epochs,
    )
