from __future__ import annotations

import numpy as np


class AliasSampler:
    """Alias method for efficient discrete sampling."""

    def __init__(self, probs: np.ndarray, rng: np.random.Generator | None = None) -> None:
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 1 or probs.size == 0:
            raise ValueError("AliasSampler requires a non-empty 1-D probability vector")
        if np.any(probs < 0) or not np.isfinite(probs).all() or probs.sum() <= 0:
            raise ValueError("AliasSampler probabilities must be finite, non-negative, and have positive mass")

        n = len(probs)
        probs = probs / probs.sum()
        self.n = n
        self.rng = rng
        self.q = np.zeros(n, dtype=np.float64)
        self.J = np.zeros(n, dtype=np.int32)
        small, large = [], []
        self.q[:] = probs * n
        for i, qi in enumerate(self.q):
            (small if qi < 1.0 else large).append(i)
        while small and large:
            s = small.pop()
            big = large.pop()
            self.J[s] = big
            self.q[big] = (self.q[big] - 1.0) + self.q[s]
            if self.q[big] < 1.0:
                small.append(big)
            else:
                large.append(big)
        for idx in small + large:
            self.q[idx] = 1.0

    def get_rng_state(self) -> dict | None:
        if self.rng is None:
            return None
        return self.rng.bit_generator.state

    def set_rng_state(self, state: dict | None) -> None:
        if state is None:
            if self.rng is not None:
                raise ValueError("Checkpoint is missing AliasSampler RNG state")
            return
        if self.rng is None:
            raise ValueError("AliasSampler has no generator to restore")
        self.rng.bit_generator.state = state

    def sample(self, size: int) -> np.ndarray:
        if size < 0:
            raise ValueError("sample size must be non-negative")
        if self.rng is None:
            kk = np.random.randint(0, self.n, size=size)
            uu = np.random.rand(size)
        else:
            kk = self.rng.integers(0, self.n, size=size)
            uu = self.rng.random(size)
        use_k = uu < self.q[kk]
        return np.where(use_k, kk, self.J[kk])


def build_unigram_sampler(
    counts: list[int],
    power: float = 0.75,
    rng: np.random.Generator | None = None,
) -> AliasSampler:
    """构建 Unigram^0.75 的负采样分布。"""
    probs = np.asarray(counts, dtype=np.float64)
    probs = np.power(probs, power)
    probs = probs / probs.sum()
    return AliasSampler(probs, rng=rng)
