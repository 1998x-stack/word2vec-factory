from __future__ import annotations
import numpy as np
from typing import List

class AliasSampler:
    """Alias method for efficient discrete sampling."""
    def __init__(self, probs: np.ndarray) -> None:
        n = len(probs)
        probs = probs / probs.sum()
        self.n = n
        self.q = np.zeros(n, dtype=np.float64)
        self.J = np.zeros(n, dtype=np.int32)
        small, large = [], []
        self.q[:] = probs * n
        for i, qi in enumerate(self.q):
            (small if qi < 1.0 else large).append(i)
        while small and large:
            s = small.pop()
            l = large.pop()
            self.J[s] = l
            self.q[l] = (self.q[l] - 1.0) + self.q[s]
            if self.q[l] < 1.0:
                small.append(l)
            else:
                large.append(l)
        # remain qs are 1.0

    def sample(self, size: int) -> np.ndarray:
        kk = np.random.randint(0, self.n, size=size)
        uu = np.random.rand(size)
        use_k = uu < self.q[kk]
        out = np.where(use_k, kk, self.J[kk])
        return out

def build_unigram_sampler(counts: List[int], power: float = 0.75) -> AliasSampler:
    """构建 Unigram^0.75 的负采样分布。"""
    probs = np.asarray(counts, dtype=np.float64)
    probs = np.power(probs, power)
    probs = probs / probs.sum()
    return AliasSampler(probs)
