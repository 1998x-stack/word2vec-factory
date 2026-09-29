from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _random(rng: np.random.Generator | None) -> float:
    if rng is None:
        return float(np.random.rand())
    return float(rng.random())


def _window(rng: np.random.Generator | None, max_window: int) -> int:
    if rng is None:
        return int(np.random.randint(1, max_window + 1))
    return int(rng.integers(1, max_window + 1))


class SentenceIndexer:
    """把分词结果转换为词 id，并进行次采样丢弃。"""

    def __init__(
        self,
        stoi: dict[str, int],
        discard_probs: np.ndarray | None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.stoi = stoi
        self.discard_probs = discard_probs
        self.rng = rng

    def encode(self, sent: list[str]) -> list[int]:
        ids = []
        for w in sent:
            if w not in self.stoi:
                continue
            wid = self.stoi[w]
            if self.discard_probs is not None:
                if _random(self.rng) < self.discard_probs[wid]:
                    continue
            ids.append(wid)
        return ids


def generate_skipgram_pairs(
    tokens: list[int],
    max_window: int,
    rng: np.random.Generator | None = None,
) -> Iterable[tuple[int, int]]:
    """为一条句子生成 Skip-gram (center, context) 正样本对。"""
    length = len(tokens)
    for i, center in enumerate(tokens):
        win = _window(rng, max_window)
        left = max(0, i - win)
        right = min(length, i + win + 1)
        for j in range(left, right):
            if j == i:
                continue
            yield center, tokens[j]


def count_skipgram_pairs(
    tokens: list[int],
    max_window: int,
    rng: np.random.Generator | None = None,
) -> int:
    """Count Skip-gram pairs using the same window draws as generation."""
    length = len(tokens)
    total = 0
    for i in range(length):
        win = _window(rng, max_window)
        left = max(0, i - win)
        right = min(length, i + win + 1)
        total += right - left - 1
    return total


def generate_cbow_pairs(
    tokens: list[int],
    max_window: int,
    rng: np.random.Generator | None = None,
) -> Iterable[tuple[list[int], int]]:
    """为一条句子生成 CBOW (contexts..., target)。"""
    length = len(tokens)
    for i, target in enumerate(tokens):
        win = _window(rng, max_window)
        left = max(0, i - win)
        right = min(length, i + win + 1)
        ctx = [tokens[j] for j in range(left, right) if j != i]
        if ctx:
            yield ctx, target
