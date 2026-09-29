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

    def encode_with_positions(self, sent: list[str]) -> tuple[list[int], list[int], int]:
        """Encode one sentence while preserving progress in the trainable-token stream.

        Positions are one-based ordinals among vocabulary-eligible source tokens.
        A token discarded by subsampling still advances the ordinal, which lets
        training decay LR by source-token progress without retaining the corpus.
        """
        ids: list[int] = []
        positions: list[int] = []
        eligible = 0
        for word in sent:
            wid = self.stoi.get(word)
            if wid is None:
                continue
            eligible += 1
            if self.discard_probs is not None and _random(self.rng) < self.discard_probs[wid]:
                continue
            ids.append(wid)
            positions.append(eligible)
        return ids, positions, eligible

    def encode(self, sent: list[str]) -> list[int]:
        ids, _, _ = self.encode_with_positions(sent)
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


def generate_skipgram_pairs_with_progress(
    tokens: list[int],
    positions: list[int],
    max_window: int,
    rng: np.random.Generator | None = None,
) -> Iterable[tuple[tuple[int, int], int]]:
    """Yield Skip-gram pairs plus the center token's source progress ordinal."""
    if len(tokens) != len(positions):
        raise ValueError("tokens and positions must have the same length")
    length = len(tokens)
    for i, center in enumerate(tokens):
        win = _window(rng, max_window)
        left = max(0, i - win)
        right = min(length, i + win + 1)
        for j in range(left, right):
            if j == i:
                continue
            yield (center, tokens[j]), positions[i]


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


def generate_cbow_pairs_with_progress(
    tokens: list[int],
    positions: list[int],
    max_window: int,
    rng: np.random.Generator | None = None,
) -> Iterable[tuple[tuple[list[int], int], int]]:
    """Yield CBOW examples plus the target token's source progress ordinal."""
    if len(tokens) != len(positions):
        raise ValueError("tokens and positions must have the same length")
    length = len(tokens)
    for i, target in enumerate(tokens):
        win = _window(rng, max_window)
        left = max(0, i - win)
        right = min(length, i + win + 1)
        ctx = [tokens[j] for j in range(left, right) if j != i]
        if ctx:
            yield (ctx, target), positions[i]
