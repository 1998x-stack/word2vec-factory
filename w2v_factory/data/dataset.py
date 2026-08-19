from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class SentenceIndexer:
    """把分词结果转换为词 id，并进行次采样丢弃。"""

    def __init__(self, stoi: dict[str, int], discard_probs: np.ndarray | None) -> None:
        self.stoi = stoi
        self.discard_probs = discard_probs

    def encode(self, sent: list[str]) -> list[int]:
        ids = []
        for w in sent:
            if w not in self.stoi:
                continue
            wid = self.stoi[w]
            # 次采样丢弃（高频词）
            if self.discard_probs is not None:
                if np.random.rand() < self.discard_probs[wid]:
                    continue
            ids.append(wid)
        return ids


def generate_skipgram_pairs(tokens: list[int], max_window: int) -> Iterable[tuple[int, int]]:
    """为一条句子生成 Skip-gram (center, context) 正样本对。"""
    L = len(tokens)
    for i, center in enumerate(tokens):
        # 随机窗口，增强鲁棒性（论文做法）
        win = np.random.randint(1, max_window + 1)
        left = max(0, i - win)
        right = min(L, i + win + 1)
        for j in range(left, right):
            if j == i:
                continue
            yield center, tokens[j]


def generate_cbow_pairs(tokens: list[int], max_window: int) -> Iterable[tuple[list[int], int]]:
    """为一条句子生成 CBOW (contexts..., target)。"""
    L = len(tokens)
    for i, target in enumerate(tokens):
        win = np.random.randint(1, max_window + 1)
        left = max(0, i - win)
        right = min(L, i + win + 1)
        ctx = [tokens[j] for j in range(left, right) if j != i]
        if ctx:
            yield ctx, target
