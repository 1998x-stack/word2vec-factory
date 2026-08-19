from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class Vocab:
    """词表结构体。"""

    stoi: dict[str, int]
    itos: list[str]
    counts: list[int]
    total_tokens: int

    @property
    def size(self) -> int:
        return len(self.itos)


def build_vocab(token_stream: Iterable[list[str]], min_count: int, max_vocab: int) -> Vocab:
    """统计词频并截断，返回 Vocab。"""
    counter: Counter[str] = Counter()
    total = 0
    for sent in token_stream:
        counter.update(sent)
        total += len(sent)
    # 过滤低频
    items = [(w, c) for w, c in counter.items() if c >= min_count]
    # 频率排序
    items.sort(key=lambda x: (-x[1], x[0]))
    if max_vocab is not None:
        items = items[:max_vocab]
    itos = [w for w, _ in items]
    stoi = {w: i for i, w in enumerate(itos)}
    counts = [c for _, c in items]
    return Vocab(stoi=stoi, itos=itos, counts=counts, total_tokens=total)
