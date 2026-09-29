from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass


class TokenStreamFingerprint:
    """Incremental fingerprint of the exact tokenized sentence stream."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self.sentences = 0
        self.tokens = 0

    def update(self, sent: list[str]) -> None:
        self.sentences += 1
        self.tokens += len(sent)
        self._digest.update(b"S")
        self._digest.update(len(sent).to_bytes(8, "little", signed=False))
        for token in sent:
            data = token.encode("utf-8")
            self._digest.update(len(data).to_bytes(8, "little", signed=False))
            self._digest.update(data)

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


@dataclass
class Vocab:
    """词表结构体。"""

    stoi: dict[str, int]
    itos: list[str]
    counts: list[int]
    total_tokens: int
    corpus_sha256: str = ""
    sentences: int = 0

    @property
    def size(self) -> int:
        return len(self.itos)


def build_vocab(token_stream: Iterable[list[str]], min_count: int, max_vocab: int) -> Vocab:
    """统计词频并截断，同时记录精确 token-stream 指纹。"""
    counter: Counter[str] = Counter()
    fingerprint = TokenStreamFingerprint()
    total = 0
    for sent in token_stream:
        fingerprint.update(sent)
        counter.update(sent)
        total += len(sent)

    items = [(w, c) for w, c in counter.items() if c >= min_count]
    items.sort(key=lambda x: (-x[1], x[0]))
    if max_vocab is not None:
        items = items[:max_vocab]
    itos = [w for w, _ in items]
    stoi = {w: i for i, w in enumerate(itos)}
    counts = [c for _, c in items]
    return Vocab(
        stoi=stoi,
        itos=itos,
        counts=counts,
        total_tokens=total,
        corpus_sha256=fingerprint.hexdigest(),
        sentences=fingerprint.sentences,
    )
