"""Section-aware analogy evaluator with explicit OOV denominators.

Unlike accuracy on eligible questions alone, this reports coverage and
unconditional accuracy. All four words must be in-vocabulary.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Analogy:
    category: str
    family: str
    a: str
    b: str
    c: str
    target: str


def read_analogies(path: str | Path, *, lowercase: bool) -> list[Analogy]:
    questions: list[Analogy] = []
    category = "uncategorized"
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(":"):
                category = line[1:].strip()
                if not category:
                    raise ValueError(f"Empty analogy section at line {line_no}")
                continue
            words = line.split()
            if len(words) != 4:
                raise ValueError(f"Expected four analogy tokens at line {line_no}")
            if lowercase:
                words = [w.lower() for w in words]
            family = "syntactic" if category.startswith("gram") else "semantic"
            questions.append(Analogy(category, family, *words))
    return questions


def read_embedding_words(path: str | Path) -> list[str]:
    """Read only vocabulary labels, not redundant vectors, from the text export."""
    words: list[str] = []
    with Path(path).open(encoding="utf-8") as stream:
        header = stream.readline().split()
        if len(header) != 2:
            raise ValueError("Expected word2vec text header V D")
        try:
            expected = int(header[0])
            int(header[1])
        except ValueError as exc:
            raise ValueError("Invalid word2vec export header") from exc
        for line in stream:
            fields = line.split(maxsplit=1)
            if fields:
                words.append(fields[0])
    if len(words) != expected or len(set(words)) != len(words):
        raise ValueError("Embedding vocabulary count or uniqueness mismatch")
    return words


def score_analogies(
    vectors: np.ndarray,
    words: list[str],
    questions: list[Analogy],
    *,
    batch_size: int = 16,
) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if vectors.ndim != 2 or vectors.shape[0] != len(words):
        raise ValueError("Vector matrix and vocabulary disagree")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain non-finite values")
    index = {word: i for i, word in enumerate(words)}
    if len(index) != len(words):
        raise ValueError("Duplicate word in embeddings")
    # Normalize once. Query and candidates use 3CosAdd; input terms are masked.
    matrix = np.asarray(vectors, dtype=np.float32)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norm, 1e-12)
    covered: list[tuple[Analogy, tuple[int, int, int, int]]] = []
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "eligible": 0, "correct": 0})
    families: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "eligible": 0, "correct": 0})
    for q in questions:
        stats[q.category]["total"] += 1
        families[q.family]["total"] += 1
        if all(word in index for word in (q.a, q.b, q.c, q.target)):
            ids = (index[q.a], index[q.b], index[q.c], index[q.target])
            covered.append((q, ids))
            stats[q.category]["eligible"] += 1
            families[q.family]["eligible"] += 1

    for offset in range(0, len(covered), batch_size):
        batch = covered[offset : offset + batch_size]
        ids = np.asarray([pair for _, pair in batch], dtype=np.int64)
        query = normalized[ids[:, 1]] - normalized[ids[:, 0]] + normalized[ids[:, 2]]
        query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
        scores = normalized @ query.T
        scores[ids[:, :3], np.arange(len(batch))[:, None]] = -np.inf
        predictions = np.argmax(scores, axis=0)
        for i, (q, pair) in enumerate(batch):
            if int(predictions[i]) == pair[3]:
                stats[q.category]["correct"] += 1
                families[q.family]["correct"] += 1

    def summarize(data: dict[str, int]) -> dict:
        total, eligible, correct = data["total"], data["eligible"], data["correct"]
        return {
            **data,
            "coverage": eligible / total if total else 0.0,
            "accuracy_covered": correct / eligible if eligible else None,
            "accuracy_all": correct / total if total else None,
        }

    grand = {
        "total": len(questions),
        "eligible": len(covered),
        "correct": sum(v["correct"] for v in stats.values()),
    }
    return {
        "overall": summarize(grand),
        "families": {name: summarize(value) for name, value in sorted(families.items())},
        "categories": {name: summarize(value) for name, value in sorted(stats.items())},
        "method": "3CosAdd, normalized cosine, exclude all three input words; exact top-1",
    }
