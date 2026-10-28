"""Category-aware, vocabulary-coverage-aware 3CosAdd evaluation.

The 1301.3781 test contains semantic and syntactic sections. Missing words
are excluded from the eligible denominator, but always reported as coverage.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..cli.eval import load_word2vec_txt


def _ratio(n: int, d: int) -> float | None:
    return n / d if d else None


def _summary(counter: dict[str, int]) -> dict[str, int | float | None]:
    return {
        "total": counter["total"],
        "eligible": counter["eligible"],
        "correct": counter["correct"],
        "coverage": _ratio(counter["eligible"], counter["total"]),
        "accuracy": _ratio(counter["correct"], counter["eligible"]),
    }


def evaluate_file(vectors: str | Path, questions: str | Path, lowercase: bool = False) -> dict:
    words, emb, stoi = load_word2vec_txt(str(vectors))
    emb = np.asarray(emb, dtype=np.float64)
    if emb.ndim != 2 or len(words) != len(emb) or not np.isfinite(emb).all():
        raise ValueError("Invalid embedding matrix")
    if len(words) < 4:
        raise ValueError("At least four words are required for analogy evaluation")
    normalized = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "eligible": 0, "correct": 0})
    categories: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "eligible": 0, "correct": 0})
    qpath = Path(questions)
    category = "uncategorized"
    with qpath.open(encoding="utf-8") as f:
        for line_number, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(":"):
                category = line[1:].strip() or "uncategorized"
                continue
            tokens = line.split()
            if len(tokens) != 4:
                raise ValueError(f"Bad analogy question at {qpath}:{line_number}")
            if lowercase:
                tokens = [token.lower() for token in tokens]
            a, b, c, target = tokens
            group = "syntactic" if category.startswith("gram") else "semantic"
            for cdict in (groups[group], categories[category]):
                cdict["total"] += 1
            if any(x not in stoi for x in tokens):
                continue
            for cdict in (groups[group], categories[category]):
                cdict["eligible"] += 1
            query = emb[stoi[b]] - emb[stoi[a]] + emb[stoi[c]]
            norm = np.linalg.norm(query)
            if norm <= 1e-12:
                continue
            scores = normalized @ (query / norm)
            scores[[stoi[a], stoi[b], stoi[c]]] = -np.inf
            if int(np.argmax(scores)) == stoi[target]:
                for cdict in (groups[group], categories[category]):
                    cdict["correct"] += 1
    total = {
        key: sum(section[key] for section in groups.values())
        for key in ("total", "eligible", "correct")
    }
    return {
        "protocol": "3CosAdd-exclude-input-top1",
        "analogy_sha256": hashlib.sha256(qpath.read_bytes()).hexdigest(),
        "lowercase": lowercase,
        "vocab_size": len(words),
        "overall": _summary(total),
        "groups": {k: _summary(v) for k, v in sorted(groups.items())},
        "categories": {k: _summary(v) for k, v in sorted(categories.items())},
    }
