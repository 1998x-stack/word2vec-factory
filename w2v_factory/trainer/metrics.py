from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

def load_analogy(path: str) -> List[Tuple[str, str, str, str]]:
    """读取 Google analogy 文件。支持": "分区标题行。"""
    items: List[Tuple[str, str, str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith(":"):
                continue
            a, b, c, d = line.strip().split()
            items.append((a, b, c, d))
    return items

def evaluate_analogy(emb: np.ndarray, stoi: Dict[str, int], itos: List[str],
                     items: List[Tuple[str, str, str, str]], topk: int = 1) -> float:
    """3CosAdd 评测 top-k@1 准确率（严格匹配）。
    Note: 同论文评测一致，若目标不在词表则跳过。
    """
    W = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    hit, tot = 0, 0
    for a, b, c, d in items:
        if a not in stoi or b not in stoi or c not in stoi or d not in stoi:
            continue
        va, vb, vc = W[stoi[a]], W[stoi[b]], W[stoi[c]]
        q = vb - va + vc
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = W @ q
        # 避免把 a,b,c 本身作为答案
        for idx in (stoi[a], stoi[b], stoi[c]):
            sims[idx] = -1.0
        rank = np.argsort(-sims)[:topk]
        if stoi[d] in rank:
            hit += 1
        tot += 1
    return hit / max(tot, 1)
