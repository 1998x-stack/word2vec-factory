from __future__ import annotations
from typing import List
import numpy as np
from pathlib import Path

def save_word2vec_txt(path: str, itos: List[str], emb: np.ndarray) -> None:
    """保存为 word2vec 文本格式（第一行：|V| dim）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(itos)} {emb.shape[1]}\n")
        for i, w in enumerate(itos):
            vec_str = " ".join(f"{x:.6f}" for x in emb[i].tolist())
            f.write(f"{w} {vec_str}\n")

def save_numpy(path: str, arr: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
