from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_word2vec_txt(path: str, itos: list[str], emb: np.ndarray) -> None:
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


def save_json(path: str, payload: dict[str, Any]) -> None:
    """Write a stable, human-readable JSON artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
