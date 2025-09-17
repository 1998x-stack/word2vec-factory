from __future__ import annotations
import argparse
import numpy as np
from typing import Dict, List
from loguru import logger
from ..trainer.metrics import load_analogy, evaluate_analogy

def load_word2vec_txt(path: str) -> tuple[List[str], np.ndarray, Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        try:
            vocab_size, dim = int(header[0]), int(header[1])
        except Exception:
            # 有些导出可能不含 header；尝试全量读取
            f.seek(0)
            vocab_size = None
            dim = None
            lines = f.readlines()
            words, vecs = [], []
            for ln in lines:
                sp = ln.strip().split()
                words.append(sp[0])
                vecs.append([float(x) for x in sp[1:]])
            emb = np.asarray(vecs, dtype=np.float32)
            stoi = {w: i for i, w in enumerate(words)}
            return words, emb, stoi

        words, vecs = [], []
        for _ in range(vocab_size):
            sp = f.readline().strip().split()
            words.append(sp[0])
            vecs.append([float(x) for x in sp[1:]])
        emb = np.asarray(vecs, dtype=np.float32)
        stoi = {w: i for i, w in enumerate(words)}
        return words, emb, stoi

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate word vectors on Google Analogy task")
    p.add_argument("--vectors", type=str, required=True)
    p.add_argument("--analogy", type=str, required=True)
    p.add_argument("--topk", type=int, default=1)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    words, emb, stoi = load_word2vec_txt(args.vectors)
    items = load_analogy(args.analogy)
    acc = evaluate_analogy(emb, stoi, words, items, topk=args.topk)
    logger.info(f"Analogy accuracy@{args.topk}: {acc:.4f}")

if __name__ == "__main__":
    main()
