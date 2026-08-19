from __future__ import annotations

import torch


def pack_hs_batch(
    word_ids: torch.Tensor, paths: list[list[int]], codes: list[list[int]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把不同长度的 Huffman 路径打包为定长张量，返回 (paths_t, codes_t, lens_t)。"""
    B = len(word_ids)
    Lmax = max((len(paths[int(w.item())]) for w in word_ids), default=0)
    paths_t = torch.zeros((B, Lmax), dtype=torch.long)
    codes_t = torch.zeros((B, Lmax), dtype=torch.long)
    lens_t = torch.zeros((B,), dtype=torch.long)
    for i, w in enumerate(word_ids):
        p = paths[int(w.item())]
        c = codes[int(w.item())]
        lens_t[i] = len(p)
        if p:
            paths_t[i, : len(p)] = torch.tensor(p, dtype=torch.long)
            codes_t[i, : len(c)] = torch.tensor(c, dtype=torch.long)
    return paths_t, codes_t, lens_t
