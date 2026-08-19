from __future__ import annotations

import numpy as np
import torch


def draw_negatives(alias_sampler, B: int, K: int, forbid: torch.Tensor | None = None) -> torch.Tensor:
    """从 alias 分布采样负样本；可选避免与正样本相同（forbid 为 [B] 正样本 id）。"""
    ne = alias_sampler.sample(B * K)
    ne = ne.reshape(B, K)
    if forbid is not None:
        # 简单重采样避免与目标相同
        for i in range(B):
            for j in range(K):
                if ne[i, j] == int(forbid[i].item()):
                    ne[i, j] = alias_sampler.sample(1)[0]
    return torch.from_numpy(ne.astype(np.int64))
