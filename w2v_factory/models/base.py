from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class W2VOutput:
    loss: torch.Tensor


class Word2VecBase(nn.Module):
    """Word2Vec base module storing shared embeddings.

    Args:
        vocab_size: 输入（词/中心词）词表大小。
        dim: 向量维度。
        share_io: 是否共享输入/输出嵌入（仅当输出节点数与 vocab_size 相同才允许）。
        out_vocab_size: 输出端节点数（NS=V；HS=2*V-1，包含内部节点）。
    """

    def __init__(self, vocab_size: int, dim: int, share_io: bool = False, out_vocab_size: int | None = None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        out_size = out_vocab_size or vocab_size

        if share_io and out_size != vocab_size:
            raise ValueError("share_input_output=True 仅在 out_vocab_size==vocab_size 时可用（HS 不支持共享）。")

        self.in_embed = nn.Embedding(vocab_size, dim)  # input/center
        if share_io:
            self.out_embed = self.in_embed
        else:
            self.out_embed = nn.Embedding(out_size, dim)  # output/context or HS nodes

        # 初始化：输入与输出嵌入都使用均匀随机分布（scale = 1/sqrt(dim)）。
        # 注意：不要将输出嵌入初始化为 0 —— 零向量会使输入嵌入在冷启动时梯度为 0，
        # 导致模型无法学习（实测 loss 卡死在初始值）。scale 取 1/sqrt(dim) 而非
        # 0.5/dim，避免在高维时向量过小、梯度过小而几乎不更新。
        bound = 1.0 / (dim**0.5)
        nn.init.uniform_(self.in_embed.weight, -bound, bound)
        if not share_io:
            nn.init.uniform_(self.out_embed.weight, -bound, bound)
