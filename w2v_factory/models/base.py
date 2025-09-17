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
    def __init__(self, vocab_size: int, dim: int, share_io: bool = False,
                 out_vocab_size: int | None = None) -> None:
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

        # 初始化（与 word2vec 类似的小范围均匀分布）
        bound = 0.5 / dim
        nn.init.uniform_(self.in_embed.weight, -bound, bound)
        if not share_io:
            nn.init.zeros_(self.out_embed.weight)
