from __future__ import annotations

import torch

from .base import W2VOutput, Word2VecBase


class SkipGram(Word2VecBase):
    """Skip-gram: 以中心词预测上下文词。"""

    def forward_ns(self, centers: torch.Tensor, pos_ctx: torch.Tensor, neg_ctx: torch.Tensor) -> W2VOutput:
        """负采样损失。
        centers: [B]
        pos_ctx: [B]
        neg_ctx: [B, K]
        """
        v_c = self.in_embed(centers)  # [B, D]
        u_o = self.out_embed(pos_ctx)  # [B, D]
        score_pos = torch.sum(v_c * u_o, dim=1)  # [B]
        # 正样本 log σ(v·u)
        pos_loss = torch.nn.functional.logsigmoid(score_pos)

        u_k = self.out_embed(neg_ctx)  # [B, K, D]
        score_neg = torch.bmm(u_k, v_c.unsqueeze(-1)).squeeze(-1)  # [B, K]
        # 负样本 log σ(-v·u_k)
        neg_loss = torch.nn.functional.logsigmoid(-score_neg).sum(dim=1)

        loss = -(pos_loss + neg_loss).mean()
        return W2VOutput(loss=loss)

    def forward_hs(
        self, centers: torch.Tensor, paths: torch.Tensor, codes: torch.Tensor, path_lens: torch.Tensor
    ) -> W2VOutput:
        """层次 softmax 损失。
        Args:
            centers: [B]
            paths:   [B, Lmax] 内部节点 id（映射到 out_embed）
            codes:   [B, Lmax] 0/1
            path_lens: [B] 实际长度
        """

        B, Lmax = paths.shape
        v_c = self.in_embed(centers)  # [B, D]
        losses = []
        for i in range(B):
            L = int(path_lens[i].item())
            if L == 0:
                continue
            nodes = paths[i, :L]  # [L]
            code = codes[i, :L].float()  # [L]
            u = self.out_embed(nodes)  # [L, D]
            # σ( (2*code-1) * u·v )
            s = torch.mv(u, v_c[i])  # [L]
            t = (2.0 * code - 1.0) * s
            loss_i = -torch.nn.functional.logsigmoid(t).sum()
            losses.append(loss_i)
        loss = torch.stack(losses).mean()
        return W2VOutput(loss=loss)
