from __future__ import annotations

import torch

from .base import W2VOutput, Word2VecBase


class CBOW(Word2VecBase):
    """CBOW: 平均上下文向量预测中心词。"""

    def forward_ns(
        self, ctxs: torch.Tensor, ctx_lens: torch.Tensor, targets: torch.Tensor, neg_targets: torch.Tensor
    ) -> W2VOutput:
        """
        ctxs: [B, Lmax]
        ctx_lens: [B]
        targets: [B]
        neg_targets: [B, K]
        """
        B, Lmax = ctxs.shape
        v_ctx = self.in_embed(ctxs)  # [B, Lmax, D]
        mask = torch.arange(Lmax, device=ctxs.device)[None, :] < ctx_lens[:, None]
        sum_ctx = (v_ctx * mask.unsqueeze(-1)).sum(dim=1)  # [B, D]
        mean_ctx = sum_ctx / ctx_lens.clamp_min(1).unsqueeze(-1)

        u_pos = self.out_embed(targets)  # [B, D]
        pos = torch.sum(mean_ctx * u_pos, dim=1)
        pos_loss = torch.nn.functional.logsigmoid(pos)

        u_neg = self.out_embed(neg_targets)  # [B, K, D]
        score_neg = torch.bmm(u_neg, mean_ctx.unsqueeze(-1)).squeeze(-1)  # [B, K]
        neg_loss = torch.nn.functional.logsigmoid(-score_neg).sum(dim=1)
        loss = -(pos_loss + neg_loss).mean()
        return W2VOutput(loss=loss)

    def forward_hs(
        self,
        ctxs: torch.Tensor,
        ctx_lens: torch.Tensor,
        paths: torch.Tensor,
        codes: torch.Tensor,
        path_lens: torch.Tensor,
    ) -> W2VOutput:
        """
        层次 softmax for CBOW。
        """
        B, Lmax = ctxs.shape
        v_ctx = self.in_embed(ctxs)
        mask2 = torch.arange(Lmax, device=ctxs.device)[None, :] < ctx_lens[:, None]
        sum_ctx = (v_ctx * mask2.unsqueeze(-1)).sum(dim=1)
        mean_ctx = sum_ctx / ctx_lens.clamp_min(1).unsqueeze(-1)

        losses = []
        for i in range(B):
            L = int(path_lens[i].item())
            if L == 0:
                continue
            nodes = paths[i, :L]
            code = codes[i, :L].float()
            u = self.out_embed(nodes)
            s = torch.mv(u, mean_ctx[i])
            t = (2.0 * code - 1.0) * s
            loss_i = -torch.nn.functional.logsigmoid(t).sum()
            losses.append(loss_i)
        loss = torch.stack(losses).mean()
        return W2VOutput(loss=loss)
