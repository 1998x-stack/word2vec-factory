from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import W2VOutput, Word2VecBase


class CBOW(Word2VecBase):
    """CBOW: 平均上下文向量预测中心词。"""

    def _mean_context(self, ctxs: torch.Tensor, ctx_lens: torch.Tensor) -> torch.Tensor:
        if ctxs.ndim != 2 or ctx_lens.ndim != 1 or ctxs.shape[0] != ctx_lens.numel():
            raise ValueError("Invalid CBOW context batch shapes")
        if torch.any(ctx_lens <= 0) or torch.any(ctx_lens > ctxs.shape[1]):
            raise ValueError("CBOW context lengths must be within [1, Lmax]")

        _, max_len = ctxs.shape
        vectors = self.in_embed(ctxs)
        mask = torch.arange(max_len, device=ctxs.device)[None, :] < ctx_lens[:, None]
        summed = (vectors * mask.unsqueeze(-1)).sum(dim=1)
        return summed / ctx_lens.unsqueeze(-1)

    def forward_ns(
        self, ctxs: torch.Tensor, ctx_lens: torch.Tensor, targets: torch.Tensor, neg_targets: torch.Tensor
    ) -> W2VOutput:
        """
        ctxs: [B, Lmax]
        ctx_lens: [B]
        targets: [B]
        neg_targets: [B, K]
        """
        mean_ctx = self._mean_context(ctxs, ctx_lens)

        u_pos = self.out_embed(targets)
        pos = torch.sum(mean_ctx * u_pos, dim=1)
        pos_loss = F.logsigmoid(pos)

        u_neg = self.out_embed(neg_targets)
        score_neg = torch.bmm(u_neg, mean_ctx.unsqueeze(-1)).squeeze(-1)
        neg_loss = F.logsigmoid(-score_neg).sum(dim=1)
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
        """Vectorized hierarchical-softmax loss for CBOW."""
        batch_size = ctxs.shape[0]
        if paths.shape[0] != batch_size or paths.shape != codes.shape or path_lens.shape != ctx_lens.shape:
            raise ValueError("Invalid hierarchical-softmax batch shapes")
        if paths.shape[1] == 0 or torch.any(path_lens <= 0) or torch.any(path_lens > paths.shape[1]):
            raise ValueError("Hierarchical softmax requires valid non-empty target paths")

        mean_ctx = self._mean_context(ctxs, ctx_lens)
        nodes = self.out_embed(paths)
        scores = (nodes * mean_ctx[:, None, :]).sum(dim=-1)
        signed_scores = (2 * codes.to(scores.dtype) - 1) * scores
        mask = torch.arange(paths.shape[1], device=paths.device)[None, :] < path_lens[:, None]
        losses = -(F.logsigmoid(signed_scores) * mask).sum(dim=1)
        return W2VOutput(loss=losses.mean())
