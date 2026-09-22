from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import W2VOutput, Word2VecBase


class SkipGram(Word2VecBase):
    """Skip-gram: predict context words from a center word."""

    def forward_ns(self, centers: torch.Tensor, pos_ctx: torch.Tensor, neg_ctx: torch.Tensor) -> W2VOutput:
        """Negative-sampling loss; shapes: centers/pos_ctx [B], neg_ctx [B, K]."""
        v_c = self.in_embed(centers)
        u_o = self.out_embed(pos_ctx)
        pos_loss = F.logsigmoid((v_c * u_o).sum(dim=1))
        u_k = self.out_embed(neg_ctx)
        neg_scores = torch.bmm(u_k, v_c.unsqueeze(-1)).squeeze(-1)
        neg_loss = F.logsigmoid(-neg_scores).sum(dim=1)
        return W2VOutput(loss=-(pos_loss + neg_loss).mean())

    def forward_hs(
        self, centers: torch.Tensor, paths: torch.Tensor, codes: torch.Tensor, path_lens: torch.Tensor
    ) -> W2VOutput:
        """HS loss for centers [B] and their *context targets'* Huffman paths [B, L]."""
        if paths.shape[0] != centers.numel() or paths.shape != codes.shape or path_lens.shape != centers.shape:
            raise ValueError("Invalid hierarchical-softmax batch shapes")
        if paths.shape[1] == 0 or torch.any(path_lens == 0):
            raise ValueError("Hierarchical softmax requires non-empty target paths")

        vectors = self.in_embed(centers)
        nodes = self.out_embed(paths)
        scores = (nodes * vectors[:, None, :]).sum(dim=-1)
        signed_scores = (2 * codes.to(scores.dtype) - 1) * scores
        mask = torch.arange(paths.shape[1], device=paths.device)[None, :] < path_lens[:, None]
        losses = -(F.logsigmoid(signed_scores) * mask).sum(dim=1)
        return W2VOutput(loss=losses.mean())
