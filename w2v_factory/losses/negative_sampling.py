from __future__ import annotations

import numpy as np
import torch


def draw_negatives(alias_sampler, B: int, K: int, forbid: torch.Tensor | None = None) -> torch.Tensor:
    """Draw ``[B, K]`` negatives, excluding each row's positive word if supplied.

    Rejection sampling preserves the unigram distribution conditional on the
    positive word not being drawn. The exclusion applies to every draw, not just
    the first attempt. Convert GPU targets to CPU once rather than per sample.
    """
    if B < 0 or K < 0:
        raise ValueError("B and K must be non-negative")
    if forbid is not None and forbid.shape != (B,):
        raise ValueError("forbid must have shape [B]")

    negatives = np.asarray(alias_sampler.sample(B * K), dtype=np.int64).reshape(B, K)
    if forbid is None or not negatives.size:
        return torch.from_numpy(negatives)

    positives = forbid.detach().cpu().numpy().astype(np.int64, copy=False)
    if np.any((positives < 0) | (positives >= alias_sampler.n)):
        raise ValueError("forbid contains a word outside the sampler vocabulary")
    if alias_sampler.n < 2:
        raise ValueError("Cannot exclude a positive word from a one-word vocabulary")

    # Genuine AliasSampler distributions have positive mass for each vocabulary
    # word. Limit retries as a safeguard against malformed custom samplers.
    for _ in range(10_000):
        conflicts = negatives == positives[:, None]
        if not conflicts.any():
            return torch.from_numpy(negatives)
        negatives[conflicts] = alias_sampler.sample(int(conflicts.sum()))

    raise RuntimeError("Negative sampler could not draw a word other than the positive word")
