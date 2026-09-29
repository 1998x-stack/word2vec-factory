from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, legacy NumPy, and PyTorch global RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_numpy_rng(seed: int, *stream_ids: int) -> np.random.Generator:
    """Create an independent deterministic NumPy RNG stream.

    Stream identifiers make randomness component-local: advancing negative
    sampling must not perturb context-window sampling or subsampling.
    """
    if any(not isinstance(stream_id, int) or isinstance(stream_id, bool) for stream_id in stream_ids):
        raise ValueError("RNG stream ids must be integers")
    sequence = np.random.SeedSequence([seed, *stream_ids])
    return np.random.default_rng(sequence)


def pick_device(cfg_device: str) -> torch.device:
    """根据配置选择设备。'auto' 则优先 CUDA。"""
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)
