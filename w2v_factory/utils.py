from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """设置全局随机种子，确保可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(cfg_device: str) -> torch.device:
    """根据配置选择设备。'auto' 则优先 CUDA。"""
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def linear_decay(it: int, total: int) -> float:
    """线性学习率衰减系数（从1到0）。"""
    return max(0.0, 1.0 - it / float(total))
