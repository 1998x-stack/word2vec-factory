from __future__ import annotations

import torch


def build_linear_warmdown(total_steps: int):
    """线性从 1.0 → 0.0 的衰减函数。"""

    def lr_lambda(step: int) -> float:
        return max(0.0, 1.0 - step / max(total_steps, 1))

    return lr_lambda


def get_scheduler(optim: torch.optim.Optimizer, name: str, total_steps: int | None):
    if name == "linear" and total_steps is not None:
        return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=build_linear_warmdown(total_steps))
    return None
