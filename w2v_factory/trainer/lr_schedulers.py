from __future__ import annotations

import torch


def build_linear_warmdown(total_steps: int):
    """线性从 1.0 → 0.0 的衰减函数。"""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    def lr_lambda(step: int) -> float:
        return max(0.0, 1.0 - step / total_steps)

    return lr_lambda


def get_scheduler(optim: torch.optim.Optimizer, name: str, total_steps: int | None):
    if name == "none":
        return None
    if name == "linear":
        if total_steps is None or total_steps <= 0:
            raise ValueError("linear lr_schedule requires positive total_steps")
        return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=build_linear_warmdown(total_steps))
    raise ValueError(f"Unsupported lr_schedule: {name}")
