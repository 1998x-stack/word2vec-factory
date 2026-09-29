from __future__ import annotations

import torch


class TokenProgressScheduler:
    """Linear LR decay driven by source-token progress rather than optimizer steps."""

    def __init__(self, optim: torch.optim.Optimizer, total_tokens: int) -> None:
        if total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        self.optim = optim
        self.total_tokens = total_tokens
        self.base_lrs = [float(group["lr"]) for group in optim.param_groups]
        self.progress = 0

    def state_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "base_lrs": list(self.base_lrs),
            "progress": self.progress,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("total_tokens") != self.total_tokens:
            raise ValueError("Scheduler total_tokens does not match checkpoint")
        base_lrs = list(state.get("base_lrs", []))
        if base_lrs != self.base_lrs:
            raise ValueError("Scheduler base learning rates do not match checkpoint")
        progress = state.get("progress")
        if not isinstance(progress, int):
            raise ValueError("Scheduler checkpoint progress must be an integer")
        # Restore through the same invariant-checked path.
        self.progress = 0
        self.set_progress(progress)

    def set_progress(self, processed_tokens: int) -> None:
        if processed_tokens < self.progress:
            raise ValueError("token progress must be monotonic")
        if processed_tokens < 0 or processed_tokens > self.total_tokens:
            raise ValueError("token progress is outside the training plan")
        self.progress = processed_tokens
        ratio = max(0.0, 1.0 - processed_tokens / self.total_tokens)
        for group, base_lr in zip(self.optim.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * ratio


def get_scheduler(
    optim: torch.optim.Optimizer,
    name: str,
    total_tokens: int | None,
) -> TokenProgressScheduler | None:
    if name == "none":
        return None
    if name == "linear":
        if total_tokens is None or total_tokens <= 0:
            raise ValueError("linear lr_schedule requires positive total_tokens")
        return TokenProgressScheduler(optim, total_tokens)
    raise ValueError(f"Unsupported lr_schedule: {name}")
