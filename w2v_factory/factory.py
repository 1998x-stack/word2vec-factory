"""Unified factory façade: share PR7 model registry and add optimizer creation."""
from __future__ import annotations

import torch

from .models import make_model
from .registry import Registry

OPTIMIZER_REG = Registry()


@OPTIMIZER_REG.register("sgd")
def _sgd(*, params, lr):
    return torch.optim.SGD(params, lr=lr)


@OPTIMIZER_REG.register("adam")
def _adam(*, params, lr):
    return torch.optim.Adam(params, lr=lr)


def make_optimizer(name: str, *, params, lr: float):
    if name not in {"sgd", "adam"}:
        raise ValueError(f"Unsupported optimizer: {name}")
    return OPTIMIZER_REG.create(name, params=params, lr=lr)
