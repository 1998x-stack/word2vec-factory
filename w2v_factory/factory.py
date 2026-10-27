"""Factories used by the production trainer, not just metadata registries."""
from __future__ import annotations

import torch

from .models.cbow import CBOW
from .models.skipgram import SkipGram
from .registry import MODEL_REG, Registry

OPTIMIZER_REG = Registry()


@MODEL_REG.register("cbow")
def _cbow(**kwargs):
    return CBOW(**kwargs)


@MODEL_REG.register("skipgram")
def _skipgram(**kwargs):
    return SkipGram(**kwargs)


@OPTIMIZER_REG.register("sgd")
def _sgd(*, params, lr):
    return torch.optim.SGD(params, lr=lr)


@OPTIMIZER_REG.register("adam")
def _adam(*, params, lr):
    return torch.optim.Adam(params, lr=lr)


def make_model(name: str, *, vocab_size: int, dim: int, share_io: bool, out_vocab_size: int):
    return MODEL_REG.create(
        name, vocab_size=vocab_size, dim=dim,
        share_io=share_io, out_vocab_size=out_vocab_size,
    )


def make_optimizer(name: str, *, params, lr: float):
    return OPTIMIZER_REG.create(name, params=params, lr=lr)
