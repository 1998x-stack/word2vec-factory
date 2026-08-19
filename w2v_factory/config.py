from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import yaml

from .data.subsample import _coerce_t


@dataclass
class RunCfg:
    out_dir: str = "runs/exp1"
    tb: bool = True

@dataclass
class DataCfg:
    input_files: list[str] = field(default_factory=list)
    lowercase: bool = True
    min_count: int = 5
    max_vocab: int = 1_000_000
    subsample_t: float | None = 1e-5
    tokenizer: str = "simple"
    max_sent_len: int = 10_000

@dataclass
class TrainCfg:
    epochs: int = 10
    batch_size: int = 1024
    lr: float = 0.025
    lr_schedule: str = "linear"  # linear | none
    optimizer: str = "sgd"
    device: str = "auto"

@dataclass
class ModelCfg:
    arch: str = "skipgram"  # skipgram | cbow
    dim: int = 300
    window: int = 5
    ns_neg_k: int = 5
    loss: str = "ns"        # ns | hs
    share_input_output: bool = False

@dataclass
class EvalCfg:
    analogy_file: str | None = None

@dataclass
class Cfg:
    SEED: int = 42
    RUN: RunCfg = field(default_factory=RunCfg)
    DATA: DataCfg = field(default_factory=DataCfg)
    TRAIN: TrainCfg = field(default_factory=TrainCfg)
    MODEL: ModelCfg = field(default_factory=ModelCfg)
    EVAL: EvalCfg = field(default_factory=EvalCfg)


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, dict) and k in a and isinstance(a[k], dict):
            a[k] = _merge(a[k], v)
        else:
            a[k] = v
    return a

def load_cfg(path: str) -> Cfg:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if "INCLUDE" in raw and raw["INCLUDE"]:
        base = load_cfg(raw["INCLUDE"])
        base_d = _to_dict(base)
        merged = _merge(base_d, {k: v for k, v in raw.items() if k != "INCLUDE"})
        return _from_dict(merged)
    return _from_dict(raw)

def _to_dict(cfg: Cfg) -> dict[str, Any]:
    return dataclasses.asdict(cfg)

def _filter_kwargs(cls, dct: dict[str, Any]) -> dict[str, Any]:
    """Strip unknown keys so configs don't crash on typos or moved fields."""
    allowed = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in dct.items() if k in allowed}

def _from_dict(d: dict[str, Any]) -> Cfg:
    run = RunCfg(**_filter_kwargs(RunCfg, d.get("RUN", {})))

    data_kwargs = _filter_kwargs(DataCfg, d.get("DATA", {}))
    data_kwargs["subsample_t"] = _coerce_t(data_kwargs.get("subsample_t"))
    data = DataCfg(**data_kwargs)

    train = TrainCfg(**_filter_kwargs(TrainCfg, d.get("TRAIN", {})))
    model = ModelCfg(**_filter_kwargs(ModelCfg, d.get("MODEL", {})))
    eval_ = EvalCfg(**_filter_kwargs(EvalCfg, d.get("EVAL", {})))
    return Cfg(SEED=d.get("SEED", 42), RUN=run, DATA=data, TRAIN=train, MODEL=model, EVAL=eval_)
