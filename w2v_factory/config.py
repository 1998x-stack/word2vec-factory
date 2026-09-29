from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
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
    loss: str = "ns"  # ns | hs
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


_TOP_LEVEL_FIELDS = {"SEED", "RUN", "DATA", "TRAIN", "MODEL", "EVAL"}


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, dict) and k in a and isinstance(a[k], dict):
            a[k] = _merge(a[k], v)
        else:
            a[k] = v
    return a


def _require_mapping(value: Any, section: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a mapping")
    if any(not isinstance(k, str) for k in value):
        raise ValueError(f"{section} keys must be strings")
    return value


def _assert_known_keys(section: str, dct: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(dct) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown {section} config key(s): {joined}")


def _strict_kwargs(cls, value: Any, section: str) -> dict[str, Any]:
    dct = _require_mapping(value, section)
    allowed = {f.name for f in dataclasses.fields(cls)}
    _assert_known_keys(section, dct, allowed)
    return dict(dct)


def load_cfg(path: str, *, _stack: tuple[Path, ...] = ()) -> Cfg:
    cfg_path = Path(path).expanduser().resolve()
    if cfg_path in _stack:
        chain = " -> ".join(str(p) for p in (*_stack, cfg_path))
        raise ValueError(f"Cyclic config INCLUDE detected: {chain}")

    with cfg_path.open(encoding="utf-8") as f:
        raw = _require_mapping(yaml.safe_load(f), f"config {cfg_path}")

    _assert_known_keys(f"top-level ({cfg_path})", raw, _TOP_LEVEL_FIELDS | {"INCLUDE"})
    include = raw.get("INCLUDE")
    overrides = {k: v for k, v in raw.items() if k != "INCLUDE"}

    if include is None or include == "":
        return _from_dict(overrides)
    if not isinstance(include, str):
        raise ValueError(f"INCLUDE in {cfg_path} must be a path string")

    include_path = Path(include).expanduser()
    if not include_path.is_absolute():
        include_path = cfg_path.parent / include_path

    base = load_cfg(str(include_path), _stack=(*_stack, cfg_path))
    merged = _merge(_to_dict(base), overrides)
    return _from_dict(merged)


def _to_dict(cfg: Cfg) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def _from_dict(d: dict[str, Any]) -> Cfg:
    root = _require_mapping(d, "config")
    _assert_known_keys("top-level", root, _TOP_LEVEL_FIELDS)

    run = RunCfg(**_strict_kwargs(RunCfg, root.get("RUN", {}), "RUN"))

    data_kwargs = _strict_kwargs(DataCfg, root.get("DATA", {}), "DATA")
    data_kwargs["subsample_t"] = _coerce_t(data_kwargs.get("subsample_t"))
    data = DataCfg(**data_kwargs)

    train = TrainCfg(**_strict_kwargs(TrainCfg, root.get("TRAIN", {}), "TRAIN"))
    model = ModelCfg(**_strict_kwargs(ModelCfg, root.get("MODEL", {}), "MODEL"))
    eval_ = EvalCfg(**_strict_kwargs(EvalCfg, root.get("EVAL", {}), "EVAL"))

    seed = root.get("SEED", 42)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("SEED must be an integer")

    return Cfg(SEED=seed, RUN=run, DATA=data, TRAIN=train, MODEL=model, EVAL=eval_)
