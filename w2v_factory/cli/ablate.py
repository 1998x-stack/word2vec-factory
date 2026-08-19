from __future__ import annotations

import argparse
import itertools
import os
from copy import deepcopy
from typing import Any

import yaml
from loguru import logger

from ..config import _from_dict, _to_dict, load_cfg
from ..log import setup_logging
from ..trainer.engine import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ablation sweeps")
    p.add_argument("--grid", type=str, required=True, help="configs/ablation_grid.yaml")
    return p.parse_args()


def set_by_path(d: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur: dict[str, Any] = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def main() -> None:
    args = parse_args()
    with open(args.grid, encoding="utf-8") as f:
        grid_cfg = yaml.safe_load(f)
    base_cfg_path = grid_cfg["base_cfg"]
    choices = grid_cfg["grid"]
    out_root = grid_cfg.get("out_root", "runs/ablations")
    base = load_cfg(base_cfg_path)
    base_d = _to_dict(base)

    keys = list(choices.keys())
    values = [choices[k] for k in keys]
    for vals in itertools.product(*values):
        d = deepcopy(base_d)
        name_parts = []
        for k, v in zip(keys, vals, strict=True):
            set_by_path(d, k, v)
            name_parts.append(f"{k.replace('.', '_')}={v}")
        cfg = _from_dict(d)
        cfg.RUN.out_dir = os.path.join(out_root, "__".join(name_parts))
        setup_logging(cfg.RUN.out_dir)
        logger.info(f"Running ablation: {cfg.RUN.out_dir}")
        Trainer(cfg).train()


if __name__ == "__main__":
    main()
