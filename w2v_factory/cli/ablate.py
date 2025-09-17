from __future__ import annotations
import argparse
import os
import yaml
from copy import deepcopy
from ..config import load_cfg, _to_dict, _from_dict
from ..log import setup_logging
from ..trainer.engine import Trainer
from loguru import logger

def parse_args():
    p = argparse.ArgumentParser(description="Run ablation sweeps")
    p.add_argument("--grid", type=str, required=True, help="configs/ablation_grid.yaml")
    return p.parse_args()

def set_by_path(d, path, value):
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value

def main():
    args = parse_args()
    with open(args.grid, "r", encoding="utf-8") as f:
        grid_cfg = yaml.safe_load(f)
    base_cfg_path = grid_cfg["base_cfg"]
    choices = grid_cfg["grid"]
    out_root = grid_cfg.get("out_root", "runs/ablations")
    base = load_cfg(base_cfg_path)
    base_d = _to_dict(base)

    # 笛卡尔积
    import itertools
    keys = list(choices.keys())
    values = [choices[k] for k in keys]
    for vals in itertools.product(*values):
        d = deepcopy(base_d)
        name_parts = []
        for k, v in zip(keys, vals):
            set_by_path(d, k, v)
            name_parts.append(f"{k.replace('.', '_')}={v}")
        cfg = _from_dict(d)
        cfg.RUN.out_dir = os.path.join(out_root, "__".join(name_parts))
        setup_logging(cfg.RUN.out_dir)
        logger.info(f"Running ablation: {cfg.RUN.out_dir}")
        Trainer(cfg).train()

if __name__ == "__main__":
    main()
