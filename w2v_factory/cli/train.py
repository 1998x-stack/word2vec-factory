from __future__ import annotations

import argparse

from loguru import logger

from ..config import load_cfg
from ..log import setup_logging
from ..trainer.engine import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Word2Vec (CBOW/Skip-gram) with HS/NS")
    p.add_argument("--cfg", type=str, required=True, help="Path to YAML config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    setup_logging(cfg.RUN.out_dir)
    logger.info(f"Loaded cfg from {args.cfg}")
    tr = Trainer(cfg)
    tr.train()


if __name__ == "__main__":
    main()
