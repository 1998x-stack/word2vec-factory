from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(out_dir: str) -> None:
    """Configure Loguru with file + stderr sinks."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(Path(out_dir) / "train.log", level="INFO", rotation="50 MB", enqueue=True)
