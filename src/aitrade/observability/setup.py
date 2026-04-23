"""loguru configuration. Call ``configure_logging()`` once at process start."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def configure_logging(log_dir: Path, level: str = "INFO") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> "
        "<cyan>{name}</cyan>: {message}",
    )
    logger.add(
        log_dir / "aitrade-{time:YYYY-MM-DD}.log",
        level=level,
        rotation="1 day",
        retention="30 days",
        compression="gz",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
