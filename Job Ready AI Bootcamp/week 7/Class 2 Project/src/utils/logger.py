"""
Structured logging using loguru.
Provides consistent, colorized logs across the project.
"""

import sys
import os
from loguru import logger

from src.config import get_config


def setup_logger() -> None:
    """Configure loguru logger with file and console sinks."""
    config = get_config()
    logger.remove()

    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
               "<level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    log_file = os.path.join(config.paths.logs_dir, "app.log")
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="1 week",
        encoding="utf-8",
    )

    logger.info("Logger initialized. Logs dir: {}", config.paths.logs_dir)


info = logger.info
debug = logger.debug
warning = logger.warning
error = logger.error
critical = logger.critical
