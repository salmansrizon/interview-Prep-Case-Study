"""Structured logging setup, mirroring the convention used across the repo."""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create (or fetch) a configured logger.

    Args:
        name: Logger name, typically ``__name__`` of the caller.
        level: Logging level, defaults to ``logging.INFO``.

    Returns:
        A ``logging.Logger`` with a single stdout handler attached.

    HIGHLIGHTS: the ``if not logger.handlers`` guard exists because
    ``logging.getLogger(name)`` returns the SAME logger object every time
    it's called with the same name (Python's logging module caches loggers
    globally). Without the guard, calling ``get_logger("src.data.loader")``
    from multiple modules — or re-importing during a Streamlit rerun —
    would attach a new duplicate handler each time, and every log line
    would print itself twice, three times, etc. The guard makes this
    function idempotent: safe to call as often as you like.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    return logger
