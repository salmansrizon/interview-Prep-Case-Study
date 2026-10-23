"""Shared logging setup — same ``get_logger(name)`` pattern as weeks 6-12."""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger without duplicating handlers.

    HIGHLIGHTS: Streamlit প্রতিটা interaction-এ app.py আবার চালায়; ``if not
    logger.handlers`` guard না থাকলে প্রতিবার নতুন handler যোগ হয়ে log line
    দুইবার-তিনবার ছাপা হত।
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    return logger
