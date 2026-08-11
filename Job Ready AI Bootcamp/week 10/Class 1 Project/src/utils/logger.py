"""Structured logging setup.

Mirrors week 6/7/8's ``src/utils/logger.py`` exactly — one shared
``get_logger(name)`` factory so every module in this project logs in the
same format, and so handlers aren't accidentally duplicated if a module is
imported more than once (Streamlit's rerun model re-executes app.py on
every interaction, which makes this idempotency guard matter more here
than in a typical script).
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger, reusing existing handlers if present.

    HIGHLIGHTS: the ``if not logger.handlers`` guard is what makes this
    safe to call repeatedly (e.g. once per Streamlit rerun, or once per
    test). Without it, every call would attach a new StreamHandler and
    log lines would duplicate 2x, 3x, ... on each rerun.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    return logger
