"""
Structured logging setup shared across the project.

HIGHLIGHTS — why a shared `get_logger()` helper instead of every module
calling `print()` or `logging.getLogger(__name__)` directly? Two reasons
specific to this project:
  1. Every module (loader, preprocessor, cnn, registry, trainer, app) logs
     things like "MNIST loaded — train=54000, val=6000" or "Saved model
     version ...". Routing all of it through one formatter means every
     line has a consistent `timestamp | module | level | message` shape,
     which makes it far easier for a student to trace *which* step of the
     pipeline produced a given line when debugging.
  2. Streamlit re-runs the entire `app.py` script top-to-bottom on every
     UI interaction (a button click, a slider drag). Without the
     `if not logger.handlers:` guard below, calling `get_logger("app")`
     on each re-run would attach a *new* `StreamHandler` every time,
     and within a few clicks every log line would print itself 5, then
     10, then 20 times. Checking `logger.handlers` first makes
     `get_logger` idempotent — safe to call repeatedly with the same name.
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured :class:`logging.Logger`.

    Handlers are only attached once per logger name so repeated calls
    (e.g. from Streamlit re-running the script on every interaction)
    never produce duplicate log lines.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        # HIGHLIGHTS: without disabling propagation, this logger's
        # records would also bubble up to the root logger, which (if a
        # student's environment or a library configures its own root
        # handler) can print every message a second time from a
        # different handler. Setting `propagate = False` keeps this
        # logger's output isolated to the one handler we just attached.
        logger.propagate = False

    return logger
