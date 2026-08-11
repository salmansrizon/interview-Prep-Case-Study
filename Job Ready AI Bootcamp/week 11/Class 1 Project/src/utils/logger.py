"""Structured logging setup.

Mirrors week 8/10's ``src/utils/logger.py`` exactly — one shared
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

    HIGHLIGHTS: ``if not logger.handlers`` guard-টাই এটাকে বারবার call করা
    safe বানায় (যেমন প্রতিটা Streamlit rerun-এ একবার, বা প্রতিটা test-এ
    একবার)। এটা ছাড়া, প্রতিটা call একটা নতুন StreamHandler attach করত, আর
    log line গুলো প্রতি rerun-এ 2x, 3x... করে ডুপ্লিকেট হয়ে যেত।
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    return logger
