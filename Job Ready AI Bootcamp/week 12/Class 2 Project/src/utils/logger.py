"""Structured logging setup.

Mirrors the repo-wide ``src/utils/logger.py`` pattern used in weeks 6-11 —
one shared ``get_logger(name)`` factory so every module in this project logs
in the same format, and so handlers aren't accidentally duplicated if a
module is imported more than once (Streamlit's rerun model re-executes
app.py on every interaction, which makes this idempotency guard matter more
here than in a typical script).
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger, reusing existing handlers if present.

    HIGHLIGHTS: ``if not logger.handlers`` guard-টা এটাকে বারবার call করা
    নিরাপদ করে (যেমন প্রতিটা Streamlit rerun-এ, বা প্রতিটা test-এ)। এই guard
    ছাড়া প্রতিটা call একটা নতুন StreamHandler যোগ করত, আর log line গুলো
    2x, 3x করে ডুপ্লিকেট হয়ে যেত।
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    return logger
