"""Save the final report to a Markdown file."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def save_report(text: str, topic: str, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:50] or "report"
    path = out_dir / f"{slug}-{datetime.now():%Y%m%d-%H%M%S}.md"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path
