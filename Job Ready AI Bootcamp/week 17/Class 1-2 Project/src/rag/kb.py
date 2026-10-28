"""Clinic handbook (data/kb/*.md) -> one Section per ``## `` heading, with a stable id like ``kb:fever:2``.

HIGHLIGHTS:
* কেন Markdown, PDF না? Handbook-এর আসল source Markdown; ``samples/clinic_handbook.pdf`` সেখান
  থেকেই বানানো (মানুষের পড়ার জন্য)। Markdown-এ প্রতিটা ``## `` section-ই একটা chunk — কোনো
  chunker লাগে না, আর citation হয় ঠিক section-এ: ``kb:fever:4`` = fever.md-এর ৪র্থ section।
* Section ছোট রেখে লেখা (প্রতিটা ~৪৫০ character-এর কম, test এটা যাচাই করে)। ছোট chunk-এ embedding
  তীক্ষ্ণ থাকে আর abstention threshold পরিষ্কারভাবে কাজ করে।
* প্রতিটা page-এর মাথায় "Educational, fictional ... not medical advice" লাইনটা প্রথম ``## ``-এর আগে, তাই
  index-এ ঢোকে না — কিন্তু প্রতিটা উত্তরের সাথে ``disclaimer`` field-এ যায়।

ponytail: id = section-এর অবস্থান। মাঝখানে নতুন section ঢোকালে পরের id সরে যায় (ingest সেটা sync করে,
কিন্তু eval-এর expected_source আবার দেখতে হয়)। নতুন section page-এর শেষে যোগ করুন।
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DISCLAIMER = "Educational, fictional clinic handbook; not medical advice."


@dataclass(frozen=True)
class Section:
    id: str          # kb:fever:2
    file: str        # fever.md
    page: str        # "Fever"
    title: str       # "Measuring a temperature"
    text: str

    @property
    def embed_text(self) -> str:
        """Page + heading + body: the heading carries most of the topic signal for short sections."""
        return f"{self.page}. {self.title}. {self.text}"


def load_kb(kb_dir: str | Path) -> list[Section]:
    sections: list[Section] = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        m = re.search(r"^# (.+)$", raw, re.M)
        page = m.group(1).strip() if m else path.stem
        for i, sec in enumerate(re.split(r"^## ", raw, flags=re.M)[1:], 1):
            title, _, body = sec.partition("\n")
            sections.append(Section(f"kb:{path.stem}:{i}", path.name, page, title.strip(), " ".join(body.split())))
    return sections
