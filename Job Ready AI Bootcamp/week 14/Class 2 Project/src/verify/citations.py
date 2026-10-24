"""
Deterministic checks on agent output — no LLM involved.

HIGHLIGHTS: MAST taxonomy-তে multi-agent failure-এর প্রায় এক-চতুর্থাংশ
"task verification"-এর ভুল, আর একটা verification step যোগ করাই সবচেয়ে
কার্যকর fix-গুলোর একটা। LLM reviewer একই model দিয়ে নিজের লেখাকে grade
করলে self-preference bias আছে — তাই যত check সম্ভব, **code**-এ:

* citation id আসল কিনা (corpus-এ আছে কিনা)
* report-এর cited id research notes-এ ছিল কিনা (writer নতুন source বানায়নি)
* প্রতিটা cited বাক্যের content word-এর কত অংশ cited passage-এ আছে (support score)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.tools.corpus_search import Corpus, tokens

CITE = re.compile(r"\[([a-z0-9_]+:\d+)\]")


def cited_ids(text: str) -> list[str]:
    return CITE.findall(text)


def sentences(text: str) -> list[str]:
    body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    parts = re.split(r"(?<=[.!?])\s+|\n+", body)
    return [s.strip(" -*•\t") for s in parts if len(tokens(s)) >= 4]


@dataclass
class SentenceCheck:
    sentence: str
    ids: list[str]
    support: float
    verdict: str          # supported | weak | uncited | invalid_id


@dataclass
class Verification:
    checks: list[SentenceCheck] = field(default_factory=list)
    invalid_ids: list[str] = field(default_factory=list)
    not_in_notes: list[str] = field(default_factory=list)

    def count(self, verdict: str) -> int:
        return sum(c.verdict == verdict for c in self.checks)

    @property
    def cited(self) -> int:
        return sum(bool(c.ids) for c in self.checks)

    @property
    def citation_precision(self) -> float:
        """Share of cited sentences whose citations are valid AND supported."""
        return round(self.count("supported") / self.cited, 2) if self.cited else 0.0

    def summary(self) -> dict:
        return {"sentences": len(self.checks), "cited": self.cited, "supported": self.count("supported"),
                "weak": self.count("weak"), "uncited": self.count("uncited"),
                "invalid_ids": sorted(set(self.invalid_ids)), "not_in_notes": sorted(set(self.not_in_notes)),
                "citation_precision": self.citation_precision}

    def feedback(self) -> str:
        lines = []
        if self.invalid_ids:
            lines.append(f"These citation ids do not exist: {sorted(set(self.invalid_ids))}. Use only ids returned by search_corpus.")
        if self.not_in_notes:
            lines.append(f"These ids were not in the research notes: {sorted(set(self.not_in_notes))}.")
        uncited = [c.sentence for c in self.checks if c.verdict == "uncited"][:3]
        if uncited:
            lines.append("Add a citation id to these factual sentences or delete them: " + " | ".join(uncited))
        weak = [c.sentence for c in self.checks if c.verdict == "weak"][:3]
        if weak:
            lines.append("These sentences are not supported by the passage they cite; rewrite them to match the source "
                         "or remove them: " + " | ".join(weak))
        return "\n".join(lines)


def support(sentence: str, passages: list[str]) -> float:
    words = {w for w in tokens(CITE.sub("", sentence)) if not w.isdigit() or len(w) > 1}
    if not words:
        return 0.0
    source = set(tokens(" ".join(passages)))
    return round(len(words & source) / len(words), 2)


def verify(text: str, corpus: Corpus, notes: str | None = None, min_support: float = 0.45) -> Verification:
    allowed = set(cited_ids(notes)) if notes is not None else None
    v = Verification()
    for s in sentences(text):
        ids = cited_ids(s)
        bad = [i for i in ids if i not in corpus.by_id]
        v.invalid_ids += bad
        if allowed is not None:
            v.not_in_notes += [i for i in ids if i in corpus.by_id and i not in allowed]
        if not ids:
            v.checks.append(SentenceCheck(s, [], 0.0, "uncited"))
        elif bad:
            v.checks.append(SentenceCheck(s, ids, 0.0, "invalid_id"))
        else:
            sc = support(s, [corpus.by_id[i].title + " " + corpus.by_id[i].text for i in ids])
            v.checks.append(SentenceCheck(s, ids, sc, "supported" if sc >= min_support else "weak"))
    return v
