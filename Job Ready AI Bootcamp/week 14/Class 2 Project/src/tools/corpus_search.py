"""
Local corpus + BM25 search, exposed to CrewAI agents as a tool.

HIGHLIGHTS: Researcher agent web-এ যায় না — শুধু ``data/corpus/``-এর
document পড়ে। প্রতিটা passage-এর একটা stable id আছে (``rag:2`` = rag.md-এর
২য় section)। Agent-কে প্রতিটা fact-এর সাথে এই id cite করতে হয়, আর পরে
code নিজে যাচাই করে id-টা আসল কিনা আর বাক্যটা সত্যিই সেই passage-এ আছে কিনা।
Grounding না থাকলে multi-agent crew একে অপরের hallucination বাড়িয়ে দেয়।

BM25 stdlib-এ ~২০ লাইনে — embedding model বা vector DB লাগে না।
ponytail: keyword ranking; semantic recall দরকার হলে Week 12-এর Chroma index বসান।
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

_WORD = re.compile(r"[a-z0-9]+")
STOP = set("a an the and or of to in on for with is are was were be by as at it its this that from than "
           "into can not no so such which while also only about up out more most very".split())


def tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOP]


@dataclass(frozen=True)
class Passage:
    id: str          # e.g. "rag:2"
    doc: str         # e.g. "rag.md"
    title: str       # section heading
    text: str


class Corpus:
    def __init__(self, corpus_dir: Path):
        self.passages: list[Passage] = []
        for path in sorted(Path(corpus_dir).glob("*.md")):
            sections = re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.M)[1:]
            for i, sec in enumerate(sections, 1):
                title, _, body = sec.partition("\n")
                self.passages.append(Passage(f"{path.stem}:{i}", path.name, title.strip(), " ".join(body.split())))
        self.by_id = {p.id: p for p in self.passages}
        self._tf = [Counter(tokens(p.title + " " + p.text)) for p in self.passages]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / max(len(self._len), 1)
        df = Counter(w for tf in self._tf for w in tf)
        n = len(self.passages)
        self._idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def search(self, query: str, k: int = 4, k1: float = 1.5, b: float = 0.75) -> list[tuple[Passage, float]]:
        q = set(tokens(query))
        scored = []
        for p, tf, length in zip(self.passages, self._tf, self._len):
            s = sum(self._idf.get(w, 0) * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * length / self._avg))
                    for w in q if w in tf)
            if s > 0:
                scored.append((p, round(s, 3)))
        return sorted(scored, key=lambda x: (-x[1], x[0].id))[:k]


class SearchInput(BaseModel):
    query: str = Field(..., description="A few keywords to search for, e.g. 'multi-agent failure modes'")


class CorpusSearchTool(BaseTool):
    name: str = "search_corpus"
    description: str = ("Search the local document collection. Returns passages, each starting with its "
                        "citation id in square brackets like [rag:2]. Cite facts with these exact ids.")
    args_schema: Type[BaseModel] = SearchInput
    corpus_dir: str = ""
    top_k: int = 4
    passage_chars: int = 450

    def _run(self, query: str) -> str:
        corpus = _cached_corpus(self.corpus_dir)
        hits = corpus.search(query, self.top_k)
        if not hits:
            return "NO_RESULTS: try different keywords. Available topics: " + ", ".join(
                sorted({p.doc.removesuffix('.md') for p in corpus.passages}))
        return "\n\n".join(f"[{p.id}] ({p.title}) {p.text[:self.passage_chars]}" for p, _ in hits)


_CACHE: dict[str, Corpus] = {}


def _cached_corpus(corpus_dir: str) -> Corpus:
    if corpus_dir not in _CACHE:
        _CACHE[corpus_dir] = Corpus(Path(corpus_dir))
    return _CACHE[corpus_dir]
