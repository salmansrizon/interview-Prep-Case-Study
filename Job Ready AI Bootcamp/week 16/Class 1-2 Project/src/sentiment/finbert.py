"""
News sentiment with FinBERT (``ProsusAI/finbert``), cached per headline in SQLite.

HIGHLIGHTS:
* FinBERT = financial text-এ fine-tune করা BERT, natively 3-class (positive / negative / neutral) —
  ADR 0009। "beats estimates", "guidance cut"-এর মতো finance phrasing tweet-model-এর চেয়ে ভালো বোঝে।
* **Lazy load:** ``transformers`` import আর ~440 MB model download হয় প্রথম **cache miss**-এ, import-এ না।
  তাই test, API startup আর আগে score-করা headline — কোথাও model লাগে না।
* Cache key = (model name, headline-এর sha256) — headline বদলালে নতুন করে score হয়, model বদলালেও।
* Risk Score-এ label না, সংখ্যা লাগে: প্রতিটা headline-এর ``positive - negative`` probability; তারপর
  recency weight দিয়ে গড় (``src/risk/metrics.py: net_sentiment``)। LLM zero-shot এই calibrated
  probability দিতে পারে না — সেটাই FinBERT বাছার মূল কারণ।

── Model Reference Table ──

| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| **`ProsusAI/finbert`** (chosen, ADR 0009) | Financial news headline sentiment | Finance domain, native 3-class, প্রতিটা class-এর probability, CPU-তে চলে | ~440 MB; শুধু English; analyst-style sentence-এ trained — slang/social media-তে দুর্বল |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` (Week 10) | Tweet / review sentiment | Students আগে দেখেছে, 3-class | Tweet domain — "guidance cut", "misses endpoint"-এ domain shift |
| `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` | হালকা financial sentiment | DistilRoBERTa — ছোট, দ্রুত | Community model, ছোট dataset; label নাম আর calibration নিজে যাচাই করতে হয় |
| `llama3.1:8b` zero-shot (Ollama) | Label-ছাড়া নতুন domain | Extra model নেই, কারণ ব্যাখ্যা করতে পারে | ধীর, non-deterministic, calibrated probability নেই — deterministic Risk Score-এ বসে না, test করা কঠিন |
"""
from __future__ import annotations

import hashlib
from typing import Callable

MODEL = "ProsusAI/finbert"
LABELS = ("positive", "negative", "neutral")

# A classifier maps a list of texts to one {label: probability} dict per text.
Classifier = Callable[[list[str]], list[dict[str, float]]]


class SentimentUnavailable(RuntimeError):
    pass


class FinBert:
    def __init__(self, model: str = MODEL):
        self.name, self._pipe = model, None

    def __call__(self, texts: list[str]) -> list[dict[str, float]]:
        if self._pipe is None:
            try:
                from transformers import pipeline          # heavy import: only on the first cache miss
            except ImportError as e:
                raise SentimentUnavailable("FinBERT needs: pip install transformers torch") from e
            try:
                self._pipe = pipeline("text-classification", model=self.name, top_k=None)
            except Exception as e:  # noqa: BLE001 - offline first run, disk full, ...
                raise SentimentUnavailable(f"Could not load {self.name} ({type(e).__name__}: {e}). "
                                           "The first run downloads it once (~440 MB).") from e
        out = []
        for row in self._pipe(list(texts), truncation=True):
            row = row if isinstance(row, list) else [row]  # one dict per label (top_k=None)
            out.append({d["label"].lower(): float(d["score"]) for d in row})
        return out


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def score_news(store, news: list[dict], classifier: Classifier) -> list[dict]:
    """Return each news item plus label/positive/negative/neutral. Only cache misses reach the model."""
    if not news:
        return []
    model = getattr(classifier, "name", type(classifier).__name__)
    keys = [text_hash(n["headline"]) for n in news]
    cached = store.cached_sentiment(model, keys)
    missing = {k: n["headline"] for k, n in zip(keys, news) if k not in cached}
    if missing:
        rows = {k: _row(p) for k, p in zip(missing, classifier(list(missing.values())))}
        store.save_sentiment(model, rows)
        cached.update(rows)
    return [{**n, **cached[k]} for n, k in zip(news, keys)]


def _row(p: dict[str, float]) -> dict:
    probs = {lab: round(p.get(lab, 0.0), 4) for lab in LABELS}
    return {"label": max(probs, key=probs.get), **probs}
