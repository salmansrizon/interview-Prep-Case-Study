"""
Test doubles shared by test_pipeline.py and test_api.py — no Ollama, no FinBERT download, no network.

HIGHLIGHTS: ``FakeLLM`` একটা আসল ``BaseLLM`` subclass, CrewAI-র ReAct format-এ ("Thought: ...\\nFinal
Answer: ...") scripted উত্তর দেয় — তাই crew, guardrail retry আর Flow-এর router সত্যিই চলে, শুধু model
নকল। ``FakeSentiment`` keyword দিয়ে FinBERT-এর মতো 3-class probability দেয় আর কতবার ডাকা হলো গোনে
(cache test-এর জন্য)।
"""
from __future__ import annotations

import config  # noqa: F401 - telemetry env vars before crewai
from crewai.llms.base_llm import BaseLLM

NEGATIVE = ("recall", "misses", "fall", "slide", "investigation", "fined", "cuts", "loss", "downgrade",
            "suspends", "delays", "warns", "reduction", "short seller", "slower", "dilutive")
POSITIVE = ("beats", "raises", "wins", "record", "approves", "rally", "growth", "positive", "jumps",
            "unveils", "launches", "renews", "expands", "settlement", "buyback", "opens")


class FakeLLM(BaseLLM):
    """Replays scripted answers (the last one repeats); records how often it was called."""

    def __init__(self, answers, **kw):
        super().__init__(model="fake", **kw)
        self._answers = list(answers)
        self.calls = 0

    def call(self, messages, *args, **kwargs):
        self.calls += 1
        answer = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]
        return f"Thought: I now know the final answer\nFinal Answer: {answer}"

    def supports_function_calling(self) -> bool:
        return False


class FakeSentiment:
    name = "fake-keywords"

    def __init__(self):
        self.calls = self.texts = 0

    def __call__(self, texts):
        self.calls += 1
        self.texts += len(texts)
        out = []
        for t in texts:
            low = t.lower()
            neg, pos = any(w in low for w in NEGATIVE), any(w in low for w in POSITIVE)
            if neg:
                out.append({"positive": 0.05, "negative": 0.85, "neutral": 0.10})
            elif pos:
                out.append({"positive": 0.80, "negative": 0.05, "neutral": 0.15})
            else:
                out.append({"positive": 0.10, "negative": 0.10, "neutral": 0.80})
        return out


def brief_for(news: list[dict]) -> str:
    return "\n".join(f"- {n['headline']} ({n['label']}) [{n['id']}]" for n in news[-3:])


def memo_for(risk: dict, news_id: str, score: float | None = None, vol_pct: float | None = None) -> str:
    """A memo that quotes the computed numbers — or wrong ones, if score/vol_pct are given."""
    m = risk["metrics"]
    score = risk["score"] if score is None else score
    vol = m["volatility"] * 100 if vol_pct is None else vol_pct
    return (f"## Risk Summary\nThe Risk Score is {score:.1f} out of 100, which puts it in the {risk['band']} band.\n\n"
            f"## Key Drivers\nAnnualized volatility of {vol:.1f}% and a maximum drawdown of "
            f"{m['max_drawdown'] * 100:.1f}% drive most of the score. The historical VaR (95%, 1-day) is "
            f"{m['var_95'] * 100:.1f}%.\n\n"
            f"## News Impact\nRecent headlines shape the sentiment component [{news_id}].")
