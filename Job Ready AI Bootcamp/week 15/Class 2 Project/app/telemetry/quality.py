"""Cheap, per-request quality signals — no LLM, no network, microseconds each.

HIGHLIGHTS: HTTP 200 মানেই ভালো উত্তর না। একটা RAG উত্তর তিনভাবে "চুপচাপ" খারাপ হতে পারে:
ভুল/অনুপস্থিত citation, passage-এ নেই এমন সংখ্যা/তথ্য, আর মাঝপথে কেটে যাওয়া (num_predict শেষ)।
এই signal-গুলো প্রতিটা request-এ চলে (tier 1)। Sampled LLM judge (tier 3) আলাদা — ``judge.py``।

| Tier | Signal | Cost | What it catches | Blind spot |
|---|---|---|---|---|
| 1 | outcome (answered / not_found / degraded / cached) | free | guardrail rate, outages | answer quality |
| 1 | citation_valid | free | `[7]` with only 4 passages, no citation at all | wrong-but-cited |
| 1 | numbers_supported | free | invented numbers (days, USD, %) — policy docs live on numbers | paraphrased facts |
| 1 | lexical_support | free | answer words absent from passages | paraphrase scores low |
| 1 | truncated | free | answer cut at num_predict | — |
| 2 | NLI / HHEM groundedness | CPU model | unsupported claims | extra model, not here |
| 3 | LLM judge (sampled, offline) | 1 LLM call | subtle unfaithfulness | biased (self-preference), slow |
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_CITE = re.compile(r"\[(\d+)\]")
_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")
_WORD = re.compile(r"[a-z]{3,}")
_STOP = set("""the and for are was were with that this from which what how have has not but you your can may must
will shall per any all each every they them their its into than then also about according only such these those
there here when where who whom whose why does did done been being our out get gets got""".split())


@dataclass
class Signals:
    outcome: str                      # answered | not_found | degraded | cached | error
    citations: int = 0                # distinct [n] markers in the answer
    citation_valid: bool | None = None
    numbers_supported: bool | None = None
    unsupported_numbers: str = ""     # e.g. "15,40" — what to look at in a trace review
    lexical_support: float | None = None
    truncated: bool = False


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def assess(result, *, cached: bool, ok: bool, num_predict: int) -> Signals:
    """Signals for one AskResult (``result`` may be None when the call failed)."""
    if not ok or result is None:
        return Signals("error")
    if cached:
        return Signals("cached")
    if result.degraded:
        return Signals("degraded")
    if not result.found:
        return Signals("not_found")

    answer, hits = result.answer, result.hits
    cited = {int(n) for n in _CITE.findall(answer)}
    passages = " ".join(h.text for h in hits)
    nums = {n.replace(",", "") for n in _NUM.findall(_CITE.sub(" ", answer))}
    have = {n.replace(",", "") for n in _NUM.findall(passages)}
    missing = sorted(nums - have)
    words = _content_words(_CITE.sub(" ", answer))
    usage = result.usage
    return Signals(
        outcome="answered",
        citations=len(cited),
        citation_valid=bool(cited) and all(1 <= n <= len(hits) for n in cited),
        numbers_supported=not missing,
        unsupported_numbers=",".join(missing),
        lexical_support=round(len(words & _content_words(passages)) / len(words), 3) if words else None,
        truncated=bool(usage and num_predict and usage.completion_tokens >= num_predict),
    )
