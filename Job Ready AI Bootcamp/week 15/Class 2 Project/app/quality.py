"""Quality score (hallucination proxy) + drift check — সবই offline, LLM-judge ছাড়া।

HIGHLIGHTS: Score = "উত্তরের শব্দ কতটা retrieved context-এ আছে" (grounding overlap)।
উত্তরের অনেক শব্দ context-এ না থাকলে model নিজে থেকে বানিয়েছে (hallucination)-এর সংকেত।
এটা proxy, সত্য-যাচাই নয়: paraphrase-এ score কমে, আর context-এ ভুল থাকলে ভুলও grounded দেখায়।

HIGHLIGHTS — Quality evaluation reference:
| Method                 | Use case                | Advantage            | Limitation                       |
| Token overlap (chosen) | cheap, প্রতি query-তে     | ফ্রি, offline, দ্রুত    | paraphrase/semantic বোঝে না       |
| Embedding similarity   | paraphrase সহ grounding  | semantic ধরে          | embedding call লাগে (ধীর)         |
| LLM-as-judge           | সূক্ষ্ম faithfulness      | মানুষের কাছাকাছি       | খরচ + নিজেও ভুল করে               |
| RAGAS / DeepEval       | পুরো eval suite          | অনেক metric           | dependency + judge LLM লাগে      |
| Human review           | golden set              | সবচেয়ে নির্ভরযোগ্য     | ধীর, scale হয় না                 |
"""

import re
from statistics import mean

REFUSAL = "not found in the documents"
_STOP = set("the a an and or of to in on for is are was were be by with as at it this that from which what how".split())


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", text.lower()) if len(t) > 2 and t not in _STOP}


def grounding_score(answer: str, context: list[str], found: bool) -> tuple[float, str]:
    """Returns (score 0..1, label)."""
    if REFUSAL in answer.lower():
        # guardrail ঠিকভাবে কাজ করেছে (found=False) = ভালো; context থাকতেও refuse = সন্দেহজনক
        return (0.5, "refused_with_context") if found else (1.0, "correct_refusal")
    ans = _tokens(answer)
    if not ans:
        return 0.0, "empty"
    ctx = _tokens(" ".join(context))
    if not ctx:
        return 0.0, "no_context"
    return round(len(ans & ctx) / len(ans), 3), "grounded_overlap"


def _avg(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r[key] is not None]
    return mean(vals) if vals else None


def drift_report(rows: list[dict], n: int = 20, latency_up: float = 0.5, quality_drop: float = 0.15) -> dict:
    """পুরনো ``n`` query (baseline) বনাম সর্বশেষ ``n`` query (recent)। rows সময়ক্রমে সাজানো।

    ponytail: ছোট window-এর mean তুলনা; সত্যিকারের statistical test (KS/PSI) দরকার হলে scipy যোগ করুন।
    """
    if len(rows) < 2 * n:
        return {"status": "insufficient_data", "need": 2 * n, "have": len(rows)}
    base, recent = rows[:n], rows[-n:]
    out = {"status": "ok", "alerts": []}
    for key, bad in (("latency_s", lambda b, r: b > 0 and (r - b) / b > latency_up),
                     ("quality", lambda b, r: (b - r) > quality_drop)):
        b, r = _avg(base, key), _avg(recent, key)
        out[key] = {"baseline": b, "recent": r}
        if b is not None and r is not None and bad(b, r):
            out["alerts"].append(f"{key} drifted: {b:.3f} -> {r:.3f}")
    if out["alerts"]:
        out["status"] = "drift"
    return out
