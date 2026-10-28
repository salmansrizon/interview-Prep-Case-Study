"""
Deterministic checks on the agents' text — no LLM involved.

HIGHLIGHTS:
* তিনটা check, তিনটাই code-এ:
  1. Cited news id (``[news:ACME:3]``) এই ticker-এর local news-এ সত্যিই আছে কিনা — বানানো id ধরা পড়ে।
  2. Memo যে সংখ্যা quote করে (Risk Score, volatility, drawdown, VaR) সেটা computed মানের সাথে
     **rounding-এর মধ্যে** মেলে কিনা: ``31%`` (0 decimal) → ±0.5 ছাড়, ``31.2%`` (1 decimal) → ±0.05।
  3. Required section (``## Risk Summary`` ...) আছে কিনা, আর Risk Score অন্তত একবার quote করা কিনা।
* একই model দিয়ে নিজের লেখা grade করালে self-preference bias থাকে — তাই যা code পারে, code-এ।
* ponytail: keyword-proximity match — keyword-এর পরে ~25 character-এর মধ্যে প্রথম সংখ্যা। "volatility
  contributes 35%" একটা false alarm দেবে (এক round retry)। আরও কঠোর করতে: memo structured JSON-এ
  লেখানো, আর সংখ্যা শুধু code বসাবে।
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from config import VerifyConfig, get_config

CITE = re.compile(r"\[(news:[A-Za-z0-9.^\-]+:\d+)\]")
# Label numbers that are not metric values: "95%" / "95th" (VaR level), "1-day", "0-100" / "0 to 100".
LABELS = re.compile(r"\b95\s*%|\b95th\b|\b(?:1|one)[- ]day\b|\b0\s*(?:-|–|to)\s*100\b", re.I)
_NUM = r"[^\d\n]{0,25}?(\d+(?:\.\d+)?)"
_PCT = r"\s*(?:%|percent)"
PATTERNS = {
    "score": re.compile(r"risk\s+score\b" + _NUM, re.I),
    "volatility": re.compile(r"volatility\b" + _NUM + _PCT, re.I),
    "max_drawdown": re.compile(r"drawdown\b" + _NUM + _PCT, re.I),
    "var_95": re.compile(r"\bVaR\b" + _NUM + _PCT, re.I),
}


def cited_ids(text: str) -> list[str]:
    return CITE.findall(text)


def missing_sections(text: str, required: tuple[str, ...]) -> list[str]:
    heads = [h.lower() for h in re.findall(r"^\s*#{1,4}\s*(.+?)\s*$", text, re.M)]
    return [s for s in required if not any(s.lower() in h for h in heads)]


def quoted_numbers(text: str) -> list[tuple[str, float, int]]:
    """(metric, value, decimals) for every number the text puts right after a metric name."""
    clean = LABELS.sub(" ", text)
    out = []
    for metric, pattern in PATTERNS.items():
        for m in pattern.finditer(clean):
            s = m.group(1)
            out.append((metric, float(s), len(s.partition(".")[2])))
    return out


def expected_numbers(risk: dict) -> dict[str, float]:
    """The computed values in the units the memo uses: score 0-100, the rest in percent."""
    m = risk["metrics"]
    return {"score": risk["score"], "volatility": m["volatility"] * 100,
            "max_drawdown": m["max_drawdown"] * 100, "var_95": m["var_95"] * 100}


@dataclass
class MemoCheck:
    invalid_ids: list[str] = field(default_factory=list)
    wrong_numbers: list[dict] = field(default_factory=list)
    missing_sections: list[str] = field(default_factory=list)
    score_quoted: bool = False
    cited: int = 0                 # distinct valid ids in the news brief
    min_citations: int = 0

    @property
    def ok(self) -> bool:
        return (not self.invalid_ids and not self.wrong_numbers and not self.missing_sections
                and self.score_quoted and self.cited >= self.min_citations)

    def summary(self) -> dict:
        return {"ok": self.ok, "invalid_ids": self.invalid_ids, "wrong_numbers": self.wrong_numbers,
                "missing_sections": self.missing_sections, "score_quoted": self.score_quoted,
                "cited_ids": self.cited, "min_citations": self.min_citations}

    def feedback(self) -> str:
        lines = []
        if self.invalid_ids:
            lines.append(f"These news ids do not exist: {self.invalid_ids}. Cite only ids from the news list.")
        for w in self.wrong_numbers:
            lines.append(f"You quoted {w['metric']} = {w['quoted']}, but the computed value is {w['expected']}. "
                         "Copy numbers exactly from the risk facts.")
        if self.missing_sections:
            lines.append("Add these Markdown sections: " + ", ".join(f"'## {s}'" for s in self.missing_sections) + ".")
        if not self.score_quoted:
            lines.append("State the Risk Score exactly as given, e.g. 'Risk Score: 57.3 out of 100'.")
        if self.cited < self.min_citations:
            lines.append(f"The news brief must cite at least {self.min_citations} different news ids.")
        return "\n".join(lines)


def verify_memo(brief: str, memo: str, risk: dict, news_ids: set[str], cfg: VerifyConfig | None = None) -> MemoCheck:
    cfg = cfg or get_config().verify
    expected = expected_numbers(risk)
    check = MemoCheck(invalid_ids=sorted({i for i in cited_ids(brief + "\n" + memo) if i not in news_ids}),
                      missing_sections=missing_sections(memo, cfg.required_sections),
                      cited=len({i for i in cited_ids(brief) if i in news_ids}),
                      min_citations=min(cfg.min_citations, len(news_ids)))
    for metric, value, decimals in quoted_numbers(memo):
        check.score_quoted |= metric == "score"
        if abs(value - expected[metric]) > 0.5 * 10 ** -decimals + 1e-9:
            check.wrong_numbers.append({"metric": metric, "quoted": value, "expected": round(expected[metric], 2)})
    return check
