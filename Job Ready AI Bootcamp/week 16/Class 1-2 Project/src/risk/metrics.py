"""
Deterministic risk metrics and the 0-100 Risk Score. Pure stdlib math — no LLM, no numpy.

HIGHLIGHTS:
* **Annualized volatility** = daily log return-এর sample stdev × sqrt(252)। Log return যোগ করা যায়
  (ln(P2/P0) = ln(P1/P0) + ln(P2/P1)), তাই volatility-র standard মাপ।
* **Max drawdown** = running peak থেকে সবচেয়ে বড় পতন, ``1 - price/peak``।
  Volatility উপরে-নিচে দুই দিকই গোনে; drawdown শুধু investor-এর আসল ব্যথা।
* **95% historical VaR (1-day)** = daily simple return-এর 5th percentile (linear interpolation,
  numpy-র default-এর মতো), loss হিসেবে positive। মানে: "১০০ দিনের ৯৫ দিনে ক্ষতি এর কম ছিল"।
  Historical = কোনো normal distribution ধরে নেওয়া হয়নি; crash-এর দিন সরাসরি tail-এ ঢোকে।
* **Net sentiment** = প্রতিটা headline-এর (positive − negative) probability-র recency-weighted গড়;
  weight = 0.5 ** (age_days / half_life)। শেষ দামের তারিখের (``as_of``) **পরের** news বাদ — নাহলে
  look-ahead bias: সেই দিনের Risk Score এমন খবর দেখত যা তখনো আসেনি। কোনো news না থাকলে 0 (neutral)।
* প্রতিটা metric config-এর bound দিয়ে 0-100-এ normalize, তারপর weight দিয়ে যোগ = **Risk Score**।
  Agent এই সংখ্যা কখনো set বা adjust করে না; সে শুধু ব্যাখ্যা করে।
"""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from datetime import date

from config import RiskConfig, get_config
from src.sentiment.finbert import score_news


class InsufficientData(ValueError):
    pass


def log_returns(closes: list[float]) -> list[float]:
    return [math.log(b / a) for a, b in zip(closes, closes[1:])]


def annualized_vol(closes: list[float], trading_days: int = 252) -> float:
    r = log_returns(closes)
    return statistics.stdev(r) * math.sqrt(trading_days) if len(r) > 1 else 0.0


def max_drawdown(closes: list[float]) -> float:
    peak, worst = closes[0], 0.0
    for c in closes:
        peak = max(peak, c)
        worst = max(worst, 1 - c / peak)
    return worst


def historical_var(closes: list[float], level: float = 0.95) -> float:
    """1-day historical VaR as a positive loss fraction (0.029 = 2.9%)."""
    r = sorted(b / a - 1 for a, b in zip(closes, closes[1:]))
    if not r:
        return 0.0
    pos = (1 - level) * (len(r) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(r) - 1)
    q = r[lo] + (r[hi] - r[lo]) * (pos - lo)
    return max(0.0, -q)


def net_sentiment(news: list[dict], as_of: str, half_life_days: float = 30.0) -> float:
    ref = date.fromisoformat(as_of[:10])
    num = den = 0.0
    for n in news:
        age = (ref - date.fromisoformat(n["date"][:10])).days
        if age < 0:
            continue            # dated after as_of: using it would be look-ahead bias
        w = 0.5 ** (age / half_life_days)
        num += w * (n["positive"] - n["negative"])
        den += w
    return num / den if den else 0.0


def normalize(value: float, lo: float, hi: float) -> float:
    """Map lo -> 0 and hi -> 100 linearly, clamped. lo > hi flips the direction (used for sentiment)."""
    return min(max((value - lo) / (hi - lo), 0.0), 1.0) * 100


def band(score: float, cuts: tuple[float, float]) -> str:
    return "low" if score < cuts[0] else "medium" if score < cuts[1] else "high"


@dataclass
class RiskResult:
    ticker: str
    as_of: str
    metrics: dict        # raw: volatility, max_drawdown, var_95 (fractions), sentiment (-1..+1)
    components: dict     # each metric normalized to 0-100
    weights: dict
    score: float         # 0-100 composite, rounded to 0.1
    band: str            # low | medium | high
    n_prices: int
    n_news: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_risk(ticker: str, prices: list[tuple[str, float]], news: list[dict],
                 cfg: RiskConfig | None = None) -> RiskResult:
    cfg = cfg or get_config().risk
    if len(prices) < cfg.min_prices:
        raise InsufficientData(f"{ticker}: {len(prices)} prices, need at least {cfg.min_prices}")
    closes, as_of = [c for _, c in prices], prices[-1][0]
    metrics = {"volatility": annualized_vol(closes, cfg.trading_days), "max_drawdown": max_drawdown(closes),
               "var_95": historical_var(closes, cfg.var_level),
               "sentiment": net_sentiment(news, as_of, cfg.half_life_days)}
    components = {k: normalize(v, *cfg.bounds[k]) for k, v in metrics.items()}
    score = round(sum(cfg.weights[k] * components[k] for k in components), 1)
    return RiskResult(ticker, as_of, {k: round(v, 6) for k, v in metrics.items()},
                      {k: round(v, 1) for k, v in components.items()}, dict(cfg.weights), score,
                      band(score, cfg.band_cuts), len(prices), len(news))


def assess(store, ticker: str, classifier) -> tuple[RiskResult, list[dict]]:
    """Everything deterministic for one ticker: prices + news (store) -> sentiment (FinBERT) -> Risk Score."""
    prices = store.closes(ticker)
    news = score_news(store, store.news(ticker), classifier)
    return compute_risk(ticker, prices, news), news
