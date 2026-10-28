"""
Offline tests for the Offline Market Intelligence Team. Run: pytest -q

HIGHLIGHTS: কোনো network, Ollama বা FinBERT download লাগে না (ADR 0006/0008)। SQLite আসল (temp dir),
metric-এর math আসল (ছোট হাতে-গোনা series-এ), CrewAI crew আর Flow আসল — শুধু LLM (``FakeLLM``) আর
sentiment model (``FakeSentiment``) নকল।
"""
from __future__ import annotations

import math
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import pytest
from crewai import Process

from config import get_config
from data.make_snapshot import build as build_snapshot
from src.crew.team import brief_guardrail, build_crew, memo_guardrail
from src.flow.market_flow import run_analysis
from src.llm.factory import build_llm, ollama_status
from src.memo.pdf import write_memo_pdf
from src.risk.metrics import (InsufficientData, annualized_vol, assess, band, compute_risk, historical_var,
                              max_drawdown, net_sentiment, normalize)
from src.sentiment.finbert import FinBert, SentimentUnavailable, score_news
from src.store.db import UnknownTicker
from src.store.refresh import refresh
from src.verify.memo_check import verify_memo
from tests.fakes import FakeLLM, FakeSentiment, brief_for, memo_for

CFG = get_config()
RISK = {"ticker": "ACME", "as_of": "2025-12-19", "score": 57.3, "band": "medium",
        "metrics": {"volatility": 0.3124, "max_drawdown": 0.2241, "var_95": 0.0287, "sentiment": -0.21},
        "components": {"volatility": 35.4, "max_drawdown": 31.6, "var_95": 37.4, "sentiment": 60.5},
        "weights": dict(CFG.risk.weights)}
NEWS = [{"id": f"news:ACME:{i}", "date": f"2025-10-0{i}", "headline": h, "summary": "", "label": lab,
         "positive": p, "negative": n, "neutral": 1 - p - n}
        for i, h, lab, p, n in [(1, "Acme opens plant", "positive", 0.8, 0.1),
                                (2, "Acme recalls arms", "negative", 0.1, 0.8),
                                (3, "Acme raises outlook", "positive", 0.9, 0.05)]]
IDS = {n["id"] for n in NEWS}
BRIEF = "- Recall of picking arms [news:ACME:2]\n- Outlook raised [news:ACME:3]"
GOOD = ("## Risk Summary\nRisk Score: 57.3 out of 100 (medium).\n\n## Key Drivers\nAnnualized volatility of 31% "
        "and a max drawdown of -22.4% matter most; the VaR (95%, 1-day) is 2.87%.\n\n"
        "## News Impact\nThe recall weighs on sentiment [news:ACME:2].")


class Out:
    def __init__(self, raw):
        self.raw = raw


# ── Offline Snapshot + SQLite store ───────────────────────────────────
def test_snapshot_regenerates_byte_identical(tmp_path):
    build_snapshot(tmp_path)
    for name in ("prices.csv", "news.json", "manifest.json"):
        assert (tmp_path / name).read_bytes() == (CFG.data.snapshot_dir / name).read_bytes(), name


def test_snapshot_shape(store):
    tickers = store.tickers()
    assert len(tickers) == 8 and {t["source"] for t in tickers} == {"synthetic"}
    for t in tickers:
        dates = {d for d, _ in store.closes(t["ticker"])}
        news = store.news(t["ticker"])
        assert t["prices"] == 252 and 6 <= len(news) <= 10
        assert [n["id"] for n in news] == [f"news:{t['ticker']}:{i}" for i in range(1, len(news) + 1)]
        assert all(n["date"] in dates for n in news)          # every headline lands on a trading day


def test_store_reload_is_idempotent_and_keeps_refreshed_tickers(store):
    store.upsert(tickers=[("REAL", "Real Co", "", "yfinance")], prices=[("REAL", "2026-01-05", 10.0)])
    before = {t["ticker"]: (t["prices"], t["news"]) for t in store.tickers()}
    assert store.load_snapshot(CFG.data.snapshot_dir)["prices"] == 8 * 252
    assert {t["ticker"]: (t["prices"], t["news"]) for t in store.tickers()} == before
    with pytest.raises(UnknownTicker, match="NOPE"):
        store.closes("NOPE")


# ── metrics math on known small series ────────────────────────────────
def test_metrics_on_known_small_series():
    assert annualized_vol([100, 200, 100]) == pytest.approx(math.log(2) * math.sqrt(2) * math.sqrt(252))
    assert max_drawdown([100, 120, 90, 130, 65, 70]) == pytest.approx(0.5)      # 130 -> 65, not 120 -> 90
    assert max_drawdown([1, 2, 3]) == 0.0
    # simple returns +10%, -10%, 0%, +10% -> sorted [-0.1, 0, 0.1, 0.1]; 5th pct at pos 0.15 -> -0.085
    assert historical_var([100, 110, 99, 99, 108.9]) == pytest.approx(0.085)
    assert historical_var([1, 2, 3, 4]) == 0.0                                    # no losses -> no VaR


def test_net_sentiment_recency_weighting():
    old = {"date": "2025-10-20", "positive": 1.0, "negative": 0.0}                # 60 days = 2 half-lives -> w 0.25
    new = {"date": "2025-12-19", "positive": 0.0, "negative": 1.0}                # w 1.0
    assert net_sentiment([old, new], "2025-12-19", 30) == pytest.approx((0.25 - 1.0) / 1.25)
    assert net_sentiment([], "2025-12-19") == 0.0


def test_news_after_as_of_does_not_change_the_score(store):
    """No look-ahead bias: a headline dated after the last close is ignored by the Risk Score."""
    prices, news = store.closes("ACME"), score_news(store, store.news("ACME"), FakeSentiment())
    before = compute_risk("ACME", prices, news)
    future = {"id": "news:ACME:99", "date": "2026-01-15", "positive": 0.0, "negative": 1.0}   # as_of is 2025-12-19
    after = compute_risk("ACME", prices, news + [future])
    assert after.score == before.score and after.metrics["sentiment"] == before.metrics["sentiment"]


def test_normalize_bands_and_weighted_score(store):
    assert normalize(0.40, 0.10, 0.70) == pytest.approx(50) and normalize(0.05, 0.10, 0.70) == 0
    assert normalize(0.90, 0.10, 0.70) == 100
    assert [normalize(s, 1.0, -1.0) for s in (1, 0, -1)] == [0, 50, 100]          # sentiment: positive = low risk
    assert [band(s, (35, 65)) for s in (34.9, 35, 64.9, 65)] == ["low", "medium", "medium", "high"]
    assert sum(CFG.risk.weights.values()) == pytest.approx(1.0)
    r = compute_risk("ACME", store.closes("ACME"), [])
    expected = sum(CFG.risk.weights[k] * normalize(v, *CFG.risk.bounds[k]) for k, v in r.metrics.items())
    assert r.score == pytest.approx(expected, abs=0.05) and r.metrics["sentiment"] == 0.0
    with pytest.raises(InsufficientData):
        compute_risk("ACME", store.closes("ACME")[:5], [])


def test_snapshot_tickers_rank_sensibly(store):
    scores = {t["ticker"]: assess(store, t["ticker"], FakeSentiment())[0] for t in store.tickers()}
    assert min(scores, key=lambda t: scores[t].score) == "MRDN" and scores["HELX"].band == "high"
    assert scores["HELX"].metrics["max_drawdown"] > 0.5 and scores["VALT"].metrics["max_drawdown"] > 0.5
    assert abs(scores["MRDN"].metrics["volatility"] - 0.14) < 0.03     # measured vol ~ the generator's sigma


# ── sentiment ─────────────────────────────────────────────────────────
def test_sentiment_cache_only_calls_model_on_misses(store):
    cls = FakeSentiment()
    first = score_news(store, store.news("HELX"), cls)
    again = score_news(store, store.news("HELX"), cls)
    assert cls.calls == 1 and cls.texts == len(first) and first == again
    trial = next(n for n in first if "misses" in n["headline"])
    assert trial["label"] == "negative" and trial["negative"] > trial["positive"]


def test_finbert_without_transformers_is_a_clear_error(monkeypatch):
    clf = FinBert()                                          # constructing it loads nothing
    monkeypatch.setitem(sys.modules, "transformers", None)   # simulate "not installed"
    with pytest.raises(SentimentUnavailable, match="pip install transformers"):
        clf(["Acme beats estimates"])


# ── verification (code, not LLM) ──────────────────────────────────────
def test_verifier_accepts_correct_memo_within_rounding():
    assert verify_memo(BRIEF, GOOD, RISK, IDS).ok                          # 31% ~ 31.24, labels 95%/1-day ignored
    assert verify_memo(BRIEF, GOOD.replace("57.3", "57"), RISK, IDS).ok    # 0 decimals -> +-0.5
    assert not verify_memo(BRIEF, GOOD.replace("57.3", "57.0"), RISK, IDS).ok


def test_verifier_catches_wrong_numbers():
    c = verify_memo(BRIEF, GOOD.replace("57.3", "62.0").replace("31%", "35.5%"), RISK, IDS)
    assert not c.ok and [w["metric"] for w in c.wrong_numbers] == ["score", "volatility"]
    assert "computed value is 57.3" in c.feedback()


def test_verifier_catches_fake_ids_missing_sections_and_missing_score():
    c = verify_memo(BRIEF, GOOD.replace("## News Impact", "## Outlook") + " [news:ACME:99]", RISK, IDS)
    assert c.invalid_ids == ["news:ACME:99"] and c.missing_sections == ["News Impact"] and not c.ok
    assert "do not exist" in c.feedback()
    assert not verify_memo(BRIEF, GOOD.replace("Risk Score: 57.3", "The score"), RISK, IDS).score_quoted
    assert not verify_memo("- one [news:ACME:2]", GOOD, RISK, IDS).ok          # brief cites < 2 ids


def test_guardrails():
    brief = brief_guardrail(IDS, 2)
    assert brief(Out(BRIEF)) == (True, BRIEF)
    ok, msg = brief(Out(BRIEF.replace("ACME:3", "ACME:42")))
    assert not ok and "news:ACME:42" in msg
    assert "at least 2 different ids" in brief(Out("Acme had a busy year [news:ACME:2]"))[1]
    memo = memo_guardrail(IDS, CFG.verify.required_sections)
    assert memo(Out(GOOD))[0] and "Missing: ['Key Drivers']" in memo(Out(GOOD.replace("Key Drivers", "Drivers")))[1]


# ── crew wiring ───────────────────────────────────────────────────────
def test_crew_wiring_grounded_vs_tool_mode():
    c = build_crew("grounded", llm=FakeLLM(["x"]), risk=RISK, news=NEWS)
    assert c.process == Process.sequential and c.tasks[1].context == [c.tasks[0]] and not c.memory
    assert c.agents[1].tools == [] and "Risk Score: 57.3 out of 100" in c.tasks[1].description
    assert "[news:ACME:2]" in c.tasks[0].description
    t = build_crew("tool", llm=FakeLLM(["x"]), risk=RISK, news=NEWS)
    tools = t.agents[1].tools
    assert [x.name for x in tools] == ["get_risk_metrics", "get_news_sentiment"] and "57.3" not in t.tasks[1].description
    assert "Risk Score: 57.3" in tools[0]._run(" acme ") and tools[0]._run("ZZZ").startswith("UNKNOWN_TICKER")
    with pytest.raises(ValueError):
        build_crew("hierarchical", llm=FakeLLM(["x"]), risk=RISK, news=NEWS)


# ── flow ──────────────────────────────────────────────────────────────
def _answers(store, ticker="ACME", **wrong):
    risk, news = assess(store, ticker, FakeSentiment())
    return brief_for(news), memo_for(risk.to_dict(), news[-1]["id"]), memo_for(risk.to_dict(), news[-1]["id"], **wrong)


def test_flow_publishes_pdf_and_indexes_it(store, tmp_path):
    brief, memo, _ = _answers(store)
    llm = FakeLLM([brief, memo])
    st = run_analysis("acme", store=store, classifier=FakeSentiment(), llm=llm, out_dir=tmp_path)
    assert st.outcome == "published" and st.rounds == 1 and llm.calls == 2 and st.verification["ok"]
    assert Path(st.pdf_path).read_bytes().startswith(b"%PDF")
    row = store.latest_memo("ACME")
    assert row["id"] == st.memo_id and row["risk_score"] == st.risk["score"] and row["mode"] == "grounded"


def test_flow_retries_once_with_feedback_then_publishes(store, tmp_path):
    brief, memo, wrong = _answers(store, score=99.9)
    llm = FakeLLM([brief, wrong, brief, memo])
    st = run_analysis("ACME", store=store, classifier=FakeSentiment(), llm=llm, out_dir=tmp_path)
    assert st.outcome == "published" and st.rounds == 2 and llm.calls == 4
    assert "score = 99.9" in st.feedback


def test_flow_escalates_when_numbers_stay_wrong(store, tmp_path):
    brief, _, wrong = _answers(store, vol_pct=12.0)
    st = run_analysis("ACME", store=store, classifier=FakeSentiment(), llm=FakeLLM([brief, wrong] * 2),
                      out_dir=tmp_path)
    assert st.outcome == "needs_human" and st.rounds == CFG.crew.flow_max_rounds
    assert st.verification["wrong_numbers"][0]["metric"] == "volatility" and not st.pdf_path
    assert store.latest_memo("ACME") is None and not list(tmp_path.glob("*.pdf"))


def test_tool_mode_invented_numbers_are_caught(store, tmp_path):
    """The Week 14 failure mode: the analyst answers without calling its tool and makes numbers up."""
    brief, _, invented = _answers(store, score=42.0, vol_pct=25.0)
    st = run_analysis("ACME", "tool", store=store, classifier=FakeSentiment(), llm=FakeLLM([brief, invented] * 2),
                      out_dir=tmp_path)
    assert st.mode == "tool" and st.outcome == "needs_human"
    assert {w["metric"] for w in st.verification["wrong_numbers"]} == {"score", "volatility"}


def test_flow_survives_guardrail_exhaustion(store, tmp_path):
    st = run_analysis("ACME", store=store, classifier=FakeSentiment(), llm=FakeLLM(["no bullets, no ids"]),
                      out_dir=tmp_path)
    assert st.outcome == "needs_human" and "error" in st.verification


def test_unknown_ticker_fails_before_the_flow(store):
    with pytest.raises(UnknownTicker):
        run_analysis("NOPE", store=store, classifier=FakeSentiment(), llm=FakeLLM(["x"]))


# ── misc ──────────────────────────────────────────────────────────────
def test_pdf_tolerates_unicode_from_the_llm(tmp_path):
    p = write_memo_pdf(tmp_path / "m.pdf", company="Acme", source="yfinance", risk=RISK, mode="grounded",
                       brief="— “quoted” • bullet ≥ 2 🚀 [news:ACME:2]", memo="## Risk Summary\nRisk Score → 57.3",
                       prices=[("2025-01-02", 1.0), ("2025-01-03", 2.0)])
    assert p.read_bytes().startswith(b"%PDF") and p.stat().st_size > 1000


def test_llm_factory_and_status():
    llm = build_llm()
    assert llm.model == "llama3.1:8b" and llm.temperature == 0.1
    ok, msg = ollama_status(base_url="http://127.0.0.1:1")
    assert not ok and "not reachable" in msg


def test_refresh_writes_into_the_same_store(store, monkeypatch):
    hist = {"Close": {datetime(2026, 1, 5): 101.0, datetime(2026, 1, 6): 102.5}}
    entry = {"title": "Real Co beats estimates", "published_parsed": time.strptime("2026-01-06", "%Y-%m-%d"),
             "link": "https://example.com/a"}
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(
        Ticker=lambda t: types.SimpleNamespace(history=lambda **kw: hist)))
    monkeypatch.setitem(sys.modules, "feedparser", types.SimpleNamespace(
        parse=lambda url: types.SimpleNamespace(entries=[entry])))
    assert refresh(store, ["real"])["REAL"] == "2 prices, 1 new headlines"
    assert refresh(store, ["REAL"])["REAL"] == "2 prices, 0 new headlines"          # same headline: skipped
    assert store.news("REAL")[0]["id"] == "news:REAL:1" and store.ticker("REAL")["source"] == "yfinance"
    assert len(store.tickers()) == 9                                                # snapshot tickers untouched
