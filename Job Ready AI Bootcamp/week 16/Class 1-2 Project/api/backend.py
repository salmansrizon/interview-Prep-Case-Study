"""Backends behind one small blocking interface: ready, tickers, risk, analyze, memo_pdf.

HIGHLIGHTS:
* ``RealBackend`` — SQLite store + FinBERT + CrewAI Flow (``src/``)। ``/v1/risk`` শুধু deterministic
  code (store → FinBERT → math), LLM ছোঁয় না; ``/v1/analyze`` পুরো Flow চালায়।
* ``FakeBackend`` — কোনো dependency ছাড়া canned উত্তর: API contract test আর Ollama/FinBERT ছাড়া UI demo।
* Methods plain ``def`` (blocking) — FastAPI route এগুলো threadpool-এ চালায় (Week 15-এর মতো)।
* Domain error (``UnknownTicker``, ``SentimentUnavailable``...) এখানে HTTP error-এ বদলায়, route-এ না।
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from src.flow.market_flow import run_analysis
from src.llm.factory import build_llm, ollama_status
from src.risk.metrics import InsufficientData, assess
from src.sentiment.finbert import FinBert, SentimentUnavailable
from src.store.db import MarketStore, UnknownTicker

from .errors import NotFound, Unavailable, Unprocessable
from .settings import Settings


@contextmanager
def _domain_errors():
    try:
        yield
    except UnknownTicker as e:
        raise NotFound(str(e), "unknown_ticker") from e
    except SentimentUnavailable as e:
        raise Unavailable(str(e), "sentiment_unavailable") from e
    except InsufficientData as e:
        raise Unprocessable(str(e), "insufficient_data") from e


def risk_payload(risk: dict, news: list[dict]) -> dict:
    return {"ticker": risk["ticker"], "as_of": risk["as_of"], "risk_score": risk["score"], "band": risk["band"],
            "metrics": risk["metrics"], "components": risk["components"], "weights": risk["weights"],
            "news": [{k: n[k] for k in ("id", "date", "headline", "label", "positive", "negative", "neutral")}
                     for n in news]}


class RealBackend:
    name = "real"

    def __init__(self, settings: Settings, classifier=None, llm=None):
        self.settings = settings
        self.store = MarketStore(settings.db_path, settings.snapshot_dir)
        self.classifier = classifier or FinBert()
        self.llm = llm                                    # tests inject a fake BaseLLM

    def ready(self) -> tuple[bool, str]:
        if not self.store.tickers():
            return False, "Local store is empty: run `python cli.py snapshot`"
        if self.llm is not None:
            return True, "injected LLM"
        return ollama_status(self.settings.model, self.settings.ollama_host)

    def tickers(self) -> list[dict]:
        return self.store.tickers()

    def risk(self, ticker: str) -> dict:
        with _domain_errors():
            r, news = assess(self.store, ticker, self.classifier)
        return risk_payload(r.to_dict(), news)

    def analyze(self, ticker: str, mode: str) -> dict:
        with _domain_errors():
            self.store.ticker(ticker)                     # 404 before we even look at Ollama
            llm = self.llm
            if llm is None:
                ok, msg = ollama_status(self.settings.model, self.settings.ollama_host)
                if not ok:
                    raise Unavailable(msg, "llm_unavailable")
                llm = build_llm(self.settings.model, self.settings.ollama_host)
            st = run_analysis(ticker, mode, store=self.store, classifier=self.classifier, llm=llm,
                              out_dir=self.settings.output_dir)
        return {"ticker": st.ticker, "mode": st.mode, "outcome": st.outcome, "rounds": st.rounds,
                "seconds": st.seconds, "risk": risk_payload(st.risk, st.news), "brief": st.brief, "memo": st.memo,
                "verification": st.verification, "memo_available": st.outcome == "published"}

    def memo_pdf(self, ticker: str) -> tuple[str, bytes]:
        with _domain_errors():
            self.store.ticker(ticker)
        row = self.store.latest_memo(ticker)
        if not row or not Path(row["path"]).exists():
            raise NotFound(f"No published memo for {ticker} yet: run POST /v1/analyze/{ticker} first", "memo_not_found")
        return Path(row["path"]).name, Path(row["path"]).read_bytes()


class FakeBackend:
    """Canned answers, no SQLite/FinBERT/Ollama. ``MKT_BACKEND=fake`` for tests and a UI demo."""
    name = "fake"
    RISK = {"ticker": "ACME", "as_of": "2025-12-19", "score": 28.1, "band": "low",
            "metrics": {"volatility": 0.279, "max_drawdown": 0.183, "var_95": 0.025, "sentiment": 0.45},
            "components": {"volatility": 29.8, "max_drawdown": 24.1, "var_95": 30.2, "sentiment": 27.6},
            "weights": {"volatility": 0.35, "max_drawdown": 0.25, "var_95": 0.25, "sentiment": 0.15}}
    NEWS = [{"id": "news:ACME:7", "date": "2025-12-04", "label": "positive", "positive": 0.9, "negative": 0.03,
             "neutral": 0.07, "headline": "Acme Robotics raises full-year outlook on strong service revenue"}]

    def __init__(self):
        self.memos: dict[str, bytes] = {}

    def ready(self) -> tuple[bool, str]:
        return True, "fake backend"

    def tickers(self) -> list[dict]:
        return [{"ticker": "ACME", "name": "Acme Robotics", "sector": "Industrials", "source": "synthetic",
                 "prices": 252, "first_date": "2025-01-02", "last_date": "2025-12-19", "news": 7}]

    def risk(self, ticker: str) -> dict:
        if ticker != "ACME":
            raise NotFound(str(UnknownTicker(ticker)), "unknown_ticker")
        return risk_payload(self.RISK, self.NEWS)

    def analyze(self, ticker: str, mode: str) -> dict:
        risk = self.risk(ticker)
        self.memos[ticker] = b"%PDF-1.4\n% fake memo\n"
        return {"ticker": ticker, "mode": mode, "outcome": "published", "rounds": 1, "seconds": 0.0, "risk": risk,
                "brief": "- Acme raised its outlook [news:ACME:7]",
                "memo": "## Risk Summary\nRisk Score: 28.1 out of 100 (low).",
                "verification": {"ok": True}, "memo_available": True}

    def memo_pdf(self, ticker: str) -> tuple[str, bytes]:
        if ticker not in self.memos:
            raise NotFound(f"No published memo for {ticker} yet: run POST /v1/analyze/{ticker} first", "memo_not_found")
        return f"{ticker}-memo.pdf", self.memos[ticker]


def make_backend(settings: Settings):
    return FakeBackend() if settings.backend == "fake" else RealBackend(settings)
