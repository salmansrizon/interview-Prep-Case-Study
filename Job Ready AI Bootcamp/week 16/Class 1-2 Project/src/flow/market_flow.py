"""
A CrewAI Flow around the crew: data -> sentiment -> Risk Score (code) -> crew (agents) -> verify (code)
-> retry once -> publish | needs_human.

HIGHLIGHTS: **Deterministic shell, agentic core.** Flow-এর প্রথম তিনটা step (``load_data``,
``score_sentiment``, ``compute_risk_score``) শুধু code — SQLite, FinBERT, math। তারপর ``run_crew``-এ
দুই agent লেখে, আর code সেই লেখা যাচাই করে (id আসল? সংখ্যা মেলে? section আছে?)। Fail হলে feedback
সহ **একবার** retry (hard cap — "not knowing when to stop" ঠেকাতে), তারপর ``@router`` ঠিক করে
publish (PDF + SQLite index) নাকি needs_human। Guardrail retry শেষ হলে CrewAI exception ছোড়ে —
Flow সেটা ধরে round-টাকে fail গণ্য করে।
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import config  # noqa: F401 - telemetry env vars first
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel, Field

from config import get_config
from src.crew.team import build_crew
from src.memo.pdf import write_memo_pdf
from src.risk.metrics import compute_risk
from src.sentiment.finbert import score_news
from src.verify.memo_check import verify_memo


class MarketState(BaseModel):
    ticker: str = ""
    mode: str = "grounded"
    company: str = ""
    source: str = ""
    news: list[dict] = Field(default_factory=list)     # headlines + FinBERT probabilities
    risk: dict = Field(default_factory=dict)           # RiskResult.to_dict()
    brief: str = ""                                    # News Scout Agent output
    memo: str = ""                                     # Risk Analyst Agent output
    verification: dict = Field(default_factory=dict)
    feedback: str = ""
    rounds: int = 0
    seconds: float = 0.0
    token_usage: dict = Field(default_factory=dict)
    outcome: str = ""                                  # published | needs_human
    pdf_path: str = ""
    memo_id: int | None = None


class MarketFlow(Flow[MarketState]):
    store = None          # injected: MarketStore
    classifier = None     # injected: FinBert() or a test fake
    llm = None            # injected: crewai LLM (tests pass a fake BaseLLM)
    out_dir = None
    on_task = None        # optional callback(task_output) for the UI/CLI

    @start()
    def load_data(self):
        """Code: ticker info + news from the local SQLite store (no network)."""
        info = self.store.ticker(self.state.ticker)
        self.state.company, self.state.source = info["name"], info["source"]
        return self.store.news(self.state.ticker)

    @listen(load_data)
    def score_sentiment(self, news):
        """Code: FinBERT per headline, cached in SQLite."""
        self.state.news = score_news(self.store, news, self.classifier)

    @listen(score_sentiment)
    def compute_risk_score(self):
        """Code: metrics + Risk Score. The agents only ever see these numbers."""
        prices = self.store.closes(self.state.ticker)
        self.state.risk = compute_risk(self.state.ticker, prices, self.state.news).to_dict()

    @listen(compute_risk_score)
    def run_crew(self):
        cfg = get_config()
        ids = {n["id"] for n in self.state.news}
        t0 = time.perf_counter()
        for self.state.rounds in range(1, cfg.crew.flow_max_rounds + 1):
            crew = build_crew(self.state.mode, llm=self.llm, risk=self.state.risk, news=self.state.news,
                              feedback=self.state.feedback)
            if self.on_task:
                crew.task_callback = self.on_task
            try:
                out = crew.kickoff(inputs={"ticker": self.state.ticker, "company": self.state.company})
            except Exception as e:  # noqa: BLE001 - guardrail retries exhausted, malformed agent output, ...
                self.state.verification = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
                self.state.feedback = f"The previous attempt failed: {str(e)[:300]}"
                continue
            self.state.brief, self.state.memo = out.tasks_output[0].raw, out.raw
            self.state.token_usage = _usage(out)
            check = verify_memo(self.state.brief, self.state.memo, self.state.risk, ids)
            self.state.verification = check.summary()
            if check.ok:
                break
            self.state.feedback = check.feedback()
        self.state.seconds = round(time.perf_counter() - t0, 1)

    @router(run_crew)
    def gate(self):
        return "ok" if self.state.verification.get("ok") else "escalate"

    @listen("ok")
    def publish(self):
        out_dir = Path(self.out_dir or get_config().data.output_dir)
        path = write_memo_pdf(out_dir / f"{self.state.ticker}-{datetime.now():%Y%m%d-%H%M%S}.pdf",
                              company=self.state.company, source=self.state.source, risk=self.state.risk,
                              brief=self.state.brief, memo=self.state.memo,
                              prices=self.store.closes(self.state.ticker), mode=self.state.mode)
        self.state.pdf_path = str(path)
        self.state.memo_id = self.store.add_memo(self.state.ticker, str(path), self.state.risk["score"],
                                                 self.state.risk["band"], self.state.mode)
        self.state.outcome = "published"

    @listen("escalate")
    def needs_human(self):
        self.state.outcome = "needs_human"       # no PDF: a person reviews the draft (brief/memo in the state)


def _usage(out) -> dict:
    u = getattr(out, "token_usage", None)
    if u is None:
        return {}
    data = u.model_dump() if hasattr(u, "model_dump") else dict(u)
    return {k: v for k, v in data.items() if isinstance(v, (int, float))}


def run_analysis(ticker: str, mode: str = "grounded", *, store=None, classifier=None, llm=None,
                 out_dir=None, on_task=None) -> MarketState:
    from src.sentiment.finbert import FinBert
    from src.store.db import default_store

    flow = MarketFlow()
    flow.store = store or default_store()
    flow.store.ticker(ticker.upper())          # fail fast with UnknownTicker before any step runs
    flow.classifier = classifier or FinBert()
    flow.llm, flow.out_dir, flow.on_task = llm, out_dir, on_task
    flow.kickoff(inputs={"ticker": ticker.upper(), "mode": mode})
    return flow.state
