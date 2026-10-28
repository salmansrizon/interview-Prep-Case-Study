"""
CrewAI tools for the Risk Analyst Agent in ``tool`` mode.

HIGHLIGHTS: Tool নিজে কিছু হিসাব করে না — Flow আগেই code-এ যা compute করেছে (Risk Score, metrics,
FinBERT sentiment) সেটাই text হিসেবে ফেরত দেয়। তাই agent tool ডাকলে সঠিক সংখ্যা পায়, আর সংখ্যা
বদলানোর কোনো পথ tool-এ নেই। Week 14-এর finding: llama3.1:8b CrewAI-র ভেতরে প্রায়ই tool **ডাকেই না**,
নিজে সংখ্যা বানিয়ে লেখে — এই mode সেটা দেখানোর জন্য; verifier সেই বানানো সংখ্যা ধরে।
"""
from __future__ import annotations

from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class TickerInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g. ACME")


class LookupTool(BaseTool):
    """Returns text the Flow already computed in code. It never calculates or changes a number."""
    name: str = "lookup"
    description: str = ""
    args_schema: Type[BaseModel] = TickerInput
    data: dict[str, str] = Field(default_factory=dict)

    def _run(self, ticker: str) -> str:
        return self.data.get(ticker.strip().upper(), f"UNKNOWN_TICKER {ticker!r}. Known tickers: {sorted(self.data)}")


def risk_tools(ticker: str, facts: str, news: str) -> list[BaseTool]:
    return [
        LookupTool(name="get_risk_metrics", data={ticker: facts},
                   description=("Get the code-computed Risk Score (0-100), its band, annualized volatility, maximum "
                                "drawdown, 95% 1-day historical VaR and net news sentiment for a ticker.")),
        LookupTool(name="get_news_sentiment", data={ticker: news},
                   description="Get the local news headlines for a ticker with their ids and FinBERT sentiment."),
    ]
