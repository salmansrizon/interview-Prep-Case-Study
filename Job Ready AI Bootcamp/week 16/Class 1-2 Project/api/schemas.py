"""Pydantic v2 response schemas — the contract that /docs (Swagger) shows.

HIGHLIGHTS: Risk Score আর তার প্রতিটা উপাদান আলাদা field — client prose পার্স না করেই জানে সংখ্যাটা
কোথা থেকে এল। ``outcome`` আর ``verification`` field বলে দেয় agent-দের লেখা publish হয়েছে নাকি মানুষের
review লাগবে।
"""
from pydantic import BaseModel, Field


class TickerInfo(BaseModel):
    ticker: str
    name: str
    sector: str
    source: str = Field(description="'synthetic' = bundled Offline Snapshot; otherwise where refresh got it")
    prices: int
    first_date: str | None
    last_date: str | None
    news: int


class TickersResponse(BaseModel):
    tickers: list[TickerInfo]


class Metrics(BaseModel):
    volatility: float = Field(description="Annualized volatility of daily log returns (0.31 = 31%)")
    max_drawdown: float = Field(description="Largest fall from a running peak (0.22 = 22%)")
    var_95: float = Field(description="95% historical 1-day VaR as a positive loss (0.029 = 2.9%)")
    sentiment: float = Field(description="Recency-weighted FinBERT net sentiment, -1..+1")


class NewsItem(BaseModel):
    id: str
    date: str
    headline: str
    label: str
    positive: float
    negative: float
    neutral: float


class RiskResponse(BaseModel):
    ticker: str
    as_of: str
    risk_score: float = Field(description="0-100, computed by code; agents never set it")
    band: str
    metrics: Metrics
    components: dict[str, float] = Field(description="Each metric normalized to 0-100 risk points")
    weights: dict[str, float]
    news: list[NewsItem]
    request_id: str | None = None


class AnalyzeResponse(BaseModel):
    ticker: str
    mode: str
    outcome: str = Field(description="published | needs_human")
    rounds: int
    seconds: float
    risk: RiskResponse
    brief: str = Field(description="News Scout Agent output")
    memo: str = Field(description="Risk Analyst Agent output")
    verification: dict
    memo_available: bool = Field(description="true = GET /v1/memo/{ticker} returns this PDF")
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    backend: str
    detail: str
    tickers: int


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
