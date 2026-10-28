"""
Central config — Offline Market Intelligence Team (Week 16, Class 31 & 32).

HIGHLIGHTS:
* Telemetry বন্ধ — CrewAI default-এ telemetry চালু রাখে। তিনটা flag ``crewai`` import-এর **আগে**
  set করতে হয়, তাই এগুলো এই file-এর একদম শুরুতে। Offline project-এ data বাইরে যাওয়ার কোনো পথ রাখা যাবে না।
* Memory/knowledge বন্ধ — চালু করলে embedder না দিলে CrewAI OpenAI-তে যায়।
* Risk Score-এর সব bound আর weight এখানে, code-এ ছড়ানো না। Bound = কোন raw মান 0 আর কোনটা 100-তে
  map হয়; মাঝখানে linear, বাইরে clamp। Weight-এর যোগফল 1, তাই composite-ও 0-100।
  Agent এই সংখ্যা বদলাতে পারে না — শুধু ব্যাখ্যা করে।

── Model Reference Table (তুলনার জন্য; course চালায় শুধু llama3.1:8b) ──

| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| **`llama3.1:8b`** (course standard, ADR 0006) | দুই agent-এর sequential crew, ছোট memo | Laptop-এ চলে, native tool calling | CrewAI-র ভেতরে tool ডাকা নির্ভরযোগ্য না (Week 14 finding) — তাই default `grounded` |
| `llama3.2:3b` (ADR 0006 fallback) | 8 GB RAM machine | দ্রুত, হালকা | Format আর সংখ্যা কপি করায় আরও বেশি ভুল |
| বড় local model (70B) | Tool mode নির্ভরযোগ্য করা | Tool calling আর instruction following ভালো | ~40 GB+ memory |
| Cloud (GPT/Claude) | Production research desk | সবচেয়ে নির্ভরযোগ্য | Offline না — market data বাইরে যায় |
"""
from __future__ import annotations

import os

for _k, _v in {"CREWAI_DISABLE_TELEMETRY": "true", "OTEL_SDK_DISABLED": "true",
               "CREWAI_TRACING_ENABLED": "false"}.items():
    os.environ.setdefault(_k, _v)

from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = "llama3.1:8b"
    temperature: float = 0.1     # কম randomness — সংখ্যা কপি আর format কম ভাঙে
    seed: int = 42
    max_tokens: int = 1200


@dataclass(frozen=True)
class CrewConfig:
    max_iter: int = 6            # agent-প্রতি সর্বোচ্চ think/act step (default 25 — ছোট model-এ অনেক বেশি)
    guardrail_retries: int = 2   # guardrail fail করলে task কতবার আবার চেষ্টা করবে
    flow_max_rounds: int = 2     # verification fail করলে পুরো crew সর্বোচ্চ কতবার (= ১ বার retry)
    news_items: int = 8          # News Scout-কে সর্বশেষ কয়টা headline দেওয়া হয় (context ছোট রাখা)


@dataclass(frozen=True)
class RiskConfig:
    trading_days: int = 252      # annualize: daily stdev × sqrt(252)
    var_level: float = 0.95      # 95% historical 1-day VaR
    half_life_days: float = 30.0  # news-এর weight প্রতি ৩০ দিনে অর্ধেক
    min_prices: int = 20         # এর কম দিনের দাম থাকলে Risk Score হয় না
    # metric -> (raw value mapped to 0, raw value mapped to 100). Linear in between, clamped outside.
    bounds: dict = field(default_factory=lambda: {
        "volatility": (0.10, 0.70),    # 10% ≈ শান্ত utility/index, 70% ≈ speculative biotech
        "max_drawdown": (0.05, 0.60),  # peak থেকে 5% পতন স্বাভাবিক, 60% = প্রায় ধ্বংস
        "var_95": (0.01, 0.06),        # 95% দিনে ক্ষতি 1%-এর কম = শান্ত; 6% = খুব ঝুঁকিপূর্ণ
        "sentiment": (1.0, -1.0),      # net sentiment +1 -> 0 risk, -1 -> 100 risk (উল্টো দিক)
    })
    weights: dict = field(default_factory=lambda: {
        "volatility": 0.35, "max_drawdown": 0.25, "var_95": 0.25, "sentiment": 0.15})
    band_cuts: tuple[float, float] = (35.0, 65.0)   # < 35 low, < 65 medium, otherwise high


@dataclass(frozen=True)
class VerifyConfig:
    required_sections: tuple[str, ...] = ("Risk Summary", "Key Drivers", "News Impact")
    min_citations: int = 2       # news brief-এ অন্তত কয়টা আলাদা headline id (news কম থাকলে কম)


@dataclass(frozen=True)
class DataConfig:
    snapshot_dir: Path = BASE_DIR / "data" / "snapshot"
    db_path: Path = Path(os.getenv("MKT_DB_PATH", BASE_DIR / "data" / "market.db"))
    output_dir: Path = Path(os.getenv("MKT_OUTPUT_DIR", BASE_DIR / "output"))


@dataclass(frozen=True)
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    crew: CrewConfig = field(default_factory=CrewConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    data: DataConfig = field(default_factory=DataConfig)


_cfg: Config | None = None


def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = Config()
    return _cfg
