"""Service config — every value can be overridden by an env var with prefix ``MKT_`` (e.g. MKT_BACKEND=fake).

HIGHLIGHTS: Week 15-এর মতো pydantic-settings — deploy-time-এ env var দিয়ে বদলানো যায়, type validation
বিনামূল্যে। Risk Score-এর bound/weight এখানে **না**, ``config.py``-তে — ওগুলো method-এর অংশ, deploy knob না।
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MKT_")

    backend: str = "real"                 # "real" = SQLite + FinBERT + Ollama crew, "fake" = canned (tests, UI demo)
    db_path: Path = PROJECT_DIR / "data" / "market.db"
    snapshot_dir: Path = PROJECT_DIR / "data" / "snapshot"
    output_dir: Path = PROJECT_DIR / "output"
    ollama_host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    # One crew run = minutes of one Ollama model. More than one at a time only queues inside Ollama.
    max_concurrent_analyses: int = 1
    queue_timeout_s: float = 5.0          # wait this long for the analysis slot, then 503 + Retry-After


def get_settings() -> Settings:
    return Settings()
