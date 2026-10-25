"""Service config — every value can be overridden by an env var with prefix ``RAG_`` (e.g. RAG_API_KEY).

HIGHLIGHTS: Week 10-14-এ config ছিল frozen dataclass। Service-এ pydantic-settings, কারণ
deploy-time-এ (Docker/K8s) env var দিয়ে বদলাতে হয়, আর type validation বিনামূল্যে।

── Model / server reference (course runs only llama3.1:8b) ──
| Choice | Use case | Advantage | Limitation |
|---|---|---|---|
| **Ollama + llama3.1:8b** (chosen) | এক মেশিনে local RAG API | Setup সহজ, offline | Default-এ model-প্রতি ১টা request একসাথে; continuous batching নেই |
| vLLM / TGI | অনেক concurrent user, GPU server | Continuous batching, PagedAttention | Setup ভারী, CUDA GPU লাগে |
| llama.cpp server | CPU/edge, ছোট deploy | হালকা, parallel slot | কম feature |
| Cloud API | সর্বোচ্চ quality, scale | কোনো local compute নেই | খরচ, data বাইরে যায় |
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_")

    backend: str = "real"                 # "real" = Ollama + Chroma, "fake" = no dependencies (tests, benchmark)
    fake_delay_s: float = 0.2
    warmup: bool = True                   # load models at startup instead of on the first request

    # LLM (Ollama)
    ollama_host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.2
    num_ctx: int = 8192
    num_predict: int = 512
    llm_timeout_s: float = 120.0

    # Retrieval
    embedding_model: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    chroma_dir: Path = PROJECT_DIR / "data" / "chroma_db"
    collection: str = "rag_api"
    top_k: int = 4
    chunk_size: int = 300                 # chars; smaller chunks = sharper embeddings + exact citations
    chunk_overlap: int = 60
    # L2 distance guardrail: no chunk closer than this -> "not found" without calling the LLM.
    # Calibrated on samples/employee_handbook.pdf (chunk 300): on-topic max 25.2, off-topic min 27.2.
    # Re-calibrate on YOUR documents + questions (notebooks/01_exploration.ipynb shows how).
    max_distance: float = 26.0

    # Protection
    # Requests allowed into the LLM at once. Match OLLAMA_NUM_PARALLEL (default 1): a bigger gate only moves
    # the queue inside Ollama, where we can't time it out or measure it.
    max_concurrent_llm: int = 1
    queue_timeout_s: float = 30.0         # wait this long for a slot, then 503 + Retry-After
    max_upload_mb: int = 20
    max_pages: int = 300
    api_key: str | None = None            # set RAG_API_KEY to require the X-API-Key header
    cache_size: int = 256                 # exact-match answer cache (0 = off)


def get_settings() -> Settings:
    return Settings()
