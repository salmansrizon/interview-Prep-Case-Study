"""Config — every value can be overridden by an env var with prefix ``HC_`` (e.g. HC_CHROMA_HOST).

HIGHLIGHTS: Week 18 এই project-কে ৪টা container-এ ভাগ করবে (UI, API, Chroma, Ollama)। তাই যা
deploy-এ বদলায় — Ollama-র ঠিকানা, Chroma local folder নাকি server, embedding model, startup-এ
KB ingest — সব env var। Code বদলাতে হয় না, শুধু ``HC_*`` বদলায়।

Privacy-first default: কোনো cloud call নেই, telemetry বন্ধ। HuggingFace-এর telemetry flag এখানে,
অন্য কোনো import-এর আগে, set হয়; Chroma-র flag ``ChromaStore``-এ client বানানোর সময়।

── Model reference (course runs only llama3.1:8b + paraphrase-MiniLM-L3-v2) ──
| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| **llama3.1:8b** (Ollama, chosen) | Local grounded answer, passage থেকে cite | Offline, ভালো instruction following, course standard | Medical training নেই — তাই শুধু passage থেকে উত্তর, নিজের জ্ঞান থেকে না |
| BioMistral-7B | Medical QA research | PubMed-এ further pre-trained, medical benchmark-এ ভালো | Research model, safety tuning দুর্বল, Ollama-য় official না |
| Meditron-7B | Clinical guideline-ভিত্তিক QA | Clinical guideline + PubMed-এ trained | Chat/instruction tuning কম, পুরোনো Llama 2 base |
| llama3.2:3b | কম RAM-এর মেশিন (8 GB) | দ্রুত, হালকা | Citation format বেশি ভাঙে |

| Model (embedding) | Use Case | Advantage | Limitation |
|---|---|---|---|
| **paraphrase-MiniLM-L3-v2** (chosen) | ছোট English KB-তে semantic search | ~61 MB, CPU-তে দ্রুত, course standard (ADR 0004) | General-domain; medical synonym কম বোঝে |
| all-MiniLM-L6-v2 | General English retrieval | L3-এর চেয়ে ভালো quality | দ্বিগুণ layer, একটু ধীর |
| pritamdeka/S-PubMedBert-MS-MARCO | Medical/biomedical retrieval | PubMed vocabulary, MS-MARCO-তে retrieval-tuned | ~440 MB, ধীর; সাধারণ মানুষের ভাষার প্রশ্নে সবসময় ভালো না |
| paraphrase-multilingual-MiniLM-L12-v2 | বাংলা + English প্রশ্ন | Multilingual, Bangla প্রশ্নেও retrieval | বড়, English-only KB-তে লাভ কম |

Embedding model বদলালে distance-এর scale বদলায় — ``HC_MAX_DISTANCE`` আবার calibrate করুন আর KB আবার
ingest করুন (``python cli.py ingest --reset``)।
"""
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")   # Chroma's env flag, belt and braces

PROJECT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HC_")

    backend: str = "real"                 # "real" = Ollama + Chroma + MiniLM, "fake" = no models (tests, demo)
    auto_ingest: bool = True              # index data/kb on startup when the collection is empty
    log_level: str = "INFO"

    # LLM (Ollama)
    ollama_host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.1
    num_ctx: int = 4096
    num_predict: int = 400
    llm_timeout_s: float = 120.0

    # Retrieval
    embedding_model: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    chroma_host: str = ""                 # empty = local PersistentClient(chroma_dir); set = HttpClient(host, port)
    chroma_port: int = 8000
    chroma_dir: Path = PROJECT_DIR / "data" / "chroma_db"
    collection: str = "clinic_kb"
    kb_dir: Path = PROJECT_DIR / "data" / "kb"
    top_k: int = 4
    # Cosine-distance abstention threshold: best passage farther than this -> "not in my knowledge base",
    # no LLM call. Calibrated on data/eval/questions.jsonl with paraphrase-MiniLM-L3-v2 (notebook section 3):
    # answerable worst 0.538 (one acronym outlier at 0.623 abstains), not-in-KB best 0.590 -> midpoint.
    max_distance: float = 0.56


def get_settings() -> Settings:
    return Settings()
