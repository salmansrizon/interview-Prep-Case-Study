"""Service config — env var দিয়ে override করা যায় (prefix ``RAG_``).

HIGHLIGHTS: Week 12-এ config একটা frozen dataclass ছিল; এখানে pydantic-settings
ব্যবহার করা হয়েছে কারণ service-এর config deploy-time-এ env var দিয়ে বদলাতে হয়
(Docker/K8s), আর type validation বিনামূল্যে পাওয়া যায়।
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parent.parent
# week 15/Class 1 Project -> ../../week 12/Class 2 Project
_DEFAULT_WEEK12 = PROJECT_DIR.parent.parent / "week 12" / "Class 2 Project"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_")

    # Week 12 project-এর root (যেখানে config.py আর src/ আছে)
    week12_dir: Path = _DEFAULT_WEEK12
    # Chroma data এই project-এর ভেতরে থাকে, যাতে week 12-এর data ছোঁয়া না হয়
    chroma_dir: Path = PROJECT_DIR / "data" / "chroma_db"
    max_upload_mb: int = 20
    # "real" = Week 12 pipeline, "fake" = Ollama/Chroma ছাড়া demo (benchmark)
    backend: str = "real"
    fake_delay_s: float = 0.2


def get_settings() -> Settings:
    return Settings()
