"""LLM factory + a network-light Ollama health check."""
import json
import urllib.request

from crewai import LLM

from config import MODEL_NAME, OLLAMA_BASE_URL, TEMPERATURE


def build_llm(model: str = MODEL_NAME, base_url: str = OLLAMA_BASE_URL) -> LLM:
    # HIGHLIGHTS: CrewAI-তে "provider/model" format — "ollama/" prefix বলে দেয় local Ollama
    # server ব্যবহার হবে। LLM object বানানোর সময় কোনো network call হয় না, তাই import/test নিরাপদ।
    return LLM(model=f"ollama/{model}", base_url=base_url, temperature=TEMPERATURE)


def ollama_status(model: str = MODEL_NAME, base_url: str = OLLAMA_BASE_URL) -> tuple[bool, str]:
    """Return (ok, message). Never raises, so the app works with or without Ollama."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
    except Exception as e:  # server বন্ধ / unreachable
        return False, f"Ollama পাওয়া যায়নি ({base_url}): {e}. `ollama serve` চালান।"
    if not any(n == model or n.startswith(model + ":") for n in names):
        return False, f"Model নেই। চালান: ollama pull {model}"
    return True, "Ollama ready"
