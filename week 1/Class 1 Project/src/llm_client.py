"""
llm_client.py
-------------
Motive: Encapsulate all communication with the local Ollama server.
WHY: If Ollama changes its API path tomorrow, you edit ONE file, not ten.
     This is the "Adapter Pattern" — your code speaks to this client,
     and this client speaks to Ollama.
WHAT IT DOES: Sends prompts to localhost:11434/api/generate and returns clean text.
ANALOGY: This is a *universal remote*. Your TV (the app) sends "power on" 
         to the remote; the remote translates that to the TV's specific infrared signal.
"""

import os
import requests
from typing import Optional
from dotenv import load_dotenv

# WHY load_dotenv() here? It runs when the module imports, ensuring env vars
# are available before any class instantiation.
load_dotenv()


class OllamaClient:
    """
    A production-grade client for local LLM inference via Ollama.

    WHY a class? Bundles state (host, model, timeout) with behavior (generate).
    WHAT IF we used functions? Every call would need to pass host+model repeatedly.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30,
    ):
        # WHY os.getenv with fallback? Allows override via .env or direct argument.
        # Priority: argument > .env > hardcoded default.
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.getenv("DEFAULT_MODEL", "llama3.2")
        self.timeout = timeout
        self.generate_url = f"{self.host}/api/generate"

    def generate(self, prompt: str) -> str:
        """
        Sends a prompt to Ollama and returns the generated text.

        WHY json payload? REST APIs speak JSON. It is the universal language of web services.
        WHY stream=False? For simplicity in Week 1. In production, streaming reduces 
             perceived latency for long responses.
        WHAT IT DOES: Packages your prompt, ships it to Ollama, unwraps the response.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            # WHY requests.post? The standard, battle-tested HTTP library in Python.
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=self.timeout,
            )
            # WHY raise_for_status()? Converts HTTP errors (404, 500) into Python 
            # exceptions instead of silently returning garbage data.
            response.raise_for_status()
            data = response.json()
            # Ollama returns {"response": "...", "done": true, ...}
            return data.get("response", "").strip()

        except requests.exceptions.ConnectionError:
            # WHY specific exception? Tells the user EXACTLY what is wrong.
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.host}. "
                "Is Ollama running? Run: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout}s. "
                "Try a smaller model or increase timeout."
            )

    def is_alive(self) -> bool:
        """
        Quick health check to see if Ollama is responding.

        WHY? Before running a 1000-line batch job, you want to know if the server is up.
        ANALOGY: Like pinging a friend before sending them a 50MB file.
        """
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False
