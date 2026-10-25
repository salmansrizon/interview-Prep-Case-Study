import os

os.environ.setdefault("RAG_BACKEND", "fake")   # tests never build the real backend (no Ollama, no model load)
