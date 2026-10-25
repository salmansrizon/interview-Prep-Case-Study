import os
import tempfile

os.environ.setdefault("RAG_BACKEND", "fake")   # tests never build the real backend (no Ollama, no model load)
os.environ.setdefault("RAG_MONITOR_DB", os.path.join(tempfile.mkdtemp(prefix="rag_test_"), "monitor.db"))
