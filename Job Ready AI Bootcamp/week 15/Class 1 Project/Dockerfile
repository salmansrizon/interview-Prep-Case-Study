# Production-style image for the RAG API. Ollama runs OUTSIDE the container (on the host or its own container).
#   docker build -t rag-api .
#   docker run -p 8000:8000 -e RAG_OLLAMA_HOST=http://host.docker.internal:11434 -v "$PWD/data:/app/data" rag-api
# Linux: add  --add-host=host.docker.internal:host-gateway
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app
COPY requirements.txt .
# CPU-only torch wheel: much smaller than the default CUDA build (the LLM runs in Ollama, not here).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

COPY app ./app
# Bake the embedding model into the image so the first request does not download it.
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/paraphrase-MiniLM-L3-v2')"

RUN useradd --create-home appuser && mkdir -p data && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
# One process: the LLM, not Python, is the bottleneck. Scale with more replicas, not more workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
