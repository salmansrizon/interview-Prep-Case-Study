# API image for the healthcare assistant. Ollama and (optionally) a Chroma server run OUTSIDE this container.
#   docker build -t healthcare-api .
#   docker run -p 8000:8000 -e HC_OLLAMA_HOST=http://host.docker.internal:11434 -v "$PWD/data/chroma_db:/app/data/chroma_db" healthcare-api
# Linux: add  --add-host=host.docker.internal:host-gateway
# Not built or run on the authoring machine (ADR 0008: no Docker daemon there). Week 18 composes UI + API + Chroma + Ollama.
FROM python:3.12-slim

ARG EMBEDDING_MODEL=sentence-transformers/paraphrase-MiniLM-L3-v2
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface HF_HUB_DISABLE_TELEMETRY=1 ANONYMIZED_TELEMETRY=False \
    HC_EMBEDDING_MODEL=${EMBEDDING_MODEL} HC_AUTO_INGEST=true

WORKDIR /app
COPY requirements.txt .
# CPU-only torch wheel: much smaller than the default CUDA build (the LLM runs in Ollama, not here).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt
# Bake the embedding model into the image: the container never needs the internet at runtime.
RUN python -c "import os; from sentence_transformers import SentenceTransformer as S; S(os.environ['HC_EMBEDDING_MODEL'])"
ENV HF_HUB_OFFLINE=1

COPY src ./src
COPY data/kb ./data/kb
COPY data/eval ./data/eval
COPY cli.py .

RUN useradd --create-home appuser && mkdir -p data/chroma_db && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
