#!/usr/bin/env sh
# Single-click deployment (Linux / macOS / Git Bash on Windows).
#   ./start.sh          CPU (default)
#   ./start.sh --gpu    + docker-compose.gpu.yml (NVIDIA GPU for ollama)
# Idempotent: re-running keeps unchanged containers, never re-downloads the model, never duplicates the KB.
set -eu
cd "$(dirname "$0")"

command -v docker >/dev/null 2>&1 || { echo "docker not found: install Docker Desktop (Windows/macOS) or Docker Engine (Linux)."; exit 1; }
docker info >/dev/null 2>&1 || { echo "Docker daemon not reachable: start Docker Desktop, or on Linux: sudo systemctl start docker"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "'docker compose' (Compose v2 plugin) not found: update Docker."; exit 1; }

[ -f .env ] || { cp .env.example .env; echo "created .env from .env.example"; }

# no -f flags by default, so COMPOSE_FILE in .env (GPU machines) still applies
files=""
if [ "${1:-}" = "--gpu" ]; then
  files="-f docker-compose.yml -f docker-compose.gpu.yml"
fi
dc() { docker compose $files "$@"; }

echo "starting the stack (first run: image build + ~4.9 GB model pull, can take 10-20 minutes)..."
dc up -d --build || {
  dc ps -a
  echo "docker compose up failed. Look at: docker compose logs ollama-init   (or chroma / api)"
  exit 1
}

timeout=$(sed -n 's/^READY_TIMEOUT_S=\([0-9]*\).*/\1/p' .env | tail -n 1)
timeout=${timeout:-900}
echo "waiting for API /ready (Ollama has the model + knowledge base indexed), up to ${timeout}s..."
elapsed=0
until dc exec -T api python -c "import urllib.request as u; u.urlopen('http://localhost:8000/ready', timeout=5)" >/dev/null 2>&1; do
  if [ "$elapsed" -ge "$timeout" ]; then
    dc ps -a
    echo "API not ready after ${timeout}s. Look at: docker compose logs api   and: curl http://127.0.0.1:8000/ready"
    exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done

ui=$(dc port ui 8501 2>/dev/null || echo "127.0.0.1:8501")
echo "Ready: http://${ui}"
