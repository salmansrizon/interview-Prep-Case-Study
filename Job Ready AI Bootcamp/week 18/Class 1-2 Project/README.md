# 🐳 Single-Click Deployment — Healthcare Assistant on Docker Compose

## সহজ ভাষায় Project Overview

Week 17-এর Privacy-First Healthcare Assistant (Greenfield Community Clinic-এর handbook থেকে উত্তর, PII redaction, Red-Flag Escalation, Abstention) laptop-এ চালাতে চারটা terminal লাগত: `ollama serve`, `uvicorn`, `streamlit`, সাথে venv, `ollama pull` আর এক গাদা env var। এই Shared Project পুরো system-কে চারটা container-এ ভরে — `ui` (Streamlit), `api` (FastAPI), `chroma` (vector DB server), `ollama` (`llama3.1:8b`) — আর একটা one-shot `ollama-init` যেটা model না থাকলেই pull করে। সব চালু হয় **একটা command-এ**: `./start.sh` বা `.\start.ps1`।

### কোন Problem Solve করে?

"আমার machine-এ চলে" সমস্যা। নতুন machine-এ Docker থাকলেই যথেষ্ট: সঠিক Python version, dependency, model download, KB ingest, সঠিক ক্রমে startup — সব stack নিজে করে, আর script "Ready" বলে তখনই যখন API সত্যিই উত্তর দিতে পারে (Ollama-য় model আছে, knowledge base indexed)। দ্বিতীয়বার চালালে কিছুই নতুন করে download হয় না, internet না থাকলেও।

### কী বদলায়নি

Healthcare Assistant-এর code (`app/src`, `ui.py`, `cli.py`, KB, eval set, ৬৭টা test) Week 17 থেকে হুবহু কপি (ADR 0002) — একটা byte-ও বদলাতে হয়নি, কারণ Week 17-এ সব deploy-ভেদে বদলানো জিনিস আগেই `HC_*` env var ছিল। এই project-এর কাজ শুধু তার চারপাশে: Dockerfile, Compose, script, test।

> **Verification-এর সৎ কথা (ADR 0008):** যে machine-এ এটা লেখা, সেখানে Docker daemon আর GPU ছিল না। Compose file `docker compose config` (CLI, daemon লাগে না) দিয়ে validate করা, আর `tests/test_deploy.py`-এর ২৪টা test file পড়ে গুরুত্বপূর্ণ property যাচাই করে। কিন্তু image কখনো build হয়নি, stack কখনো চলেনি — প্রথম আসল `up` আপনার machine-এ। নিচের "What to Verify on Your Machine" অংশটা সেই checklist।
>
> ⚠️ Educational, fictional clinic handbook; not medical advice. এটা medical device না।

---

![Docker Compose](https://img.shields.io/badge/docker%20compose-v2-2496ED.svg)
![Ollama](https://img.shields.io/badge/ollama-0.35.1%20%C2%B7%20llama3.1%3A8b-black.svg)
![ChromaDB](https://img.shields.io/badge/chroma-1.5.9%20server%20%2B%20client-orange.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.162-009688.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.66-red.svg)
![GPU](https://img.shields.io/badge/GPU-optional%20NVIDIA%20override-76B900.svg)
![Tests](https://img.shields.io/badge/tests-91%20passing-green.svg)

## ✨ Features

- 🖱️ **Single click** — `start.sh` / `start.ps1` / `make up`: Docker আছে আর daemon চলছে কিনা দেখে (না হলে পরিষ্কার message), `.env` না থাকলে `.env.example` থেকে বানায়, `docker compose up -d --build`, তারপর API-র `/ready`-এর জন্য timeout-সহ অপেক্ষা, শেষে UI-এর URL।
- 🔁 **Idempotent** — আবার চালালে অপরিবর্তিত container ছোঁয়া হয় না (Compose), model আবার নামে না (`ollama show || ollama pull`), KB-তে duplicate হয় না (Week 17-এর id-ভিত্তিক `upsert` + stale delete)।
- 🚦 **Running ≠ ready** — প্রতিটা long-running service-এ healthcheck, image-এ যে tool আছে তা দিয়ে (`ollama list`, bash `/dev/tcp`, Python `urllib`); `depends_on` সবসময় `service_healthy` বা `service_completed_successfully`।
- 📦 **Lean images** — API: `python:3.12-slim` + CPU-only torch + baked MiniLM (code COPY-এর **আগে**, তাই code edit-এ model আবার নামে না), non-root। UI: শুধু `streamlit` + `httpx`।
- 💾 **State volume-এ** — `ollama_models` (≈4.9 GB) আর `chroma_data`; top-level `name: healthcare-assistant` তাই folder rename-এও একই volume।
- 🔒 **Privacy defaults** — host-এ publish শুধু `ui` আর `api`, দুটোই `127.0.0.1`-এ; Ollama/Chroma-র কোনো auth নেই, তাই publish না। Chroma আর Streamlit telemetry বন্ধ, `.env` image-এ ঢোকে না।
- 📌 **Pinned everything** — `ollama/ollama:0.35.1`, `chromadb/chroma:1.5.9` (= `chromadb==1.5.9` client), `torch==2.14.1` CPU wheel, সব Python package; tag না থাকলে Compose থামে (`${VAR:?}`), চুপচাপ `latest` টানে না।
- 🎮 **Optional GPU** — `docker-compose.gpu.yml` শুধু `ollama`-কে NVIDIA GPU দেয়; CPU default।
- 🧪 **91 offline tests** — ৬৭টা Week 17-এর (fully mocked) + ২৪টা deployment test (compose, Dockerfile, script; `docker compose config` CLI থাকলে)।

## 🚀 Quick Start

শর্ত: Docker Desktop (Windows: WSL2 backend; macOS) বা Linux-এ Docker Engine + Compose v2। Docker-কে অন্তত **১০-১২ GB RAM** দিন (Settings → Resources; Windows-এ `.wslconfig`), disk-এ **≈২০ GB** খালি। প্রথম run-এ ≈৯-১০ GB download (Ollama image ≈3.8 GB + `llama3.1:8b` ≈4.9 GB + Python base, torch, Chroma)।

```bash
cd "Job Ready AI Bootcamp/week 18/Class 1-2 Project"
./start.sh                       # Linux / macOS / Git Bash   (NVIDIA GPU: ./start.sh --gpu)
```

```powershell
cd "Job Ready AI Bootcamp\week 18\Class 1-2 Project"
.\start.ps1                      # Windows   (GPU: .\start.ps1 --gpu)
# execution policy আটকালে: powershell -ExecutionPolicy Bypass -File .\start.ps1
```

শেষ লাইন `Ready: http://127.0.0.1:8501` — browser-এ খুলুন। বন্ধ: `./stop.sh` / `.\stop.ps1` (volume থাকে)। `make` থাকলে: `make up | gpu | down | ps | logs | eval | config | test`।

## 🧩 Services, Ports, Volumes

| Service | Image | Container port | Host port | Volume | Healthcheck | Restart |
|---|---|---|---|---|---|---|
| `ollama` | `ollama/ollama:0.35.1` | 11434 | — (internal only) | `ollama_models:/root/.ollama` | `ollama list` | `unless-stopped` |
| `ollama-init` | `ollama/ollama:0.35.1` | — | — | — (server-এর volume-এ pull) | one-shot, exit 0 | `"no"` |
| `chroma` | `chromadb/chroma:1.5.9` | 8000 | — (internal only) | `chroma_data:/data` | bash `/dev/tcp` → `GET /api/v2/heartbeat` | `unless-stopped` |
| `api` | build `Dockerfile.api` | 8000 | `127.0.0.1:8000` (`API_PORT`) | — | `GET /health` (start_period 120 s) | `unless-stopped` |
| `ui` | build `Dockerfile.ui` | 8501 | `127.0.0.1:8501` (`UI_PORT`) | — | `GET /_stcore/health` | `unless-stopped` |

`api` আর `chroma` দুটোই container-এর ভেতরে 8000-এ শোনে — আলাদা container, তাই কোনো সংঘাত নেই। সংঘাত হতো শুধু host-এ দুটো publish করলে; Chroma publish করা হয়নি।

## 🏗️ Architecture

```
 browser ─▶ http://127.0.0.1:8501            host: only ui + api are published, both on loopback
 ═══════════════│═══════════════════════════════════════════════════════════════════════════════════
                ▼                                   Compose network: healthcare-assistant_default
        ┌──────────────┐  http://api:8000   ┌───────────────────────────┐
        │ ui           │ ─────────────────▶ │ api   FastAPI :8000       │ ◀── 127.0.0.1:8000 (/docs, curl)
        │ Streamlit    │                    │ PII → Red-Flag → refusal  │
        │ :8501        │                    │ → retrieve → LLM → cite   │
        └──────────────┘                    │ MiniLM baked · CPU torch  │
                                            └─────┬───────────────┬─────┘
                                   chroma:8000    │               │   http://ollama:11434
                                                  ▼               ▼
                                 ┌─────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
                                 │ chroma :8000    │   │ ollama :11434       │◀──│ ollama-init          │
                                 │ cosine vectors  │   │ llama3.1:8b Q4_K_M  │   │ show || pull, exit 0 │
                                 │ (no ports)      │   │ (no ports; GPU via  │   └──────────────────────┘
                                 └────────┬────────┘   │  gpu override file) │
                                          ▼            └──────────┬──────────┘
                                   [chroma_data]                  ▼
                                    volume, /data          [ollama_models] volume, /root/.ollama
```

**Startup order** (Compose নিজে অপেক্ষা করে; প্রতিটা তীর একটা `depends_on` condition):

```
ollama ──healthy (ollama list)──▶ ollama-init ──completed (exit 0)──┐
   └──healthy───────────────────────────────────────────────────────┤
chroma ──healthy (GET /api/v2/heartbeat)────────────────────────────┴──▶ api ──healthy (/health)──▶ ui
                                                                          │
                                        start.sh / start.ps1 then wait for api /ready (≤ READY_TIMEOUT_S)
```

`api`-র lifespan startup-এ collection খালি হলে KB ingest করে (`HC_AUTO_INGEST=true`); `/health` তার পরেই সাড়া দেয়। `/ready` 200 দেয় যখন Ollama-য় model আছে **আর** Chroma-য় section আছে।

## 📁 Project Structure

```
Class 1-2 Project/
├── docker-compose.yml        # 4 services + ollama-init, healthchecks, depends_on, named volumes  ← start reading
├── docker-compose.gpu.yml    # NVIDIA override: deploy.resources.reservations.devices, ollama only
├── Dockerfile.api            # slim + CPU torch + baked MiniLM (before code), non-root, HEALTHCHECK
├── Dockerfile.ui             # slim + streamlit + httpx only, non-root, HEALTHCHECK
├── .dockerignore             # .env, tests, caches, local chroma_db out of the build context
├── .env.example              # every tunable: image tags, ports, model, HC_MAX_DISTANCE, ready timeout
├── start.sh / start.ps1      # single click: docker check -> .env -> up -d --build -> wait /ready -> URL
├── stop.sh / stop.ps1        # docker compose down (volumes kept)
├── Makefile                  # up | gpu | down | ps | logs | eval | config | test
├── pytest.ini                # pythonpath = app; runs tests/ and app/tests/
├── tests/test_deploy.py      # 24 static deployment tests (no daemon)
└── app/                      # self-contained copy of the Week 17 Healthcare Assistant
    ├── src/                  # assistant.py (guardrail order), guardrails/, rag/, api/, evaluation/, settings.py
    ├── data/kb/*.md          # 30 handbook pages = 128 sections (baked into the API image)
    ├── data/eval/questions.jsonl   # 42 labelled questions (cli.py eval)
    ├── ui.py · cli.py · .streamlit/config.toml
    ├── samples/make_handbook.py    # data/kb -> printable PDF (used by a test)
    ├── tests/                # the 67 Week 17 tests, unchanged
    ├── requirements-api.txt  # API image
    ├── requirements-ui.txt   # UI image
    └── requirements.txt      # local dev = both + pytest, httpx2, fpdf2, PyYAML
```

## ⚙️ Configuration (`.env`)

`start.sh`/`start.ps1` প্রথমবার `.env.example` → `.env` কপি করে। Compose `.env` পড়ে `docker-compose.yml`-এর `${VAR}` বসানোর জন্য; container পায় শুধু যা `environment:`-এ লেখা। Service-এর ঠিকানাগুলো (`HC_OLLAMA_HOST=http://ollama:11434`, `HC_CHROMA_HOST=chroma`, `HC_API_URL=http://api:8000`, `HC_AUTO_INGEST=true`) wiring — compose-এ স্থির, `.env`-এ না।

| Variable | Default | মানে |
|---|---|---|
| `OLLAMA_TAG` | `0.35.1` | `ollama/ollama` image tag (না থাকলে `up` থামে) |
| `CHROMA_TAG` | `1.5.9` | `chromadb/chroma` tag — `app/requirements-api.txt`-এর `chromadb==` এর সমান রাখুন (test যাচাই করে) |
| `UI_PORT` / `API_PORT` | `8501` / `8000` | host port, `127.0.0.1`-এ bind; container port বদলায় না |
| `HC_MODEL` | `llama3.1:8b` | `ollama-init` এটাই pull করে, API এটাই ডাকে (ADR 0006) |
| `HC_TEMPERATURE` / `HC_NUM_CTX` / `HC_NUM_PREDICT` | `0.1` / `4096` / `400` | generation options |
| `HC_LLM_TIMEOUT_S` | `120` | এর বেশি হলে 503 `llm_unavailable` (CPU-তে লম্বা উত্তরে বাড়াতে হতে পারে) |
| `HC_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-MiniLM-L3-v2` | build arg হিসেবে image-এ bake **আর** runtime env — দুটো এক জায়গা থেকে, তাই কখনো আলাদা হয় না |
| `HC_MAX_DISTANCE` | `0.56` | abstention threshold (cosine), Week 17-এ MiniLM দিয়ে calibrate করা |
| `HC_TOP_K` / `HC_COLLECTION` / `HC_LOG_LEVEL` | `4` / `clinic_kb` / `INFO` | retrieval আর logging |
| `READY_TIMEOUT_S` | `900` | start script `/ready`-এর জন্য কত সেকেন্ড অপেক্ষা করবে |
| `COMPOSE_FILE` | *(commented)* | GPU machine-এ `docker-compose.yml:docker-compose.gpu.yml` (Windows-এ `;`) — তখন সাধারণ `docker compose up`-ও GPU পায় |

`tests/test_deploy.py` যাচাই করে: compose-এর প্রতিটা `${VAR}` `.env.example`-এ আছে, compose-এর `:-` default আর `.env.example`-এর মান এক, আর `HC_*` মানগুলো `app/src/settings.py`-এর default-এর সমান।

## 🎮 GPU (NVIDIA)

Host-এ লাগবে: NVIDIA driver (host-এ `nvidia-smi` চলে), NVIDIA Container Toolkit, তারপর `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`। Windows-এ Docker Desktop-এর WSL2 backend + সাধারণ Windows NVIDIA driver (WSL-এর ভেতরে Linux driver না)। Ollama-র শর্ত: compute capability ≥ 5.0, driver ≥ 550 (Class 36 lecture §3.4)।

```bash
./start.sh --gpu        # = docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
docker compose exec ollama nvidia-smi            # container GPU দেখে?
docker compose exec ollama ollama ps             # PROCESSOR = 100% GPU ?
```

Override শুধু `ollama`-তে `deploy.resources.reservations.devices: [{driver: nvidia, count: all, capabilities: [gpu]}]` যোগ করে। NVIDIA runtime ছাড়া machine-এ এই file দিলে `up` ব্যর্থ হয় (`could not select device driver "nvidia"`) — তাই base file-এ নেই। macOS-এর Docker Desktop-এ GPU passthrough নেই: সেখানে host-এ native Ollama চালিয়ে `HC_OLLAMA_HOST=http://host.docker.internal:11434` দেওয়া যায়, কিন্তু তখন আর single click না।

## 🧠 Model & Hardware Choices

| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| **`llama3.1:8b` Q4_K_M on CPU** (default, ADR 0006) | যেকোনো laptop/CPU VM, Docker ১০-১২ GB RAM | কোনো GPU setup নেই, একই compose সব machine-এ; ≈4.9 GB weight | উত্তর ধীর — RAM bandwidth-bound decode; long উত্তরে `HC_LLM_TIMEOUT_S` ছুঁতে পারে |
| **`llama3.1:8b` Q4_K_M on NVIDIA GPU** (`--gpu`) | ৮ GB+ VRAM GPU PC বা GPU VM | একই model, অনেক দ্রুত decode; ≈৬.৭ GB VRAM @৮k context (estimate) | driver + Container Toolkit লাগে; ভুল হলে নীরবে CPU — `ollama ps` দিয়ে প্রমাণ দরকার |
| `llama3.1:8b` Q8_0 on GPU (`HC_MODEL=llama3.1:8b-instruct-q8_0`) | ১২-১৬ GB GPU, একটু বেশি quality | FP16-এর কাছের quality | ≈8.5 GB; decode ধীর; eval আবার চালাতে হয় |
| `llama3.2:3b` on CPU (`HC_MODEL=llama3.2:3b`) | ৮ GB RAM laptop, সস্তা CPU VPS (ADR 0006 fallback) | ≈2 GB, CPU-তেও চলনসই | citation format বেশি ভাঙে, grounding দুর্বল — `cli.py eval`-এ `no_valid_citation` বাড়বে |

VRAM-এর সংখ্যা Class 36 lecture-এর formula থেকে estimate, মাপা না। Embedding model (`paraphrase-MiniLM-L3-v2`, ADR 0004) API image-এ baked, সবসময় CPU-তে; বিকল্পের table `Dockerfile.api`-এর HIGHLIGHTS-এ।

## ✅ What to Verify on Your Machine

এখানে কিছুই চালানো যায়নি (ADR 0008), তাই প্রথম run-এ এই checklist মেলান:

```bash
./start.sh                                       # শেষ লাইন: Ready: http://127.0.0.1:8501
docker compose ps -a                             # ollama/chroma/api/ui = running (healthy); ollama-init = Exited (0)
curl http://127.0.0.1:8000/ready                 # {"status":"ready","backend":"real","llm":"ok","indexed_sections":128}
docker compose exec api python cli.py eval       # safety metric pass হলে exit code 0 (fail হলে 2)
echo $?                                          # PowerShell: $LASTEXITCODE
docker compose exec ollama ollama list           # llama3.1:8b আছে
./start.sh                                       # দ্বিতীয়বার: কয়েক সেকেন্ডে Ready, কোনো download নেই
```

| কী মাপবেন | Command | কেন |
|---|---|---|
| প্রথম run-এর সময় (download সহ) আর দ্বিতীয় run-এর সময় | `time ./start.sh` | idempotency-র প্রমাণ: দ্বিতীয়টা সেকেন্ডে |
| Image size | `docker image ls` · `docker history healthcare-assistant-api` | API ≈১-১.৫ GB, UI ≈৪০০-৫০০ MB হওয়া উচিত (Class 35 §3.5-এর budget) |
| `unsafe_answers`, `missed_escalations` | `docker compose exec api python cli.py eval` | **০** হতে হবে |
| CPU বনাম GPU-র উত্তরের গতি | `docker compose exec ollama ollama run llama3.1:8b --verbose` → `eval rate` | GPU override-এর আসল লাভ |
| Offline re-run | internet বন্ধ করে `./stop.sh && ./start.sh` | `show || pull` কাজ করছে কিনা |
| Memory | `docker stats` | Ollama exit 137 = Docker-এর RAM কম |

## 🔬 Static Verification (এই repo-তে যা চালানো হয়েছে)

```bash
docker compose --env-file .env.example -f docker-compose.yml config --quiet                          # exit 0
docker compose --env-file .env.example -f docker-compose.yml -f docker-compose.gpu.yml config --quiet # exit 0
docker compose -f docker-compose.yml config     # .env ছাড়া: "required variable OLLAMA_TAG is missing a value" (ইচ্ছাকৃত)
pytest -q                                       # 91 passed (67 Week 17 + 24 deployment)
```

GPU merge-এর resolved output-এ base-এর তুলনায় একমাত্র পার্থক্য `ollama.deploy.resources.reservations.devices` (`count: all` → `-1`)। Docker CLI 29.1.3, Compose v2.40.3; daemon বন্ধ। Test-এর জন্য Python 3.13 venv-এ শুধু হালকা dependency (sentence-transformers/torch বা streamlit লাগে না — heavy import lazy, test mocked)।

## 🔧 Troubleshooting

| দেখছেন | কারণ | Fix |
|---|---|---|
| `Docker daemon not reachable` | Docker Desktop বন্ধ / Linux-এ dockerd বন্ধ | Docker চালু করে আবার script |
| `required variable OLLAMA_TAG is missing a value` | `.env` নেই বা পুরোনো | script চালান (কপি করে দেয়) বা `cp .env.example .env` |
| API log-এ `Ollama not reachable at http://localhost:11434` | কেউ `HC_OLLAMA_HOST`-এ localhost দিয়েছে — container-এর ভেতরে localhost মানে API নিজে | service name: `http://ollama:11434` (compose-এ আছে; override করবেন না) |
| `service "ollama-init" didn't complete successfully: exit 1` | প্রথম run-এ internet নেই বা registry-তে সমস্যা | `docker compose logs ollama-init`; network ঠিক করে আবার `./start.sh` |
| প্রতিবার model আবার download | volume মুছেছে (`down -v`), বা `name:` বদলেছে | `docker volume ls` → `healthcare-assistant_ollama_models` থাকা উচিত; `down -v` এড়ান |
| `/ready` 503 `knowledge base not indexed` | startup-এ Chroma-য় ingest ব্যর্থ হয়েছিল (log-এ `auto-ingest skipped`) | `docker compose exec api python cli.py ingest` বা `docker compose restart api` |
| `could not select device driver "nvidia"` | Container Toolkit নেই/configure হয়নি, বা CPU machine-এ `--gpu` | toolkit + `nvidia-ctk runtime configure` + Docker restart; না হলে `--gpu` ছাড়া |
| `ollama ps` বলে `100% CPU` GPU machine-এ | override ছাড়া `up`, driver < 550, পুরোনো card | `./start.sh --gpu` বা `.env`-এ `COMPOSE_FILE`; host-এ `nvidia-smi` |
| Ollama container exit 137 | Docker Desktop VM-এর RAM কম (OOM kill) | Resources-এ RAM বাড়ান, বা `HC_MODEL=llama3.2:3b` |
| `port is already allocated` | host-এ 8501/8000 দখল (যেমন Week 17-এর uvicorn চলছে) | `.env`-এ `UI_PORT`/`API_PORT` বদলান |
| `/v1/ask` 503 `llm_unavailable` CPU-তে | উত্তর `HC_LLM_TIMEOUT_S`-এর বেশি সময় নিচ্ছে | `HC_LLM_TIMEOUT_S=300`, `HC_NUM_PREDICT` কমান, বা GPU |
| `./start.sh: $'\r': command not found` | Windows checkout-এ CRLF | `.gitattributes` এটা আটকায়; না হলে `dos2unix start.sh` বা `.\start.ps1` |

**`docker compose down -v` সম্পর্কে:** `-v` named volume-ও মুছে দেয় — `ollama_models` (≈4.9 GB আবার download) আর `chroma_data` (পরের start-এ auto-ingest আবার বানায়)। ইচ্ছাকৃত reset ছাড়া কখনো না; `stop.sh`/`stop.ps1` তাই `-v` ছাড়া `down`।

## ☁️ Moving this to a Cloud VM

Cloud deployment এই project-এ theory (ADR 0010) — কোনো deploy script নেই, cloud account ধরে নেওয়া হয়নি। একই compose file একটা Ubuntu VM-এ চলে; চারপাশে যা যোগ করতে হয় (SSH key-only login, provider firewall-এ শুধু 22/80/443, Docker-এর published port ufw-কে পাশ কাটায়, Caddy reverse proxy-তে HTTPS + `basic_auth`, `ui`-এর port `!reset`, server-এ `chmod 600 .env`, volume backup VM-এর বাইরে, `json-file` log rotation, version tag দিয়ে rollback, reboot test, খরচের হিসাব) — সব **Week 18 Class 2 Lecture** (`../Class 2 Lecture/context.md` §3.8-3.16) আর তার notebook-এ। এই stack আগে থেকেই VM-বান্ধব: Ollama/Chroma publish করা নেই, `api`/`ui` `127.0.0.1`-এ, `restart: unless-stopped`।

## 📝 Changes from the Week 17 Copy

Code বদলায়নি; শুধু layout আর packaging:

1. Week 17-এর code `app/`-এর নিচে: `src/`, `data/kb`, `data/eval`, `ui.py`, `cli.py`, `.streamlit/`, `samples/make_handbook.py`, `tests/` — সব byte-for-byte একই (diff দিয়ে যাচাই)।
2. `requirements.txt` তিন ভাগ: `requirements-api.txt` (API image), `requirements-ui.txt` (UI image), `requirements.txt` (dev = দুটো + pytest, httpx2, fpdf2)। `pandas` বাদ (শুধু Week 17-এর notebook ব্যবহার করত, যেটা কপি হয়নি; Streamlit এমনিতেই আনে), `PyYAML` যোগ (deployment test)।
3. `pytest.ini` project root-এ (`pythonpath = app`, `testpaths = tests app/tests`) — এক `pytest -q`-এ দুই suite।
4. Week 17-এর single `Dockerfile` আর `.dockerignore`-এর জায়গায় root-এ `Dockerfile.api`, `Dockerfile.ui`, `.dockerignore` (build context = project root)। পার্থক্য: model bake code COPY-এর আগে (আগে ছিল পরে); `appuser` আগে তৈরি, bake তার নামে, `chown -R` শুধু খালি folder-এ; `HF_HOME` appuser-এর home-এ; torch version pin (`2.14.1`); `HEALTHCHECK`-এ `--start-period=120s`; `HC_AUTO_INGEST` image-এ না, compose-এ।
5. কপি হয়নি: `notebooks/01_exploration.ipynb` (threshold calibration — Week 17-এর বিষয়), `samples/clinic_handbook.pdf` (`python app/samples/make_handbook.py` আবার বানায়), Week 17-এর README।

**Notebook কেন নেই:** এই project-এর "logic" হলো config, আর সেটা যাচাই করে `tests/test_deploy.py`। Compose linter আর startup-order walk-through Class 1 Lecture-এর notebook-এ (`docker_ai_stack_lecture.ipynb`) আগে থেকেই আছে, VRAM আর cost calculator Class 2-এর notebook-এ — এখানে আরেকটা notebook শুধু তাদের পুনরাবৃত্তি হতো।

## ⚠️ Known Limits

- **কখনো build বা run হয়নি।** Image build, healthcheck-এর আসল আচরণ, `ollama-init`-এর pull, GPU passthrough — সব প্রথমবার আপনার machine-এ। Static test file-এর গঠন যাচাই করে, runtime না।
- **Image tag-এর অস্তিত্ব Docker Hub API দিয়ে দেখা (2026-10-28):** `ollama/ollama:0.35.1` (amd64 + arm64, ≈3.8 GB compressed) আর `chromadb/chroma:1.5.9` (amd64 + arm64); `torch==2.14.1+cpu` wheel CPU index-এ cp312 x86_64/aarch64-এর জন্য আছে। Pull করে দেখা হয়নি।
- **Base image `python:3.12-slim` minor-এ pin, patch-এ না** (lecture-এর মতো) — security rebuild পায়, কিন্তু দুই মাস পরের build একটু আলাদা হতে পারে। পুরো reproducible চাইলে `python:3.12.15-slim` বা `@sha256:` digest।
- **Chroma healthcheck raw HTTP over bash `/dev/tcp`** — image-এ curl নেই বলে; HTTP/1.0 response-এ ` 200 ` খোঁজে। Chroma image-এর base বদলে bash সরে গেলে এটা ভাঙবে।
- **Auto-ingest একবারই চেষ্টা করে** (Week 17-এর code)। Chroma healthy হওয়ার পরেই API start হয় বলে race কম, কিন্তু ব্যর্থ হলে নিজে আবার চেষ্টা করে না — `/ready` 503 থাকে আর start script timeout দেয় (Troubleshooting দেখুন)।
- **Healthcheck restart করে না।** Plain Docker unhealthy container restart করে না; `restart: unless-stopped` শুধু process exit-এ। Healthcheck এখানে `depends_on`-এর gate আর `ps`-এ দৃশ্যমানতা।
- **API image full `chromadb` package নেয়** (HTTP-only `chromadb-client` না), কারণ Week 17-এর code আর test `HC_CHROMA_HOST` খালি থাকলে local `PersistentClient` ব্যবহার করে। Ceiling: ≈২০০ MB বেশি; দরকার হলে code থেকে local path বাদ দিয়ে `chromadb-client`।
- **No login, no HTTPS** — `127.0.0.1`-এ bind করা বলে laptop-এ ঠিক আছে; LAN বা internet-এ খোলার আগে Class 36-এর reverse proxy অংশ।
- **CPU-তে ধীর।** `llama3.1:8b` CPU-তে প্রতি উত্তরে অনেক সেকেন্ড; ৪২টা প্রশ্নের `cli.py eval` কয়েক মিনিট থেকে বেশি নিতে পারে।
