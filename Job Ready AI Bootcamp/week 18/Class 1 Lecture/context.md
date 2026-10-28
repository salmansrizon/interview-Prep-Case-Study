# সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **1. Topic: Containerizing the AI Stack — Image, Compose, Healthcheck আর Single-Click Deployment**। Technical term English-এ রাখা হয়েছে, আর explanation Bangla-তে দেওয়া হয়েছে—যাতে concept বোঝা, code পড়া এবং interview-তে explain করা তিনটিই সহজ হয়।

## কেন এই Topic দরকার?

Week 17-এর Healthcare Assistant আপনার laptop-এ চালাতে এখন চারটা terminal লাগে: `ollama serve`, Chroma, `uvicorn`, `streamlit`। সাথে সঠিক Python version, venv, `ollama pull llama3.1:8b`, আর এক গাদা env var। অন্য কারো machine-এ এটা চালাতে বললে প্রথম এক ঘণ্টা যায় "আমার এখানে চলে না" ঠিক করতে। Container প্রতিটা অংশকে তার সব dependency-সহ একটা বাক্সে ভরে; Compose পুরো system-কে একটা file-এ লেখে; আর একটা command সবকিছু সঠিক ক্রমে চালু করে, *ready* হওয়া পর্যন্ত অপেক্ষা করে।

কিন্তু AI stack সাধারণ web app-এর মতো containerize হয় না। Image ১০ GB হয়ে যায় (CUDA torch, model weight), model download-এ পাঁচ GB আর দশ মিনিট লাগে, Ollama চালু হওয়া আর model *ready* থাকা এক জিনিস না, vector DB-র data container মুছলে হারিয়ে যায়, আর Week 15-এর `host.docker.internal` এখানে আর কাজের না। এই class-এ শিখব কীভাবে UI + DB + LLM পুরো stack-কে এমনভাবে package করতে হয় যাতে নতুন machine-এ একটা command-এই চলে, আর দ্বিতীয়বার চালালে কিছুই নতুন করে download হয় না।

## শেখার সহজ Workflow

Image → container → volume → network → AI app-এর Dockerfile → `.dockerignore` → image size budget → Compose service → service name দিয়ে discovery → healthcheck + `depends_on` → model pull-এর init container → named volume → 12-factor config → single-click (idempotent, এক command, readiness wait) → lint।

প্রতিটি section পড়ার সময় তিনটি প্রশ্ন করুন: **এটি কোন problem solve করে? কীভাবে কাজ করে? Alternative-এর তুলনায় কখন better?**

> **Practical mindset:** `docker compose up`-এর আগে কাগজে লিখুন — কোন container কার উপর নির্ভর করে, আর প্রতিটার ক্ষেত্রে "*ready*" মানে ঠিক কী (port খোলা? model pulled? knowledge base ingest শেষ?)। "Running" আর "ready"-র পার্থক্যই এই ক্লাসের অর্ধেক।

> **Verification-এর সৎ কথা (ADR 0008):** এই lecture যে machine-এ লেখা হয়েছে সেখানে Docker daemon বা GPU ছিল না। Compose file `docker compose config` (CLI, daemon লাগে না) আর notebook-এর linter দিয়ে যাচাই করা হয়েছে; `docker compose up`, `ps` বা `docker image ls`-এর output যেখানে দেখানো হয়েছে, সেগুলো **expected/illustrative** — নিজের machine-এ চালিয়ে মেলান।

---

## 1. Topic: Containerizing the AI Stack

এই ক্লাসে আমরা শিখব:

* **চারটা শব্দ** — image, container, volume, network; কোনটা কতক্ষণ বাঁচে।
* **AI app-এর Dockerfile** — slim base, CPU-only torch wheel, layer caching-এর ক্রম, model bake বনাম mount, non-root, `HEALTHCHECK`, exec-form `CMD`, `.dockerignore`।
* **Image size budget** — AI image কেন ১০ GB হয়, আর কীভাবে ১ GB-এর কাছে আনা যায়।
* **Multi-service Compose** — service name দিয়ে discovery, `depends_on` + healthcheck, model pull-এর জন্য one-shot init container, model weight আর vector DB-র জন্য named volume।
* **12-factor config** — একই image, আলাদা env; secret কখনো image-এ না।
* **Single-click** — idempotent, এক command, readiness wait, পরিষ্কার failure message।

**Project Goal (Shared Project `../Class 1-2 Project/`):** Week 17-এর Healthcare Assistant-এর একটা self-contained কপিকে ৪-service Compose stack বানানো — `ui` (Streamlit), `api` (FastAPI), `chroma` (Chroma server), `ollama` (official image) + one-shot `ollama-init` যেটা `llama3.1:8b` pull করে। Healthcheck, `depends_on: condition: service_healthy`, named volume, `.env.example`, optional GPU override (Class 36), আর single click = `make up` / `start.sh` / `start.ps1`।

---

## 2. Why It Is Related

Week 15-এর RAG API-র একটা Dockerfile ছিল: `python:3.12-slim`, CPU-only torch, embedding model image-এ baked, non-root user, `HEALTHCHECK`, exec-form `CMD`। কিন্তু Ollama চলত container-এর **বাইরে**, host-এ — API তাকে ডাকত `host.docker.internal:11434` দিয়ে। Week 15 Class 2-এ একটা ৬-service Compose file দেখেছিলাম (Langfuse), কিন্তু সেটা অন্যের app, আমাদের না।

Week 17-এ Healthcare Assistant বানিয়েছি: Streamlit UI → FastAPI API → Chroma (knowledge base) + Ollama (`llama3.1:8b`), সাথে Red-Flag Escalation আর Abstention। এই ক্লাসে চারটা অংশই container-এ যায়, নাম ধরে একে অপরকে খুঁজে নেয়, আর সঠিক ক্রমে চালু হয়। পরের ক্লাস (Class 36) একই stack-কে GPU দেবে আর একটা cloud VM-এ কীভাবে তুলতে হয় সেটা দেখাবে।

---

## 3. How It Works

### 3.1 চারটা শব্দ: Image, Container, Volume, Network

| জিনিস | কী | কতক্ষণ বাঁচে | Healthcare stack-এ |
|---|---|---|---|
| **Image** | read-only, layer-এ layer সাজানো template: OS + Python + package + code | যতক্ষণ না মুছছেন; registry-তে push করা যায় | `api`, `ui` (আমরা build করি); `ollama/ollama`, `chromadb/chroma` (pull করি) |
| **Container** | image-এর একটা চলমান instance + উপরে একটা পাতলা writable layer | container মুছলে writable layer-ও শেষ | ৫টা: ৪টা চলমান + `ollama-init` (কাজ শেষে exit) |
| **Volume** | container-এর lifecycle-এর বাইরে রাখা data | `docker compose down`-এর পরেও থাকে; শুধু `down -v` বা `volume rm` মোছে | `ollama_models` (≈4.9 GB model), `chroma_data` (vector) |
| **Network** | container-দের private network, নিজস্ব DNS সহ | Compose project-এর সাথে তৈরি/মোছা হয় | default network-এ service name = hostname |

সবচেয়ে গুরুত্বপূর্ণ সত্য: **container-এর writable layer ক্ষণস্থায়ী।** Ollama model pull করলে সেটা যায় `/root/.ollama`-তে; volume না থাকলে container recreate হওয়া মাত্র (image update, config বদল, `down`) ৪.৯ GB আবার download। Chroma-র vector-ও তাই। Rule: যে data হারালে কষ্ট হয়, সেটা volume-এ।

**Bind mount বনাম named volume:**

| | Bind mount (`./data:/data`) | Named volume (`chroma_data:/data`) |
|---|---|---|
| কোথায় থাকে | host-এর একটা নির্দিষ্ট folder | Docker নিজে manage করে |
| কখন | dev-এ code hot reload, host থেকে file দেখা | production data, model weight |
| ঝুঁকি | Windows/Mac-এ file I/O ধীর (VM boundary), host-এর permission/UID মেলে না | host থেকে সরাসরি দেখা কঠিন; backup আলাদা করে ভাবতে হয় |
| প্রথম mount | host folder যেমন আছে তেমন (image-এর content ঢেকে দেয়) | volume খালি হলে image-এর ওই path-এর content আর ownership কপি হয় |

### 3.2 AI App-এর Dockerfile — Week 15 থেকে এক ধাপ এগিয়ে

Week 15-এর Dockerfile-এর ভালো অভ্যাসগুলো রাখছি, আর তিনটা জিনিস ঠিক করছি:

```dockerfile
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface
RUN useradd --create-home appuser
WORKDIR /app

# 1. requirements first: the heavy layer, rarely changes
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

# 2. bake the small embedding model BEFORE copying code
USER appuser
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/paraphrase-MiniLM-L3-v2')"
# 3. only after the bake: never reach for the Hub at runtime
ENV HF_HUB_OFFLINE=1

# 4. code last: changes most often
COPY --chown=appuser app ./app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=120s \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

(Dockerfile-এ `#` comment শুধু নিজের line-এ চলে; instruction-এর পাশে লিখলে সেটা argument হয়ে যায়।)

* **Layer cache-এর ক্রম।** Docker প্রতিটা instruction-কে একটা layer বানায়; কোনো layer-এর input বদলালে তার **পরের সব** layer আবার build হয়। `COPY . .` যদি `pip install`-এর আগে থাকে, code-এর একটা comment বদলালেও torch আবার install — কয়েক মিনিট। Rule: যা কম বদলায় উপরে, যা বেশি বদলায় নিচে।
* **Week 15-এর একটা লুকোনো দাম:** সেখানে model bake ছিল `COPY app`-এর **পরে** — তাই code-এর প্রতিটা edit-এ embedding model আবার download হতো। এখানে bake আগে, code পরে।
* **CPU-only torch।** API container শুধু embedding বানায় (MiniLM, CPU-তে কয়েক ms); LLM চলে Ollama container-এ। Default torch wheel Linux-এ CUDA library-গুলো (`nvidia-*` package) টেনে আনে — কয়েক GB, যার একটাও এই container ব্যবহার করে না।
* **`HF_HUB_OFFLINE=1` bake-এর পরে।** আগে দিলে bake-এর download-ই ব্যর্থ হবে; পরে দিলে runtime-এ library কখনো network খোঁজে না — internet ছাড়া machine-এ startup আটকায় না।
* **Non-root।** Container-এর root মানে host kernel-এর সাথে root হিসেবে কথা বলা; কোনো library-র bug দিয়ে container থেকে বের হলে ক্ষতি বেশি। User আগে বানিয়ে `USER`-এর পরে bake করলে model file শুরু থেকেই appuser-এর — পরে `chown -R` লাগে না।
* **`chown -R`-এর size ফাঁদ।** আলাদা `RUN chown -R` পুরো directory-কে একটা নতুন layer-এ কপি করে। ৬০ MB-এর model-এ ৬০ MB বাড়ে; কেউ ৫ GB LLM bake করে `chown -R` করলে image-এ ৫ GB **দুবার**। `COPY --chown` বা আগে user বানানো এই কপি এড়ায়।
* **`HEALTHCHECK` আর exec-form `CMD`।** Slim image-এ `curl` নেই, তাই Python-এর `urllib`। Shell-form `CMD uvicorn ...` হলে PID 1 হয় `/bin/sh`, SIGTERM uvicorn-এ পৌঁছায় না — `docker stop` ১০ s অপেক্ষা করে তারপর kill করে।

**Build cache mount (ঐচ্ছিক উন্নতি):** `RUN --mount=type=cache,target=/root/.cache/pip pip install ...` — pip-এর download cache build-এর মাঝে থাকে কিন্তু image-এ ঢোকে না; requirements বদলালেও torch আবার download হয় না। এটা ব্যবহার করলে `PIP_NO_CACHE_DIR=1` সরাতে হয়।

**UI image আলাদা, অনেক হালকা।** Streamlit UI শুধু API-কে HTTP call করে — তার image-এ torch, sentence-transformers, chromadb কিছুই লাগে না। এক `requirements.txt` দুই image-এ দিলে UI image অকারণে এক GB ভারী হয়। প্রতিটা service-এর নিজের dependency তালিকা রাখুন।

**Base image-এর বিকল্প:**

| Base | কখন | সুবিধা | সীমা |
|---|---|---|---|
| `python:3.12-slim` (course) | বেশিরভাগ Python AI service | ছোট (≈১২০-১৫০ MB), Debian, সব wheel চলে | build tool নেই — C extension compile করতে হলে multi-stage |
| `python:3.12` (full) | compile করতে হয় এমন package, দ্রুত prototype | gcc, header সব আছে | ≈১ GB, attack surface বড় |
| `python:3.12-alpine` | ছোট pure-Python tool | সবচেয়ে ছোট | musl libc — torch/numpy-র manylinux wheel চলে না, source build বা ব্যর্থ |
| Distroless / minimal hardened image | security-কড়া production | shell নেই, CVE কম | debug কঠিন; healthcheck-এ shell command চলে না |

### 3.3 Model: Bake নাকি Mount?

| Artifact | আকার | সিদ্ধান্ত | কেন |
|---|---|---|---|
| Embedding model (`paraphrase-MiniLM-L3-v2`) | ≈৬১ MB | **Image-এ bake** | ছোট; image immutable আর reproducible; offline startup; প্রথম request-এ download নেই |
| LLM (`llama3.1:8b`, Q4_K_M) | ≈৪.৯ GB | **কখনো bake না** — volume + init pull | প্রতিটা push/pull-এ ৫ GB; hardware-ভেদে আলাদা quantization লাগতে পারে; Ollama নিজেই model store manage করে |
| Knowledge base document | ছোট | image-এ বা bind mount | data, code না; বদলালে re-ingest |
| Chroma vector | data-র সাথে বাড়ে | **volume** | runtime-এ লেখা হয়; container-এর বাইরে বাঁচতে হবে |

Embedding model বাছাই container-এর আকারও ঠিক করে:

| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| `sentence-transformers/paraphrase-MiniLM-L3-v2` (chosen, ADR 0004) | ছোট knowledge base-এ local semantic search | ≈৬১ MB, CPU-তে দ্রুত, bake করলে image প্রায় বাড়ে না | বড় model-এর চেয়ে retrieval quality একটু কম; acronym মেলাতে দুর্বল |
| `all-MiniLM-L6-v2` | সাধারণ sentence embedding | quality ভালো, এখনো ছোট (≈৯০ MB) | একটু ধীর; model বদলালে পুরো KB re-embed |
| `all-mpnet-base-v2` | quality সবচেয়ে জরুরি | এই family-তে সেরা accuracy | ≈৪২০ MB, CPU-তে কয়েক গুণ ধীর, image ভারী |
| Ollama embedding model (যেমন `nomic-embed-text`) | API image থেকে torch পুরো বাদ দিতে চাইলে | API image ১ GB+ হালকা; embedding আর LLM এক জায়গায় | Ollama-র slot-এর জন্য LLM-এর সাথে লড়াই; পুরো KB re-embed; Ollama down হলে retrieval-ও down |

আর LLM container-এর পছন্দ:

| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| `llama3.1:8b` (Q4_K_M) in `ollama/ollama` (chosen, ADR 0006) | Healthcare Assistant-এর উত্তর, course standard | official image, এক volume, ≈৪.৯ GB, RAG আর instruction-এ যথেষ্ট | GPU ছাড়া ধীর; Mac-এর Docker Desktop-এ GPU নেই |
| `llama3.2:3b` in same container | ৮ GB RAM machine (ADR 0006 fallback) | ≈২ GB, CPU-তেও চলনসই | medical প্রশ্নে instruction following আর grounding দুর্বল |
| Host-এ native Ollama + `host.docker.internal` | Mac (Metal GPU), আগে থেকে Ollama install | GPU পায়, model আবার download লাগে না | আর "single click" না — host-এ আলাদা setup; Linux-এ `host-gateway` লাগে |
| vLLM container + Llama 3.1 8B | NVIDIA GPU-তে অনেক একসাথে user | continuous batching, বেশি throughput | NVIDIA GPU বাধ্যতামূলক, image বড়, FP16-এ ≈১৬ GB VRAM |
| Hosted API | GPU নেই, scale দরকার | কোনো model container নেই | health data বাইরে যায়, প্রতি token খরচ, Sovereign AI framing ভাঙে |

### 3.4 `.dockerignore` — Build Context ছোট আর নিরাপদ

`docker build .` পুরো folder-টা (build context) builder-এ পাঠায়। `.venv` (কয়েক GB), `data/chroma_db`, `.git`, notebook checkpoint — সব। আরও খারাপ: `.env` থাকলে আর `COPY . .` থাকলে secret image-এ ঢুকে যায়।

```
.venv/
venv/
__pycache__/
.pytest_cache/
.git/
.env
data/
*.ipynb_checkpoints
notebooks/
tests/
```

Week 15-এর `.dockerignore`-এ `.env` ছিল না (সেখানে secret ছিল না)। Healthcare stack-এ `.env` থাকবে, তাই এখানে বাধ্যতামূলক।

### 3.5 Image Size Budget: AI Image কেন ১০ GB হয়

নিচের সংখ্যাগুলো **approximate** (Linux x86_64, version-ভেদে বদলায়); নিজের image `docker image ls` আর `docker history <image>` দিয়ে মাপুন। Notebook-এর section 5 একই হিসাব করে।

| অংশ | Naive API image | Lean API image | কীভাবে কমল |
|---|---|---|---|
| Base | `python:3.12` ≈১,০০০ MB | `python:3.12-slim` ≈১২৫ MB | slim |
| torch | default wheel + CUDA lib ≈৪,৫০০ MB | CPU-only wheel ≈৭০০ MB | `--index-url .../whl/cpu` |
| transformers + sentence-transformers + numpy/scipy | ≈২৫০ MB | ≈২৫০ MB | — |
| Vector DB client | full `chromadb` ≈২০০ MB | `chromadb-client` (শুধু HTTP) ≈৩০ MB | server আলাদা container-এ |
| Embedding model | ≈৬১ MB | ≈৬১ MB | bake (ছোট) |
| LLM weight | baked ≈৪,৯০০ MB | ০ | Ollama volume |
| pip cache layer-এ | ≈৭০০ MB | ০ | `PIP_NO_CACHE_DIR=1` বা cache mount |
| **মোট** | **≈১১.৬ GB** | **≈১.২ GB** | **≈১০ গুণ ছোট** |

UI image (slim + Streamlit + তার pandas/pyarrow) ≈৪০০ MB-এর ঘরে থাকা উচিত। Budget কেন জরুরি: প্রতিটা deploy-এ image pull হয়; ২ MB/s-এর upload-এ ১১.৬ GB push ≈ ১.৬ ঘণ্টা, ১.২ GB ≈ ১০ মিনিট। আর VM-এর disk-এ পুরোনো tag জমে — rollback-এর জন্য তিনটা version রাখলে ৩৫ GB বনাম ৪ GB।

### 3.6 Compose: পুরো System এক File-এ

```yaml
name: healthcare-assistant            # project name স্থির: folder rename করলেও volume একই

services:
  ollama:
    image: ollama/ollama:${OLLAMA_TAG:?pin OLLAMA_TAG in .env}
    volumes: [ollama_models:/root/.ollama]
    healthcheck:
      test: ["CMD", "ollama", "list"]          # image-এ curl নেই; CLI নিজেই server-কে ডাকে
      interval: 10s
      timeout: 5s
      retries: 30
      start_period: 20s
    restart: unless-stopped
    # ports নেই: Ollama-র auth নেই, শুধু internal network-এ

  ollama-init:                               # one-shot: model না থাকলেই pull, তারপর exit 0
    image: ollama/ollama:${OLLAMA_TAG:?pin OLLAMA_TAG in .env}
    environment: {OLLAMA_HOST: "http://ollama:11434"}
    entrypoint: ["sh", "-c", "ollama show llama3.1:8b >/dev/null 2>&1 || ollama pull llama3.1:8b"]
    depends_on:
      ollama: {condition: service_healthy}
    restart: "no"

  chroma:
    image: chromadb/chroma:${CHROMA_TAG:?pin CHROMA_TAG in .env}
    volumes: [chroma_data:/data]
    healthcheck:
      test: ["CMD-SHELL", "bash -c ':> /dev/tcp/127.0.0.1/8000' || exit 1"]   # TCP check; image-এ curl নেই
      interval: 10s
      timeout: 5s
      retries: 12
    restart: unless-stopped

  api:
    build: {context: ., dockerfile: Dockerfile.api}
    environment:
      HC_OLLAMA_HOST: http://ollama:11434    # service name, localhost না
      HC_CHROMA_HOST: chroma
      HC_AUTO_INGEST: "true"
      HC_API_KEY: ${HC_API_KEY:-}            # .env থেকে; image-এ কখনো না
    ports: ["127.0.0.1:8000:8000"]           # debug-এর জন্য, শুধু এই machine থেকে
    depends_on:
      chroma: {condition: service_healthy}
      ollama-init: {condition: service_completed_successfully}
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request as u; u.urlopen('http://localhost:8000/health')"]
      interval: 10s
      timeout: 5s
      retries: 6
      start_period: 120s                     # embedding model load + auto-ingest
    restart: unless-stopped

  ui:
    build: {context: ., dockerfile: Dockerfile.ui}
    environment: {HC_API_URL: "http://api:8000"}
    ports: ["8501:8501"]                     # host-এ শুধু এটাই সবার জন্য খোলা
    depends_on:
      api: {condition: service_healthy}
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')"]
      interval: 10s
      timeout: 5s
      retries: 6
    restart: unless-stopped

volumes:
  ollama_models: {}
  chroma_data: {}
```

এটা lecture-এর sketch — Shared Project-এর আসল file-এ details (Dockerfile path, tag, timeout) আলাদা হতে পারে; README দেখুন। প্রতিটা অংশ নিচের section-গুলোতে।

### 3.7 Service Discovery: Container-এর ভেতরে `localhost` মানে নিজে

Compose প্রতিটা project-এর জন্য একটা default network বানায়; সেখানে প্রতিটা service তার **নাম দিয়েই** DNS-এ পাওয়া যায়। `api` container থেকে `http://ollama:11434` → Ollama container; `chroma:8000` → Chroma।

* **`localhost` ফাঁদ।** Container-এর ভেতরে `localhost` = সেই container নিজে। API-তে `HC_OLLAMA_HOST=http://localhost:11434` দিলে connection refused — Ollama অন্য container-এ। Laptop-এ যে config চলত, container-এ সেটাই ভাঙে।
* **Week 15 বনাম এখন।** তখন Ollama host-এ ছিল, তাই `host.docker.internal`। এখন Ollama নিজেই একটা service, তাই service name। Host-এর Ollama ব্যবহার করতে চাইলে (যেমন Mac-এ GPU-র জন্য) আবার `host.docker.internal` — Class 36।
* **Container port, host port না।** Container-দের মধ্যে সবসময় container-এর নিজের port (`api:8000`)। `ports: "18000:8000"` দিলেও UI ডাকবে `api:8000`; `18000` শুধু host-এর জন্য।
* **Browser বনাম server।** Streamlit-এর Python code server-এ (container-এ) চলে, তাই `http://api:8000` কাজ করে। কিন্তু যে frontend browser-এ JavaScript দিয়ে API ডাকে (React ইত্যাদি), তার জন্য `api` নামটা অচেনা — browser user-এর machine-এ চলে। তখন host-এর URL বা reverse proxy লাগে।
* **Healthcheck-এ `localhost` ঠিক আছে** — healthcheck container-এর ভেতরেই চলে, নিজেকেই check করে।
* **`ports` বনাম শুধু network।** যে service-কে বাইরের কেউ ডাকবে না (Ollama, Chroma), তার `ports` দরকার নেই — একই network-এর container এমনিতেই পায়। Ollama-র কোনো auth নেই; `11434` host-এ খুললে একই Wi-Fi-র যে কেউ আপনার GPU চালাতে পারে। Debug-এর জন্য খুলতে হলে `127.0.0.1:11434:11434`।

### 3.8 `depends_on` + Healthcheck: Running ≠ Ready

Docker docs স্পষ্ট বলে: Compose startup-এ container "ready" হওয়ার জন্য অপেক্ষা করে না, শুধু "running" হওয়া পর্যন্ত। Ollama process চালু হয়েছে মানে API নিতে পারছে না; API process চালু মানে embedding model load আর ingest শেষ না। `condition` তিন রকম:

| Condition | মানে | Stack-এ কোথায় |
|---|---|---|
| `service_started` (list-form `depends_on: [x]`-এর default) | container start হয়েছে, ব্যস | প্রায় কখনো যথেষ্ট না |
| `service_healthy` | dependency-র healthcheck pass করেছে | `ollama-init` → `ollama`, `api` → `chroma`, `ui` → `api` |
| `service_completed_successfully` | one-shot dependency exit code 0 দিয়ে শেষ | `api` → `ollama-init` |

Healthcheck-এর parameter:

* `interval` — কত পরপর check; `timeout` — একটা check-এর সর্বোচ্চ সময়; `retries` — টানা কতবার fail হলে `unhealthy`।
* `start_period` — শুরুর এই সময়ে fail গোনা হয় না। API-র জন্য বড় (embedding model load + `HC_AUTO_INGEST`-এর ingest); ছোট দিলে ধীর machine-এ API `unhealthy` হয়ে UI কখনো start হয় না। নতুন Docker-এ `start_interval` দিয়ে এই সময়ে বেশি ঘনঘন check করা যায়, যাতে ready হওয়া মাত্র ধরা পড়ে।
* **Healthcheck-এর command image-এর tool-এর উপর নির্ভর করে।** `ollama/ollama` image Ubuntu-ভিত্তিক কিন্তু `curl` নেই → `ollama list` (CLI server-কে API call করে, server না থাকলে fail)। `chromadb/chroma` image `debian:stable-slim`-ভিত্তিক, `curl` নেই কিন্তু `bash` আছে → bash-এর `/dev/tcp` দিয়ে TCP check (ceiling: port খোলা মানে heartbeat endpoint ঠিক আছে এমন না; HTTP check চাইলে Chroma 1.x-এর `/api/v2/heartbeat`)। আমাদের Python image-এ `urllib`। Image version বদলালে tool-ও বদলাতে পারে — `docker compose exec chroma sh -c 'command -v curl wget bash'` দিয়ে দেখে নিন।
* **Liveness বনাম readiness (Week 15-এর শিক্ষা)।** API-র healthcheck `/health` — সস্তা, Ollama ছোঁয় না। আর মনে রাখুন: **plain Docker unhealthy container restart করে না** — `restart: unless-stopped` শুধু process exit হলে কাজ করে। Healthcheck এখানে শুধু (১) `depends_on`-এর gate আর (২) `docker compose ps`-এ দৃশ্যমানতা। Unhealthy-তে restart চাইলে orchestrator (Swarm, Kubernetes) বা আলাদা watchdog লাগে।
* Healthcheck ছাড়া service-এর উপর `condition: service_healthy` দিলে `docker compose up` ব্যর্থ হয় — dependency কখনো "healthy" হতে পারে না। `docker compose config` এটা ধরে না; notebook-এর linter ধরে।

### 3.9 Init Container: Model Pull একবার, Idempotent

`ollama/ollama` image-এ model থাকে না। কাউকে একবার `ollama pull llama3.1:8b` চালাতে হবে — README-তে "এরপর এই command চালান" লিখলে আর single click থাকে না। সমাধান: একই image থেকে একটা one-shot service, যেটা Ollama-কে network দিয়ে বলে model আনতে (download আসলে করে `ollama` server, model যায় তার volume-এ), তারপর exit করে।

```yaml
entrypoint: ["sh", "-c", "ollama show llama3.1:8b >/dev/null 2>&1 || ollama pull llama3.1:8b"]
```

* **কেন `show || pull`, শুধু `pull` না?** `ollama pull` প্রতিবার registry-তে গিয়ে manifest মেলায়। Internet না থাকলে (VPS-এর firewall, লোডশেডিং-এ router বন্ধ, classroom-এর offline demo) model volume-এ থাকা সত্ত্বেও pull fail করে → init exit 1 → API কখনো start হয় না। `show` local-এ দেখে; model থাকলে network ছোঁয়াই হয় না। এটাই **idempotent**: দ্বিতীয়বার চালালে কিছুই বদলায় না, কিছু download হয় না।
* **ক্রম:** `ollama` healthy → `ollama-init` চলে → exit 0 → `api` start। প্রথমবার এই ধাপে ৪.৯ GB; ২০ MB/s-এ ≈৪ মিনিট, ৫ MB/s-এ ≈১৬ মিনিট। পরের বার এক সেকেন্ডের কম।
* **`docker compose ps -a`-এ `ollama-init` "Exited (0)"** — এটা error না, কাজ শেষ। "Exited (1)" হলে `docker compose logs ollama-init`।
* **কেন `pre_start` না?** নতুন Compose release-এ `pre_start` hook আছে — service container start হওয়ার *আগে* একটা ephemeral container-এ setup চালায় (migration-এর মতো কাজে)। কিন্তু model pull-এর জন্য Ollama server *চলমান* থাকতে হয় — `ollama`-র `pre_start` তখন server পাবে না। Docker docs নিজেই বলে, setup কাজটা আলাদা service হিসেবে রাখা ঠিক যখন সেটা shared concern বা আলাদাভাবে দেখা দরকার। তাই one-shot service।
* **বিকল্প: custom entrypoint script** যেটা `ollama serve` background-এ চালায়, অপেক্ষা করে, pull করে। চলে, কিন্তু signal handling আর failure দেখা কঠিন, আর official image-এর আচরণ বদলে যায়।

### 3.10 Named Volume, Project Name আর `down -v`

```
docker compose stop        # container থামে, থেকে যায়
docker compose down        # container + network মোছে; named volume থাকে
docker compose down -v     # volume-ও মোছে: ৪.৯ GB model + পুরো vector DB শেষ
```

* Volume-এর আসল নাম `<project>_<volume>` — যেমন `healthcare-assistant_ollama_models`। Project name না দিলে Compose folder-এর নাম থেকে বানায়। Folder rename করলে বা অন্য path-এ clone করলে **নতুন খালি volume** → model আবার download, KB আবার ingest। Top-level `name:` এটা স্থির করে।
* "সব পরিষ্কার করে নতুন করে শুরু" করতে `down -v` অভ্যাস বিপজ্জনক — production VM-এ এটা data-loss incident। Cleanup-এর জন্য `down` (volume ছাড়া) যথেষ্ট।
* কোন volume হারালে কী: `ollama_models` → আবার download (বিরক্তিকর, ফেরানো যায়)। `chroma_data` → `HC_AUTO_INGEST` আবার বানায় (KB source document যদি image/repo-তে থাকে)। কিন্তু ভবিষ্যতে user-এর upload করা document বা chat history এলে সেটা **ফেরানো যায় না** — তখন backup বাধ্যতামূলক (Class 36)।

### 3.11 12-Factor Config: একই Image, আলাদা Env

The Twelve-Factor App-এর তৃতীয় factor: config (যা deploy-ভেদে বদলায় — URL, credential, flag) code-এ না, **environment variable**-এ। ফলে একই image laptop, staging আর VM-এ চলে; শুধু env বদলায়।

* **App-এর দিক:** Week 15-এর মতো pydantic-settings, prefix `HC_` (`HC_OLLAMA_HOST`, `HC_CHROMA_HOST`, `HC_API_URL`, `HC_AUTO_INGEST`)। Default laptop-এর জন্য, Compose container-এর জন্য override করে।
* **Compose-এর দুই রকম `.env` ব্যবহার — গুলিয়ে ফেলা খুব সাধারণ:**
  * project folder-এর `.env` Compose নিজে পড়ে **file-এর ভেতরে `${VAR}` বসানোর জন্য** (interpolation) — যেমন `${OLLAMA_TAG}`।
  * container-এর ভেতরে variable পাঠাতে `environment:` বা `env_file:` লাগে। `.env`-এ লিখলেই container পায় না।
* `${VAR:?message}` — না থাকলে Compose error দিয়ে থামে (আমরা image tag-এ ব্যবহার করি, যাতে চুপচাপ `latest` না হয়)। `${VAR:-default}` — না থাকলে default।
* **`.env.example` commit, `.env` কখনো না** (`.gitignore` আর `.dockerignore` দুটোতেই)। Single-click script `.env` না থাকলে example থেকে কপি করে।
* **Secret কখনো image-এ না।** `ENV HC_API_KEY=...` বা `ARG` দিয়ে দিলে `docker history` আর `docker image inspect`-এ যে কেউ দেখে — image registry-তে গেলে secret-ও গেল। Secret runtime-এ env বা file (Compose `secrets:`) দিয়ে।

### 3.12 Single-Click: এক Command-এর পেছনে চারটা প্রতিশ্রুতি

"Single-click" মানে শুধু একটা shortcut না। চারটা প্রতিশ্রুতি:

1. **এক command, কোনো manual ধাপ নেই** — `make up` / `./start.sh` / `.\start.ps1`; model pull, ingest সব stack নিজে করে।
2. **Idempotent** — দ্বিতীয়বার চালালে একই অবস্থা: অপরিবর্তিত container ছোঁয়া হয় না (`docker compose up` নিজেই এটা করে), model আবার download হয় না (`show || pull`), KB-তে duplicate chunk হয় না (Week 15-এর content-hash `upsert`)।
3. **Readiness wait** — "done" বলবে তখনই যখন user আসলে ব্যবহার করতে পারবে, শুধু container start হলে না।
4. **পরিষ্কার failure** — Docker daemon বন্ধ, port দখল, init fail — কোনটা হয়েছে আর কোন log দেখতে হবে, সেটা বলে থামা।

```sh
#!/usr/bin/env sh
set -eu
docker info >/dev/null 2>&1 || { echo "Docker is not running - start Docker Desktop / dockerd first"; exit 1; }
[ -f .env ] || cp .env.example .env
docker compose up -d --build            # dependency-র condition পূরণ না হওয়া পর্যন্ত এটা নিজেই অপেক্ষা করে
for i in $(seq 1 120); do               # তারপর UI নিজে ready কিনা (সর্বোচ্চ ১০ মিনিট)
  curl -fs http://localhost:8501/_stcore/health >/dev/null && { echo "Ready: http://localhost:8501"; exit 0; }
  sleep 5
done
docker compose ps -a
echo "Not ready after 10 minutes - see: docker compose logs <service>"; exit 1
```

* `docker compose up -d` dependent service start করার আগে `service_healthy`/`completed` condition-এর জন্য নিজেই অপেক্ষা করে — তাই command ফিরে আসার সময় API-র dependency-গুলো ready। শেষ service (UI)-এর নিজের readiness-এর জন্য loop।
* `docker compose up --wait` আছে — service-গুলো running/healthy হওয়া পর্যন্ত অপেক্ষা করে। One-shot service (exit করে যায়) নিয়ে এর আচরণ Compose version-ভেদে আলাদা হয়েছে — নিজের version-এ test না করে script-এ ভরসা করবেন না।
* Windows-এর `start.ps1` একই কাজ PowerShell-এ (`Invoke-WebRequest`)। `make up` শুধু wrapper — Windows-এ `make` সাধারণত থাকে না, তাই তিনটাই।

---

## 4. Details & Valid Points

### 4.1 Compose যা ধরে, আর যা ধরে না

`docker compose config` file পড়ে, `.env` থেকে variable বসায়, reference মেলায় (volume declared কিনা, `depends_on`-এর service আছে কিনা), আর resolved file দেখায় — **daemon লাগে না**, তাই CI-র প্রথম check হিসেবে চমৎকার। Notebook-এ এই repo-র build machine-এ (Docker CLI ছিল, daemon ছিল না) আসলেই চালানো হয়েছে।

কিন্তু design-এর ভুল সে ধরে না। Notebook-এর linter যেগুলো ধরে:

| Rule | উদাহরণ | কেন খারাপ |
|---|---|---|
| `localhost` container-দের মধ্যে | `HC_OLLAMA_HOST=http://localhost:11434` | container নিজেকে ডাকছে → connection refused |
| healthcheck ছাড়া `service_healthy` | `api` → `chroma: service_healthy`, chroma-য় healthcheck নেই | `up` ব্যর্থ |
| list-form `depends_on` | `depends_on: [api]` | শুধু "started" পর্যন্ত অপেক্ষা |
| `latest` / tag নেই | `ollama/ollama:latest` | আজ আর কাল আলাদা version; rollback অসম্ভব |
| literal secret | `HC_API_KEY: sk-demo-123` | file commit হলেই secret ফাঁস |
| stateful service-এ volume নেই | `ollama`-এ volume নেই | recreate-এ ৪.৯ GB আবার download |
| internal port public | `ollama: ports: ["11434:11434"]` | auth ছাড়া LLM সবার জন্য খোলা |
| undeclared named volume | `chroma_db:/data`, top-level-এ নেই | Compose নিজেও ধরে |

### 4.2 Image Tag: `latest` একটা অবস্থা, version না

* `latest` মানে "শেষবার যে push হয়েছে" — আজ এক version, কাল আরেক। দুই জনের machine-এ একই compose file দুটো আলাদা stack।
* **Chroma-র client আর server version মিলতে হয়।** API image-এ `chromadb-client` (বা `chromadb`) এক version-এ pin, server image `latest` — server-এর major update-এ API ভাঙে। দুটোই একসাথে pin, একসাথে upgrade।
* Ollama update-এ model format বা API-র ছোট পরিবর্তন আসে; pin করে, test করে, তারপর tag বদলান।
* সবচেয়ে কড়া pin: digest (`image@sha256:...`) — tag সরিয়ে দিলেও একই byte।

### 4.3 এক Container, এক Process

Ollama + API + UI এক container-এ `supervisord` দিয়ে চালানো যায়, কিন্তু: একটার crash অন্যটাকে নিয়ে যায়, আলাদা scale বা restart করা যায় না, log মিশে যায়, image-এ সবার dependency (UI image-এ CUDA!)। একটা container = একটা concern। Week 15-এর শিক্ষাও এখানে খাটে: API container-এ একটাই uvicorn process — bottleneck Ollama-র slot, worker না।

### 4.4 Memory আর Platform

* **RAM:** `llama3.1:8b` Q4 CPU mode-এ ৫-৬ GB RAM নেয়, সাথে API (torch + MiniLM ≈ কয়েকশো MB), Chroma, UI। Docker Desktop চলে একটা VM-এ, যার memory limit আছে (Settings → Resources; Windows-এ WSL2-র `.wslconfig`)। Limit কম হলে Ollama-কে kernel মেরে ফেলে — container **exit code 137** (SIGKILL, প্রায়ই OOM)।
* **Apple Silicon:** `ollama/ollama`, `chromadb/chroma`, `python:3.12-slim` arm64-এ আছে; CPU-only torch wheel-ও আছে। কিন্তু Docker Desktop-এ Mac-এর GPU container-এ যায় না — Ollama container CPU-তে চলে। বিকল্প Class 36-এ।
* **Windows:** Docker Desktop + WSL2 backend। Bind mount-এ Windows folder (`C:\...`) WSL2 VM-এর ভেতর থেকে ধীর — named volume এই সমস্যা এড়ায়। Path-এ space থাকলে (`Class 1-2 Project`) script-এ quote।

### 4.5 Debug Toolkit

| প্রশ্ন | Command |
|---|---|
| কোনটা চলছে, healthy কিনা, কে exit করেছে | `docker compose ps -a` (machine-readable: `--format json`) |
| কেন unhealthy | `docker inspect --format '{{json .State.Health}}' <container>` |
| log | `docker compose logs -f api`, `docker compose logs ollama-init` |
| ভেতরে ঢুকে দেখা | `docker compose exec api sh` |
| network থেকে অন্য service পাওয়া যায় কিনা | `docker compose exec api python -c "import urllib.request as u; print(u.urlopen('http://ollama:11434').read())"` |
| model আছে কিনা | `docker compose exec ollama ollama list` |
| resolved config | `docker compose config` |
| image কেন বড় | `docker history <image>` |

---

## 5. Related Analogy: বিয়েবাড়ির Catering

একটা catering company প্রতি সপ্তাহে আলাদা আলাদা বিয়েবাড়িতে একই মানের খাবার দেয়।

* **Image** = sealed রান্নার kit: রেসিপি + মাপা মশলা + হাঁড়ি। যে হলেই যাক, একই kit, একই স্বাদ।
* **Container** = আজকের অনুষ্ঠানে জ্বলা একটা চুলা। অনুষ্ঠান শেষে চুলা গুটিয়ে ফেলা হয় — চুলার উপর রেখে যাওয়া কিছু থাকে না।
* **Volume** = গুদাম আর ফ্রিজ। চুলা গুটালেও চাল-ডাল-মাংস থাকে। `down -v` = গুদাম খালি করে ফেলা।
* **Service name** = কাউন্টারের নাম ধরে ডাকা ("বিরিয়ানি কাউন্টার"), টেবিল নম্বর না। আর "আমার পাশের জন" (`localhost`) প্রত্যেক কাউন্টারে আলাদা মানুষ।
* **Healthcheck + `depends_on`** = পরিবেশন শুরু হবে যখন বিরিয়ানি সত্যিই রান্না হয়েছে — শুধু চুলা জ্বলেছে বলে না।
* **Init container** = সকালে একবার বাজার। ফ্রিজে আগে থেকে মাংস থাকলে আবার বাজারে যাওয়া না (`show || pull`) — বাজার বন্ধ থাকলেও অনুষ্ঠান আটকায় না।
* **`.env`** = আজকের অনুষ্ঠানের নির্দেশনার কাগজ (কত অতিথি, কোন হল)। রেসিপিতে লেখা থাকে না — রেসিপি সব অনুষ্ঠানে এক।
* **Single-click** = manager-এর একটা ফোন: "অনুষ্ঠান চালু করো।" সবাই নিজের ক্রমে কাজ ধরে, আর manager তখনই "ready" বলে যখন প্রথম প্লেট পরিবেশনযোগ্য।
* **Image size** = kit-এর ওজন। প্রতিটা অনুষ্ঠানে পুরো গুদাম (LLM weight) kit-এ ভরে বয়ে নিলে ভ্যান ভারী, দেরি — গুদাম থাকে গুদামে।

---

## 6. Common Pitfalls

| Pitfall | কী হয় | Fix |
|---|---|---|
| Container-এ `localhost:11434` | connection refused — container নিজেকে ডাকছে | service name: `http://ollama:11434` |
| Week 15-এর `host.docker.internal` রেখে দেওয়া | Ollama এখন container-এ, host-এ না | service name; host-এর Ollama চাইলে তবেই `host.docker.internal` |
| প্রতি build-এ model আবার download | bake-এর RUN `COPY app`-এর পরে | model আগে, code পরে |
| প্রতি `up`-এ LLM আবার download | volume নেই, বা project name বদলেছে | named volume + top-level `name:` |
| শুধু `ollama pull` init-এ | offline-এ pull fail → API কখনো start হয় না | `ollama show ... \|\| ollama pull ...` |
| `docker compose down -v` অভ্যাস | model + vector DB মুছে গেল | `down`; `-v` শুধু ইচ্ছাকৃত reset-এ |
| `ollama/ollama:latest`, `chromadb/chroma:latest` | আজ চলে কাল ভাঙে; client-server mismatch | `.env`-এ tag pin, `${TAG:?}` |
| List-form `depends_on` | API চালু হয় Ollama ready হওয়ার আগে | `condition: service_healthy` |
| Healthcheck-এ `curl` | image-এ curl নেই → সবসময় unhealthy | image-এর tool দেখে: `ollama list`, Python `urllib`, bash `/dev/tcp` |
| ছোট `start_period` | ধীর machine-এ API unhealthy, UI কখনো আসে না | slowest startup (model load + ingest) মেপে |
| "Unhealthy হলে Docker restart করবে" | plain Docker করে না | healthcheck = gate + visibility; restart চাইলে orchestrator |
| Default torch wheel | API image-এ কয়েক GB অব্যবহৃত CUDA | CPU index |
| এক requirements দুই image-এ | UI image-এ torch | service-প্রতি dependency |
| `.env` build context-এ + `COPY . .` | secret image-এ | `.dockerignore`-এ `.env` |
| `ENV API_KEY=...` Dockerfile-এ | `docker history`-তে সবাই দেখে | runtime env / secret file |
| Ollama port `0.0.0.0`-এ publish | auth ছাড়া LLM network-এ খোলা | `ports` বাদ, বা `127.0.0.1:` |
| Exit code 137 | Docker Desktop VM-এর memory কম, OOM kill | Resources-এ RAM বাড়ান বা `llama3.2:3b` |
| Shell-form `CMD` | SIGTERM পৌঁছায় না, stop-এ ১০ s পরে kill | exec-form `CMD [...]` |

---

## 7. Interview Q&A

1. **Image, container আর volume-এর পার্থক্য কী? কোনটায় model রাখবেন?** Image read-only layered template; container তার চলমান instance + ক্ষণস্থায়ী writable layer; volume container-এর lifecycle-এর বাইরের data। ছোট embedding model image-এ bake (immutable, offline startup); ৫ GB LLM আর vector DB volume-এ — container recreate-এ হারায় না, image ছোট থাকে।
2. **API container `localhost:11434`-এ Ollama পায় না কেন? Service discovery কীভাবে কাজ করে?** Container-এর `localhost` সে নিজে। Compose project-এর default network-এ প্রতিটা service তার নামে DNS-এ resolve হয় — `http://ollama:11434`। Container-দের মধ্যে container port, host-এর published port না। Host-এর service চাইলে `host.docker.internal` (Linux-এ `host-gateway`)।
3. **`depends_on` দিলেও API আগে চালু হয়ে Ollama-তে fail করে — কেন, ঠিক কীভাবে?** Default condition শুধু "started"। Dependency-তে healthcheck দিয়ে `condition: service_healthy`; one-shot setup-এ `service_completed_successfully`। আর app নিজেও startup-এ ছোট retry রাখুক — ordering race কমায়, কিন্তু পরে dependency restart হলে app-কেই সামলাতে হয়।
4. **Docker কি unhealthy container restart করে?** Plain Docker/Compose না — restart policy শুধু process exit-এ কাজ করে। Healthcheck এখানে `depends_on`-এর gate আর `ps`-এ status। Unhealthy-তে replace করে Swarm বা Kubernetes (liveness probe)।
5. **আপনার AI image ১০ GB — কীভাবে ছোট করবেন?** `docker history` দিয়ে দেখুন কোন layer। সাধারণ অপরাধী: full base image, CUDA torch যেখানে GPU লাগে না, image-এ baked LLM weight, pip cache, `chown -R`-এ duplicate layer, এক requirements সব service-এ। Fix: slim base, CPU wheel, LLM volume-এ, `PIP_NO_CACHE_DIR`/cache mount, service-প্রতি dependency, দরকার হলে multi-stage।
6. **প্রতিবার `docker compose up`-এ model download হওয়া কীভাবে আটকাবেন?** Ollama-র `/root/.ollama`-এ named volume; top-level `name:` যাতে volume নাম folder-এর উপর নির্ভর না করে; init-এ `show || pull` যাতে থাকলে network না ছোঁয়; `down -v` না।
7. **Model pull কোথায় করবেন — Dockerfile, entrypoint script, নাকি আলাদা service?** Dockerfile-এ না (image ৫ GB, hardware-নির্দিষ্ট)। আলাদা one-shot service যেটা server healthy হলে চলে, idempotent command দিয়ে, আর API তার `service_completed_successfully`-এর উপর নির্ভর করে। `pre_start` server চালুর আগে চলে, তাই এখানে খাটে না; custom entrypoint চলে কিন্তু signal আর failure দেখা কঠিন।
8. **Secret কোথায় রাখবেন? `.env` আর `env_file` কি একই?** Image-এ কখনো না (`ENV`/`ARG` `docker history`-তে দেখা যায়)। Runtime env বা secret file দিয়ে। Project-এর `.env` Compose file-এর `${VAR}` interpolation-এর জন্য; container-এ পাঠাতে `environment:`/`env_file:`। `.env.example` commit, `.env` `.gitignore` আর `.dockerignore`-এ।
9. **Dockerfile-এ instruction-এর ক্রম কেন গুরুত্বপূর্ণ?** প্রতিটা instruction একটা layer; কোনো layer-এর input বদলালে পরের সব layer আবার build। Requirements + install আগে, model bake তারপর, code শেষে — code edit-এ শুধু শেষ layer। Torch-এর মতো ভারী dependency-তে এটা মিনিট বনাম সেকেন্ড।
10. **"Single-click deployment" বলতে আপনি কী বোঝেন?** এক command, কোনো manual ধাপ নেই; idempotent (আবার চালালে একই অবস্থা, কোনো re-download বা duplicate ingest নেই); readiness wait (user ব্যবহার করতে পারলে তবেই "ready"); আর failure হলে কোন service, কোন log সেটা বলা। Compose-এর declarative model প্রথম দুটোর বেশিরভাগ দেয়; বাকিটা ছোট wrapper script।
11. **Ollama, API, UI এক container-এ রাখলে সমস্যা কী?** একটার crash সবাইকে নেয়, আলাদা restart/scale/upgrade হয় না, log মেশে, image-এ সবার dependency। এক container এক concern — Compose তাদের জোড়া লাগায়।

---

## Achievement: Single-Click Deployment (Week 18 Shared Project)

**`../Class 1-2 Project/`** — Week 17-এর Healthcare Assistant-এর self-contained কপি (ADR 0002) একটা ৪-service Compose stack হিসেবে: `ui` (Streamlit, port 8501, `HC_API_URL=http://api:8000`), `api` (FastAPI, port 8000, `HC_OLLAMA_HOST=http://ollama:11434`, `HC_CHROMA_HOST=chroma`, `HC_AUTO_INGEST=true`), `chroma` (Chroma server image, persistent volume), `ollama` (official `ollama/ollama` image, model volume), আর one-shot `ollama-init` যেটা `llama3.1:8b` pull করে। Healthcheck, `depends_on: condition: service_healthy`, named volume, `.env.example`, optional `docker-compose.gpu.yml` (NVIDIA GPU override — Class 36), আর **single click = একটা local command**: `make up` / `start.sh` / `start.ps1`, ভেতরে `docker compose up -d`। এই ক্লাস containerization-এর অংশ ব্যাখ্যা করে; GPU আর cloud VM-এর অংশ Class 36-এ। Stack এই repo-তে statically verified (ADR 0008) — প্রথম আসল `docker compose up` আপনার machine-এ। এই ক্লাসের companion notebook **`docker_ai_stack_lecture.ipynb`** সম্পূর্ণ self-contained — project folder বা Docker daemon ছাড়াই চলে।

---

## 🧠 Brain Teasers & Exercises (নিজে চেষ্টা করুন)

1. **Cache Detective:** Week 15-এর Dockerfile-এ `app/main.py`-তে একটা comment যোগ করে আবার build করলে কোন কোন layer আবার চলে? কত MB আবার download হয়? এই lecture-এর ক্রমে কত?
2. **Healthcheck মুছে দেখুন:** `chroma` service থেকে `healthcheck` সরান, কিন্তু `api`-র `condition: service_healthy` রাখুন। `docker compose config` কী বলে? `docker compose up` কী বলে? Notebook-এর linter কী বলে?
3. **Rename Trap:** Project folder-এর নাম বদলে (top-level `name:` ছাড়া) `docker compose up -d` চালান। `docker volume ls`-এ কী দেখেন? Model আবার download হচ্ছে কেন?
4. **Offline Drill:** Stack একবার চালিয়ে model pull করুন। তারপর internet বন্ধ করে `docker compose down && docker compose up -d` — init-এ শুধু `ollama pull` থাকলে কী হয়, `show || pull` থাকলে কী হয়?
5. **Size Budget:** নিজের API image build করে `docker history`-র সবচেয়ে বড় তিনটা layer লিখুন। Default torch wheel দিয়ে আবার build করে তুলনা করুন। ২ MB/s upload-এ দুটো push করতে কত সময়?
6. **`localhost` শিকার:** Stack-এর প্রতিটা URL-এর তালিকা বানান (UI → API, API → Ollama, API → Chroma, browser → UI, healthcheck)। কোনটা container-এর ভেতর থেকে, কোনটা host থেকে, কোনটা browser থেকে ডাকা হয়? কোথায় `localhost` ঠিক, কোথায় ভুল?

---

## References

* Docker docs — Dockerfile best practices: https://docs.docker.com/build/building/best-practices/ · Multi-stage builds: https://docs.docker.com/build/building/multi-stage/ · Build cache: https://docs.docker.com/build/cache/ · Cache optimization (cache mounts): https://docs.docker.com/build/cache/optimize/ · Build context and `.dockerignore`: https://docs.docker.com/build/concepts/context/
* Docker docs — Volumes: https://docs.docker.com/engine/storage/volumes/ · Port publishing: https://docs.docker.com/engine/network/port-publishing/
* Docker Compose — Control startup order (`depends_on`, conditions): https://docs.docker.com/compose/how-tos/startup-order/ · `healthcheck` and `depends_on` reference: https://docs.docker.com/reference/compose-file/services/ · Init containers (`pre_start`): https://docs.docker.com/compose/how-tos/init-containers/ · Networking: https://docs.docker.com/compose/how-tos/networking/ · Environment variables and interpolation: https://docs.docker.com/compose/how-tos/environment-variables/ · Project name: https://docs.docker.com/compose/how-tos/project-name/ · `docker compose up`: https://docs.docker.com/reference/cli/docker/compose/up/ · `docker compose ps`: https://docs.docker.com/reference/cli/docker/compose/ps/ · `docker compose config`: https://docs.docker.com/reference/cli/docker/compose/config/
* Ollama Docker: https://docs.ollama.com/docker · Ollama FAQ: https://docs.ollama.com/faq · Ollama image Dockerfile (base, no curl): https://github.com/ollama/ollama/blob/main/Dockerfile
* Chroma — Docker deployment: https://docs.trychroma.com/guides/deploy/docker · Chroma image Dockerfile: https://github.com/chroma-core/chroma/blob/main/rust/Dockerfile · `chromadb-client` (HTTP-only client): https://pypi.org/project/chromadb-client/
* PyTorch install (CPU-only wheels): https://pytorch.org/get-started/locally/ · Hugging Face Hub environment variables (`HF_HOME`, `HF_HUB_OFFLINE`): https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
* Streamlit — Deploy using Docker (`/_stcore/health` healthcheck): https://docs.streamlit.io/deploy/tutorials/docker
* The Twelve-Factor App — Config: https://12factor.net/config · Backing services: https://12factor.net/backing-services · Dev/prod parity: https://12factor.net/dev-prod-parity · Disposability: https://12factor.net/disposability
