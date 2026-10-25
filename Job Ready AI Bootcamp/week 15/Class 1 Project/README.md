# 🚀 Production-Ready RAG API Service (FastAPI + Ollama + ChromaDB)

## সহজ ভাষায় Project Overview

Week 12-এর RAG Q&A bot একটা notebook/Streamlit demo ছিল। এই project সেটাকে একটা **service** বানায়: অন্য program HTTP দিয়ে PDF upload করে, প্রশ্ন করে, আর citation-সহ উত্তর পায় — একবারে (`/v1/ask`) অথবা token-by-token stream (`/v1/ask/stream`)। সব local: `llama3.1:8b` (Ollama) + ChromaDB + একটা ছোট embedding model, internet লাগে না।

### কোন Problem Solve করে?

Demo আর service-এর পার্থক্য happy path-এ না, **খারাপ দিনে**: একসাথে ১০ জন প্রশ্ন করলে, Ollama বন্ধ থাকলে, কেউ ২০০ MB-র "PDF" পাঠালে, একই file দুবার upload হলে, client মাঝপথে চলে গেলে। এই project প্রতিটা ক্ষেত্রে একটা নির্দিষ্ট, test-করা আচরণ দেয় — 503 + `Retry-After`, retrieval-only fallback, 413/415/422, idempotent ingestion, slot release।

### কেন FastAPI?

Async server (একটা process অনেক connection ধরে রাখে), Pydantic v2 validation, auto OpenAPI docs (`/docs`), dependency injection (test-এ backend বদলানো সহজ), আর 0.135 থেকে built-in Server-Sent Events।

> **Run করার নিয়ম:** venv বানিয়ে dependency install, `ollama pull llama3.1:8b`, তারপর Quick Start। এই course-এ একটাই model: `llama3.1:8b`।

---

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.162-009688.svg)
![Ollama](https://img.shields.io/badge/ollama-llama3.1%3A8b-black.svg)
![ChromaDB](https://img.shields.io/badge/chromadb-1.5-orange.svg)
![Tests](https://img.shields.io/badge/tests-24%20passing-green.svg)

## ✨ Features

- 📄 **Safe upload** — `.pdf` only (415), size limit read as `limit+1` bytes (413), `%PDF-` magic check + page cap + "no text layer" (422)।
- ♻️ **Idempotent ingestion** — chunk id = `sha256(source + text)`; একই file আবার upload → `new_chunks: 0`, কোনো duplicate না।
- 🎯 **Grounding guardrail** — সবচেয়ে কাছের chunk-ও `max_distance`-এর বাইরে হলে LLM ডাকাই হয় না → "not found" (০ ms GPU)।
- 📌 **Exact page citations** — chunk কখনো page পার হয় না; প্রতিটা উত্তরে `[n]` → source, page, chunk_id, distance, snippet।
- 🌊 **SSE streaming** — `citations` → `token`… → `done` (usage + timings সহ); `X-Accel-Buffering: no`; client চলে গেলে আর token টানা হয় না আর LLM slot ছেড়ে দেওয়া হয়।
- 🚦 **Backpressure** — `asyncio.Semaphore` = Ollama-র parallel slot; slot না পেলে `queue_timeout_s` পরে **503 + `Retry-After`**। Stream-এর slot response শুরুর *আগে* নেওয়া হয়, তাই 503-ও একটা সাধারণ HTTP status।
- 🩹 **Graceful degradation** — Ollama down/timeout → `degraded: true` + সবচেয়ে প্রাসঙ্গিক passage, 500 না।
- ⚡ **Answer cache** — exact-match LRU, key-তে `corpus_version`; নতুন document এলে পুরোনো উত্তর নিজে থেকে বাতিল। Semantic cache ইচ্ছে করে নেই।
- 🔑 **Optional API key** — `RAG_API_KEY` দিলে `/v1/*`-এ `X-API-Key` লাগে (401); `/health` `/ready` খোলা (orchestrator-এর জন্য)।
- 🩺 **`/health` বনাম `/ready`** — liveness (process বেঁচে) বনাম readiness (Ollama reachable + model pulled)।
- 🧾 **Request ID + timing** — প্রতিটা response-এ `X-Request-ID`, `X-Process-Time-Ms`; সব error-এর একই আকার `{error, detail, request_id}`।
- 📊 **Benchmark** — concurrency sweep: req/s, p50/p95/p99, TTFT, 503 count; fake (Ollama ছাড়া) বা real server।
- 🧪 **24 offline tests** — real FastAPI + real ChromaDB (temp dir) + real PDF (fpdf2); শুধু LLM আর embedder নকল।

## 🚀 Quick Start

```bash
cd "Job Ready AI Bootcamp/week 15/Class 1 Project"
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.1:8b && ollama serve

uvicorn app.main:app --port 8000                    # http://localhost:8000/docs
RAG_BACKEND=fake uvicorn app.main:app --port 8000   # Ollama/Chroma ছাড়া demo (০.২s নকল latency)
pytest -q                                           # 24 tests, Ollama লাগে না
```

Windows PowerShell-এ env var: `$env:RAG_BACKEND="fake"`।

```bash
curl -F "file=@samples/employee_handbook.pdf" localhost:8000/v1/documents
curl -X POST localhost:8000/v1/ask -H "content-type: application/json" \
     -d '{"question": "How many sick days do employees get?"}'
curl -N -X POST localhost:8000/v1/ask/stream -H "content-type: application/json" \
     -d '{"question": "How many characters must a password have?"}'
```

Benchmark:

```bash
python benchmark/bench.py --fake                                   # in-process, ০.২s fake LLM
python benchmark/bench.py --fake --queue-timeout 0.5 --levels 1,4,16   # 503 দেখা দেয়
python benchmark/bench.py --url http://localhost:8000 --levels 1,2,4 --requests 8 --stream
```

Docker (Ollama host-এ চলে; image এই repo-তে build করে verify করা হয়নি — Docker daemon ছাড়া লেখা):

```bash
docker build -t rag-api .
docker run -p 8000:8000 -e RAG_OLLAMA_HOST=http://host.docker.internal:11434 -v "$PWD/data:/app/data" rag-api
```

## 📁 Project Structure

```
Class 1 Project/
├── app/
│   ├── main.py            # routes (/health, /ready, /v1/...), middleware, error handlers, SSE
│   ├── service.py         # RagService: semaphore gate, cache, threadpool, observers  ← monitoring seam
│   ├── backend.py         # RealBackend (app/rag) / FakeBackend (tests, benchmark)
│   ├── schemas.py         # Pydantic request/response models (= OpenAPI contract)
│   ├── errors.py          # ServiceError -> status code (+ Retry-After)
│   ├── settings.py        # pydantic-settings, env prefix RAG_
│   └── rag/               # self-contained RAG core (ADR 0002: no Week 12 import)
│       ├── ingest.py      # PDF checks, page extraction, page-bounded chunking, hash ids
│       ├── components.py  # Embedder, ChromaStore, OllamaLLM (usage + TTFT), Usage
│       └── engine.py      # retrieve, guardrail, prompt, answer, stream, degraded mode
├── benchmark/bench.py     # concurrency sweep (fake or real)
├── samples/               # employee_handbook.pdf + make_handbook.py (fictional demo data)
├── notebooks/01_exploration.ipynb   # live walkthrough of every behaviour below
├── tests/                 # test_api.py (HTTP), test_pipeline.py (RAG core)
├── Dockerfile             # CPU torch, embedding model baked in, non-root, HEALTHCHECK
└── requirements.txt       # pinned, tested versions
```

## 🏗️ Architecture

```
client ──HTTP──▶ middleware (request id, timing)
                   │
                   ▼
              route (validate: Pydantic, 413/415)  ── /health, /ready (no gate)
                   │
                   ▼
              RagService ── cache hit? ──▶ return (cached: true)
                   │
             semaphore gate ── busy > queue_timeout_s ──▶ 503 + Retry-After
                   │
          run_in_threadpool / iterate_in_threadpool     (blocking code off the event loop)
                   │
                   ▼
   RagEngine: embed ─▶ Chroma top-k ─▶ guardrail ─▶ prompt ─▶ Ollama llama3.1:8b
                                          │                     │
                                     not found (no LLM)   LLMUnavailable ─▶ degraded passages
                   │
                   ▼
             CallRecord ─▶ observers (Class 2 monitoring plugs in here)
```

**কেন endpoint `async def` কিন্তু backend plain `def`?** Ollama client, Chroma, sentence-transformers, pypdf — সবই blocking। `async def`-এর ভেতরে সরাসরি ডাকলে event loop আটকে যায় আর সব request (এমনকি `/health`) একটার পর একটা চলে। তাই service layer এগুলো threadpool-এ পাঠায়।

## ⚙️ Configuration (`RAG_` env vars)

| Variable | Default | মানে |
|---|---|---|
| `RAG_BACKEND` | `real` | `fake` = কোনো dependency ছাড়া (tests, demo, benchmark) |
| `RAG_WARMUP` | `true` | startup-এ models load (real backend) |
| `RAG_OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `RAG_MODEL` | `llama3.1:8b` | course-এর একমাত্র model |
| `RAG_NUM_CTX` / `RAG_NUM_PREDICT` | `8192` / `512` | context window / উত্তরের সর্বোচ্চ token |
| `RAG_LLM_TIMEOUT_S` | `120` | এর বেশি হলে degraded উত্তর |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `300` / `60` | characters; overlap শুধু page-এর ভেতরে |
| `RAG_TOP_K` | `4` | প্রতি প্রশ্নে কয়টা passage |
| `RAG_MAX_DISTANCE` | `26.0` | L2 guardrail — নিজের data-তে calibrate করুন (notebook §2) |
| `RAG_MAX_CONCURRENT_LLM` | `1` | `OLLAMA_NUM_PARALLEL`-এর সমান রাখুন |
| `RAG_QUEUE_TIMEOUT_S` | `30` | slot-এর জন্য অপেক্ষা, তারপর 503 |
| `RAG_MAX_UPLOAD_MB` / `RAG_MAX_PAGES` | `20` / `300` | upload limit |
| `RAG_API_KEY` | — | দিলে `/v1/*`-এ `X-API-Key` লাগে |
| `RAG_CACHE_SIZE` | `256` | `0` = cache বন্ধ |
| `RAG_CHROMA_DIR` | `data/chroma_db` | persistent index (git-ignored) |

## 🔌 API

| Method | Path | Success | Errors |
|---|---|---|---|
| GET | `/health` | `200 {"status":"ok"}` | — |
| GET | `/ready` | `200 {status, backend, llm, indexed_documents}` | `503` Ollama/model নেই |
| GET | `/v1/documents` | `200 {"sources": [...]}` | `401` |
| POST | `/v1/documents` (multipart `file`) | `201 {filename, pages, chunks, new_chunks, request_id}` | `401 413 415 422` |
| POST | `/v1/ask` `{question, source?, top_k?}` | `200 {answer, found, degraded, cached, citations[], timings_ms, usage, request_id}` | `401 422 503` |
| POST | `/v1/ask/stream` (same body) | `200 text/event-stream`: `citations`, `token`…, `done` | `401 422 503` |

## 📊 Live Results (MacBook, Ollama 0.40.8, `llama3.1:8b`, warm model)

Sample: `samples/employee_handbook.pdf` (৩ page → ৭ chunk)।

| Check | Result |
|---|---|
| ৭টা on-topic প্রশ্ন (sick days, password, remote, allowance, meals, parental, carry-over) | **৭/৭ সঠিক**, প্রতিটায় `[1]` citation |
| ২টা off-topic প্রশ্ন (France, World Cup) | **২/২ "not found"**, LLM call ০ ms |
| Latency breakdown (`/v1/ask`) | embed ≈10 ms, search ≈1 ms, **LLM ≈1.5–3.4 s (99%+)** |
| Decode speed | ≈19–21 tok/s; prompt ≈370–530 token |
| Prefill, নতুন prompt | ≈1.7–2.4 s (~200 tok/s) → **TTFT ≈2 s** |
| Prefill, prompt prefix আগে দেখা (Ollama KV cache) | ≈50–80 ms → **TTFT ≈0.1 s** |
| Cold start (model unloaded) | startup warmup ছাড়া 1ম request TTFT 8.4 s, warmup দিয়ে ≈2.5 s |
| Same PDF again | `new_chunks: 0` |
| Cache hit | ≈3 ms বনাম ≈1.3 s |
| Ollama unreachable | `/ready` → 503; `/v1/ask` → `degraded: true` + ৪টা passage |
| 6 concurrent, `RAG_QUEUE_TIMEOUT_S=3` | 1 × 200, **5 × 503** (`Retry-After: 3`), ~3 s-এ |

Real benchmark (`--stream`, ৮ request/level, gate = 1):

| concurrency | req/s | p50 s | p95 s | TTFT p50 s |
|---|---|---|---|---|
| 1 | 0.41 | 1.83 | 7.93* | 0.60 |
| 2 | 0.58 | 3.55 | 4.20 | 2.48 |
| 4 | 0.56 | 7.01 | 7.34 | 5.85 |

\* প্রথম request-এ model cold load। Benchmark ৪টা প্রশ্ন ঘুরিয়ে পাঠায়, তাই বেশিরভাগ request Ollama-র prompt-prefix cache পায় — concurrency 1-এ TTFT 0.6 s সেই কারণে; নতুন প্রশ্নে ≈2 s। **পড়ার নিয়ম:** throughput ~0.6 req/s-এ আটকে যায় (একটাই GPU stream); concurrency বাড়ালে শুধু অপেক্ষা (latency, TTFT) বাড়ে। এক Ollama ≈ মিনিটে ~২০-৩৫টা ছোট উত্তর (নতুন প্রশ্নে কম, cached prefix-এ বেশি)। আরও চাইলে: replica, ছোট context, অথবা continuous batching server (vLLM/TGI)।

Fake benchmark (২ slot × ০.২ s): throughput ≈9.7 req/s-এ plateau = 2 ÷ 0.2 s — সূত্র মিলে যায়।

> সংখ্যা hardware আর Ollama version-এ বদলায় — নিজের মেশিনে `bench.py` চালিয়ে মাপুন।

## 🧭 Design Decisions

- **Self-contained RAG core (ADR 0002)** — Week 12 import না করে `app/rag/`-এ ছোট করে আবার লেখা; folder সরালেও ভাঙে না, Docker image-এ শুধু `app/` লাগে।
- **Gate = 1, Ollama-র সমান** — gate 2 দিয়ে মেপে দেখা গেছে throughput বাড়ে না, শুধু queue Ollama-র ভেতরে লুকায় (TTFT 2.4 s, কিন্তু আমাদের metric-এ queue wait 0)। Queue আমাদের layer-এ থাকলে timeout আর 503 সম্ভব।
- **Chunk 300 + page boundary** — 700-char chunk-এ on-topic আর off-topic distance overlap করছিল (gap −0.2); 300-এ gap +2.0 (notebook §2)। Page পার না হওয়ায় citation-এর page সবসময় সঠিক।
- **Exact cache, semantic না** — ভুল "কাছাকাছি" hit আত্মবিশ্বাসী ভুল উত্তর দেয়, আর semantic cache poison করা যায় (NDSS paper, reports দেখুন)।
- **Prompt-এ প্রশ্ন সবার শেষে** — Ollama মিলে যাওয়া prompt prefix-এর KV cache আবার ব্যবহার করে। System prompt + passage আগে, প্রশ্ন শেষে রাখলে একই passage-এর উপর নতুন প্রশ্নে prefill ≈2 s থেকে ≈55 ms-এ নামে (notebook §5)। Usage-এর `prefill_ms` এটা দেখায়।
- **Startup warmup** — lifespan-এ embedding model, Chroma আর Ollama model load (`RAG_WARMUP`); cold Ollama-য় 1ম request 8.4 s → ≈2.5 s।
- **এক process** — bottleneck LLM, Python না; বেশি worker মানে একই GPU-তে আরও queue। Scale = আরও replica + load balancer।

## 🔌 Extension Seam for Class 2 (Monitoring)

প্রতিটা `ask` / `stream` / `ingest` শেষে `RagService` একটা `CallRecord(op, request_id, latency_s, ok, error, result, queue_wait_s, cached)` প্রতিটা observer-কে দেয়। `result.usage`-এ prompt/completion token, prefill/decode ms, TTFT। Observer ভাঙলেও request ভাঙে না (test আছে)।

```python
def log_call(rec):
    if rec.ok and rec.result is not None and getattr(rec.result, "usage", None):
        print(rec.op, rec.request_id, round(rec.latency_s, 2), rec.result.usage.completion_tokens)

service.observers.append(log_call)
```

## ⚠️ Known Limits

- Upload synchronous — বড় PDF-এ response দেরি; production-এ background job + status endpoint।
- Guardrail threshold এই sample-এ calibrate করা; অন্য document-এ আবার মাপুন।
- Prompt injection: passage-কে "data" বলা হয় system prompt-এ, কিন্তু RAG injection পুরো আটকায় না — model-এর হাতে কোনো tool/action নেই, তাই ক্ষতি সীমিত।
- Rate limit per API key নেই (শুধু global concurrency gate); multi-tenant হলে per-key limit (429) আর tenant-ভিত্তিক collection লাগবে।
- In-memory cache আর semaphore — প্রতি process আলাদা; একাধিক replica-য় shared cache (Redis) লাগবে।
