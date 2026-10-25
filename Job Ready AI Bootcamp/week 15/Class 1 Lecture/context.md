# সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **1. Topic: Serving LLM Apps with FastAPI — Async, Streaming, Backpressure আর Production Contract**। Technical term English-এ রাখা হয়েছে, আর explanation Bangla-তে দেওয়া হয়েছে—যাতে concept বোঝা, code পড়া এবং interview-তে explain করা তিনটিই সহজ হয়।

## কেন এই Topic দরকার?

Notebook বা Streamlit-এ চলা RAG bot অন্য কেউ ব্যবহার করতে পারে না — মোবাইল অ্যাপ, ওয়েবসাইট বা অন্য সার্ভিস তাকে HTTP দিয়ে call করবে। কিন্তু LLM service সাধারণ web API-র মতো না: একটা উত্তর কয়েক সেকেন্ড নেয়, Ollama একসাথে **একটাই** উত্তর বানায়, আর upload করা প্রতিটা PDF আসলে model-এর prompt-এ ঢোকা অচেনা text। তাই সাধারণ web-scaling অভ্যাস (বেশি worker, গভীর queue, উদার retry) এখানে উল্টো ফল দেয়। এই class-এ শিখব কীভাবে একটা local LLM app-কে এমন service বানাতে হয় যেটা খারাপ দিনেও (ভিড়, model down, খারাপ upload) একটা নির্দিষ্ট, সৎ আচরণ করে।

## শেখার সহজ Workflow

Function → REST contract → async বনাম blocking → lifespan → streaming (SSE) → Ollama-র সীমা → backpressure (503) → latency-র গঠন (prefill/decode) → cache → security → reliability → test।

প্রতিটি section পড়ার সময় তিনটি প্রশ্ন করুন: **এটি কোন problem solve করে? কীভাবে কাজ করে? Alternative-এর তুলনায় কখন better?**

> **Practical mindset:** benchmark চালানোর আগে অনুমান লিখুন — concurrency ১ থেকে ৪ করলে throughput আর TTFT কী হবে? তারপর `benchmark/bench.py` চালিয়ে মেলান। এই lecture-এর প্রায় প্রতিটা সংখ্যা project-এ আসল `llama3.1:8b`-এ মাপা।

---

## 1. Topic: Serving LLM Apps with FastAPI

এই ক্লাসে আমরা শিখব:

* **REST contract** — Pydantic v2 schema, `/v1` versioning, structured error, auto Swagger (`/docs`)।
* **Async বনাম blocking** — কেন একটা blocking call সব user-কে লাইনে দাঁড় করায়; threadpool।
* **`lifespan`** — heavy resource একবার load, startup warmup।
* **Streaming** — FastAPI-র built-in Server-Sent Events (`fastapi.sse`)।
* **Backpressure** — Ollama-র একটা slot, `asyncio.Semaphore`, 503 + `Retry-After`।
* **Latency anatomy** — embed, search, prefill, decode; TTFT বনাম total; prompt-prefix cache।
* **Safe caching, security, reliability** — exact cache, OWASP LLM risk, graceful degradation, idempotent ingestion।
* **Testing** — `TestClient`, `dependency_overrides`, fake backend।

**Project Goal (Class 1 Project — Portfolio Project 14: Production-Ready RAG API Service):** Week 12-এর PDF Q&A bot-কে একটা self-contained FastAPI service বানানো — `POST /v1/documents` (safe upload, idempotent), `POST /v1/ask` (citation-সহ উত্তর), `POST /v1/ask/stream` (SSE), `/health` আর `/ready`, semaphore backpressure, LLM down হলে retrieval-only উত্তর, exact-match cache, optional API key — সব `llama3.1:8b` + ChromaDB-তে, offline।

---

## 2. Why It Is Related

Week 12-এ RAG pipeline বানিয়েছি (`answer_question()`), Week 13-14-এ agent। এগুলো সব Python function — একজন user, এক terminal। Function থেকে service হতে তিনটা নতুন প্রশ্ন: (১) বাইরের client কীভাবে call করবে (REST + schema + docs), (২) অনেকে একসাথে এলে কী হবে (async, queue, 503), (৩) কিছু ভাঙলে কী হবে (timeout, degradation, error contract)। পরের ক্লাস (Class 30) এই service-কে **monitor** করবে — তাই project-এ শুরু থেকেই একটা observer seam রাখা হয়েছে: প্রতিটা call শেষে `CallRecord`।

---

## 3. How It Works

### 3.1 Function → Service: Contract আগে

Client আপনার code দেখে না, দেখে **contract**: কোন URL, কী পাঠাতে হবে, কী ফেরত আসবে, কোন error-এ কোন status। FastAPI-তে contract = Pydantic model + route; `/docs` (Swagger) সেটা থেকে নিজেই তৈরি হয়।

```python
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)   # LLM10: input-এর সীমা
    source: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=10)

class AskResponse(BaseModel):
    answer: str
    found: bool          # grounding schema-র অংশ: client prose পার্স করে না
    degraded: bool       # LLM down -> retrieval-only
    cached: bool
    citations: list[Citation]
    timings_ms: dict[str, float]
    usage: Usage | None
    request_id: str
```

* **`/v1/` prefix** — prompt, model বা schema বদলালে উত্তরের আচরণ বদলায়; client একটা version-এ pin করে থাকে।
* **Structured error** — সব error-এর একই আকার `{error, detail, request_id}`; client একটাই parser লেখে। 413 (বড় file), 415 (PDF না), 422 (validation / খারাপ PDF), 401 (key), 503 (ভিড়)।
* **Request ID** — প্রতিটা response-এ `X-Request-ID`; log, trace আর user-এর অভিযোগ এক সুতোয় বাঁধে।

### 3.2 Async বনাম Blocking — সবচেয়ে সাধারণ bug

FastAPI-তে `async def` endpoint সরাসরি event loop-এ চলে; plain `def` endpoint একটা threadpool-এ (AnyIO, default **40** thread)। আপনি নিজে যে helper function ডাকেন, সেটা কখনো thread-এ যায় না। তাই `async def`-এর ভেতরে blocking call (sync Ollama client, Chroma, sentence-transformers, pypdf) পুরো loop আটকে দেয় — অন্য সব request, এমনকি `/health`-ও।

| Endpoint style | ৪টা concurrent request (প্রতিটা 0.5 s sleep) | কেন |
|---|---|---|
| `async def` + `time.sleep(0.5)` | **2.01 s** | loop blocked, একটার পর একটা |
| plain `def` | **0.52 s** | প্রতিটা আলাদা thread |
| `async def` + `await run_in_threadpool(time.sleep, 0.5)` | **0.51 s** | blocking কাজ thread-এ |

(Research report-এ FastAPI 0.162-এ verified।) Project-এ সব endpoint `async def`, আর service layer blocking backend-কে `run_in_threadpool` / `iterate_in_threadpool`-এ পাঠায়:

```python
res = await run_in_threadpool(self.backend.ask, question, source, top_k)
```

### 3.3 `lifespan` — একবার load, আগে থেকে গরম

`@app.on_event("startup")` Starlette 1.0-এ মুছে গেছে (FastAPI এখনো চালায় কিন্তু warning দেয়)। এখন `lifespan`: `yield`-এর আগে startup, পরে shutdown। Embedding model, Chroma client, Ollama client — প্রতি request-এ না, process-প্রতি একবার।

Project-এ lifespan আরও একটা কাজ করে: **warmup**। Ollama model ৫ মিনিট idle থাকলে unload করে; প্রথম request-কে তখন model load-এর দাম দিতে হয়। Ollama-কে খালি prompt পাঠালে সে শুধু model load করে:

```python
self._client().generate(model=self.model, prompt="", keep_alive="30m",
                        options={"num_ctx": self.options["num_ctx"]})   # একই num_ctx, নইলে আবার reload
```

**মাপা ফল (cold Ollama):** warmup ছাড়া প্রথম request-এর TTFT **8.4 s**, warmup দিয়ে **≈2.5 s**। `/ready` তখনই 200 দেয় যখন Ollama reachable আর model pulled।

### 3.4 Streaming: Server-Sent Events

৩ সেকেন্ডের উত্তর একসাথে দিলে user ৩ সেকেন্ড ফাঁকা screen দেখে; token ধরে পাঠালে প্রথম শব্দ আসে কয়েকশো ms-এ। FastAPI 0.135 থেকে SSE built-in (`fastapi.sse`) — `sse-starlette` আর লাগে না। Event-এর ক্রম:

```
event: citations   data: [{"n":1,"source":"employee_handbook.pdf","page":2,...}]   <- retrieval-এর পরেই
event: token       data: "According"
event: token       data: " to"   ...
event: done        data: {"answer": "...", "found": true, "usage": {...}, "timings_ms": {...}}
```

**গুরুত্বপূর্ণ protocol সত্য:** প্রথম byte যাওয়া মানে status 200 commit হয়ে গেছে। তাই 503 (ভিড়), 401 (key) — এই সিদ্ধান্ত **stream শুরুর আগে** নিতে হয়। Project-এ তাই slot আগে নেওয়া হয়, তারপর response:

```python
handle = await svc.open_stream(body.question, ...)     # slot না পেলে এখানেই 503
async def events():
    async for kind, value in handle.events():
        if await request.is_disconnected():          # চলে যাওয়া client-এর জন্য token বানানো বন্ধ
            break
        ...
return EventSourceResponse(events(), background=BackgroundTask(handle.close),
                           headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```

Docs-এর pattern (`response_class=EventSourceResponse` + endpoint নিজেই generator) ছোট, কিন্তু তাতে generator-এর ভেতরের কোড response শুরুর পরে চলে — তখন 503 দেওয়া যায় না। আর FastAPI 0.162-এ `response_class` দিয়ে coroutine endpoint ফেরত দিলে FastAPI সেটাকেই generator ভাবে ("coroutine not iterable") — project বানানোর সময় এটা ধরা পড়েছিল। `handle.close()` idempotent: stream-এর `finally` আর response-এর background task দুই জায়গা থেকে ডাকা হয়, তাই client মাঝপথে চলে গেলেও slot ফেরত আসে (test আছে)।

Browser-এর `EventSource` শুধু GET পারে, custom header পাঠাতে পারে না — POST + `X-API-Key`-এর জন্য `fetch()` + `ReadableStream`। Proxy (nginx) buffering বন্ধ করতে `X-Accel-Buffering: no`।

### 3.5 Ollama-র একটা Slot আর Backpressure

Ollama FAQ: `OLLAMA_NUM_PARALLEL` (model-প্রতি একসাথে কয়টা request) **default 1**; `OLLAMA_MAX_QUEUE` **512** — তারপর 503। মানে ৫০ জন একসাথে এলে ৪৯ জন Ollama-র ভেতরে লাইনে দাঁড়ায়, কোনো error ছাড়াই মিনিটের পর মিনিট। ১০ s-এর উত্তরে ৫১২-তম জন এক ঘণ্টার বেশি অপেক্ষা করবে।

সমাধান: API-র নিজের gate, Ollama-র slot-এর সমান, আর একটা সংক্ষিপ্ত অপেক্ষা:

```python
async def _acquire(self):
    try:
        await asyncio.wait_for(self.gate.acquire(), timeout=self.queue_timeout_s)
    except TimeoutError:
        raise Overloaded("All LLM slots busy; retry later", retry_after_s=...)   # -> 503 + Retry-After
```

**কেন gate = 1, 2 না?** Project-এ মাপা হয়েছে: gate 2 দিলে throughput বাড়ে না (একটাই GPU stream), শুধু দ্বিতীয় request Ollama-র ভেতরে লুকিয়ে অপেক্ষা করে — TTFT 2.4 s, কিন্তু আমাদের metric-এ queue wait 0। Queue আমাদের layer-এ থাকলে timeout দেওয়া যায়, মাপা যায়, 503 দেওয়া যায়।

**মাপা ফল:** `RAG_QUEUE_TIMEOUT_S=3`, ৬টা একসাথে → ১টা 200, **৫টা 503** (`Retry-After: 3`), ~3 s-এ — এক ঘণ্টা পরে না।

429 বনাম 503: 429 = "*আপনি* আপনার quota পার করেছেন" (per-key rate limit); 503 = "*server* ভরা"।

**`--workers 8` কেন খারাপ:** semaphore process-প্রতি, তাই ৮ worker = ৮ গুণ concurrency একই একটা slot-এর বিরুদ্ধে; প্রতিটা worker নিজের embedder আর Chroma load করে (RAM ৮ গুণ, SQLite lock)। Scale = আরও model capacity বা replica, worker না।

### 3.6 `/health` বনাম `/ready`

* `/health` (liveness): process বেঁচে আছে কিনা। কখনো Ollama/Chroma ছোঁয় না — নইলে Ollama down হলে orchestrator পুরো API restart করবে, যেটা কিছুই ঠিক করে না।
* `/ready` (readiness): Ollama reachable **আর** model pulled? না হলে 503 → load balancer traffic পাঠায় না।

---

## 4. Details & Valid Points

### 4.1 ২০২৪-এর Tutorial বনাম ২০২৬-এর FastAPI

FastAPI breaking change দেয় minor 0.x bump-এ (শুধু October 2026-এ ২০টা release) — তাই `requirements.txt`-এ exact version pin।

| বিষয় | ২০২৪ tutorial | এখন (FastAPI 0.162, Starlette 1.9, Uvicorn 0.54) |
|---|---|---|
| Startup/shutdown | `@app.on_event("startup")` | `FastAPI(lifespan=...)`; Starlette 1.0 থেকে `on_event` নেই |
| Token streaming | `sse-starlette` / হাতে `f"data: {x}\n\n"` | `fastapi.sse` (0.135+) |
| JSON body | যেকোনো `Content-Type` | JSON `Content-Type` না থাকলে **422** (0.132+) |
| Credential নেই | 403 | **401** (0.122+) |
| Pydantic | v1 বা v2 | শুধু v2 |
| Process | Gunicorn + `UvicornWorker` | এক `uvicorn`/`fastapi run` process per container |
| Python | 3.8+ | Starlette 1.8+ চায় 3.11; course-এ 3.12 |

`curl`/`fetch`-এ `content-type: application/json` ভুলে গেলে এখন 422 — student-দের সবচেয়ে সাধারণ প্রথম error।

### 4.2 Latency-র গঠন: সময় কোথায় যায়

Ollama প্রতিটা response-এ nanosecond-এ timing দেয়: `load_duration` (> 0 = cold start), `prompt_eval_count/_duration` (prefill), `eval_count/_duration` (decode)। Project প্রতিটা উত্তরে এগুলো `usage`-এ ফেরত দেয়।

**মাপা (MacBook, llama3.1:8b, ৩৭০-৫৩০ prompt token):**

| Stage | সময় |
|---|---|
| Question embed | ≈10 ms |
| Chroma search | ≈1 ms |
| Prefill, নতুন prompt | **≈1.7-2.4 s** (~200 tok/s) |
| Prefill, prefix আগে দেখা | **≈50-80 ms** |
| Decode | ≈19-21 tok/s → ২০ token ≈1 s |
| LLM-এর ভাগ | মোট সময়ের **≈99%** |

* **Prefill** = prompt-এর সব token একবারে পড়া → TTFT ঠিক করে। Context যত লম্বা, তত বেশি।
* **Decode** = প্রতিটা নতুন token-এ সব weight একবার পড়া → memory-bandwidth-bound, tok/s প্রায় স্থির।
* Hosted API-র নিয়ম ("prompt অর্ধেক করলে latency ১-৫% কমে") laptop-এ খাটে না — এখানে retrieved context সেকেন্ডের দাম। **top-k আর chunk size quality-র knob, latency-রও knob।**
* `top_k × chunk + system prompt` যদি `num_ctx` পার হয়, prompt-এর শুরু (প্রায়ই instruction) চুপচাপ কাটা যায়। Ollama-র default context 4096; project `num_ctx=8192` স্পষ্ট দেয়।

### 4.3 Prompt-Prefix Cache: প্রশ্ন সবার শেষে

Ollama আগের prompt-এর KV cache রাখে; নতুন prompt-এর শুরু মিলে গেলে সেই অংশের prefill হয় না (`prompt_eval_duration` শুধু uncached token গোনে)। Project-এর notebook-এ মাপা:

| Case | prefill | TTFT |
|---|---|---|
| ১. নতুন prompt (cold prefix) | 2255 ms | 2271 ms |
| ২. হুবহু একই prompt আবার | 50 ms | 64 ms |
| ৩. একই passage, **শেষে** নতুন প্রশ্ন | 138 ms | 156 ms |
| ৪. নতুন প্রশ্ন **শুরুতে** | 2101 ms | 2118 ms |

তাই prompt-এর ক্রম: system prompt → passage → প্রশ্ন। **Benchmark-এর ফাঁদ:** একই কয়েকটা প্রশ্ন বারবার পাঠালে cache-এর কারণে TTFT মিথ্যা রকম ভালো দেখায় (project-এর benchmark-এ concurrency 1-এ TTFT 0.6 s, নতুন প্রশ্নে ≈2 s)। Project বানানোর সময় এটা প্রথমে "server-এ ২ s overhead" bug মনে হয়েছিল — আসলে ছিল cache hit বনাম miss।

### 4.4 Capacity Math

Little's law: L = λW। একটা slot, W ≈ 1.7 s → সর্বোচ্চ ≈0.6 req/s। Project-এর real benchmark (gate 1, ৮ request/level):

| concurrency | req/s | p50 s | TTFT p50 s |
|---|---|---|---|
| 1 | 0.41 | 1.83 | 0.60 |
| 2 | 0.58 | 3.55 | 2.48 |
| 4 | 0.56 | 7.01 | 5.85 |

Throughput ~0.6-তে আটকে; concurrency বাড়লে শুধু অপেক্ষা বাড়ে। এক Ollama ≈ মিনিটে ২০-৩৫টা ছোট উত্তর। ৮০% utilization-এ queueing theory বলে অপেক্ষা বিস্ফোরিত হয় — ৫০-৭০%-এ plan করুন, বাকি shed করুন।

### 4.5 Serving Engine Reference Table

| Engine | কখন | সুবিধা | সীমা |
|---|---|---|---|
| **Ollama** (course) | dev, এক user, classroom | install সহজ, GGUF, Apple GPU | default ১ slot; continuous batching নেই |
| **llama.cpp `llama-server`** | GGUF/CPU/Apple-এ থাকতেই হবে | explicit slot, continuous batching, `/metrics`, grammar JSON | কম tooling |
| **vLLM** | NVIDIA/ROCm GPU, আসল concurrency, p95 SLO | PagedAttention (একই latency-তে 2-4× throughput), continuous batching | CUDA GPU, setup ভারী |
| **TGI** | Hugging Face stack | production feature | ২০২৬-এর maintenance status verify করা হয়নি |
| Cloud API | scale, quality | কোনো GPU নেই | খরচ, data বাইরে যায় |

### 4.6 Cache: কোনটা নিরাপদ

| Cache | নিরাপদ? | কেন |
|---|---|---|
| Model warm (`keep_alive`, warmup) | ✅ | cold-load spike p99 থেকে সরে |
| Prompt prefix (Ollama নিজে) | ✅ | stable অংশ আগে রাখলেই |
| Embedding by content hash | ✅ | একই chunk আবার embed হয় না |
| Exact answer cache, key-তে `corpus_version` | ✅ | নতুন document এলে নিজে থেকে বাতিল |
| **Semantic answer cache** | ❌ default-এ | "কাছাকাছি" প্রশ্নে ভুল উত্তর; NDSS paper দেখিয়েছে poison করা যায় — অন্য user attacker-এর উত্তর পায় |

Project: exact cache, key = (normalized question, source, top_k, corpus_version); degraded উত্তর cache হয় না। মাপা: cache hit ≈3 ms বনাম ≈1.3 s।

### 4.7 Security: প্রতিটা PDF একটা অচেনা Prompt

* **LLM01 indirect prompt injection** — OWASP নিজেই বলে RAG এটা পুরো আটকায় না, নিশ্চিত প্রতিকার নেই। তাই লক্ষ্য ক্ষতি সীমিত রাখা: model-এর হাতে কোনো tool/action নেই; system prompt passage-কে "data, instruction না" বলে; citation code-এ যাচাই; কোন document থেকে কোন chunk সেটা রাখা যাতে খারাপ document মুছে ফেলা যায়।
* **LLM08 vector store** — multi-tenant হলে tenant id **auth থেকে**, কখনো request body থেকে না; tenant-প্রতি collection বা বাধ্যতামূলক `where` filter, আর একটা test যে tenant B কখনো A-র chunk পায় না।
* **LLM10 / API4 unbounded consumption** — প্রশ্নের দৈর্ঘ্য, `top_k`, `num_predict`, `num_ctx`, upload size, page count, concurrency — সবকিছুর সীমা।
* **Upload** — extension (`.pdf` → না হলে 415), `%PDF-` magic bytes (client-এর content-type বিশ্বাস করবেন না), `read(limit+1)` দিয়ে 413, page cap, text layer না থাকলে পরিষ্কার error। Starlette 1.6-এর `max_body_size` FastAPI 0.162 expose করে না — আসল byte limit reverse proxy-তে (nginx `client_max_body_size`)।
* **API key** — `APIKeyHeader(auto_error=False)` + নিজে 401; `/health`/`/ready` খোলা রাখুন (orchestrator-এর জন্য)। Ollama-র নিজের কোনো auth নেই — তার port কখনো public না।

### 4.8 Reliability

* **Graceful degradation** — Ollama down/timeout হলে 500 না: `degraded: true` + সবচেয়ে প্রাসঙ্গিক passage। Retrieval-এর LLM লাগে না। মাপা: Ollama unreachable → `/ready` 503, `/v1/ask` → degraded + ৪টা passage।
* **Idempotent ingestion** — chunk id = `sha256(source + "\0" + text)[:16]` + Chroma `upsert` → একই file আবার upload = `new_chunks: 0`। Client timeout-এর পর retry খুব সাধারণ; duplicate chunk top-k ভরিয়ে ফেলত।
* **Grounding guardrail, calibrate করা** — সবচেয়ে কাছের chunk-ও `max_distance`-এর বাইরে → LLM না ডেকেই "not found"। Threshold মাপতে হয়। Project-এ ৭টা on-topic আর ৪টা off-topic প্রশ্নে:

| chunk size | chunks | on-topic সবচেয়ে খারাপ | off-topic সবচেয়ে ভালো | gap |
|---|---|---|---|---|
| 700 | 3 | 25.8 | 25.6 | **-0.2** (আলাদা করা অসম্ভব) |
| 400 | 6 | 25.0 | 25.9 | 0.8 |
| 300 | 7 | 25.2 | 27.2 | 2.0 |
| 200 | 12 | 23.5 | 27.3 | 3.8 |

বড় chunk অনেক topic মিশিয়ে embedding ঘোলা করে। Project: 300/60, `max_distance=26.0`। আর chunk কখনো page পার হয় না, তাই citation-এর page সবসময় সঠিক (প্রথম version-এ overlap পরের page-এ ঢুকে "page 2" citation-এ page 1-এর text দেখাত)।

* **Retry** — শুধু connection error, 429, 503; সর্বোচ্চ ২ বার, jitter-সহ; token stream শুরু হওয়ার পর কখনো না।

### 4.9 Testing

* `with TestClient(app) as c:` lifespan চালায়; `app.dependency_overrides[get_service] = lambda: fake_svc` backend বদলায়; `c.stream("POST", "/v1/ask/stream", ...)` SSE পড়ে।
* `httpx.ASGITransport` **lifespan চালায় না** — concurrency test-এ (503) settings হাতে বসাতে হয় বা `LifespanManager`।
* Project: ২৪টা offline test — real FastAPI, real ChromaDB (temp dir), real PDF (fpdf2); শুধু LLM আর embedder নকল। 503, SSE order, early disconnect-এ slot release, cache invalidation, 401/413/415/422 সব।
* Load test-এর ফাঁদ: tool-এর default "response time" = stream-এর শেষ; TTFT আলাদা মাপতে হয়।

---

## 5. Related Analogy: এক রাঁধুনির রেস্তোরাঁ

একটা ছোট রেস্তোরাঁয় **একজনই রাঁধুনি** (Ollama, এক slot)। প্রতিটা dish ২-৩ মিনিট।

* **Menu card** (OpenAPI `/docs`) — কী অর্ডার করা যায়, কী আসবে।
* **দরজার host** (semaphore) — রাঁধুনি ব্যস্ত হলে নতুন অতিথিকে বলে "১০ মিনিট পরে আসুন" (503 + `Retry-After`)। host না থাকলে ৫১২ জন রান্নাঘরের দরজায় লাইন দিত, কেউ জানত না কতক্ষণ।
* **আরও ওয়েটার রাখা** (`--workers 8`) — রাঁধুনি একজনই; ৮ জন ওয়েটার ৮ গুণ অর্ডার নিয়ে রান্নাঘরে ভিড় করে, dish দ্রুত হয় না।
* **Appetizer আগে** (`citations` event) — main dish আসার আগে টেবিলে কিছু; user দেখে কাজ চলছে।
* **রাঁধুনি অসুস্থ** (Ollama down) — রেস্তোরাঁ বন্ধ না করে রেডিমেড খাবার (retrieval-only passage) দেওয়া, আর সেটা স্পষ্ট বলা (`degraded: true`)।
* **চুলা আগে গরম** (warmup) — প্রথম অতিথিকে চুলা জ্বালানোর অপেক্ষা করতে হয় না।
* **"আগের মতোই"** (exact cache) — ঠিক একই অর্ডার হলে আগের রান্না থেকে; "কাছাকাছি" অর্ডারে (semantic cache) ভুল dish যাওয়ার ঝুঁকি।

---

## 6. Common Pitfalls

| Pitfall | কী হয় | Fix |
|---|---|---|
| `async def`-এ sync Ollama/Chroma/embedder | event loop block, ৪ request 2.01 s | `run_in_threadpool`, `AsyncClient`, বা plain `def` |
| `@app.on_event`, `sse-starlette`, Gunicorn+UvicornWorker | deprecated/legacy | `lifespan`, `fastapi.sse`, এক process |
| Client `Content-Type` দেয় না | 422 (0.132+) | client ঠিক করুন |
| "Ollama নিজেই parallel চালাবে" | `NUM_PARALLEL=1`, ৫১২-গভীর queue | semaphore = slot + দ্রুত 503 |
| Gate > Ollama slot | queue Ollama-র ভেতরে লুকায়, TTFT বাড়ে, throughput না | gate = `OLLAMA_NUM_PARALLEL` |
| Stream শুরু করে তারপর slot চাওয়া | 200 commit হয়ে গেছে, 503 অসম্ভব | slot আগে, response পরে |
| `--workers N` | process-প্রতি semaphore, N× RAM | এক worker, model capacity বাড়ান |
| Context চুপচাপ কাটা | top-k × chunk > `num_ctx` | token গুনুন, `num_ctx` স্পষ্ট |
| প্রথম request ধীর | model unload (৫ মিনিট idle) | lifespan warmup, `keep_alive` |
| একই প্রশ্নে benchmark | prefix cache → TTFT মিথ্যা ভালো | নতুন প্রশ্ন, cache-hit আলাদা রিপোর্ট |
| Guardrail threshold অন্য project থেকে কপি | chunk size বদলালে distance বণ্টন বদলায় | নিজের data-য় calibrate |
| Chunk page পার হয় | citation-এর page ভুল | page boundary-তে chunk শেষ |
| Semantic cache | ভুল/poisoned উত্তর | exact + corpus_version |
| `ASGITransport` test-এ lifespan ধরে নেওয়া | lifespan চলে না | `LifespanManager` বা state হাতে |
| Shell-form `CMD` Docker-এ | signal পৌঁছায় না, shutdown চলে না | exec-form `CMD [...]` |

---

## 7. Interview Q&A

1. **RAG endpoint `async def` করার পর load-এ latency খারাপ হলো — কেন?** Endpoint blocking library (sync Ollama, Chroma, sentence-transformers) সরাসরি event loop-এ ডাকছে; প্রতিটা call অন্য সব request আর stream আটকায়। `run_in_threadpool`, `ollama.AsyncClient`, বা endpoint-কে plain `def` করুন। Mixing ঠিক আছে; bug শুধু `async def`-এর ভেতরে blocking।
2. **৫০ জন user এক GPU-র এক Ollama-য় — কী হবে, কীভাবে design করবেন?** Ollama default-এ এক request চালায়, ৫১২ পর্যন্ত queue করে — user বিশাল TTFT দেখে, error ছাড়া। API-তে slot-এর সমান semaphore, ছোট অপেক্ষা, তারপর 503 + `Retry-After`; per-key 429; `num_predict` সীমা; deadline; streaming যাতে অগ্রগতি দেখা যায়।
3. **`--workers 8` দিলে দ্রুত হবে?** না। Bottleneck এক GPU-র এক model। Worker RAM গুণ করে, semaphore-কে ৮ গুণ limit বানায়, SQLite lock বাড়ায়। Model capacity বাড়ান: `NUM_PARALLEL`, replica, vLLM।
4. **Streaming কীভাবে, কী ভাঙে?** SSE: `citations` → `token`… → `done`। ভাঙে proxy buffering-এ (`X-Accel-Buffering: no`), পুরোনো gzip-এ, চলে যাওয়া client-এর জন্য generation চালিয়ে গেলে (`is_disconnected()`)। `EventSource` GET-only → POST-এ `fetch`। প্রথম byte-এ status commit — তাই error পরে `event: error` হিসেবে, আর 503/401 stream-এর আগে।
5. **RAG request-এ সময় কোথায় যায়?** Embed + vector search কয়েক ms; তারপর optional model load, prefill (prompt token-এর অনুপাতে → TTFT), decode (output token-এর অনুপাতে, bandwidth-bound → total)। Ollama-র `load/prompt_eval/eval_duration` আর নিজের timer দিয়ে মাপুন, p50/p95।
6. **TTFT 8 s, decode ঠিক আছে — কী করবেন?** প্রথমে `load_duration` দেখুন — > 0 মানে cold start → warmup, `keep_alive`। না হলে prefill: top-k কমান, chunk ছোট, rerank করে কম রাখুন, stable অংশ আগে (prefix cache)। Laptop-এ prefill কয়েকশো tok/s।
7. **Prefix cache কী, prompt design-এ কী প্রভাব?** Ollama আগের prompt-এর KV cache রাখে; নতুন prompt-এর শুরু মিললে সেই অংশ আবার prefill হয় না। তাই system → passage → প্রশ্ন। আমাদের মাপে নতুন prefix 2.2 s, একই passage-এ শেষে নতুন প্রশ্ন 0.14 s। Benchmark-এ একই প্রশ্ন ঘুরালে latency মিথ্যা ভালো দেখায়।
8. **একটা উত্তর 10 s, traffic মিনিটে ২০ — এক Ollama পারবে?** না: capacity মিনিটে ৬, utilization > 1, queue অসীম বাড়ে। এখনই 429/503-এ shed; তারপর parallelism, ছোট output, cache, ভালো hardware বা continuous batching।
9. **Continuous batching আর PagedAttention কী?** Continuous batching প্রতি decode step-এ sequence যোগ/বাদ দেয় — সবচেয়ে ধীরটার জন্য অপেক্ষা নেই। PagedAttention KV cache-কে virtual memory-র page-এর মতো block-এ রাখে, fragmentation নেই, বেশি sequence আঁটে। vLLM একই latency-তে 2-4× throughput রিপোর্ট করে।
10. **Semantic cache দেবেন?** Default-এ না। আগে exact, embedding, retrieval cache। Semantic answer cache কাছাকাছি প্রশ্নে ভুল উত্তর দেয়, re-ingestion-এর পর stale, tenant পার হয়, আর ইচ্ছাকৃতভাবে poison করা যায় (NDSS)। লাগলে tenant + corpus version-এ scope, উঁচু threshold, প্রতিটা hit audit।
11. **খারাপ PDF কীভাবে উত্তর hijack করা থেকে আটকাবেন?** পুরো আটকানো যায় না (LLM01)। ক্ষতি সীমিত: model-এর tool নেই, passage delimited আর "data" হিসেবে চিহ্নিত, hidden text strip, output schema, citation retrieved id-র subset কিনা code-এ check, provenance রাখা যাতে মুছা যায়, tenant isolation।
12. **Ollama down — API কী ফেরত দেয়?** Retrieval-only: top-k passage + citation, `degraded: true`, documented status (project-এ 200)। `/ready` 503 দেখায়। Retry সীমিত, jitter-সহ, শুধু retryable error-এ।
13. **Ingestion idempotent কীভাবে?** Content-hash chunk id + `upsert`; একই file = কোনো নতুন chunk না। চাইলে `Idempotency-Key` header আর doc-hash table। Embedding model version collection-এ রাখুন, বদলালে সব re-embed।
14. **"শুধু document থেকে উত্তর" contract-এ কীভাবে বাঁধবেন?** Score threshold-এর নিচে LLM না ডেকে `found: false`; schema-তে `answer`, `found`, `citations`; code-এ citation ⊆ retrieved id; validation fail-এ retrieval-only। Threshold নিজের data-য় calibrate — আমাদের data-য় chunk 700-এ কোনো threshold-ই কাজ করেনি।
15. **GPU ছাড়া streaming endpoint test কীভাবে? Container কেন `localhost:11434` পায় না?** LLM/backend dependency override করে fake generator, `TestClient.stream()` দিয়ে SSE frame assert; async test-এ `ASGITransport` + `LifespanManager`। Container-এ `localhost` = container নিজে → `host.docker.internal` (Linux-এ `--add-host=host.docker.internal:host-gateway`)।
16. **`/health` আর `/ready` আলাদা কেন?** Liveness fail → orchestrator restart করে; readiness fail → শুধু traffic বন্ধ। Ollama down হলে API restart কিছু ঠিক করে না, তাই সেটা readiness-এ, liveness-এ না।

---

## Achievement: Production-Ready RAG API Service (Portfolio Project 14)

**`Class 1 Project/`** — self-contained RAG core (ADR 0002: Week 12 import না), FastAPI service: `/v1/documents` (safe, idempotent upload), `/v1/ask`, `/v1/ask/stream` (SSE, slot আগে), `/health` + `/ready`, semaphore backpressure (503 + `Retry-After`), graceful degradation, exact cache + corpus version, optional API key, request id + timing header, startup warmup। Concurrency benchmark (req/s, p50/p95/p99, TTFT, 503), Dockerfile, ২৪টা offline test, live exploration notebook। Live: ৭/৭ on-topic সঠিক, ২/২ off-topic LLM ছাড়াই "not found"। এই ক্লাসের companion notebook **`fastapi_serving_lecture.ipynb`** সম্পূর্ণ self-contained — project folder ছাড়াই চলে।

---

## 🧠 Brain Teasers & Exercises (নিজে চেষ্টা করুন)

1. **Blocking Detective:** একটা `async def /slow` endpoint বানান যেটা `time.sleep(1)` করে। একসাথে ৫টা `/slow` আর ১টা `/health` পাঠান। `/health` কত সময় নেয়? `run_in_threadpool` দিয়ে আবার মাপুন।
2. **Gate Experiment:** `RAG_MAX_CONCURRENT_LLM=2` দিয়ে server চালিয়ে `bench.py --stream --levels 2` চালান, তারপর `=1` দিয়ে। Throughput আর TTFT-এর পার্থক্য ব্যাখ্যা করুন।
3. **Cache Honesty:** benchmark-এর `QUESTIONS` list-এ প্রতিটা প্রশ্নে একটা random শব্দ **শুরুতে** যোগ করুন। TTFT কী হয়? কেন?
4. **Threshold Trade-off:** "Is MFA required for the VPN?" প্রশ্নটা কেন "not found" হয়? `max_distance` কত করলে ঢোকে, আর তখন off-topic প্রশ্নগুলো কী হয়?
5. **Mid-stream Failure:** Stream চলাকালীন Ollama বন্ধ করলে client কী পায়? এখন যা পায় সেটা কি ভালো contract? একটা `event: error` design করুন।
6. **Little's Law:** আপনার মেশিনে গড় উত্তর-সময় মাপুন। ১০০ জন user প্রতি ৫ মিনিটে একটা প্রশ্ন করলে এক Ollama কি পারবে? কত replica লাগবে ৬০% utilization-এ?

---

## References

* FastAPI release notes — https://fastapi.tiangolo.com/release-notes/ · Concurrency and async/await — https://fastapi.tiangolo.com/async/ · Server-Sent Events — https://fastapi.tiangolo.com/tutorial/server-sent-events/ · `fastapi.sse` reference — https://fastapi.tiangolo.com/reference/sse/ · Strict Content-Type — https://fastapi.tiangolo.com/advanced/strict-content-type/
* FastAPI Testing — https://fastapi.tiangolo.com/tutorial/testing/ · Testing dependencies — https://fastapi.tiangolo.com/advanced/testing-dependencies/ · Async tests — https://fastapi.tiangolo.com/advanced/async-tests/ · Server workers — https://fastapi.tiangolo.com/deployment/server-workers/ · Docker — https://fastapi.tiangolo.com/deployment/docker/
* Starlette release notes — https://starlette.dev/release-notes/ · Uvicorn release notes — https://uvicorn.dev/release-notes/ · Uvicorn settings — https://uvicorn.dev/settings/ · AnyIO threads — https://anyio.readthedocs.io/en/stable/threads.html
* Ollama FAQ (NUM_PARALLEL, MAX_QUEUE, keep_alive) — https://docs.ollama.com/faq · Ollama API generate (timing fields) — https://docs.ollama.com/api/generate · ollama-python — https://github.com/ollama/ollama-python
* llama.cpp Apple Silicon benchmarks — https://github.com/ggml-org/llama.cpp/discussions/4167 · llama.cpp server — https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
* vLLM / PagedAttention (Kwon et al.) — https://arxiv.org/abs/2309.06180 · Anyscale continuous batching — https://www.anyscale.com/blog/continuous-batching-llm-inference
* OpenAI latency optimization guide — https://developers.openai.com/api/docs/guides/latency-optimization · Little's law — https://en.wikipedia.org/wiki/Little%27s_law · Google SRE book, monitoring — https://sre.google/sre-book/monitoring-distributed-systems/
* NDSS semantic cache poisoning — https://www.ndss-symposium.org/ndss-paper/when-cache-poisoning-meets-llm-systems-semantic-cache-poisoning-and-its-countermeasures/
* OWASP LLM01:2025 — https://genai.owasp.org/llmrisk/llm01-prompt-injection/ · LLM08:2025 — https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/ · LLM10:2025 — https://genai.owasp.org/llmrisk/llm102025-unbounded-consumption/ · OWASP API Top 10 2023 — https://owasp.org/API-Security/editions/2023/en/0x11-t10/
* Chroma architecture — https://docs.trychroma.com/docs/overview/architecture · Docker Desktop networking — https://docs.docker.com/desktop/features/networking/networking-how-tos/
* Course research report: `reports/Serving LLM apps with FastAPI.md`
