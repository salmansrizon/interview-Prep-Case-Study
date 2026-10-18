# সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **1. Topic: Model Serving with FastAPI — AI Agent-কে REST API বানানো, Async Serving, আর Throughput vs Latency**। এখানে technical term English-এ রাখা হয়েছে, আর explanation Bangla-তে দেওয়া হয়েছে—যাতে concept বোঝা, code পড়া এবং interview-তে explain করা তিনটিই সহজ হয়।

## কেন এই Topic দরকার?

Notebook বা Streamlit-এ চলা একটা RAG bot অন্য কেউ ব্যবহার করতে পারে না — মোবাইল অ্যাপ, ওয়েবসাইট, বা অন্য সার্ভিস তাকে call করবে HTTP দিয়ে। Real-world AI engineer-এর কাজ model বানানোর পর শেষ হয় না; সেটাকে এমন একটা **service** বানাতে হয় যেটা একসাথে অনেক user সামলায়, ভুল input-এ সুন্দরভাবে error দেয়, আর নিজের contract (API docs) নিজেই প্রকাশ করে।

## শেখার সহজ Workflow

Problem বোঝা → Core concept ও intuition → Step-by-step workflow → Practical example → কখন ব্যবহার করবেন বা করবেন না।

প্রতিটি section পড়ার সময় তিনটি প্রশ্ন করুন: **এটি কোন problem solve করে? কীভাবে কাজ করে? Alternative-এর তুলনায় কখন better?**

> **Practical mindset:** benchmark চালানোর আগে অনুমান লিখুন — concurrency ১ থেকে ৬৪ করলে throughput আর latency কোনটা কীভাবে বদলাবে? তারপর ফলাফলের সাথে মেলান।

---

## 1. Topic: Model Serving with FastAPI

Week 12-এ আমরা PDF থেকে প্রশ্নের উত্তর দেওয়া RAG bot বানিয়েছি (`answer_question()`)। এই ক্লাসে সেই একই pipeline-কে `rag-api` নামে একটা **web service** বানাই, যেখানে থাকবে:

* `POST /ask` — প্রশ্ন পাঠাও, citation-সহ উত্তর পাও
* `POST /upload` — PDF আপলোড ও index করা
* `GET /health` — service বেঁচে আছে কিনা
* `/docs` — FastAPI নিজে থেকেই বানানো live Swagger UI

**Project Goal:** Week 12 pipeline-কে কোনো কপি না করে ঝুঁকিহীনভাবে wrap করা, Pydantic schema দিয়ে request/response বাঁধা, error-কে সঠিক HTTP status-এ map করা, আর একটা concurrency-sweep benchmark দিয়ে throughput vs latency মাপা।

---

## 2. Why It Is Related

Week 12 pipeline-এর `answer_question()` একটা সাধারণ Python function। Function থেকে service হতে গেলে তিনটা নতুন প্রশ্ন আসে: (১) বাইরের client কীভাবে call করবে (REST + schema), (২) অনেক client একসাথে এলে কী হবে (async + concurrency), (৩) কত দ্রুত আর কত request সামলাচ্ছে তা কীভাবে মাপব (throughput/latency)। পরের ক্লাস (Class 30) ঠিক এই তৃতীয় প্রশ্নকেই monitoring দিয়ে স্থায়ী রূপ দেবে — তাই এই project-এ শুরু থেকেই একটা পরিষ্কার extension seam রাখা হয়েছে।

---

## 3. How It Works

### 3.1 Agent-কে REST API দিয়ে Wrap করা

```
Client ──HTTP JSON──► FastAPI route ──► RagService._run() ──threadpool──► Week 12 answer_question()
        ◄─────────────  Pydantic response ◄───────────────  CallRecord → observers
```

চারটা স্তর, প্রতিটার একটাই কাজ:

* **Route** (`app/main.py`) — HTTP বোঝে, validation করে, response বানায়। AI-র কিছু জানে না।
* **Service** (`app/service.py`) — প্রতিটা request-এর একমাত্র প্রবেশপথ; blocking call threadpool-এ পাঠায়, latency মাপে।
* **Backend adapter** (`app/backend.py`) — Week 12 pipeline-কে তিনটা sync function (`ask`, `ingest`, `sources`) হিসেবে দেখায়।
* **Schemas** (`app/schemas.py`) — Pydantic v2 model; এগুলোই Swagger-এর contract।

Week 12 কোড আমরা **কপি করিনি, import করেছি** — দুই জায়গায় একই logic থাকলে একটা fix করলে অন্যটা পুরনো থেকে যায়। Week 12 নিজের `config` আর `src.*` নামে import করে, তাই তার root `sys.path`-এর শেষে যোগ করা হয় এবং Chroma data path আমাদের project-এর ভেতরে override করা হয় (Week 12-এর data অক্ষত থাকে)।

```python
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    source: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)

@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request, svc: Svc):
    return await svc.ask(body.question, body.source, body.top_k)
```

`body: AskRequest` লিখলেই FastAPI JSON parse, validate (খালি question → 422) এবং Swagger schema তৈরি করে দেয়।

### 3.2 Async Serving: `async def` vs Blocking Call

FastAPI একটা **event loop**-এ চলে — একটা thread যে অনেক request-এর মধ্যে ঘুরে ঘুরে কাজ করে, যখন একটা `await`-এ অপেক্ষা করে তখন অন্যটা চালায়। সমস্যা: `async def` route-এর ভেতরে blocking call (Ollama, Chroma, embedding model — সবই blocking) সরাসরি চালালে loop আটকে যায়, তখন একটা ৫ সেকেন্ডের `/ask` চলাকালীন `/health`-ও ৫ সেকেন্ড অপেক্ষা করে।

সমাধান:

```python
from fastapi.concurrency import run_in_threadpool
result = await run_in_threadpool(backend.ask, question, source, top_k)
```

Blocking কাজ worker thread-এ যায়, event loop মুক্ত থাকে। `tests/test_api.py::test_slow_ask_does_not_block_health` ঠিক এটাই প্রমাণ করে।

| Route লেখার ধরন | কখন | ফল |
| --- | --- | --- |
| `async def` + `await` async library | কাজ নিজেই async (httpx, async DB) | সবচেয়ে হালকা |
| `async def` + `run_in_threadpool(blocking)` (chosen) | blocking library আছে কিন্তু route-এ `await` দরকার | loop মুক্ত, thread-এ কাজ |
| plain `def` route | পুরো route blocking | FastAPI নিজেই threadpool-এ চালায় |
| `async def` + সরাসরি blocking call | **কখনো না** | পুরো server freeze |

### 3.3 Streaming

LLM উত্তর ধীরে তৈরি হয়; user ৫ সেকেন্ড ফাঁকা পর্দা দেখার বদলে শব্দ শব্দ করে দেখলে **perceived latency** কমে (time-to-first-token)। FastAPI-তে `StreamingResponse` + একটা async generator:

```python
async def gen():
    for word in re.findall(r"\S+\s*", answer):
        yield word
        await asyncio.sleep(0)
return StreamingResponse(gen(), media_type="text/plain")
```

**সৎ সীমাবদ্ধতা:** Week 12-এর `generate()` পুরো উত্তর একসাথে ফেরত দেয়, তাই `/ask/stream` আগে পুরো উত্তর বানায় তারপর stream করে — total latency কমে না। আসল token streaming-এর জন্য `ollama.chat(..., stream=True)` আর generator-কে threadpool থেকে loop-এ আনতে হয় (exercise)।

### 3.4 Throughput vs Latency

* **Latency** — একটা request শুরু থেকে শেষ হতে কত সময় (user-এর অভিজ্ঞতা)। p50 = সাধারণ, p95 = খারাপ দিনের।
* **Throughput** — প্রতি সেকেন্ডে কতটা request শেষ হয় (system-এর ক্ষমতা)।

একটা request ০.২ সেকেন্ডের হলে concurrency ১-এ throughput ≈ ৫ req/s। concurrency বাড়ালে throughput প্রায় সমানুপাতিক বাড়ে — যতক্ষণ না কোনো সম্পদ (threadpool-এর ৪০ thread, Ollama-র GPU/CPU) saturate হয়। তার পর throughput আর বাড়ে না, বরং request-গুলো queue-তে দাঁড়ায় ফলে **latency বাড়ে**। এই বাঁকের জায়গাকে বলে **knee**।

`benchmark/bench.py --fake`-এর একটা বাস্তব run (নকল ০.২s LLM):

| concurrency | req/s | p50 (s) | p95 (s) |
| --- | --- | --- | --- |
| 1 | 4.9 | 0.204 | 0.207 |
| 8 | 36.7 | 0.215 | 0.225 |
| 32 | 130.2 | 0.237 | 0.248 |
| 64 | 131.4 | 0.263 | 0.476 |

৩২ থেকে ৬৪-তে throughput থেমে গেছে (১৩০ → ১৩১) কিন্তু p95 দ্বিগুণ — knee ≈ ৪০ (AnyIO threadpool-এর default আকার)। Real Ollama-তে knee অনেক আগে আসবে, কারণ model নিজেই CPU/GPU-র bottleneck; সেখানে threadpool বাড়ানো নয়, Ollama-র parallelism বা batching দেখতে হবে।

---

## 4. Details & Valid Points

### 4.1 ASGI Server Reference Table

| Server | Use case | Advantage | Limitation |
| --- | --- | --- | --- |
| Uvicorn (chosen) | dev ও single-node production | সহজ, দ্রুত (uvloop) | multi-process নিজে manage করে না |
| Gunicorn + Uvicorn workers | production, multi-core | process supervision, auto-restart | Unix only |
| Hypercorn | HTTP/2, HTTP/3 দরকার হলে | trio support | ecosystem ছোট |
| Granian | Rust-based, কম overhead | নতুন ও দ্রুত | কম battle-tested |

**Worker সংখ্যা:** একাধিক worker মানে একাধিক process — প্রতিটাতে আলাদা embedding model লোড হয় (RAM গুণ হয়)। LLM-bound service-এ আসল bottleneck Ollama, তাই worker বাড়ানোর আগে মাপুন।

### 4.2 LLM Reference Table (Week 12 pipeline যা ব্যবহার করে)

| Model | Use case | Advantage | Limitation |
| --- | --- | --- | --- |
| `llama3.1:8b` (chosen) | সাধারণ RAG answer | ভালো quality, সম্পূর্ণ local | ~8GB RAM, CPU-তে ধীর |
| `llama3.2:3b` | low-RAM machine | দ্রুত, ছোট | reasoning দুর্বল |
| `qwen2.5:7b` | Bangla/multilingual প্রশ্ন | বহুভাষিক ভালো | RAM প্রায় llama-র সমান |
| Cloud API (GPT/Claude) | সর্বোচ্চ quality | local compute লাগে না | cost, data বাইরে যায় |

### 4.3 Error Handling — Failure-কে সঠিক HTTP Status-এ Map করা

| ঘটনা | Status | `error` code |
| --- | --- | --- |
| খালি/অবৈধ JSON | 422 | `validation_error` |
| `.pdf` নয় | 415 | `http_error` |
| ফাইল খুব বড় | 413 | `http_error` |
| পড়া যায় না এমন PDF | 422 | `bad_document` |
| Ollama বন্ধ | 503 | `llm_unavailable` |
| অন্য যেকোনো bug | 500 | `internal_error` (stack trace client-কে দেওয়া হয় না) |

সব error একই `ErrorResponse` shape-এ ফেরে, সাথে `request_id` — client আর log একই ID দিয়ে মেলানো যায়।

### 4.4 Testability: Dependency Override

`get_service` একটা FastAPI dependency। Tests-এ `app.dependency_overrides[get_service] = lambda: fake_service` দিলে Ollama, Chroma, এমনকি Week 12 import ছাড়াই পুরো API পরীক্ষা হয়। `with TestClient(app)` ব্যবহার করলে lifespan (startup/shutdown) ও চলে।

### 4.5 Monitoring Seam (Class 30-এর জন্য)

`RagService._run()` প্রতিটা backend call-এর পর একটা `CallRecord(op, request_id, latency_s, ok, error, result)` বানিয়ে সব `service.observers`-কে দেয়। Class 30 শুধু একটা function append করবে — route বা backend বদলাতে হবে না।

---

## 5. Related Analogy: রেস্টুরেন্টের ওয়েটার

Event loop একজন ওয়েটার। Blocking call মানে ওয়েটার রান্নাঘরের সামনে দাঁড়িয়ে খাবার তৈরি হওয়া দেখছে — বাকি সব টেবিল অপেক্ষা করছে। Threadpool মানে ওয়েটার অর্ডার রান্নাঘরের শেফদের (worker thread) হাতে দিয়ে অন্য টেবিলে চলে যায়। Latency হলো আপনার খাবার আসতে কত সময় লাগল; throughput হলো রেস্টুরেন্ট ঘণ্টায় কত প্লেট পরিবেশন করে। শেফ সবাই ব্যস্ত হয়ে গেলে (knee) আরো অর্ডার নিলে প্লেট বাড়ে না, শুধু অপেক্ষা বাড়ে।

---

## Achievement: `rag-api` Service + Concurrency Benchmark

* `week 15/Class 1 Project/` — চলমান FastAPI service, ৯টা offline test, benchmark script
* `fastapi_serving_lecture.ipynb` — blocking vs threadpool, TestClient, benchmark হাতে-কলমে

চালানো: `uvicorn app.main:app --port 8000` (Ollama চালু রেখে) অথবা `RAG_BACKEND=fake uvicorn app.main:app` — তারপর http://localhost:8000/docs।

---

## 🧠 Brain Teasers & Exercises (নিজে চেষ্টা করুন)

1. `service.py`-তে `run_in_threadpool(fn, ...)`-কে সরাসরি `fn(...)` call বানান। `test_slow_ask_does_not_block_health` কেন fail করে?
2. `--levels 1,8,40,80,160` দিয়ে benchmark চালান। knee কোথায়? `anyio` threadpool limit ১০ করলে knee কোথায় সরে?
3. `/ask/stream`-কে আসল token streaming বানান: `ollama.chat(stream=True)` একটা sync generator — একে async generator-এ আনতে কী লাগবে?
4. একটা observer লিখুন যা প্রতি `ask`-এর latency ও `found` স্টোর করে এবং `/metrics`-এ p95 দেখায় (Class 30-এর পূর্বাভাস)।
5. `/upload` একটা বড় PDF-এ কয়েক সেকেন্ড নেয়। এটাকে background job (task id ফেরত, পরে status poll) বানালে API contract কেমন হবে?
